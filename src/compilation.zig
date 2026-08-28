//! Staged compilation pipeline. Each Alloy source file is a module. Modules
//! run their per-module stages independently (tokenize, then parse), which
//! keeps them parallelizable, and a stage must complete without errors before
//! its output is handed to the next stage. Imports discovered while parsing
//! are loaded through the supplied loader and run the same per-module stages.
//! After the per-module stages the modules merge into one compilation unit
//! for the whole-program stages: name resolution, type checking, and then
//! execution through the tree-walking interpreter ('alloyc run').

const std = @import("std");
const tokenizer_module = @import("tokenizer.zig");
const Token = tokenizer_module.Token;
const Tokenizer = tokenizer_module.Tokenizer;
const Diagnostic = @import("diagnostic.zig").Diagnostic;
const ast = @import("ast.zig");
const Parser = @import("parser.zig").Parser;
const resolution = @import("resolution.zig");
const Checker = @import("checker.zig").Checker;
const types = @import("types.zig");
const Interpreter = @import("interpreter.zig").Interpreter;
const Codegen = @import("codegen.zig").Codegen;
const library_format = @import("library.zig");

/// The version stamped into every .alloylib; a mismatched payload falls
/// back to recompiling the embedded source (section 6.4)
pub const compiler_version = "0.1.0";

/// Loads the source of an imported module given its relative file path
/// ('std/vec.alloy'). Returns null when the file does not exist; the source
/// must be allocated with the passed allocator. The optional library hook
/// fetches '.alloylib' container bytes for 'pkg::' imports by package name.
pub const ModuleLoader = struct {
    context: ?*anyopaque,
    function: *const fn (context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8,
    library: ?*const fn (context: ?*anyopaque, allocator: std.mem.Allocator, package_name: []const u8) anyerror!?[]const u8 = null,
};

pub const Module = struct {
    path: []const u8,
    source: []const u8,
    // canonical import key ('std::option'), null for the entry module
    key: ?[]const u8 = null,
    // the package this module arrived from, null inside the executable's
    // own unit; only 'exp' symbols cross a library boundary (section 6.4)
    library: ?[]const u8 = null,
    tokens: std.ArrayList(Token),
    ast: ?ast.Ast = null,

    // stage 1: source to token list. error tokens become diagnostics and are
    // kept out of the output so later stages only ever see well-formed tokens
    fn tokenize(module: *Module, allocator: std.mem.Allocator, diagnostics: *std.ArrayList(Diagnostic)) !void {
        var tokenizer = Tokenizer.init(module.source);
        while (true) {
            const token = tokenizer.next();
            if (token.tag.errorMessage()) |message| {
                try diagnostics.append(allocator, .{
                    .path = module.path,
                    .source = module.source,
                    .span = token.location,
                    .message = message,
                });
                continue;
            }
            try module.tokens.append(allocator, token);
            if (token.tag == .end_of_file) return;
        }
    }

    // stage 2: token list to abstract syntax tree. a parse error becomes a
    // diagnostic; the partial tree's arena is kept so the message stays alive
    fn parse(module: *Module, allocator: std.mem.Allocator, diagnostics: *std.ArrayList(Diagnostic)) !void {
        module.ast = ast.Ast.init(allocator);
        var parser = Parser.init(module.ast.?.arena.allocator(), module.source, module.tokens.items);
        const parsed = parser.parseModule() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ParseError => {
                const failure = parser.failure.?;
                try diagnostics.append(allocator, .{
                    .path = module.path,
                    .source = module.source,
                    .span = failure.span,
                    .message = failure.message,
                });
                return;
            },
        };
        module.ast.?.module = parsed;
    }
};

// a lock-guarded allocator wrapper for the parallel front-end stages: the
// standard library's blocking mutex needs an Io handle the compilation
// does not carry, so a spinlock guards the child allocator instead (the
// stages allocate in amortized chunks, and batches are small)
pub const LockedAllocator = struct {
    child: std.mem.Allocator,
    flag: std.atomic.Value(bool) = .init(false),

    pub fn allocator(self: *LockedAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn lock(self: *LockedAllocator) void {
        while (self.flag.swap(true, .acquire)) {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(self: *LockedAllocator) void {
        self.flag.store(false, .release);
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(context: *anyopaque, length: usize, alignment: std.mem.Alignment, return_address: usize) ?[*]u8 {
        const self: *LockedAllocator = @ptrCast(@alignCast(context));
        self.lock();
        defer self.unlock();
        return self.child.rawAlloc(length, alignment, return_address);
    }

    fn resize(context: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_length: usize, return_address: usize) bool {
        const self: *LockedAllocator = @ptrCast(@alignCast(context));
        self.lock();
        defer self.unlock();
        return self.child.rawResize(memory, alignment, new_length, return_address);
    }

    fn remap(context: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_length: usize, return_address: usize) ?[*]u8 {
        const self: *LockedAllocator = @ptrCast(@alignCast(context));
        self.lock();
        defer self.unlock();
        return self.child.rawRemap(memory, alignment, new_length, return_address);
    }

    fn free(context: *anyopaque, memory: []u8, alignment: std.mem.Alignment, return_address: usize) void {
        const self: *LockedAllocator = @ptrCast(@alignCast(context));
        self.lock();
        defer self.unlock();
        self.child.rawFree(memory, alignment, return_address);
    }
};

pub const Compilation = struct {
    allocator: std.mem.Allocator,
    // wraps the caller's allocator for the parallel front-end stages; lives
    // on the compilation because module arenas keep it as their child
    thread_safe: LockedAllocator,
    // cross-stage allocations: imported sources, paths, keys, the merged unit
    arena: std.heap.ArenaAllocator,
    modules: std.ArrayList(Module),
    diagnostics: std.ArrayList(Diagnostic),
    // canonical keys of every loaded import, for dedup
    loaded_keys: std.StringHashMapUnmanaged(void),
    merged: ?resolution.MergedUnit = null,
    views: []resolution.ModuleView = &.{},
    // checker outputs consumed by the interpreter and later code generation
    expression_types: std.AutoHashMapUnmanaged(*const ast.Expression, *const types.Type) = .empty,
    call_targets: std.AutoHashMapUnmanaged(*const ast.Expression, resolution.Symbol) = .empty,
    call_type_bindings: std.AutoHashMapUnmanaged(*const ast.Expression, []const types.Type.Binding) = .empty,
    comptime_values: std.AutoHashMapUnmanaged(*const ast.Expression, Interpreter.Value) = .empty,
    cast_shapes: std.AutoHashMapUnmanaged(*const ast.Expression, types.CastShapes) = .empty,
    type_targets: std.AutoHashMapUnmanaged(*const ast.Expression, types.TypeIdentity) = .empty,
    // the checker outlives its stage: code generation queries it for
    // section 4.9 layouts instead of re-deriving them
    checker: ?*Checker = null,
    // the io handle behind '#read_file' (section 7.4); null keeps
    // compile-time evaluation filesystem-free (the hermetic-test default)
    comptime_io: ?std.Io = null,
    // the message of the last interpreter runtime fault
    fault: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator) Compilation {
        return .{
            .allocator = allocator,
            .thread_safe = .{ .child = allocator },
            .arena = std.heap.ArenaAllocator.init(allocator),
            .modules = .empty,
            .diagnostics = .empty,
            .loaded_keys = .empty,
        };
    }

    pub fn deinit(compilation: *Compilation) void {
        for (compilation.modules.items) |*module| {
            module.tokens.deinit(compilation.allocator);
            if (module.ast) |*tree| tree.deinit();
        }
        compilation.modules.deinit(compilation.allocator);
        compilation.diagnostics.deinit(compilation.allocator);
        compilation.arena.deinit();
    }

    /// The module borrows `path` and `source`; the caller keeps ownership.
    pub fn addModule(compilation: *Compilation, path: []const u8, source: []const u8) !*Module {
        const module = try compilation.modules.addOne(compilation.allocator);
        module.* = .{
            .path = path,
            .source = source,
            .tokens = .empty,
        };
        return module;
    }

    /// Registers a module under its canonical import key. All slices must
    /// outlive the compilation (typically allocated in its arena).
    pub fn addImportedModule(compilation: *Compilation, key: []const u8, path: []const u8, source: []const u8) !*Module {
        const module = try compilation.addModule(path, source);
        module.key = key;
        try compilation.loaded_keys.put(compilation.arena.allocator(), key, {});
        return module;
    }

    /// Runs the pipeline. Returns true when every stage completed without
    /// errors; on false the collected diagnostics describe what failed.
    pub fn run(compilation: *Compilation, loader: ?ModuleLoader) !bool {
        // per-module stages: modules are independent of each other here and
        // can later be distributed across threads. each batch of newly added
        // modules is staged, then its imports are loaded as the next batch
        var processed: usize = 0;
        while (processed < compilation.modules.items.len) {
            const batch_start = processed;
            const batch_end = compilation.modules.items.len;
            processed = batch_end;

            try compilation.runFrontEndStage(Module.tokenize, batch_start, batch_end);
            if (compilation.diagnostics.items.len != 0) return false;

            try compilation.runFrontEndStage(Module.parse, batch_start, batch_end);
            if (compilation.diagnostics.items.len != 0) return false;

            try compilation.loadImports(batch_start, batch_end, loader);
            if (compilation.diagnostics.items.len != 0) return false;
        }

        // merge point: from here on, the whole-program stages (name
        // resolution, type checking, later code generation) operate on
        // the single merged unit
        try compilation.resolveNames();
        if (compilation.diagnostics.items.len != 0) return false;

        try compilation.checkTypes();
        return compilation.diagnostics.items.len == 0;
    }

    // a per-module front-end stage fans out across threads; every worker
    // collects its own diagnostics, merged back in module order so the
    // rendered output stays deterministic
    fn runFrontEndStage(
        compilation: *Compilation,
        comptime stage: fn (*Module, std.mem.Allocator, *std.ArrayList(Diagnostic)) anyerror!void,
        batch_start: usize,
        batch_end: usize,
    ) !void {
        const batch = compilation.modules.items[batch_start..batch_end];
        if (batch.len <= 1) {
            for (batch) |*module| {
                try stage(module, compilation.allocator, &compilation.diagnostics);
            }
            return;
        }
        const worker_allocator = compilation.thread_safe.allocator();
        const Worker = struct {
            module: *Module,
            allocator: std.mem.Allocator,
            diagnostics: std.ArrayList(Diagnostic) = .empty,
            failure: ?anyerror = null,

            fn run(worker: *@This()) void {
                stage(worker.module, worker.allocator, &worker.diagnostics) catch |err| {
                    worker.failure = err;
                };
            }
        };
        const workers = try compilation.allocator.alloc(Worker, batch.len);
        defer compilation.allocator.free(workers);
        for (batch, workers) |*module, *worker| {
            worker.* = .{ .module = module, .allocator = worker_allocator };
        }
        const threads = try compilation.allocator.alloc(std.Thread, batch.len);
        defer compilation.allocator.free(threads);
        var spawned: usize = 0;
        for (workers) |*worker| {
            // a spawn failure degrades to inline execution: every module
            // still runs its stage exactly once
            if (std.Thread.spawn(.{}, Worker.run, .{worker})) |thread| {
                threads[spawned] = thread;
                spawned += 1;
            } else |_| {
                worker.run();
            }
        }
        for (threads[0..spawned]) |thread| thread.join();
        var failure: ?anyerror = null;
        for (workers) |*worker| {
            try compilation.diagnostics.appendSlice(compilation.allocator, worker.diagnostics.items);
            worker.diagnostics.deinit(worker_allocator);
            if (worker.failure) |err| failure = err;
        }
        if (failure) |err| return err;
    }

    // resolves 'import a::b::c' to the file 'a/b/c.alloy' through the loader
    // and queues the loaded module for the next per-module batch
    fn loadImports(compilation: *Compilation, batch_start: usize, batch_end: usize, loader: ?ModuleLoader) !void {
        const arena = compilation.arena.allocator();

        const Request = struct {
            key: []const u8,
            // the key qualified by the importing module's directory; tried
            // first, so 'import token_kind' inside tokenizer/ finds
            // tokenizer/token_kind.alloy before token_kind.alloy (5.4)
            relative_key: ?[]const u8,
            span: Token.Location,
            importer_path: []const u8,
            importer_source: []const u8,
        };
        // requests are collected first; adding modules while iterating would
        // invalidate the module slice
        var requests: std.ArrayList(Request) = .empty;
        defer requests.deinit(compilation.allocator);

        for (compilation.modules.items[batch_start..batch_end]) |*module| {
            const tree = module.ast orelse continue;
            for (tree.module.imports) |import| {
                var key: std.ArrayList(u8) = .empty;
                // a library's relative imports resolve inside its own
                // namespace: the members were registered under the package
                // prefix when the container unpacked (section 6.4)
                if (module.library) |package_name| {
                    const first_segment = import.path[0].slice(module.source);
                    if (!std.mem.eql(u8, first_segment, "std") and !std.mem.eql(u8, first_segment, "pkg")) {
                        try key.appendSlice(arena, "pkg::");
                        try key.appendSlice(arena, package_name);
                        try key.appendSlice(arena, "::");
                    }
                }
                for (import.path, 0..) |segment, segment_index| {
                    if (segment_index != 0) try key.appendSlice(arena, "::");
                    try key.appendSlice(arena, segment.slice(module.source));
                }
                const owned_key = try key.toOwnedSlice(arena);
                var relative_key: ?[]const u8 = null;
                if (module.library == null and
                    !std.mem.startsWith(u8, owned_key, "std::") and
                    !std.mem.startsWith(u8, owned_key, "pkg::"))
                {
                    if (module.key) |module_key| {
                        if (std.mem.lastIndexOf(u8, module_key, "::")) |prefix_end| {
                            relative_key = try std.fmt.allocPrint(arena, "{s}::{s}", .{ module_key[0..prefix_end], owned_key });
                        }
                    }
                }
                if (relative_key == null and compilation.loaded_keys.contains(owned_key)) continue;
                var already_requested = false;
                for (requests.items) |request| {
                    const same_relative = if (request.relative_key) |left|
                        if (relative_key) |right| std.mem.eql(u8, left, right) else false
                    else
                        relative_key == null;
                    if (same_relative and std.mem.eql(u8, request.key, owned_key)) {
                        already_requested = true;
                        break;
                    }
                }
                if (already_requested) continue;
                try requests.append(compilation.allocator, .{
                    .key = owned_key,
                    .relative_key = relative_key,
                    .span = .{
                        .start = import.path[0].location.start,
                        .end = import.path[import.path.len - 1].location.end,
                    },
                    .importer_path = module.path,
                    .importer_source = module.source,
                });
            }
        }

        for (requests.items) |request| {
            if (std.mem.startsWith(u8, request.key, "pkg::")) {
                try compilation.loadPackage(request.key, request.span, request.importer_path, request.importer_source, loader);
                continue;
            }
            // the importing module's directory is searched first (5.4)
            var relative_path: ?[]const u8 = null;
            if (request.relative_key) |relative_key| {
                if (compilation.loaded_keys.contains(relative_key)) continue;
                const candidate_path = try importFilePath(arena, relative_key);
                relative_path = candidate_path;
                const relative_source: ?[]const u8 = if (loader) |active_loader|
                    try active_loader.function(active_loader.context, arena, candidate_path)
                else
                    null;
                if (relative_source) |loaded_source| {
                    _ = try compilation.addImportedModule(relative_key, candidate_path, loaded_source);
                    continue;
                }
            }
            if (compilation.loaded_keys.contains(request.key)) continue;
            const file_path = try importFilePath(arena, request.key);
            const source: ?[]const u8 = if (loader) |active_loader|
                try active_loader.function(active_loader.context, arena, file_path)
            else
                null;
            if (source) |loaded_source| {
                _ = try compilation.addImportedModule(request.key, file_path, loaded_source);
            } else {
                const message = if (relative_path) |candidate|
                    try std.fmt.allocPrint(arena, "module '{s}' not found (expected file '{s}' or '{s}')", .{ request.key, candidate, file_path })
                else
                    try std.fmt.allocPrint(arena, "module '{s}' not found (expected file '{s}')", .{ request.key, file_path });
                try compilation.diagnostics.append(compilation.allocator, .{
                    .path = request.importer_path,
                    .source = request.importer_source,
                    .span = request.span,
                    .message = message,
                });
            }
        }
    }

    // 'pkg::name[::module]' resolves through the loader's library hook: the
    // '.alloylib' container unpacks once, registering every member under
    // the package's namespace (section 6.4)
    fn loadPackage(compilation: *Compilation, key: []const u8, span: Token.Location, importer_path: []const u8, importer_source: []const u8, loader: ?ModuleLoader) !void {
        const arena = compilation.arena.allocator();
        const remainder = key["pkg::".len..];
        const name_length = std.mem.indexOf(u8, remainder, "::") orelse remainder.len;
        const package_name = remainder[0..name_length];
        if (package_name.len == 0) {
            try compilation.diagnostics.append(compilation.allocator, .{
                .path = importer_path,
                .source = importer_source,
                .span = span,
                .message = "a 'pkg::' import needs a package name (section 6.4)",
            });
            return;
        }
        const package_key = try std.fmt.allocPrint(arena, "pkg::{s}", .{package_name});
        if (!compilation.loaded_keys.contains(package_key)) {
            const bytes: ?[]const u8 = if (loader) |active_loader|
                if (active_loader.library) |library_hook|
                    try library_hook(active_loader.context, arena, package_name)
                else
                    null
            else
                null;
            const container = bytes orelse {
                try compilation.diagnostics.append(compilation.allocator, .{
                    .path = importer_path,
                    .source = importer_source,
                    .span = span,
                    .message = try std.fmt.allocPrint(arena, "package '{s}' not found (expected 'pkg/{s}.alloylib')", .{ package_name, package_name }),
                });
                return;
            };
            const unpacked = library_format.unpack(arena, container) catch |err| {
                const reason: []const u8 = switch (err) {
                    error.UnsupportedFormatVersion => "its container format is newer than this compiler",
                    else => "its container is malformed",
                };
                try compilation.diagnostics.append(compilation.allocator, .{
                    .path = importer_path,
                    .source = importer_source,
                    .span = span,
                    .message = try std.fmt.allocPrint(arena, "package '{s}' cannot be loaded: {s}", .{ package_name, reason }),
                });
                return;
            };
            // the embedded source is authoritative: a compiler-version
            // mismatch only invalidates cache sections, never the library
            for (unpacked.members) |member| {
                const member_key = if (member.key.len == 0)
                    package_key
                else
                    try std.fmt.allocPrint(arena, "pkg::{s}::{s}", .{ package_name, member.key });
                if (compilation.loaded_keys.contains(member_key)) continue;
                const module = try compilation.addImportedModule(member_key, member.path, member.source);
                module.library = package_name;
            }
        }
        if (!compilation.loaded_keys.contains(key)) {
            try compilation.diagnostics.append(compilation.allocator, .{
                .path = importer_path,
                .source = importer_source,
                .span = span,
                .message = try std.fmt.allocPrint(arena, "package '{s}' has no module '{s}'", .{ package_name, key }),
            });
        }
    }

    /// Packs the fully checked unit into a '.alloylib' container: the entry
    /// module plus every module of its own (std:: and pkg:: dependencies
    /// stay imports the consumer resolves).
    pub fn packLibrary(compilation: *Compilation, allocator: std.mem.Allocator, name: []const u8) ![]u8 {
        var members: std.ArrayList(library_format.Member) = .empty;
        defer members.deinit(allocator);
        for (compilation.modules.items) |module| {
            const member_key: []const u8 = if (module.key) |key| key: {
                if (std.mem.startsWith(u8, key, "std::") or std.mem.startsWith(u8, key, "pkg::")) continue;
                break :key key;
            } else "";
            try members.append(allocator, .{ .key = member_key, .path = module.path, .source = module.source });
        }
        return library_format.pack(allocator, compiler_version, name, members.items);
    }

    // stage 3, after the merge point: build the global symbol table and
    // resolve every name in every module
    fn resolveNames(compilation: *Compilation) !void {
        const arena = compilation.arena.allocator();
        const views = try arena.alloc(resolution.ModuleView, compilation.modules.items.len);
        for (compilation.modules.items, views) |*module, *view| {
            view.* = .{
                .key = module.key,
                .path = module.path,
                .source = module.source,
                .library = module.library,
                .module = &module.ast.?.module,
            };
        }
        compilation.views = views;
        var resolver = resolution.Resolver.init(arena, views, &compilation.diagnostics, compilation.allocator);
        compilation.merged = try resolver.run();
    }

    // stage 4: type checking over the merged unit
    fn checkTypes(compilation: *Compilation) !void {
        const checker = try compilation.arena.allocator().create(Checker);
        checker.* = Checker.init(
            compilation.arena.allocator(),
            compilation.views,
            &compilation.merged.?,
            &compilation.diagnostics,
            compilation.allocator,
        );
        // '#read_file' (section 7.4) resolves against the entry module's
        // directory; a null io keeps compile-time evaluation
        // filesystem-free (the hermetic-test default)
        checker.comptime_io = compilation.comptime_io;
        if (compilation.modules.items.len != 0) {
            checker.comptime_root = std.fs.path.dirname(compilation.modules.items[0].path) orelse ".";
        }
        try checker.run();
        compilation.checker = checker;
        // the side tables live in the compilation arena and feed stage 5
        compilation.expression_types = checker.expression_types;
        compilation.call_targets = checker.call_targets;
        compilation.call_type_bindings = checker.call_type_bindings;
        compilation.comptime_values = checker.comptime_values;
        compilation.cast_shapes = checker.cast_shapes;
        compilation.type_targets = checker.type_targets;
    }

    /// The host facilities an interpreted program may use: the filesystem
    /// behind the std::io externs and the argv behind std::process (section
    /// 5.1a). The defaults keep tests hermetic.
    pub const RunEnvironment = struct {
        host_io: ?std.Io = null,
        arguments: []const []const u8 = &.{},
    };

    /// Stage 5: executes 'main' through the tree-walking interpreter.
    /// On error.RuntimeFault the message is available in 'fault'.
    pub fn interpret(compilation: *Compilation, output: *std.Io.Writer) !i64 {
        return compilation.interpretWithEnvironment(output, .{});
    }

    pub fn interpretWithEnvironment(compilation: *Compilation, output: *std.Io.Writer, environment: RunEnvironment) !i64 {
        var interpreter = Interpreter.init(
            compilation.arena.allocator(),
            compilation.views,
            &compilation.merged.?,
            &compilation.expression_types,
            &compilation.call_targets,
            &compilation.call_type_bindings,
            &compilation.comptime_values,
            &compilation.cast_shapes,
            &compilation.type_targets,
            output,
        );
        interpreter.host_io = environment.host_io;
        if (compilation.checker) |checked| interpreter.pierced_results = &checked.pierced_results;
        interpreter.process_arguments = environment.arguments;
        return interpreter.run() catch |err| switch (err) {
            error.RuntimeFault => {
                compilation.fault = interpreter.fault_message;
                return error.RuntimeFault;
            },
            error.Return => return 0,
            else => return err,
        };
    }

    /// Stage 5 for 'alloyc build': lowers the checked unit to LLVM IR text.
    /// Returns null when lowering reported diagnostics (an unsupported
    /// construct or a missing 'main').
    pub fn generate(compilation: *Compilation, release_mode: bool) !?[]const u8 {
        var codegen = Codegen.init(
            compilation.arena.allocator(),
            compilation.views,
            &compilation.merged.?,
            compilation.checker.?,
            &compilation.expression_types,
            &compilation.call_targets,
            &compilation.call_type_bindings,
            &compilation.comptime_values,
            &compilation.cast_shapes,
            &compilation.diagnostics,
            compilation.allocator,
            release_mode,
        );
        return codegen.run() catch |err| switch (err) {
            error.Unsupported => null,
            error.OutOfMemory => error.OutOfMemory,
        };
    }

    pub fn renderDiagnostics(compilation: *const Compilation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        for (compilation.diagnostics.items) |diagnostic| {
            try diagnostic.render(writer);
        }
        const count = compilation.diagnostics.items.len;
        try writer.print("{d} error{s} generated.\n", .{ count, if (count == 1) "" else "s" });
    }
};

// 'a::b::c' to 'a/b/c.alloy'
fn importFilePath(allocator: std.mem.Allocator, key: []const u8) ![]const u8 {
    var file_path: std.ArrayList(u8) = .empty;
    var segments = std.mem.splitSequence(u8, key, "::");
    var first = true;
    while (segments.next()) |segment| {
        if (!first) try file_path.append(allocator, '/');
        try file_path.appendSlice(allocator, segment);
        first = false;
    }
    try file_path.appendSlice(allocator, ".alloy");
    return file_path.toOwnedSlice(allocator);
}

fn expectRuns(source: []const u8, expected_exit: i64, expected_output: []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    try expectRunSucceeds(&compilation, null);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqualStrings(expected_output, output.writer.buffered());
    try std.testing.expectEqual(expected_exit, exit_code);
}

fn expectRunFault(source: []const u8, expected_message: []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    try expectRunSucceeds(&compilation, null);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const result = compilation.interpret(&output.writer);
    try std.testing.expectError(error.RuntimeFault, result);
    const message = compilation.fault orelse "";
    if (std.mem.indexOf(u8, message, expected_message) == null) {
        std.debug.print("fault message: {s}\nexpected: {s}\n", .{ message, expected_message });
        return error.TestUnexpectedResult;
    }
}

test "the interpreter runs arithmetic and control flow" {
    try expectRuns(
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn add(a: i32, b: i32) -> i32 { return a + b; }
        \\fn add(a: f32, b: f32) -> f32 { return a + b; }
        \\fn main() -> i32 {
        \\    var total = 0;
        \\    for ([..5]) |i| {
        \\        total += i;
        \\    }
        \\    while (total < 12) {
        \\        total += 1;
        \\    }
        \\    const doubled = add(total, total);
        \\    printf("result %d\n", doubled);
        \\    return doubled;
        \\}
    , 24, "result 24\n");
}

test "the interpreter handles structs enums and matches" {
    try expectRuns(
        \\type Shape = struct { width: u32, height: u32 };
        \\type State = enum { Idle, Busy: u32 };
        \\fn area(self s: &Shape) -> u32 { return s.width * s.height; }
        \\fn main() -> i32 {
        \\    var s = Shape { .width = 3, .height = 4 };
        \\    var st: State = ::Busy(s.area());
        \\    const described = match (st) {
        \\        ::Idle { yield 0; }
        \\        ::Busy |load| { yield load; }
        \\    };
        \\    if (st is ::Busy |amount|) {
        \\        return (described + amount) to i32;
        \\    }
        \\    return -1;
        \\}
    , 24, "");
}

test "the interpreter moves pointers and deep copies values" {
    try expectRuns(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 5 };
        \\    p.value += 2;
        \\    var q: *var Box = move p;
        \\    var a = [1, 2, 3];
        \\    var b = a;
        \\    b[0] = 9;
        \\    return q.value + a[0];
        \\}
    , 8, "");
    try expectCheckErrors(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 5 };
        \\    var q: *var Box = move p;
        \\    return p.value;
        \\}
    , &.{"use of 'p' after 'move' (section 5.2)"});
    // a conditional move stays a checked runtime fault
    try expectRunFault(
        \\type Box = struct { value: i32 };
        \\fn consume(b: *Box) -> i32 { return b.value; }
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 5 };
        \\    var flag = true;
        \\    if (flag) {
        \\        var taken = consume(move p);
        \\    }
        \\    return p.value;
        \\}
    , "moved-from pointer");
}

test "the interpreter faults on overflow bounds and lockstep" {
    try expectRunFault(
        \\fn main() -> i32 {
        \\    var x: u8 = 250;
        \\    x += 10;
        \\    return 0;
        \\}
    , "integer overflow");
    try expectRunFault(
        \\fn main() -> i32 {
        \\    const digits = [1, 2, 3];
        \\    return digits[5];
        \\}
    , "out of bounds");
    try expectRunFault(
        \\fn main() -> i32 {
        \\    const left = [1, 2, 3];
        \\    const right = [4, 5];
        \\    var total = 0;
        \\    for (left, right) |a, b| {
        \\        total += a + b;
        \\    }
        \\    return total;
        \\}
    , "disagree on length");
}

test "'to' keeps the value and faults when the target cannot" {
    // in-range conversions pass; a float loses only its fractional part
    try expectRuns(
        \\fn main() -> i32 {
        \\    var n: i64 = 120;
        \\    var wide = n to u64;
        \\    var back = wide to i32;
        \\    var f: f64 = -2.9;
        \\    var whole = f to i32;
        \\    return back + whole;
        \\}
    , 118, "");
    // a negative cannot keep its meaning in an unsigned (section 4.5)
    try expectRunFault(
        \\fn main() -> i32 {
        \\    var n: i64 = -5;
        \\    var u = n to u64;
        \\    return 0;
        \\}
    , "'to' keeps the value");
    try expectRunFault(
        \\fn main() -> i32 {
        \\    var n: i64 = 300;
        \\    var b = n to u8;
        \\    return 0;
        \\}
    , "does not fit u8");
    try expectRunFault(
        \\fn main() -> i32 {
        \\    var f: f64 = 4000000000.0;
        \\    var v = f to i32;
        \\    return 0;
        \\}
    , "does not fit i32");
}

test "the interpreter runs lambdas with captured environments" {
    try expectRuns(
        \\fn main() -> i32 {
        \\    var base = 10;
        \\    const add_base = |base| (x: i32) -> i32 { return base + x; };
        \\    var counter = 0;
        \\    const bump = |&var counter| (amount: i32) { counter += amount; };
        \\    bump(3);
        \\    bump(4);
        \\    base = 100;
        \\    return add_base(5) + counter;
        \\}
    , 22, "");
    try expectCheckErrors(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 5 };
        \\    const get = |move p| () -> i32 { return p.value; };
        \\    return get() + p.value;
        \\}
    , &.{"use of 'p' after 'move' (section 5.2)"});
}

test "the interpreter drives custom iterables through the cursor protocol" {
    var sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
        .{ "std/iterable.alloy", "import std::option;\npub interface Iterator<T> { fn next(self: &var) -> Option<&T>; }\npub interface Iterable<T, It: Iterator<T>> { fn iterator(self: &) -> It; }" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::option;
        \\import std::iterable;
        \\type Range : Iterable<u64, RangeCursor> = struct { limit: u64 };
        \\type RangeCursor : Iterator<u64> = struct { current: u64, limit: u64 };
        \\fn iterator(self r: &Range) -> RangeCursor {
        \\    return RangeCursor { .current = 0, .limit = r.limit };
        \\}
        \\fn next(self c: &var RangeCursor) -> Option<&u64> {
        \\    if (c.current == c.limit) {
        \\        return Option::None;
        \\    }
        \\    c.current += 1;
        \\    return Option::Some(&c.current);
        \\}
        \\fn main() -> i32 {
        \\    const range = Range { .limit = 4 };
        \\    var total: u64 = 0;
        \\    for (range) |n| {
        \\        total += n;
        \\    }
        \\    return total to i32;
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 10), exit_code);
}

test "imports resolve relative to the importing module first" {
    // nested/helper imports 'sibling': the nested/ copy must win over the
    // entry-root copy; 'lonely' has no nested copy and falls back
    var sources = TestSources.initComptime(.{
        .{ "nested/helper.alloy", "import sibling;\nimport lonely;\npub fn combined() -> i64 { return near() + far(); }" },
        .{ "nested/sibling.alloy", "pub fn near() -> i64 { return 10; }" },
        .{ "sibling.alloy", "pub fn near() -> i64 { return 999; }" },
        .{ "lonely.alloy", "pub fn far() -> i64 { return 3; }" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        "import nested::helper;\nfn main() -> i64 { return combined(); }");
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 13), exit_code);
}

test "the interpreter resolves generic type parameters at runtime" {
    try expectRuns(
        \\fn make<T>(count: u64) -> T {
        \\    return count as T;
        \\}
        \\fn pick<T>(flag: bool, a: T, b: T) -> T {
        \\    const choose = |a, b| (wanted: bool) -> T {
        \\        if (wanted) { return a; }
        \\        return b;
        \\    };
        \\    return choose(flag);
        \\}
        \\fn main() -> i32 {
        \\    const value = make<u64>(7);
        \\    const chosen = pick(true, 30, 5);
        \\    return value to i32 + chosen;
        \\}
    , 37, "");
}

test "comptime expressions evaluate during compilation" {
    try expectRuns(
        \\fn main() -> i32 {
        \\    const base = 10;
        \\    const computed = #double(base + 1);
        \\    const choice = #if (base > 4) 50 else 100;
        \\    return computed + choice;
        \\}
        \\fn double(x: i32) -> i32 { return x * 2; }
    , 72, "");
}

test "comptime expressions reject runtime state externs and indirections" {
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    var wide = 5;
        \\    const choice = #(wide + 1);
        \\    return choice;
        \\}
    , &.{"comptime evaluation failed"});
    try expectCheckErrors(
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn shout() -> i32 { return printf("hi"); }
        \\fn main() -> i32 { return #shout(); }
    , &.{"extern functions cannot be called at compile time"});
    try expectCheckErrors(
        \\type Box = struct { value: i32 };
        \\fn boxed() -> *var Box { return new Box { .value = 1 }; }
        \\fn main() -> i32 {
        \\    const p = #boxed();
        \\    return p.value;
        \\}
    , &.{"is a value or a slice, never"});
}

test "macros synthesise types and reflect at compile time" {
    try expectRuns(
        \\macro struct_type() -> #Type;
        \\macro type_of(value) -> #Type;
        \\macro vector2() -> #Type {
        \\    var t = #struct_type();
        \\    t.add_member("x", #f32);
        \\    t.add_member("y", #f32);
        \\    return t;
        \\}
        \\macro widthOf(wide: bool) -> #Type {
        \\    if (wide) {
        \\        return #u64;
        \\    }
        \\    return #u32;
        \\}
        \\macro answer() -> i32 { return 42; }
        \\type Vec2 = #vector2();
        \\type Header = #widthOf(false);
        \\interface Marked { }
        \\type Tag : Marked = struct { id: u32 };
        \\fn main() -> i32 {
        \\    var v = Vec2 { .x = 1.5, .y = 2.5 };
        \\    v.x += 1.0;
        \\    var h: Header = 7;
        \\    h += 2;
        \\    var score = 0;
        \\    if (#(#Tag.implements_interface(#Marked))) { score += 10; }
        \\    if (#(#type_of(7).equals(#i32))) { score += 100; }
        \\    if (#(#Vec2.is_struct())) { score += 1000; }
        \\    return (v.x + v.y) to i32 + h to i32 + score + #answer() - 42;
        \\}
    , 1124, "");
}

test "macros synthesise enum types" {
    try expectRuns(
        \\extern printf(format: &[u8], ...) -> i32;
        \\macro enum_type() -> #Type;
        \\macro signal() -> #Type {
        \\    var t = #enum_type();
        \\    t.add_member("Idle", #void);
        \\    t.add_member("Busy", #u32);
        \\    return t;
        \\}
        \\type Signal = #signal();
        \\fn main() -> i64 {
        \\    var idle: Signal = Signal::Idle;
        \\    var busy: Signal = Signal::Busy(7);
        \\    var total: i64 = 0;
        \\    match (idle) {
        \\        Signal::Idle { total += 1; }
        \\        Signal::Busy |load| { total += load to i64; }
        \\    }
        \\    if (busy is Signal::Busy |load|) {
        \\        total += load to i64;
        \\    }
        \\    var word = busy as u64;
        \\    var round = word as Signal;
        \\    if (round is Signal::Busy |load|) {
        \\        total += load to i64;
        \\    }
        \\    printf("members %d\n", #Signal.member_names().length());
        \\    return total;
        \\}
    , 15, "members 2\n");
    // a synthesised enum compares structurally with a matching inline enum
    // (section 4.3 rule 7)
    try expectChecks(
        \\macro enum_type() -> #Type;
        \\macro signal() -> #Type {
        \\    var t = #enum_type();
        \\    t.add_member("Idle", #void);
        \\    t.add_member("Busy", #u32);
        \\    return t;
        \\}
        \\type Signal = #signal();
        \\fn f() {
        \\    var s: Signal = Signal::Busy(5);
        \\    var inline_view: enum { Idle, Busy: u32 } = s;
        \\}
    );
    try expectCheckErrors(
        \\fn main() -> i64 {
        \\    const kept = #void;
        \\    return 0;
        \\}
    , &.{"cannot be retained in a runtime declaration"});
}

test "macro misuse is reported" {
    try expectCheckErrors(
        \\macro answer() -> i32 { return 42; }
        \\fn main() -> i32 { return answer(); }
    , &.{"a macro call must be invoked with '#'"});
    try expectCheckErrors(
        \\macro struct_type() -> #Type;
        \\fn main() -> i32 {
        \\    const t = #struct_type();
        \\    return 0;
        \\}
    , &.{"cannot be retained in a runtime declaration"});
}

test "extension receivers handle temporaries and ownership" {
    // a temporary receiver materializes for an '&' self (section 5.5)
    try expectRuns(
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Counter = struct { value: u32 };
        \\fn read(self c: &Counter) -> u32 { return c.value; }
        \\fn loud() -> Counter {
        \\    printf("made\n");
        \\    return Counter { .value = 10 };
        \\}
        \\fn main() -> i64 {
        \\    var first = loud().read();
        \\    return first to i64;
        \\}
    , 10, "made\n");
    // a '*T' self takes ownership: 'move' or an inline allocation passes,
    // a bare owning place does not (section 5.2)
    try expectRuns(
        \\type Counter = struct { value: u32 };
        \\fn consume(self c: *Counter) -> u32 { return c.value; }
        \\fn main() -> i64 {
        \\    var p: *Counter = new Counter { .value = 9 };
        \\    var a = (move p).consume();
        \\    var b = (new Counter { .value = 5 }).consume();
        \\    return (a + b) to i64;
        \\}
    , 14, "");
    try expectCheckErrors(
        \\type Counter = struct { value: u32 };
        \\fn consume(self c: *Counter) -> u32 { return c.value; }
        \\fn main() -> i64 {
        \\    var p: *Counter = new Counter { .value = 9 };
        \\    return p.consume() to i64;
        \\}
    , &.{"takes ownership of its receiver"});
}

test "fixed arrays need a positive length" {
    try expectCheckErrors(
        \\fn f() {
        \\    var empty: [u32 : 0] = [0 : 0];
        \\}
    , &.{ "at least one element", "at least one element" });
}

test "the interpreter dispatches interface objects and downcasts" {
    try expectRuns(
        \\interface Shape {
        \\    fn area(self: &) -> i32;
        \\    fn sides(self: &) -> i32;
        \\}
        \\fn sides(self s: &Shape) -> i32 { return 0; }
        \\type Circle : Shape = struct { radius: i32 };
        \\fn area(self c: &Circle) -> i32 { return c.radius * c.radius; }
        \\type Square : Shape = struct { side: i32 };
        \\fn area(self s: &Square) -> i32 { return s.side * s.side; }
        \\fn sides(self s: &Square) -> i32 { return 4; }
        \\fn measure(shape: &Shape) -> i32 { return shape.area() + shape.sides(); }
        \\fn main() -> i32 {
        \\    const circle = Circle { .radius = 2 };
        \\    const square = Square { .side = 3 };
        \\    var total = measure(&circle) + measure(&square);
        \\    const shape: &Shape = &square;
        \\    if (shape is Circle) {
        \\        total += 100;
        \\    }
        \\    match (shape) {
        \\        Circle { total += 1000; }
        \\        Square |&sq| { total += sq.side; }
        \\        else {}
        \\    }
        \\    return total;
        \\}
    , 20, "");
}

test "clean module passes all stages" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    const module = try compilation.addModule("main.alloy", "fn main() -> i32 { return 0; }");
    try expectRunSucceeds(&compilation, null);
    try std.testing.expectEqual(@as(usize, 0), compilation.diagnostics.items.len);
    // 10 tokens plus the end_of_file marker
    try std.testing.expectEqual(@as(usize, 12), module.tokens.items.len);
    try std.testing.expect(compilation.merged != null);
}

test "a stage with errors gates the pipeline" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("bad.alloy", "var s = \"abc\nvar t = 0x");
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 2), compilation.diagnostics.items.len);
}

test "diagnostics render cleanly" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("bad.alloy", "const x = 0b");
    try std.testing.expect(!try compilation.run(null));

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try compilation.renderDiagnostics(&output.writer);
    try std.testing.expectEqualStrings(
        "bad.alloy:1:11: error: integer literal has a radix prefix but no digits\n" ++
            "    const x = 0b\n" ++
            "              ^~\n" ++
            "1 error generated.\n",
        output.writer.buffered(),
    );
}

test "errors in any module gate the whole pipeline" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("good.alloy", "fn ok() { }");
    _ = try compilation.addModule("bad.alloy", "fn no() { var y = '' }");
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 1), compilation.diagnostics.items.len);
}

test "a missing import is a diagnostic" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", "import std::nowhere;\nfn main() { }");
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 1), compilation.diagnostics.items.len);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "std::nowhere") != null);
}

const TestSources = std.StaticStringMap([]const u8);

// runs the front-end stages and, on an unexpected failure, prints every
// diagnostic so a broken fixture names its own problem
fn expectRunSucceeds(compilation: *Compilation, loader: ?ModuleLoader) !void {
    if (try compilation.run(loader)) return;
    for (compilation.diagnostics.items) |diagnostic| {
        std.debug.print("unexpected diagnostic: {s}\n", .{diagnostic.message});
    }
    return error.TestUnexpectedResult;
}

fn testLoader(context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8 {
    const sources: *const TestSources = @ptrCast(@alignCast(context.?));
    const source = sources.get(file_path) orelse return null;
    return try allocator.dupe(u8, source);
}

test "undeclared names become diagnostics, all of them" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\fn main() {
        \\    var x = missing_one();
        \\    var y: NoSuchType = x;
        \\    missing_two(x, y);
        \\}
    );
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 3), compilation.diagnostics.items.len);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "missing_one") != null);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[1].message, "NoSuchType") != null);
}

test "redeclaration across modules is an error" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("first.alloy", "pub fn helper() { }");
    _ = try compilation.addModule("second.alloy", "type helper = u32;");
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 1), compilation.diagnostics.items.len);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "redeclaration of 'helper'") != null);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "first.alloy") != null);
}

test "qualified access checks visibility, unqualified does not" {
    var sources = TestSources.initComptime(.{
        .{ "util.alloy", "fn hidden() { }\npub fn shown() { }" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import util;
        \\fn main() {
        \\    util::shown();
        \\    util::hidden();
        \\    hidden();
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try std.testing.expect(!try compilation.run(loader));
    try std.testing.expectEqual(@as(usize, 1), compilation.diagnostics.items.len);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "private") != null);
}

test "import aliases qualify names" {
    var sources = TestSources.initComptime(.{
        .{ "std/vec.alloy", "pub type Vec<T> = struct { count: u64 };" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::vec as vectors;
        \\fn main() {
        \\    var v: vectors::Vec<u32> = Vec { .count = 0 };
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
}

test "enum variant references are checked" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\type Holder = enum { Boxed: u32, Empty };
        \\fn main() {
        \\    var a = Holder::Boxed(1);
        \\    var b = Holder::Missing;
        \\}
    );
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 1), compilation.diagnostics.items.len);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "no variant 'Missing'") != null);
}

test "lambda bodies see captures, not enclosing locals" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\fn main() {
        \\    var seen = 1;
        \\    var unseen = 2;
        \\    const lambda = |seen| (x: u32) -> u32 {
        \\        return x + seen + unseen;
        \\    };
        \\}
    );
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 1), compilation.diagnostics.items.len);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "capture list") != null);
}

test "scopes shadow outward but not within one frame" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\fn main() {
        \\    var x = 1;
        \\    {
        \\        var x = 2;
        \\    }
        \\    var x = 3;
        \\}
    );
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 1), compilation.diagnostics.items.len);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "already declared") != null);
}

test "captures and type parameters bind names" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\type Holder = enum { Boxed: u32, Empty };
        \\fn pick<T>(value: T, holder: Holder) -> T {
        \\    if (holder is Holder::Boxed |inner|) {
        \\        consume(inner);
        \\    }
        \\    match (holder) {
        \\        Holder::Boxed |payload| { consume(payload); }
        \\        else { }
        \\    }
        \\    for ([1, 2, 3]) |&element| {
        \\        consume(element);
        \\    }
        \\    return value;
        \\}
        \\fn consume(anything: u32) { }
        \\fn consume(anything: i32) { }
    );
    try expectRunSucceeds(&compilation, null);
}

fn expectCheckErrors(source: []const u8, expected: []const []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expectEqual(expected.len, compilation.diagnostics.items.len);
    for (compilation.diagnostics.items, expected) |diagnostic, fragment| {
        if (std.mem.indexOf(u8, diagnostic.message, fragment) == null) {
            std.debug.print("message '{s}' does not contain '{s}'\n", .{ diagnostic.message, fragment });
            return error.TestUnexpectedResult;
        }
    }
}

fn expectCheckErrorsWith(sources: *TestSources, source: []const u8, expected: []const []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(sources)), .function = testLoader };
    try std.testing.expect(!try compilation.run(loader));
    try std.testing.expectEqual(expected.len, compilation.diagnostics.items.len);
    for (compilation.diagnostics.items, expected) |diagnostic, fragment| {
        if (std.mem.indexOf(u8, diagnostic.message, fragment) == null) {
            std.debug.print("message '{s}' does not contain '{s}'\n", .{ diagnostic.message, fragment });
            return error.TestUnexpectedResult;
        }
    }
}

fn expectChecks(source: []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const success = try compilation.run(null);
    if (!success) {
        for (compilation.diagnostics.items) |diagnostic| {
            std.debug.print("unexpected: {s}\n", .{diagnostic.message});
        }
    }
    try std.testing.expect(success);
}

test "type mismatches are reported" {
    try expectCheckErrors(
        \\fn f() {
        \\    var x: u32 = true;
        \\    var y: bool = 1.5;
        \\    var z: u8 = x;
        \\}
    , &.{ "expected u32, found bool", "expected bool", "expected u8, found u32" });
}

test "a bare unsized array type is reported" {
    try expectCheckErrors(
        \\fn f() {
        \\    const path: [u8] = "spec";
        \\}
    , &.{"a bare '[T]' has no size"});
}

test "name_of yields an enum value's variant name" {
    try expectRuns(
        \\macro name_of(value) -> &[u8];
        \\type State = enum { Idle, Busy: i32 };
        \\fn main() -> i32 {
        \\    const label = #name_of(State::Busy(4));
        \\    return label.length() to i32;
        \\}
    , 4, "");
}

test "macro bodies are type-checked like function bodies" {
    // section 7.3: a body is checked statically against every rule before
    // it runs, as compile-time context ('#Type' locals are fine)
    try expectCheckErrors(
        \\macro struct_type() -> #Type;
        \\type Point = struct { x: i32 };
        \\macro helper(width: i64) -> i64 {
        \\    var t = #struct_type();
        \\    const wrong: bool = width + Point { .x = true };
        \\    return wrong + t.no_such_method();
        \\}
        \\fn main() -> i32 { return 0; }
    , &.{
        "expected i32, found bool",
        "'+' requires numeric operands",
        "'no_such_method' is not a '#Type' method",
        "'+' requires numeric operands",
    });
    try expectCheckErrors(
        \\type Entry = struct { name: &[u8] };
        \\macro names(entries: &[Entry]) -> u64 {
        \\    var total: u64 = 0;
        \\    for (entries) |&entry| {
        \\        total += entry.length();
        \\    }
        \\    return total;
        \\}
        \\fn main() -> i32 { return 0; }
    , &.{"no extension function 'length' for Entry"});
    try expectCheckErrors(
        \\macro falls(flag: bool) -> i64 {
        \\    if (flag) { return 1; }
        \\}
        \\fn main() -> i32 { return 0; }
    , &.{"control can fall off the end of macro 'falls', which must return i64 on every path"});
    try expectChecks(
        \\macro struct_type() -> #Type;
        \\macro vec2() -> #Type {
        \\    var t = #struct_type();
        \\    t.add_member("x", #f32);
        \\    t.add_member("y", #struct { a: u8 });
        \\    const names = &t.member_names();
        \\    if (names.length() == 2) { return t; }
        \\    return #u32;
        \\}
        \\type V = #vec2();
        \\fn main() -> i32 { return 0; }
    );
}

test "a comptime fault reports its call chain" {
    try expectCheckErrors(
        \\macro inner(text: &[u8]) -> u64 {
        \\    return 10 / text.length() - 1;
        \\}
        \\macro outer() -> u64 {
        \\    return #inner("");
        \\}
        \\fn main() -> i32 {
        \\    const v = #outer();
        \\    return v to i32;
        \\}
    , &.{"comptime call chain: outer -> inner"});
}

test "an 'is' capture works on a temporary subject in both engines" {
    // the subject is a call result, not a place: the interpreter used to
    // treat it as a silent non-match and always take the else branch
    try expectRuns(
        \\type Maybe = enum { Some: i64, None };
        \\fn produce(flag: bool) -> Maybe {
        \\    if (flag) { return Maybe::Some(42); }
        \\    return Maybe::None;
        \\}
        \\fn main() -> i32 {
        \\    var total = 0;
        \\    if (produce(true) is ::Some |value|) {
        \\        total += value to i32;
        \\    }
        \\    if (produce(false) is ::Some |value|) {
        \\        total += value to i32;
        \\    } else {
        \\        total += 1;
        \\    }
        \\    return total;
        \\}
    , 43, "");
    try expectBuildsAndRuns("is_temporary_subject",
        \\type Maybe = enum { Some: i64, None };
        \\fn produce(flag: bool) -> Maybe {
        \\    if (flag) { return Maybe::Some(42); }
        \\    return Maybe::None;
        \\}
        \\fn main() -> i32 {
        \\    var total = 0;
        \\    if (produce(true) is ::Some |value|) {
        \\        total += value to i32;
        \\    }
        \\    if (produce(false) is ::Some |value|) {
        \\        total += value to i32;
        \\    } else {
        \\        total += 1;
        \\    }
        \\    return total;
        \\}
    , 43, "");
}

test "reference results are borrowed explicitly, in both engines" {
    // '&shared(...)' keeps the borrow and observes the mutation; the bare
    // call pierces to a copy taken before it (section 5.2)
    try expectRuns(
        \\fn shared(source: &i64) -> &i64 {
        \\    return &source;
        \\}
        \\fn main() -> i32 {
        \\    var backing: i64 = 5;
        \\    const copied = shared(&backing);
        \\    const kept = &shared(&backing);
        \\    backing = 40;
        \\    return copied to i32 + kept to i32;
        \\}
    , 45, "");
    try expectBuildsAndRuns("reference_explicitness",
        \\fn shared(source: &i64) -> &i64 {
        \\    return &source;
        \\}
        \\fn main() -> i32 {
        \\    var backing: i64 = 5;
        \\    const copied = shared(&backing);
        \\    const kept = &shared(&backing);
        \\    backing = 40;
        \\    return copied to i32 + kept to i32;
        \\}
    , 45, "");
}

test "a bare slice-returning call at a use site is an error" {
    try expectCheckErrors(
        \\fn view(source: &[u8]) -> &[u8] {
        \\    return &source;
        \\}
        \\fn main() -> i32 {
        \\    const text = view("abc");
        \\    return text.length() to i32;
        \\}
    , &.{"a call's '&[T]' result must be marked at a use site"});
}

test "a same-named associated function hints at the extension form" {
    try expectCheckErrors(
        \\interface Serializable {
        \\    fn to_string(self: &) -> &[u8];
        \\}
        \\type Token : Serializable = struct { value: i64 };
        \\pub fn Token::to_string() -> &[u8] { return ""; }
        \\fn main() -> i32 { return 0; }
    , &.{"'Token::to_string' is an associated function, not an extension"});
}

test "a faulting type initializer is diagnosed even when unused" {
    // the macro yields void, not a '#Type'; the type is never used, but
    // the declaration still evaluates and reports
    try expectCheckErrors(
        \\macro enum_type() -> #Type;
        \\macro broken() -> #Type {
        \\    var t = #enum_type();
        \\    return t;
        \\}
        \\type Never = #broken().member_names();
        \\fn main() -> i32 { return 0; }
    , &.{"must yield a '#Type'"});
}

test "an unimplemented declared macro faults at the invocation" {
    try expectCheckErrors(
        \\macro mystery(value) -> i32;
        \\fn main() -> i32 {
        \\    const v = #mystery(1);
        \\    return 0;
        \\}
    , &.{"declared but not implemented"});
}

test "numeric widening follows sign classes" {
    try expectChecks(
        \\fn f() {
        \\    var small: u8 = 1;
        \\    var wide: u64 = small;
        \\    var signed: i8 = -1;
        \\    var wider: i64 = signed;
        \\    var float_small: f32 = 1.5;
        \\    var float_wide: f64 = float_small;
        \\}
    );
    try expectCheckErrors(
        \\fn f() {
        \\    var unsigned: u32 = 1;
        \\    var signed: i64 = unsigned;
        \\}
    , &.{"expected i64, found u32"});
}

test "mutability is enforced" {
    try expectCheckErrors(
        \\fn f(parameter: u32) {
        \\    const constant = 1;
        \\    constant = 2;
        \\    parameter = 3;
        \\    var mutable = 4;
        \\    mutable = 5;
        \\}
    , &.{ "immutable binding", "immutable binding" });
}

test "references and pointers obey section 5.2 assignment rules" {
    try expectChecks(
        \\fn f() {
        \\    var value: i32 = 5;
        \\    var borrowed: &i32 = &value;
        \\    var copied = borrowed;
        \\    var owned: *i32 = new 5;
        \\    var transferred: *i32 = move owned;
        \\}
    );
    try expectCheckErrors(
        \\fn f() {
        \\    var owned: *var i32 = new 5;
        \\    var aliased: *var i32 = owned;
        \\    var value: i32 = 7;
        \\    var borrowed: &i32 = value;
        \\}
    , &.{ "use 'move' to hand over the allocation or 'new' to copy it", "take a reference explicitly with '&'" });
}

test "move requires a mutable pointer" {
    try expectCheckErrors(
        \\fn f() {
        \\    const owned: *i32 = new 5;
        \\    var taken: *i32 = move owned;
        \\    var plain: i32 = 5;
        \\    var wrong = move plain;
        \\}
    , &.{ "'move' clears its source", "'move' requires a pointer" });
}

test "pointee transparency pierces reads" {
    try expectChecks(
        \\type Packet = struct { id: u32 };
        \\fn f() {
        \\    var p: *var Packet = new Packet { .id = 1 };
        \\    var copy: Packet = p;
        \\    var identifier: u32 = p.id;
        \\    var r: &Packet = &p;
        \\    var through: u32 = r.id;
        \\}
    );
}

test "overloads resolve and ambiguity is reported" {
    try expectChecks(
        \\fn describe(value: u32) -> u32 { return value; }
        \\fn describe(value: bool) -> u32 { return 1; }
        \\fn f() {
        \\    var a = describe(true);
        \\    var b = describe(7);
        \\}
    );
    try expectCheckErrors(
        \\fn pick(value: u64) { }
        \\fn pick(value: f32) { }
        \\fn f() {
        \\    pick(1);
        \\}
    , &.{"ambiguous"});
}

test "generic inference binds and propagates type parameters" {
    try expectChecks(
        \\type Pair<T> = struct { first: T, second: T };
        \\fn first<T>(pair: &Pair<T>) -> T { return pair.first; }
        \\fn f() {
        \\    var pair: Pair<u32> = Pair { .first = 1, .second = 2 };
        \\    var head: u32 = first(&pair);
        \\}
    );
}

test "casts follow section 4.5" {
    try expectChecks(
        \\fn f() {
        \\    var raw: u32 = 1065353216;
        \\    var reinterpreted: f32 = raw as f32;
        \\    var negative: i64 = -5;
        \\    var converted: u32 = negative to u32;
        \\}
    );
    try expectCheckErrors(
        \\fn f() {
        \\    var raw: u32 = 1;
        \\    var wrong = raw as f64;
        \\    var flag = true to bool;
        \\}
    , &.{ "'as' reinterprets bytes in place", "'to' converts Number values" });
}

test "'as' widths follow C-compatible layout on every type" {
    // Pair is 8 bytes, Padded pads u8 to a u32 boundary (8 bytes), Status
    // is a 1-byte tag padded plus a u32 payload (8 bytes) - section 4.9
    try expectChecks(
        \\type Pair = struct { low: u32, high: u32 };
        \\type Padded = struct { tag: u8, value: u32 };
        \\type Status = enum { Idle, Busy: u32 };
        \\fn f() {
        \\    var pair: Pair = Pair { .low = 1, .high = 2 };
        \\    var as_wide = pair as u64;
        \\    var padded: Padded = Padded { .tag = 1, .value = 2 };
        \\    var padded_wide = padded as u64;
        \\    var status: Status = Status::Busy(9);
        \\    var status_wide = status as u64;
        \\    var viewed: &u64 = &pair as &u64;
        \\}
    );
    try expectCheckErrors(
        \\type Pair = struct { low: u32, high: u32 };
        \\fn f() {
        \\    var pair: Pair = Pair { .low = 1, .high = 2 };
        \\    var narrow = pair as u32;
        \\    var bytes: [u8 : 3] = [1, 2, 3];
        \\    var word = bytes as u32;
        \\    var viewed = &pair as &u32;
        \\}
    , &.{
        "Pair is 8 byte(s) but u32 is 4 byte(s)",
        "[u8 : 3] is 3 byte(s) but u32 is 4 byte(s)",
        "Pair is 8 byte(s) but u32 is 4 byte(s)",
    });
}

test "the interpreter reinterprets structs arrays and enums" {
    // Pair { low = 1, high = 2 } is 00000001 00000002 little-endian:
    // 1 | 2 << 32 = 8589934593
    try expectRuns(
        \\type Pair = struct { low: u32, high: u32 };
        \\fn main() -> i64 {
        \\    var pair: Pair = Pair { .low = 1, .high = 2 };
        \\    var word = pair as u64;
        \\    var back = word as Pair;
        \\    var sum: u64 = (back.low to u64) + (back.high to u64);
        \\    return (word - 8589934593 + sum) to i64;
        \\}
    , 3, "");
    // Status::Busy(9): tag 1 at byte 0, u32 payload 9 at byte 4
    try expectRuns(
        \\type Status = enum { Idle, Busy: u32 };
        \\fn main() -> i64 {
        \\    var status: Status = Status::Busy(9);
        \\    var word = status as u64;
        \\    var round = word as Status;
        \\    var payload: u64 = 0;
        \\    if (round is Status::Busy |load|) {
        \\        payload = load to u64;
        \\    }
        \\    return (word - 38654705665 + payload) to i64;
        \\}
    , 9, "");
    // [1, 2, 3, 4] of u8 reads back as the little-endian u32 0x04030201
    try expectRuns(
        \\fn main() -> i64 {
        \\    var bytes: [u8 : 4] = [1, 2, 3, 4];
        \\    var word = bytes as u32;
        \\    return (word to i64) - 67305985;
        \\}
    , 0, "");
    // a padded struct zero-fills its padding bytes
    try expectRuns(
        \\type Padded = struct { tag: u8, value: u32 };
        \\fn main() -> i64 {
        \\    var padded: Padded = Padded { .tag = 5, .value = 7 };
        \\    var word = padded as u64;
        \\    return (word - 30064771077) to i64;
        \\}
    , 0, "");
}

test "character literals scale to their byte width" {
    try expectChecks(
        \\fn f() {
        \\    var ascii: u8 = 'a';
        \\    var short_escape: u8 = '\u{7F}';
        \\    var two_bytes: u16 = '\u{7FF}';
        \\    var four_bytes: u32 = '\u{1F600}';
        \\    var packed_sequence: u64 = 'abcdefgh';
        \\}
    );
}

test "escape payloads are validated" {
    try expectCheckErrors(
        \\fn f() {
        \\    var truncated: &[u8] = "bad \x4 escape";
        \\    var empty: u32 = '\u{}';
        \\    var oversized: u32 = '\u{110000}';
        \\}
    , &.{
        "'\\x' needs exactly two hex digits",
        "'\\u' needs '{' hex digits '}'",
        "not a valid Unicode scalar value",
    });
}

test "value-yielding constructs unify break values" {
    try expectChecks(
        \\fn f() -> u64 {
        \\    var counter: u64 = 0;
        \\    var label = match (counter) {
        \\        0 { yield "zero"; }
        \\        else { yield "more"; }
        \\    };
        \\    var capped = while (counter < 10) {
        \\        counter += 1;
        \\    } else {
        \\        yield counter;
        \\    };
        \\    return capped;
        \\}
    );
    try expectCheckErrors(
        \\fn f() {
        \\    var counter = 0;
        \\    while (counter < 3) {
        \\        counter += 1;
        \\    } else {
        \\        counter = 0;
        \\    }
        \\}
    , &.{"an 'else' on a loop is only valid when the loop is used as a value"});
}

test "enum payload captures type the payload" {
    try expectChecks(
        \\type Holder = enum { Boxed: *u32, Empty };
        \\fn f() {
        \\    var h: Holder = Holder::Boxed(new 5);
        \\    if (h is Holder::Boxed |move taken|) {
        \\        var inner: u32 = taken;
        \\    }
        \\    if (h is Holder::Empty) { }
        \\}
    );
    try expectCheckErrors(
        \\type Holder = enum { Boxed: u32, Empty };
        \\fn f() {
        \\    const h: Holder = Holder::Boxed(1);
        \\    if (h is Holder::Boxed |move taken|) { }
        \\}
    , &.{"owning capture requires a pointer-typed value"});
}

test "string literals are static u8 slices" {
    try expectChecks(
        \\fn consume(text: &[u8]) -> u64 { return text.length(); }
        \\fn f() {
        \\    var greeting: &[u8] = "hello";
        \\    var size = consume(&greeting);
        \\    var bytes: u64 = greeting.length();
        \\    var first: u8 = greeting[0];
        \\}
    );
}

test "extension functions resolve via dot notation" {
    try expectChecks(
        \\type Vector = struct { x: f32, y: f32 };
        \\fn scaled(self v: &Vector, factor: f32) -> Vector {
        \\    return Vector { .x = v.x * factor, .y = v.y * factor };
        \\}
        \\fn f() {
        \\    const v = Vector { .x = 1.0, .y = 2.0 };
        \\    const doubled: Vector = v.scaled(2.0);
        \\}
    );
    try expectCheckErrors(
        \\type Vector = struct { x: f32 };
        \\fn f() {
        \\    const v = Vector { .x = 1.0 };
        \\    v.missing();
        \\}
    , &.{"no extension function 'missing' for Vector"});
}

test "a mutating extension needs a mutable receiver" {
    try expectChecks(
        \\type Counter = struct { count: u64 };
        \\fn bump(self c: &var Counter) { c.count += 1; }
        \\fn f() {
        \\    var counter = Counter { .count = 0 };
        \\    counter.bump();
        \\}
    );
    try expectCheckErrors(
        \\type Counter = struct { count: u64 };
        \\fn bump(self c: &var Counter) { c.count += 1; }
        \\fn f() {
        \\    const counter = Counter { .count = 0 };
        \\    counter.bump();
        \\}
    , &.{"no overload of 'bump' matches Counter"});
}

test "extension functions are not free functions" {
    try expectCheckErrors(
        \\type Vector = struct { x: f32 };
        \\fn scaled(self v: &Vector, factor: f32) -> f32 { return v.x * factor; }
        \\fn f() {
        \\    const v = Vector { .x = 1.0 };
        \\    const result = scaled(&v, 2.0);
        \\}
    , &.{"no overload of 'scaled' matches these argument types"});
}

test "interface verification accepts implementations and defaults" {
    try expectChecks(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\    fn name(self: &) -> &[u8];
        \\}
        \\fn name(self s: &Shape) -> &[u8] { return "shape"; }
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius * c.radius; }
    );
}

test "interface verification reports missing and mismatched extensions" {
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
    , &.{"'Circle' does not implement 'Shape': no extension function 'area'"});
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> u64 { return 1; }
    , &.{"the extension 'area' for 'Circle' does not match the signature"});
}

test "a concrete extension overrides an interface default" {
    try expectChecks(
        \\interface Shape {
        \\    fn name(self: &) -> &[u8];
        \\}
        \\fn name(self s: &Shape) -> &[u8] { return "shape"; }
        \\type Circle : Shape = struct { radius: f32 };
        \\fn name(self c: &Circle) -> &[u8] { return "circle"; }
        \\fn f() {
        \\    const circle = Circle { .radius = 1.0 };
        \\    const label: &[u8] = &circle.name();
        \\}
    );
}

test "interface objects convert and dispatch dynamically" {
    try expectChecks(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn measure(shape: &Shape) -> f32 { return shape.area(); }
        \\fn f() {
        \\    const circle = Circle { .radius = 1.0 };
        \\    const surface = measure(&circle);
        \\}
    );
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Square = struct { side: f32 };
        \\fn measure(shape: &Shape) -> f32 { return shape.area(); }
        \\fn f() {
        \\    const square = Square { .side = 1.0 };
        \\    const surface = measure(&square);
        \\}
    , &.{"no overload of 'measure' matches these argument types"});
}

test "interface objects downcast with 'is'" {
    try expectChecks(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn measure(shape: &Shape) -> f32 {
        \\    if (shape is Circle |&c|) {
        \\        return c.area();
        \\    }
        \\    return 0.0;
        \\}
        \\fn grow(shape: *var Shape) {
        \\    if (shape is Circle |&var c|) {
        \\        c.radius += 1.0;
        \\    }
        \\}
    );
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Square = struct { side: f32 };
        \\fn measure(shape: &Shape) -> f32 {
        \\    if (shape is Square |&s|) {
        \\        return 0.0;
        \\    }
        \\    return 1.0;
        \\}
        \\fn f(x: u32) -> bool {
        \\    return x is Square;
        \\}
    , &.{
        "does not implement 'Shape', so this 'is' test can never succeed",
        "'is' tests enum variants and interface objects; the subject is u32",
    });
}

test "interface objects match on concrete types" {
    try expectChecks(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\type Square : Shape = struct { side: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn area(self s: &Square) -> f32 { return s.side; }
        \\fn measure(shape: &Shape) -> f32 {
        \\    return match (shape) {
        \\        Circle |&c| { yield c.area(); }
        \\        Square |&s| { yield s.area(); }
        \\        else { yield 0.0; }
        \\    };
        \\}
    );
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\type Blob = struct { x: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn measure(shape: &Shape) -> f32 {
        \\    return match (shape) {
        \\        Blob |&b| { yield b.x; }
        \\        5 { yield 1.0; }
        \\        Circle |&c| { yield c.area(); }
        \\        else { yield 0.0; }
        \\    };
        \\}
    , &.{
        "'Blob' does not implement 'Shape', so this arm can never match",
        "must name a concrete type implementing 'Shape'",
    });
}

test "implied enum variants infer their type" {
    try expectChecks(
        \\type State = enum { Idle, Busy: u32 };
        \\type Maybe<T> = enum { Just: T, Nothing };
        \\fn f() -> u32 {
        \\    var s: State = ::Idle;
        \\    s = ::Busy(3);
        \\    var m: Maybe<u32> = ::Just(7);
        \\    const lone = ::Idle;
        \\    if (s is ::Busy |load|) {
        \\        return load;
        \\    }
        \\    return match (s) {
        \\        ::Idle { yield 0; }
        \\        ::Busy |load| { yield load; }
        \\    };
        \\}
    );
    try expectCheckErrors(
        \\type A = enum { Dup, OnlyA };
        \\type B = enum { Dup };
        \\fn f() {
        \\    const x = ::Dup;
        \\    const y = ::Missing;
        \\    const z: A = ::Dup;
        \\}
    , &.{
        "'::Dup' is ambiguous: 2 enums in scope have this variant",
        "no enum in scope has a variant 'Missing'",
    });
}

test "inline enum types are structural" {
    try expectChecks(
        \\type Status = enum { Ok, Err: u32 };
        \\fn consume(s: enum { Ok, Err: u32 }) -> u32 {
        \\    return match (s) {
        \\        ::Ok { yield 0; }
        \\        ::Err |code| { yield code; }
        \\    };
        \\}
        \\fn f() -> u32 {
        \\    var s: Status = ::Ok;
        \\    var inline_value: enum { Ok, Err: u32 } = ::Err(5);
        \\    const a = consume(s);
        \\    const b = consume(inline_value);
        \\    return a + b;
        \\}
    );
    try expectCheckErrors(
        \\fn consume(s: enum { Ok, Err: u32 }) -> u32 { return 0; }
        \\fn f() {
        \\    var bad: enum { Ok } = ::Ok;
        \\    const c = consume(bad);
        \\}
    , &.{"no overload of 'consume' matches these argument types"});
}

test "contextual arguments resolve against overload parameters" {
    try expectChecks(
        \\type Status = enum { Ok, Err: u32 };
        \\type Maybe<T> = enum { Just: T, Nothing };
        \\fn consume(s: enum { Ok, Err: u32 }) -> u32 { return 1; }
        \\fn pick(s: Status) -> u32 { return 1; }
        \\fn pick(n: u32) -> u32 { return n; }
        \\fn unwrap(m: Maybe<u32>) -> u32 { return 0; }
        \\fn f() -> u32 {
        \\    const direct = consume(::Err(4));
        \\    const chosen = pick(::Ok);
        \\    const bound = unwrap(::Nothing);
        \\    const qualified = unwrap(Maybe::Just(7));
        \\    return direct + chosen + bound + qualified;
        \\}
    );
    try expectCheckErrors(
        \\type Status = enum { Ok, Err: u32 };
        \\fn pick(s: Status) -> u32 { return 1; }
        \\fn pick(n: u32) -> u32 { return n; }
        \\fn f() {
        \\    const chosen = pick(::Absent);
        \\}
    , &.{
        "no overload of 'pick' matches these argument types",
        "no enum in scope has a variant 'Absent'",
    });
}

test "variant constructions must bind every type parameter" {
    try expectChecks(
        \\type Maybe<T> = enum { Just: T, Nothing };
        \\fn f() {
        \\    var a: Maybe<u8> = Maybe::Nothing;
        \\    var b: Maybe<u8> = ::Nothing;
        \\    const c = Maybe::Just(7);
        \\}
    );
    try expectCheckErrors(
        \\type Maybe<T> = enum { Just: T, Nothing };
        \\fn f() {
        \\    const a = ::Nothing;
        \\    const b = Maybe::Nothing;
        \\}
    , &.{
        "cannot infer type parameter 'T' of 'Maybe': annotate the expected type",
        "cannot infer type parameter 'T' of 'Maybe': annotate the expected type",
    });
}

test "value-yielding matches must be exhaustive" {
    try expectChecks(
        \\type State = enum { Idle, Busy: u32 };
        \\fn f(s: State) -> u32 {
        \\    return match (s) {
        \\        State::Idle { yield 0; }
        \\        State::Busy |load| { yield load; }
        \\    } else {
        \\        yield 0;
        \\    };
        \\}
        \\fn g(s: State) -> u32 {
        \\    return match (s) {
        \\        State::Busy |load| { yield load; }
        \\        else { yield 0; }
        \\    };
        \\}
    );
    try expectCheckErrors(
        \\type State = enum { Idle, Busy: u32, Done };
        \\fn f(s: State) -> u32 {
        \\    return match (s) {
        \\        State::Idle { yield 0; }
        \\    };
        \\}
        \\fn g(n: u32) -> u32 {
        \\    return match (n) {
        \\        0 { yield 1; }
        \\        1 { yield 2; }
        \\    };
        \\}
    , &.{
        "this match does not cover variants 'Busy', 'Done' of 'State'",
        "a match over this subject can never cover every value: add an 'else' arm",
    });
}

test "a statement match needs no exhaustiveness and skips unmatched subjects" {
    // statement position: any subset of the subject's values may be
    // covered; an unmatched subject is a no-op (section 5.3)
    try expectBuildsAndRuns("statement_match_no_else",
        \\type State = enum { Idle, Busy: u32, Done };
        \\fn f(s: State) -> i32 {
        \\    var hit: i32 = 7;
        \\    match (s) {
        \\        State::Busy |load| { hit = load to i32; }
        \\    }
        \\    return hit;
        \\}
        \\fn main() -> i32 {
        \\    var total = f(::Idle) + f(::Busy(10)) + f(::Done);
        \\    match (total to u32) {
        \\        0 { total = 100; }
        \\        1 { total = 200; }
        \\    }
        \\    return total;
        \\}
    , 24, "");
}

test "a bare interface type needs an indirection" {
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\fn f(shape: Shape) { }
    , &.{"interface 'Shape' can only be used behind an indirection"});
}

test "generic constraints restrict instantiation" {
    try expectChecks(
        \\interface Marker { }
        \\type Tagged : Marker = struct { id: u32 };
        \\fn accept<T: Marker>(value: T) -> T { return value; }
        \\fn f() {
        \\    const tagged = Tagged { .id = 1 };
        \\    const same: Tagged = accept(tagged);
        \\}
    );
    try expectCheckErrors(
        \\interface Marker { }
        \\type Plain = struct { id: u32 };
        \\fn accept<T: Marker>(value: T) -> T { return value; }
        \\fn f() {
        \\    const plain = Plain { .id = 1 };
        \\    const same: Plain = accept(plain);
        \\}
    , &.{"no overload of 'accept' matches"});
}

test "array ranges generate integer arrays" {
    try expectChecks(
        \\fn f() {
        \\    var digits = [0..10];
        \\    var first: i32 = digits[0];
        \\    var bytes: [u8 : 3] = [2..5];
        \\    var count: u64 = digits.length();
        \\    var heap: *[i32] = new [1..4];
        \\    var limit: i64 = 9;
        \\    var runtime: *[i64] = new [3..limit];
        \\    var total: u64 = 0;
        \\    for ([..5]) |i| {
        \\        total += i to u64;
        \\    }
        \\    for ([..limit]) |n| {
        \\        total += n to u64;
        \\    }
        \\}
    );
}

test "array range bounds must be integers and ordered" {
    try expectCheckErrors(
        \\fn f() {
        \\    var wrong = [1.5..4];
        \\    var backwards = [5..2];
        \\}
    , &.{ "a range bound must be an integer", "the range end 2 is less than its start 5" });
    try expectCheckErrors(
        \\fn f(limit: u32) {
        \\    var runtime = [0..limit];
        \\}
    , &.{"a runtime-bounded range requires 'new' outside a 'for' subject"});
}

test "constrained generics expose interface functions" {
    try expectChecks(
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn measure<T: Shape>(shape: T) -> f32 { return shape.area(); }
        \\fn f() {
        \\    const circle = Circle { .radius = 1.0 };
        \\    const surface = measure(circle);
        \\}
    );
}

test "lang item interfaces are satisfied implicitly" {
    var sources = TestSources.initComptime(.{
        .{ "std/number.alloy", "pub interface Number { }" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::number;
        \\fn double<T: Number>(value: T) -> T { return value + value; }
        \\fn f() {
        \\    const doubled: u32 = double(7 as u32);
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    const success = try compilation.run(loader);
    if (!success) {
        for (compilation.diagnostics.items) |diagnostic| {
            std.debug.print("unexpected: {s}\n", .{diagnostic.message});
        }
    }
    try std.testing.expect(success);
}

test "custom iterables drive for loops via the cursor protocol" {
    var sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
        .{ "std/iterable.alloy", "import std::option;\npub interface Iterator<T> { fn next(self: &var) -> Option<&T>; }\npub interface Iterable<T, It: Iterator<T>> { fn iterator(self: &) -> It; }" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::option;
        \\import std::iterable;
        \\type Range : Iterable<u64, RangeCursor> = struct { limit: u64 };
        \\type RangeCursor : Iterator<u64> = struct { current: u64, limit: u64 };
        \\fn iterator(self r: &Range) -> RangeCursor {
        \\    return RangeCursor { .current = 0, .limit = r.limit };
        \\}
        \\fn next(self c: &var RangeCursor) -> Option<&u64> {
        \\    if (c.current == c.limit) {
        \\        return Option::None;
        \\    }
        \\    c.current += 1;
        \\    return Option::Some(&c.current);
        \\}
        \\fn f() {
        \\    const range = Range { .limit = 3 };
        \\    var total: u64 = 0;
        \\    for (range) |n| {
        \\        total += n;
        \\    }
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    const success = try compilation.run(loader);
    if (!success) {
        for (compilation.diagnostics.items) |diagnostic| {
            std.debug.print("unexpected: {s}\n", .{diagnostic.message});
        }
    }
    try std.testing.expect(success);
}

test "a type without the cursor protocol is not iterable" {
    try expectCheckErrors(
        \\type Plain = struct { id: u32 };
        \\fn f() {
        \\    const plain = Plain { .id = 1 };
        \\    for (plain) |n| { }
        \\}
    , &.{"Plain is not iterable"});
}

// the shared std stub for the generic-iterable tests below
const iterable_stub_sources = .{
    .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
    .{ "std/iterable.alloy", "import std::option;\npub interface Iterator<T> { fn next(self: &var) -> Option<&T>; }\npub interface Iterable<T, It: Iterator<T>> { fn iterator(self: &) -> It; }" },
};

test "a cursor-capable type still declares Iterable conformance" {
    var sources = TestSources.initComptime(iterable_stub_sources);
    try expectCheckErrorsWith(&sources,
        \\import std::option;
        \\import std::iterable;
        \\type Range = struct { limit: u64 };
        \\type RangeCursor : Iterator<u64> = struct { current: u64, limit: u64 };
        \\fn iterator(self r: &Range) -> RangeCursor {
        \\    return RangeCursor { .current = 0, .limit = r.limit };
        \\}
        \\fn next(self c: &var RangeCursor) -> Option<&u64> {
        \\    if (c.current == c.limit) {
        \\        return ::None;
        \\    }
        \\    c.current += 1;
        \\    return ::Some(&c.current);
        \\}
        \\fn f() {
        \\    const range = Range { .limit = 3 };
        \\    for (range) |n| { }
        \\}
    , &.{"provides 'iterator()' but does not declare 'Iterable' conformance"});
}

test "generic interface conformance checks the instantiated signature" {
    var sources = TestSources.initComptime(iterable_stub_sources);
    // 'next' returns 'Option<u64>' where 'Iterator<u64>' declares
    // 'Option<&u64>': the substituted signature catches it
    try expectCheckErrorsWith(&sources,
        \\import std::option;
        \\import std::iterable;
        \\type RangeCursor : Iterator<u64> = struct { current: u64, limit: u64 };
        \\fn next(self c: &var RangeCursor) -> Option<u64> {
        \\    return ::None;
        \\}
    , &.{"the extension 'next' for 'RangeCursor' does not match the signature declared by 'Iterator'"});
}

test "a conformance marker binds every interface type parameter" {
    var sources = TestSources.initComptime(iterable_stub_sources);
    try expectCheckErrorsWith(&sources,
        \\import std::option;
        \\import std::iterable;
        \\type Bag : Iterable<u64> = struct { limit: u64 };
    , &.{ "'Iterable' expects 2 type arguments, found 1", "'Bag' does not implement 'Iterable'" });
}

test "a generic interface object requires full instantiation and never downcasts" {
    var sources = TestSources.initComptime(iterable_stub_sources);
    // a bare generic interface behind an indirection is a partial erasure
    try expectCheckErrorsWith(&sources,
        \\import std::option;
        \\import std::iterable;
        \\fn f(cursor: &Iterator) { }
    , &.{"'Iterator' expects 1 type argument, found 0"});
    // runtime identity carries no instantiation, so no downcasting
    var downcast_sources = TestSources.initComptime(iterable_stub_sources);
    try expectCheckErrorsWith(&downcast_sources,
        \\import std::option;
        \\import std::iterable;
        \\type RangeCursor : Iterator<u64> = struct { current: u64 };
        \\fn next(self c: &var RangeCursor) -> Option<&u64> {
        \\    return ::None;
        \\}
        \\fn f(cursor: &Iterator<u64>) -> bool {
        \\    return cursor is RangeCursor;
        \\}
    , &.{"downcasting a generic interface object ('Iterator') is not supported"});
}

test "generic interface objects dispatch dynamically in both engines" {
    var sources = TestSources.initComptime(iterable_stub_sources);
    const source =
        \\import std::option;
        \\import std::iterable;
        \\type Range : Iterable<u64, RangeCursor> = struct { limit: u64 };
        \\type RangeCursor : Iterator<u64> = struct { current: u64, limit: u64 };
        \\fn iterator(self r: &Range) -> RangeCursor {
        \\    return RangeCursor { .current = 0, .limit = r.limit };
        \\}
        \\fn next(self c: &var RangeCursor) -> Option<&u64> {
        \\    if (c.current == c.limit) {
        \\        return ::None;
        \\    }
        \\    c.current += 1;
        \\    return ::Some(&c.current);
        \\}
        \\type Repeater<T> : Iterator<T> = struct { value: T, remaining: u64 };
        \\fn next<T>(self r: &var Repeater<T>) -> Option<&T> {
        \\    if (r.remaining == 0) {
        \\        return ::None;
        \\    }
        \\    r.remaining -= 1;
        \\    return ::Some(&r.value);
        \\}
        \\fn total(it: &var Iterator<u64>) -> u64 {
        \\    var sum: u64 = 0;
        \\    while (true) {
        \\        const step = it.next();
        \\        if (step is ::Some |&value|) {
        \\            sum += value;
        \\        } else {
        \\            break;
        \\        }
        \\    }
        \\    return sum;
        \\}
        \\fn main() -> i32 {
        \\    var range_cursor = RangeCursor { .current = 0, .limit = 3 };
        \\    var repeater: Repeater<u64> = Repeater { .value = 7, .remaining = 2 };
        \\    const combined = total(&var range_cursor) + total(&var repeater);
        \\    return combined to i32;
        \\}
    ;
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 20), exit_code);
    try expectBuildsAndRunsWith("dyn_iterator", source, &sources, 20, "");
}

test "constraints take type arguments and dispatch interface functions" {
    var sources = TestSources.initComptime(iterable_stub_sources);
    const source =
        \\import std::option;
        \\import std::iterable;
        \\type Range : Iterable<u64, RangeCursor> = struct { limit: u64 };
        \\type RangeCursor : Iterator<u64> = struct { current: u64, limit: u64 };
        \\fn iterator(self r: &Range) -> RangeCursor {
        \\    return RangeCursor { .current = 0, .limit = r.limit };
        \\}
        \\fn next(self c: &var RangeCursor) -> Option<&u64> {
        \\    if (c.current == c.limit) {
        \\        return ::None;
        \\    }
        \\    c.current += 1;
        \\    return ::Some(&c.current);
        \\}
        \\fn drain<T, It: Iterator<T>>(cursor: It) -> u64 {
        \\    var mine = cursor;
        \\    var count: u64 = 0;
        \\    while (mine.next() is ::Some) {
        \\        count += 1;
        \\    }
        \\    return count;
        \\}
        \\fn main() -> i32 {
        \\    const range = Range { .limit = 3 };
        \\    return drain<u64, RangeCursor>(range.iterator()) to i32;
        \\}
    ;
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 3), exit_code);
    try expectBuildsAndRunsWith("constraint_dispatch", source, &sources, 3, "");
}

test "struct literals bind type parameters explicitly" {
    var sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
    });
    const source =
        \\import std::option;
        \\type Sack<T> = struct { storage: Option<*var [T]>, count: u64 };
        \\fn Sack::with_capacity<T>(size: u64) -> Sack<T> {
        \\    var sack = Sack<T> { .storage = ::None, .count = size };
        \\    return sack;
        \\}
        \\fn main() -> i32 {
        \\    const sack = Sack<u8> { .storage = ::None, .count = 2 };
        \\    const made = Sack::with_capacity<u8>(4);
        \\    return (sack.count + made.count) to i32;
        \\}
    ;
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 6), exit_code);
    try expectBuildsAndRunsWith("generic_struct_literal", source, &sources, 6, "");
}

test "a struct literal's type arguments match the type's parameters" {
    try expectCheckErrors(
        \\type Pair = struct { x: i64 };
        \\fn f() {
        \\    const p = Pair<i64> { .x = 1 };
        \\}
    , &.{"'Pair' expects 0 type arguments, found 1"});
}

test "inline is captures chain through a condition and rebind in while" {
    var sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
    });
    const source =
        \\import std::option;
        \\type Pair = struct { left: Option<u8>, right: Option<u8> };
        \\fn main() -> i32 {
        \\    const pair = Pair { .left = ::Some(40), .right = ::Some(2) };
        \\    var total: i32 = 0;
        \\    if (pair.left is ::Some |first| && first == 40 && pair.right is ::Some |second|) {
        \\        total += first to i32 + second to i32;
        \\    }
        \\    if (pair.left is ::Some) {
        \\        total += 1;
        \\    }
        \\    var countdown: Option<u8> = ::Some(3);
        \\    while (countdown is ::Some |step|) {
        \\        total += step to i32;
        \\        if (step == 1) {
        \\            countdown = ::None;
        \\        } else {
        \\            countdown = ::Some(step - 1);
        \\        }
        \\    }
        \\    return total;
        \\}
    ;
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 49), exit_code);
    try expectBuildsAndRunsWith("inline_captures", source, &sources, 49, "");
}

test "an is capture demands a dominating conjunct position" {
    // under '||' a failed test still reaches uses of the capture
    try expectCheckErrors(
        \\type Holder = enum { Boxed: u32, Empty };
        \\fn f() {
        \\    const h: Holder = Holder::Boxed(1);
        \\    if (h is ::Boxed |taken| || taken == 2) { }
        \\}
    , &.{"an 'is' capture is only valid on a direct '&&' conjunct of an 'if' or 'while' condition"});
    // outside a condition there is no branch to scope the binding
    try expectCheckErrors(
        \\type Holder = enum { Boxed: u32, Empty };
        \\fn f() {
        \\    const h: Holder = Holder::Boxed(1);
        \\    const tested = h is ::Boxed |taken|;
        \\}
    , &.{"an 'is' capture is only valid on a direct '&&' conjunct of an 'if' or 'while' condition"});
}

test "the retired postfix if capture names the inline form" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\type Holder = enum { Boxed: u32, Empty };
        \\fn f() {
        \\    const h: Holder = Holder::Boxed(1);
        \\    if (h is ::Boxed) |taken| { }
        \\}
    );
    try std.testing.expect(!try compilation.run(null));
    try std.testing.expect(compilation.diagnostics.items.len >= 1);
    try std.testing.expect(std.mem.indexOf(u8, compilation.diagnostics.items[0].message, "the capture follows the 'is' test inside the condition") != null);
}

test "read_file reads project files at compile time and stays sandboxed" {
    const io = std.testing.io;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ".zig-cache/read_file_probe.txt", .data = "hello alloy" });
    var sources = TestSources.initComptime(.{
        .{ "std/macros.alloy", "pub macro read_file(path) -> &[u8];" },
    });
    const source =
        \\import std::macros;
        \\fn main() -> i32 {
        \\    const text = #read_file(".zig-cache/read_file_probe.txt");
        \\    return text.length() to i32;
        \\}
    ;
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    compilation.comptime_io = io;
    _ = try compilation.addModule("main.alloy", source);
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 11), exit_code);
    try expectBuildsAndRunsWith("read_file_comptime", source, &sources, 11, "");

    // '..' escaping the project root is a compile-time error
    var escape_sources = TestSources.initComptime(.{
        .{ "std/macros.alloy", "pub macro read_file(path) -> &[u8];" },
    });
    var escaping = Compilation.init(std.testing.allocator);
    defer escaping.deinit();
    escaping.comptime_io = io;
    _ = try escaping.addModule("main.alloy",
        \\import std::macros;
        \\fn main() -> i32 {
        \\    const text = #read_file("../secret.txt");
        \\    return 0;
        \\}
    );
    const escape_loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&escape_sources)), .function = testLoader };
    try std.testing.expect(!try escaping.run(escape_loader));
    try std.testing.expect(std.mem.indexOf(u8, escaping.diagnostics.items[0].message, "cannot escape the project root") != null);
}

test "a macro may share its name with functions" {
    const source =
        \\macro tag(value: i64) -> i64 {
        \\    return value * 2;
        \\}
        \\fn tag(x: i64) -> i64 {
        \\    return x + 1;
        \\}
        \\fn main() -> i32 {
        \\    const doubled = #tag(4);
        \\    return (tag(4) + doubled) to i32;
        \\}
    ;
    try expectRuns(source, 13, "");
    try expectBuildsAndRuns("macro_fn_name_share", source, 13, "");
}

test "using a reference variable bare consumes the value" {
    // a slice variable's value is the unsized array: '&' or 'new' required
    try expectCheckErrors(
        \\fn take(view: &[u8]) -> u64 {
        \\    return view.length();
        \\}
        \\fn main() -> i32 {
        \\    const text: &[u8] = "abc";
        \\    return take(text) to i32;
        \\}
    , &.{"a '&[T]' variable used here means the array value"});
    // a '&T' variable used bare is the pointee value, so it no longer
    // fits a '&T' parameter; passing the borrow is spelled '&x'
    try expectCheckErrors(
        \\fn read(source: &i64) -> i64 {
        \\    return source;
        \\}
        \\fn main() -> i32 {
        \\    var backing: i64 = 5;
        \\    const borrowed = &backing;
        \\    return read(borrowed) to i32;
        \\}
    , &.{"no overload of 'read' matches"});
}

test "reference variables copy bare and borrow with an ampersand" {
    // 'copied' snapshots the pointee before the mutation; 'alias'
    // re-borrows the same backing and observes it
    const source =
        \\fn main() -> i32 {
        \\    var backing: i64 = 5;
        \\    const borrowed = &backing;
        \\    const copied: i64 = borrowed;
        \\    backing = 40;
        \\    const alias = &borrowed;
        \\    return (copied + alias) to i32;
        \\}
    ;
    try expectRuns(source, 45, "");
    try expectBuildsAndRuns("reference_variable_use", source, 45, "");
}

test "re-borrowing a reference or slice binding is idempotent" {
    const source =
        \\fn take(view: &[u8]) -> u64 {
        \\    return view.length();
        \\}
        \\fn main() -> i32 {
        \\    const owned: *[u8] = new [65 : 3];
        \\    const view = &owned;
        \\    return take(&view) to i32;
        \\}
    ;
    try expectRuns(source, 3, "");
    try expectBuildsAndRuns("slice_reborrow", source, 3, "");
}

test "a bound violating a constraint's arguments is rejected" {
    var sources = TestSources.initComptime(iterable_stub_sources);
    // RangeCursor conforms to Iterator<u64>, not Iterator<u8>
    try expectCheckErrorsWith(&sources,
        \\import std::option;
        \\import std::iterable;
        \\type RangeCursor : Iterator<u64> = struct { current: u64, limit: u64 };
        \\fn next(self c: &var RangeCursor) -> Option<&u64> {
        \\    return ::None;
        \\}
        \\fn drain<T, It: Iterator<T>>(cursor: It) -> u64 {
        \\    return 0;
        \\}
        \\fn f() {
        \\    const cursor = RangeCursor { .current = 0, .limit = 3 };
        \\    const drained = drain<u8, RangeCursor>(cursor);
        \\}
    , &.{"no overload of 'drain' matches"});
}

test "imports load transitively through the loader" {
    var sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
        .{ "std/vec.alloy", "import std::option;\npub type Vec<T> = struct { items: *[T], count: u64 };" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::vec;
        \\fn main() {
        \\    var maybe: Option<u32> = Option::Some(7);
        \\    var fallback: Option<u32> = Option::None;
        \\    var buffer: Vec<u8> = Vec { .items = new [0 : 16], .count = 16 };
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    // entry plus std::vec plus the transitive std::option
    try std.testing.expectEqual(@as(usize, 3), compilation.modules.items.len);
}

fn expectGenerates(source: []const u8, release_mode: bool, present: []const []const u8, absent: []const []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    try expectRunSucceeds(&compilation, null);
    const ir_text = (try compilation.generate(release_mode)) orelse {
        for (compilation.diagnostics.items) |diagnostic| {
            std.debug.print("unexpected diagnostic: {s}\n", .{diagnostic.message});
        }
        return error.TestUnexpectedResult;
    };
    for (present) |fragment| {
        if (std.mem.indexOf(u8, ir_text, fragment) == null) {
            std.debug.print("missing fragment '{s}' in generated IR:\n{s}\n", .{ fragment, ir_text });
            return error.TestUnexpectedResult;
        }
    }
    for (absent) |fragment| {
        if (std.mem.indexOf(u8, ir_text, fragment) != null) {
            std.debug.print("unexpected fragment '{s}' in generated IR:\n{s}\n", .{ fragment, ir_text });
            return error.TestUnexpectedResult;
        }
    }
}

fn expectGenerateErrors(source: []const u8, expected_messages: []const []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    try expectRunSucceeds(&compilation, null);
    const generated = try compilation.generate(false);
    if (generated != null) {
        std.debug.print("expected diagnostics, but code generation succeeded\n", .{});
        return error.TestUnexpectedResult;
    }
    for (expected_messages) |expected| {
        const found = for (compilation.diagnostics.items) |diagnostic| {
            if (std.mem.indexOf(u8, diagnostic.message, expected) != null) break true;
        } else false;
        if (!found) {
            for (compilation.diagnostics.items) |diagnostic| {
                std.debug.print("diagnostic: {s}\n", .{diagnostic.message});
            }
            std.debug.print("expected: {s}\n", .{expected});
            return error.TestUnexpectedResult;
        }
    }
}

// the native end-to-end tests need a clang; absent one they skip
fn testClang(arena: std.mem.Allocator) ?[]const u8 {
    const candidates = [_][]const u8{
        "clang",
        "C:\\LLVM.bak18\\bin\\clang.exe",
        "C:\\LLVM\\bin\\clang.exe",
    };
    for (candidates) |candidate| {
        const result = std.process.run(arena, std.testing.io, .{
            .argv = &.{ candidate, "--version" },
        }) catch continue;
        if (result.term == .exited and result.term.exited == 0) return candidate;
    }
    return null;
}

fn expectBuildsAndRuns(name: []const u8, source: []const u8, expected_exit: u8, expected_output: []const u8) !void {
    return expectBuildsAndRunsMode(name, source, null, false, expected_exit, expected_output);
}

fn expectBuildsAndRunsWith(name: []const u8, source: []const u8, extra_sources: ?*const TestSources, expected_exit: u8, expected_output: []const u8) !void {
    return expectBuildsAndRunsMode(name, source, extra_sources, false, expected_exit, expected_output);
}

fn expectBuildsAndRunsMode(name: []const u8, source: []const u8, extra_sources: ?*const TestSources, release_mode: bool, expected_exit: u8, expected_output: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const clang = testClang(arena) orelse return error.SkipZigTest;

    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    // '#read_file' fixtures read real files relative to the test cwd
    compilation.comptime_io = std.testing.io;
    _ = try compilation.addModule("main.alloy", source);
    const loader: ?ModuleLoader = if (extra_sources) |sources|
        .{ .context = @constCast(@ptrCast(sources)), .function = testLoader }
    else
        null;
    try expectRunSucceeds(&compilation, loader);
    try expectNativeRun(arena, clang, &compilation, name, release_mode, expected_exit, expected_output);
}

// lowers an already-checked compilation to a native executable and runs it
fn expectNativeRun(arena: std.mem.Allocator, clang: []const u8, compilation: *Compilation, name: []const u8, release_mode: bool, expected_exit: u8, expected_output: []const u8) !void {
    const ir_text = (try compilation.generate(release_mode)) orelse {
        for (compilation.diagnostics.items) |diagnostic| {
            std.debug.print("unexpected diagnostic: {s}\n", .{diagnostic.message});
        }
        return error.TestUnexpectedResult;
    };

    const io = std.testing.io;
    try std.Io.Dir.cwd().createDirPath(io, ".zig-cache/alloyc-native-tests");
    const ir_path = try std.fmt.allocPrint(arena, ".zig-cache/alloyc-native-tests/{s}.ll", .{name});
    const executable_path = try std.fmt.allocPrint(arena, ".zig-cache/alloyc-native-tests/{s}.exe", .{name});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = ir_path, .data = ir_text });

    const optimization: []const u8 = if (release_mode) "-O2" else "-O0";
    const link_result = try std.process.run(arena, io, .{
        .argv = &.{ clang, ir_path, "-o", executable_path, optimization, "-Wno-override-module" },
    });
    if (link_result.term != .exited or link_result.term.exited != 0) {
        std.debug.print("clang failed:\n{s}\n", .{link_result.stderr});
        return error.TestUnexpectedResult;
    }

    const run_result = try std.process.run(arena, io, .{
        .argv = &.{executable_path},
    });
    // the C runtime writes "\r\n" on Windows; compare line content only
    const normalized = try std.mem.replaceOwned(u8, arena, run_result.stdout, "\r\n", "\n");
    try std.testing.expectEqualStrings(expected_output, normalized);
    if (run_result.term != .exited or run_result.term.exited != expected_exit) {
        std.debug.print("term: {any}, expected exit {d}\nstderr: {s}\n", .{ run_result.term, expected_exit, run_result.stderr });
        return error.TestUnexpectedResult;
    }
}

const TestPackage = struct {
    name: []const u8,
    bytes: []const u8,

    fn loadLibrary(context: ?*anyopaque, allocator: std.mem.Allocator, package_name: []const u8) anyerror!?[]const u8 {
        const package: *const TestPackage = @ptrCast(@alignCast(context.?));
        if (!std.mem.eql(u8, package.name, package_name)) return null;
        return try allocator.dupe(u8, package.bytes);
    }

    fn loadNoFiles(context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8 {
        _ = context;
        _ = allocator;
        _ = file_path;
        return null;
    }
};

const TestPackageSet = struct {
    packages: []const TestPackage,

    fn loadLibrary(context: ?*anyopaque, allocator: std.mem.Allocator, package_name: []const u8) anyerror!?[]const u8 {
        const set: *const TestPackageSet = @ptrCast(@alignCast(context.?));
        for (set.packages) |package| {
            if (std.mem.eql(u8, package.name, package_name)) return try allocator.dupe(u8, package.bytes);
        }
        return null;
    }
};

test "packages load from alloylib containers and expose only exp symbols" {
    const members = [_]library_format.Member{
        .{ .key = "", .path = "mathx.alloy", .source =
        \\import inner;
        \\exp fn twice(x: i64) -> i64 { return inner::double(x); }
        \\pub fn hidden(x: i64) -> i64 { return x; }
        },
        .{ .key = "inner", .path = "inner.alloy", .source = "pub fn double(x: i64) -> i64 { return x * 2; }" },
    };
    const container = try library_format.pack(std.testing.allocator, compiler_version, "mathx", &members);
    defer std.testing.allocator.free(container);
    var package = TestPackage{ .name = "mathx", .bytes = container };
    const loader: ModuleLoader = .{
        .context = @ptrCast(&package),
        .function = TestPackage.loadNoFiles,
        .library = TestPackage.loadLibrary,
    };

    {
        var compilation = Compilation.init(std.testing.allocator);
        defer compilation.deinit();
        _ = try compilation.addModule("main.alloy",
            \\import pkg::mathx;
            \\fn main() -> i32 {
            \\    return mathx::twice(21) to i32;
            \\}
        );
        try expectRunSucceeds(&compilation, loader);
        var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer output.deinit();
        try std.testing.expectEqual(@as(i64, 42), try compilation.interpret(&output.writer));
    }

    // 'pub' reaches across modules inside a unit, but not across the
    // library boundary (section 6.4)
    {
        var compilation = Compilation.init(std.testing.allocator);
        defer compilation.deinit();
        _ = try compilation.addModule("main.alloy",
            \\import pkg::mathx;
            \\fn main() -> i32 {
            \\    return mathx::hidden(1) to i32;
            \\}
        );
        try std.testing.expect(!try compilation.run(loader));
        const found = for (compilation.diagnostics.items) |diagnostic| {
            if (std.mem.indexOf(u8, diagnostic.message, "is not exported; mark it 'exp'") != null) break true;
        } else false;
        try std.testing.expect(found);
    }
}

test "unqualified lookup sees exp symbols and the own library shadows foreign ones" {
    const liba_members = [_]library_format.Member{.{ .key = "", .path = "liba.alloy", .source =
    \\pub fn helper() -> i64 { return 10; }
    \\exp fn contribute() -> i64 { return helper(); }
    \\exp type Pair = struct { left: i64 };
    \\exp fn make_pair(left: i64) -> Pair { return Pair { .left = left }; }
    }};
    const libb_members = [_]library_format.Member{.{ .key = "", .path = "libb.alloy", .source =
    \\pub fn helper() -> i64 { return 20; }
    \\exp fn boost() -> i64 { return helper(); }
    \\exp type Pair = struct { left: i64 };
    }};
    const liba_container = try library_format.pack(std.testing.allocator, compiler_version, "liba", &liba_members);
    defer std.testing.allocator.free(liba_container);
    const libb_container = try library_format.pack(std.testing.allocator, compiler_version, "libb", &libb_members);
    defer std.testing.allocator.free(libb_container);
    const packages = [_]TestPackage{
        .{ .name = "liba", .bytes = liba_container },
        .{ .name = "libb", .bytes = libb_container },
    };
    var set = TestPackageSet{ .packages = &packages };
    const loader: ModuleLoader = .{
        .context = @ptrCast(&set),
        .function = TestPackage.loadNoFiles,
        .library = TestPackageSet.loadLibrary,
    };

    // an unaliased import injects 'exp' names unqualified; an aliased one
    // is reachable through the alias only, so the two 'Pair' types do not
    // collide; both libraries reuse the internal name 'helper' and each
    // exported function calls its own one; the qualified struct literal
    // 'liba::Pair { ... }' names the injected library explicitly
    {
        var compilation = Compilation.init(std.testing.allocator);
        defer compilation.deinit();
        _ = try compilation.addModule("main.alloy",
            \\import pkg::liba;
            \\import pkg::libb as bb;
            \\fn main() -> i32 {
            \\    const bare = Pair { .left = 100 };
            \\    const qualified = liba::Pair { .left = 200 };
            \\    return (bare.left + qualified.left + contribute() + bb::boost()) to i32;
            \\}
        );
        const ran = try compilation.run(loader);
        if (!ran) {
            for (compilation.diagnostics.items) |diagnostic| {
                std.debug.print("diagnostic: {s}\n", .{diagnostic.message});
            }
        }
        try std.testing.expect(ran);
        var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer output.deinit();
        try std.testing.expectEqual(@as(i64, 330), try compilation.interpret(&output.writer));
    }

    // a foreign library-internal name stays invisible unqualified
    {
        var compilation = Compilation.init(std.testing.allocator);
        defer compilation.deinit();
        _ = try compilation.addModule("main.alloy",
            \\import pkg::liba;
            \\fn main() -> i32 {
            \\    return helper() to i32;
            \\}
        );
        try std.testing.expect(!try compilation.run(loader));
        const found = for (compilation.diagnostics.items) |diagnostic| {
            if (std.mem.indexOf(u8, diagnostic.message, "'helper' is not exported") != null) break true;
        } else false;
        try std.testing.expect(found);
    }

    // an aliased import does not inject: the bare name errors and points
    // at qualified access
    {
        var compilation = Compilation.init(std.testing.allocator);
        defer compilation.deinit();
        _ = try compilation.addModule("main.alloy",
            \\import pkg::liba as la;
            \\fn main() -> i32 {
            \\    return contribute() to i32;
            \\}
        );
        try std.testing.expect(!try compilation.run(loader));
        const found = for (compilation.diagnostics.items) |diagnostic| {
            if (std.mem.indexOf(u8, diagnostic.message, "'contribute' is exported by 'liba' but not imported unqualified") != null) break true;
        } else false;
        try std.testing.expect(found);
    }

    // the same name injected from two libraries is an error at the import
    {
        var compilation = Compilation.init(std.testing.allocator);
        defer compilation.deinit();
        _ = try compilation.addModule("main.alloy",
            \\import pkg::liba;
            \\import pkg::libb;
            \\fn main() -> i32 {
            \\    return contribute() to i32;
            \\}
        );
        try std.testing.expect(!try compilation.run(loader));
        const found = for (compilation.diagnostics.items) |diagnostic| {
            if (std.mem.indexOf(u8, diagnostic.message, "'Pair' is visible from both") != null and
                std.mem.indexOf(u8, diagnostic.message, "alias the import") != null) break true;
        } else false;
        try std.testing.expect(found);
    }

    // an own declaration colliding with an injected name is the same error
    {
        var compilation = Compilation.init(std.testing.allocator);
        defer compilation.deinit();
        _ = try compilation.addModule("main.alloy",
            \\import pkg::liba;
            \\fn contribute() -> i64 { return 1; }
            \\fn main() -> i32 {
            \\    return contribute() to i32;
            \\}
        );
        try std.testing.expect(!try compilation.run(loader));
        const found = for (compilation.diagnostics.items) |diagnostic| {
            if (std.mem.indexOf(u8, diagnostic.message, "'contribute' is visible from both") != null) break true;
        } else false;
        try std.testing.expect(found);
    }
}

test "runtime identity distinguishes same-named types from different libraries" {
    const libshape_members = [_]library_format.Member{.{ .key = "", .path = "libshape.alloy", .source =
    \\exp interface Shape {
    \\    fn area(self: &) -> i64;
    \\}
    }};
    const libcircle_members = [_]library_format.Member{.{ .key = "", .path = "libcircle.alloy", .source =
    \\import pkg::libshape;
    \\exp type Circle : Shape = struct { radius: i64 };
    \\exp fn area(self c: &Circle) -> i64 { return 10; }
    \\exp fn make() -> *Shape { return new Circle { .radius = 1 }; }
    }};
    const libshape_container = try library_format.pack(std.testing.allocator, compiler_version, "libshape", &libshape_members);
    defer std.testing.allocator.free(libshape_container);
    const libcircle_container = try library_format.pack(std.testing.allocator, compiler_version, "libcircle", &libcircle_members);
    defer std.testing.allocator.free(libcircle_container);
    const packages = [_]TestPackage{
        .{ .name = "libshape", .bytes = libshape_container },
        .{ .name = "libcircle", .bytes = libcircle_container },
    };
    var set = TestPackageSet{ .packages = &packages };
    const loader: ModuleLoader = .{
        .context = @ptrCast(&set),
        .function = TestPackage.loadNoFiles,
        .library = TestPackageSet.loadLibrary,
    };

    // the program declares its OWN 'Circle' implementing the same interface
    // (the library's Circle stays behind an alias); the interface object
    // holds the LIBRARY's Circle, so the arm naming the program's Circle
    // must not match, 'is' must say false, and dispatch must reach the
    // library's 'area' - name-based identity got all three wrong
    const source =
        \\import pkg::libshape;
        \\import pkg::libcircle as lc;
        \\type Circle : Shape = struct { side: i64 };
        \\fn area(self c: &Circle) -> i64 { return 20; }
        \\fn main() -> i32 {
        \\    var shape: *Shape = lc::make();
        \\    var matched = match (shape) {
        \\        Circle |&c| { yield 1; }
        \\        else { yield 2; }
        \\    };
        \\    if (shape is Circle) {
        \\        matched += 100;
        \\    }
        \\    return (matched * 100 + shape.area()) to i32;
        \\}
    ;

    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const ran = try compilation.run(loader);
    if (!ran) {
        for (compilation.diagnostics.items) |diagnostic| {
            std.debug.print("identity diagnostic: {s}\n", .{diagnostic.message});
        }
    }
    try std.testing.expect(ran);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try std.testing.expectEqual(@as(i64, 210), try compilation.interpret(&output.writer));

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const clang = testClang(arena) orelse return error.SkipZigTest;
    try expectNativeRun(arena, clang, &compilation, "cross_library_identity", false, 210, "");
}

test "implementers_of reflects the whole merged unit" {
    const libshape_members = [_]library_format.Member{.{ .key = "", .path = "libshape.alloy", .source =
    \\exp interface Shape {
    \\    fn area(self: &) -> i64;
    \\}
    }};
    // the implementer is library-INTERNAL: reflection still sees it
    const libcircle_members = [_]library_format.Member{.{ .key = "", .path = "libcircle.alloy", .source =
    \\import pkg::libshape;
    \\type Circle : Shape = struct { radius: i64 };
    \\fn area(self c: &Circle) -> i64 { return 10; }
    }};
    const libshape_container = try library_format.pack(std.testing.allocator, compiler_version, "libshape", &libshape_members);
    defer std.testing.allocator.free(libshape_container);
    const libcircle_container = try library_format.pack(std.testing.allocator, compiler_version, "libcircle", &libcircle_members);
    defer std.testing.allocator.free(libcircle_container);
    const packages = [_]TestPackage{
        .{ .name = "libshape", .bytes = libshape_container },
        .{ .name = "libcircle", .bytes = libcircle_container },
    };
    var set = TestPackageSet{ .packages = &packages };
    const loader: ModuleLoader = .{
        .context = @ptrCast(&set),
        .function = TestPackage.loadNoFiles,
        .library = TestPackageSet.loadLibrary,
    };

    // three implementers: Square (before the use), Hexagon (after the
    // use), and the library-internal Circle; the first in module then
    // declaration order is Square (6 letters)
    const source =
        \\import pkg::libshape;
        \\import pkg::libcircle;
        \\macro implementers_of(target) -> &[#Type];
        \\type Square : Shape = struct { side: i64 };
        \\type Plain = struct { value: i64 };
        \\fn area(self s: &Square) -> i64 { return 4; }
        \\fn main() -> i32 {
        \\    const count = #(implementers_of(Shape).length());
        \\    const first_name_length = #(implementers_of(Shape)[0].name().length());
        \\    const square_conforms = #(Square.implements_interface(Shape));
        \\    const plain_conforms = #(Plain.implements_interface(Shape));
        \\    var total = count * 10 + first_name_length;
        \\    if (square_conforms) {
        \\        total += 100;
        \\    }
        \\    if (plain_conforms) {
        \\        total += 1000;
        \\    }
        \\    return total to i32;
        \\}
        \\type Hexagon : Shape = struct { sides: i64 };
        \\fn area(self h: &Hexagon) -> i64 { return 6; }
    ;

    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const ran = try compilation.run(loader);
    if (!ran) {
        for (compilation.diagnostics.items) |diagnostic| {
            std.debug.print("implementers diagnostic: {s}\n", .{diagnostic.message});
        }
    }
    try std.testing.expect(ran);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try std.testing.expectEqual(@as(i64, 136), try compilation.interpret(&output.writer));

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const clang = testClang(arena) orelse return error.SkipZigTest;
    try expectNativeRun(arena, clang, &compilation, "implementers_reflection", false, 136, "");
}

test "native codegen emits checked LLVM IR" {
    try expectGenerates(
        \\fn double(value: i32) -> i32 { return value * 2; }
        \\fn main() -> i32 {
        \\    var total = 0;
        \\    for ([..4]) |i| {
        \\        total += double(i);
        \\    }
        \\    return total;
        \\}
    , false, &.{
        "define i32 @main(i32 %argc, ptr %argv)",
        "llvm.smul.with.overflow.i32",
        "@\"alloy.fault\"",
        "call void @llvm.trap()",
        // checked builds carry DWARF debug info (section 6.4)
        "!DISubprogram(name: \"main\"",
        "!DILocation(line:",
        "Debug Info Version",
        ", !dbg !",
        "!DILocalVariable(name: \"total\"",
        "llvm.dbg.declare",
        "!DIBasicType(name: \"i32\"",
    }, &.{});
    // release builds wrap arithmetic instead of trapping (section 5.2)
    // and skip debug info
    try expectGenerates(
        \\fn main() -> i32 {
        \\    var total = 40;
        \\    total += 2;
        \\    return total;
        \\}
    , true, &.{"add i32"}, &.{ "with.overflow", "!dbg", "DISubprogram" });
}

test "release codegen wraps arithmetic and keeps move bookkeeping" {
    try expectGenerates(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var total = 0;
        \\    const digits = [7, 8, 9];
        \\    for (digits) |d| {
        \\        total += d * 3;
        \\    }
        \\    var p: *var Box = new Box { .value = 4 };
        \\    var q: *var Box = move p;
        \\    return total + q.value;
        \\}
    , true, &.{
        "define i32 @main(i32 %argc, ptr %argv)",
        // the moved-from mark stays in release: drops depend on it
        "store ptr null",
        "= mul i32",
    }, &.{
        "with.overflow",
        "@\"alloy.fault\"",
        "call void @llvm.trap()",
    });
    // division by zero faults in every build mode (section 5.2)
    try expectGenerates(
        \\fn main() -> i32 {
        \\    var numerator = 10;
        \\    var denominator = 2;
        \\    return numerator / denominator;
        \\}
    , true, &.{
        "@\"alloy.fault\"",
    }, &.{
        "with.overflow",
    });
}

test "native 'to' guards conversions in checked builds only" {
    // negative-into-unsigned and float-into-integer range guards
    // (section 4.5); release converts straight
    const source =
        \\fn main() -> i32 {
        \\    var n: i64 = -5;
        \\    var u = n to u64;
        \\    var f: f64 = 2.5;
        \\    var v = f to i32;
        \\    return (u to i32) + v;
        \\}
    ;
    try expectGenerates(source, false, &.{
        "icmp sge i64",
        "fcmp oge double",
        "fcmp ole double",
        "@\"alloy.fault\"",
    }, &.{});
    try expectGenerates(source, true, &.{}, &.{
        "icmp sge",
        "fcmp oge",
        "@\"alloy.fault\"",
    });
    // the engines agree on the passing conversions
    try expectBuildsAndRuns("to_conversions",
        \\fn main() -> i32 {
        \\    var n: i64 = 120;
        \\    var wide = n to u64;
        \\    var back = wide to i32;
        \\    var f: f64 = -2.9;
        \\    var whole = f to i32;
        \\    return back + whole;
        \\}
    , 118, "");
}

test "release executables match checked executables" {
    try expectBuildsAndRunsMode("release_soundness",
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn makeAdder(base: i64) -> (i64) -> i64 {
        \\    return |base| (x: i64) -> i64 { return x + base; };
        \\}
        \\interface Critter {
        \\    fn legs(self: &) -> i64;
        \\}
        \\type Spider : Critter = struct { hairs: *var i64 };
        \\fn legs(self s: &Spider) -> i64 { return 8; }
        \\fn main() -> i32 {
        \\    var total: i64 = 0;
        \\    const fixed = [makeAdder(1) : 4];
        \\    total += fixed[0](1) + fixed[3](2);
        \\    var pet: *Critter = new Spider { .hairs = new 5 };
        \\    var second: *Critter = move pet;
        \\    total += second.legs();
        \\    pet = new Spider { .hairs = new 7 };
        \\    if (pet is Spider |&s|) {
        \\        total += s.hairs;
        \\    }
        \\    printf("total %d\n", total to i32);
        \\    return total to i32;
        \\}
    , null, true, 20, "total 20\n");
}

test "native codegen reports unsupported constructs" {
    try expectGenerateErrors(
        \\type Point = struct { x: i32, y: i32 };
        \\extern absorb(p: Point) -> i32;
        \\fn main() -> i32 {
        \\    const p = Point { .x = 1, .y = 2 };
        \\    return absorb(p);
        \\}
    , &.{"this type cannot cross the extern boundary yet"});
    try expectGenerateErrors(
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Point = struct { x: i32, y: i32 };
        \\fn main() -> i32 {
        \\    const p = Point { .x = 1, .y = 2 };
        \\    printf("point %d\n", p);
        \\    return 0;
        \\}
    , &.{"this argument cannot cross the extern boundary yet"});
}

test "native executables match the interpreter" {
    try expectBuildsAndRuns("structs_enums_match",
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Account = struct { id: u32, balance: i64 };
        \\type Event = enum { Deposit: i64, Withdraw: i64, Audit };
        \\fn apply(self a: &var Account, e: Event) -> bool {
        \\    match (e) {
        \\        ::Deposit |amount| { a.balance += amount; }
        \\        ::Withdraw |amount| {
        \\            if (amount > a.balance) { return false; }
        \\            a.balance -= amount;
        \\        }
        \\        ::Audit { printf("account %u holds %d\n", a.id, a.balance); }
        \\    }
        \\    return true;
        \\}
        \\fn main() -> i32 {
        \\    var account = Account { .id = 7, .balance = 100 };
        \\    const events = [Event::Deposit(50), Event::Withdraw(30), Event::Audit];
        \\    var refused = 0;
        \\    for (events) |e| {
        \\        if (!account.apply(e)) { refused += 1; }
        \\    }
        \\    for ([..3]) |round| {
        \\        account.apply(::Deposit(round to i64 * 10));
        \\    }
        \\    printf("refused %d\n", refused);
        \\    return account.balance to i32;
        \\}
    , 150, "account 7 holds 120\nrefused 0\n");
}

test "native executables reinterpret and slice like the interpreter" {
    try expectBuildsAndRuns("shapes_and_slices",
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Pair = struct { low: u32, high: u32 };
        \\fn main() -> i32 {
        \\    var pair = Pair { .low = 3, .high = 0 };
        \\    var packed: u64 = pair as u64;
        \\    var back = packed as Pair;
        \\    const text = "alloy";
        \\    var vowels = 0;
        \\    for (text) |letter| {
        \\        if (letter == 'a' || letter == 'o' || letter == 'y') { vowels += 1; }
        \\    }
        \\    printf("len %d vowels %d\n", text.length() to i32, vowels);
        \\    var countdown = back.low;
        \\    var total = while (countdown > 1) {
        \\        countdown -= 1;
        \\    } else { yield (countdown to i32) + 41; };
        \\    return total;
        \\}
    , 42, "len 5 vowels 3\n");
}

test "native codegen frees on every exit path" {
    try expectGenerates(
        \\type Box = struct { value: i32 };
        \\fn consume(p: *Box) -> i32 { return p.value; }
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 5 };
        \\    p = new Box { .value = 6 };
        \\    var taken = consume(move p);
        \\    var arr: *var [u32] = new [0 : 8];
        \\    return taken;
        \\}
    , false, &.{
        "@\"malloc\"",
        "@\"free\"",
        "alloy.drop.",
        "store ptr null",
    }, &.{});
}

test "native executables run heap pointers and deep copies" {
    try expectBuildsAndRuns("heap_basics",
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Inner = struct { value: i32 };
        \\type Outer = struct { label: i32, boxed: *var Inner };
        \\fn consume(p: *Inner) -> i32 { return p.value; }
        \\fn main() -> i32 {
        \\    var outer = Outer { .label = 2, .boxed = new Inner { .value = 40 } };
        \\    var clone = outer;
        \\    clone.boxed.value = 100;
        \\    const taken = consume(move clone.boxed);
        \\    printf("taken %d\n", taken);
        \\    return outer.label + outer.boxed.value;
        \\}
    , 42, "taken 100\n");
}

test "native executables manage heap arrays" {
    try expectBuildsAndRuns("heap_arrays",
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn main() -> i32 {
        \\    var n: u64 = 6;
        \\    var values: *var [u32] = new [1 : n];
        \\    for ([..6]) |i| {
        \\        values[i] = (i to u32) * 2;
        \\    }
        \\    var total: u32 = 0;
        \\    for (values) |v| {
        \\        total += v;
        \\    }
        \\    values = new [9 : 2];
        \\    total += values[0] + values[1];
        \\    printf("len %d total %d\n", values.length() to i32, total);
        \\    return total to i32;
        \\}
    , 48, "len 2 total 48\n");
}
test "native executables monomorphize generics" {
    try expectBuildsAndRuns("generics",
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Pair<T> = struct { first: T, second: T };
        \\fn swap<T>(p: &Pair<T>) -> Pair<T> {
        \\    return Pair { .first = p.second, .second = p.first };
        \\}
        \\fn pick<T>(a: T, b: T, take_first: bool) -> T {
        \\    if (take_first) { return a; }
        \\    return b;
        \\}
        \\fn main() -> i32 {
        \\    var numbers: Pair<i64> = Pair { .first = 2, .second = 40 };
        \\    var flipped = swap(&numbers);
        \\    var letters: Pair<u8> = Pair { .first = 'a', .second = 'z' };
        \\    var chosen = pick(letters.first, letters.second, true);
        \\    var wide = pick(numbers.first, numbers.second, false);
        \\    printf("first %d chosen %c wide %d\n", flipped.first to i32, chosen, wide);
        \\    return (flipped.first + wide) to i32;
        \\}
    , 80, "first 40 chosen a wide 40\n");
}
test "native executables dispatch through interface objects" {
    try expectBuildsAndRuns("interfaces",
        \\extern printf(format: &[u8], ...) -> i32;
        \\interface Shape {
        \\    fn area(self: &) -> i32;
        \\    fn sides(self: &) -> i32;
        \\}
        \\type Square : Shape = struct { width: i32 };
        \\type Circle : Shape = struct { radius: i32 };
        \\fn area(self s: &Square) -> i32 { return s.width * s.width; }
        \\fn sides(self s: &Square) -> i32 { return 4; }
        \\fn area(self c: &Circle) -> i32 { return 3 * c.radius * c.radius; }
        \\fn sides(self s: &Shape) -> i32 { return 0; }
        \\fn describe(s: &Shape) -> i32 {
        \\    if (s is Square |&sq|) {
        \\        printf("square %d\n", sq.width);
        \\    }
        \\    return s.area() + s.sides();
        \\}
        \\fn main() -> i32 {
        \\    var square = Square { .width = 5 };
        \\    var circle = Circle { .radius = 2 };
        \\    var total = describe(&square) + describe(&circle);
        \\    var viewed: &Shape = &circle;
        \\    var label = match (viewed) {
        \\        Square { yield 1; }
        \\        Circle { yield 2; }
        \\        else { yield 0; }
        \\    };
        \\    return total + label;
        \\}
    , 43, "square 5\n");
}

test "native executables match strings and run cursors" {
    const cursor_sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None, };" },
        .{ "std/iterable.alloy", "import std::option;\npub interface Iterator<T> { fn next(self: &var) -> Option<&T>; }\npub interface Iterable<T, It: Iterator<T>> { fn iterator(self: &) -> It; }" },
    });
    try expectBuildsAndRunsWith("strings_cursors",
        \\import std::option;
        \\import std::iterable;
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Countdown : Iterable<i32, CountdownCursor> = struct { current: i32 };
        \\type CountdownCursor : Iterator<i32> = struct { remaining: i32, current: i32 };
        \\fn iterator(self c: &Countdown) -> CountdownCursor {
        \\    return CountdownCursor { .remaining = c.current, .current = 0 };
        \\}
        \\fn next(self it: &var CountdownCursor) -> Option<&i32> {
        \\    if (it.remaining == 0) {
        \\        return ::None;
        \\    }
        \\    it.remaining -= 1;
        \\    it.current = it.remaining;
        \\    return ::Some(&it.current);
        \\}
        \\fn nameOf(code: i32) -> &[u8] {
        \\    var name = match (code) {
        \\        1 { yield "one"; }
        \\        2 { yield "two"; }
        \\        else { yield "many"; }
        \\    };
        \\    return &name;
        \\}
        \\fn main() -> i32 {
        \\    var sum = 0;
        \\    var c = Countdown { .current = 5 };
        \\    for (c) |step| {
        \\        sum += step;
        \\    }
        \\    var checks = 0;
        \\    match (nameOf(2)) {
        \\        "one" { checks += 1; }
        \\        "two" { checks += 10; }
        \\        else { checks += 100; }
        \\    }
        \\    printf("sum %d checks %d\n", sum, checks);
        \\    return sum + checks;
        \\}
    , &cursor_sources, 20, "sum 10 checks 10\n");
}

test "native executables run closures and function values" {
    try expectBuildsAndRuns("closures",
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn double(x: i64) -> i64 { return x * 2; }
        \\fn apply(f: (i64) -> i64, x: i64) -> i64 {
        \\    return f(x);
        \\}
        \\fn makeAdder(base: i64) -> (i64) -> i64 {
        \\    return |base| (x: i64) -> i64 { return x + base; };
        \\}
        \\type Box = struct { p: *var i64 };
        \\type Counter = struct { step: (i64) -> i64 };
        \\fn main() -> i32 {
        \\    var total: i64 = 0;
        \\    const add = (a: i64, b: i64) -> i64 { return a + b; };
        \\    total += add(3, 4);
        \\    var named: (i64) -> i64 = double;
        \\    total += named(5);
        \\    total += apply(double, 6);
        \\    total += apply(named, 7);
        \\    const add_ten = makeAdder(10);
        \\    total += add_ten(1);
        \\    var box = Box { .p = new 100 };
        \\    const peek = |box| () -> i64 { return box.p; };
        \\    total += peek();
        \\    box.p = new 1;
        \\    total += peek();
        \\    var owned: *var i64 = new 9;
        \\    const eat = |move owned| () -> i64 { return owned; };
        \\    total += eat();
        \\    total += eat();
        \\    const counter = Counter { .step = |&total| (n: i64) -> i64 { return total + n; } };
        \\    total += counter.step(1);
        \\    total += ((x: i64) -> i64 { return x + 100; })(0);
        \\    var f: (i64) -> i64 = (x: i64) -> i64 { return x + 1; };
        \\    f = (x: i64) -> i64 { return x + 2; };
        \\    total += f(0);
        \\    printf("total %d\n", total to i32);
        \\    return (total - 502) to i32;
        \\}
    , 145, "total 647\n");
}

test "native executables drop owning interface objects" {
    try expectBuildsAndRuns("owning_interfaces",
        \\extern printf(format: &[u8], ...) -> i32;
        \\interface Critter {
        \\    fn legs(self: &) -> i64;
        \\}
        \\type Spider : Critter = struct { hairs: *var i64 };
        \\type Worm : Critter = struct { tag: i64 };
        \\fn legs(self s: &Spider) -> i64 { return 8; }
        \\fn legs(self w: &Worm) -> i64 { return 0; }
        \\fn main() -> i32 {
        \\    var pet: *Critter = new Spider { .hairs = new 1000 };
        \\    var count = pet.legs();
        \\    if (pet is Spider |&s|) {
        \\        count += s.hairs;
        \\    }
        \\    pet = new Worm { .tag = 3 };
        \\    count += pet.legs();
        \\    var second: *Critter = new Spider { .hairs = new 50 };
        \\    match (second) {
        \\        Spider |&sp| { count += sp.hairs; }
        \\        else { count += 1; }
        \\    }
        \\    printf("count %d\n", count to i32);
        \\    return (count - 1000) to i32;
        \\}
    , 58, "count 1058\n");
}

test "native executables materialize comptime aggregate values" {
    try expectBuildsAndRuns("comptime_materialize",
        \\macro symbols() -> [&[u8] : 4] {
        \\    return ["+", "-=", "::", "..."];
        \\}
        \\macro numbers() -> [i32 : 3] {
        \\    return [3, 5, 7];
        \\}
        \\fn main() -> i32 {
        \\    const symbols = #symbols();
        \\    const counts = #numbers();
        \\    var total: u64 = 0;
        \\    for (symbols) |&symbol| {
        \\        total += symbol.length();
        \\    }
        \\    var sum: i32 = 0;
        \\    for (counts) |n| {
        \\        sum += n;
        \\    }
        \\    return total to i32 * 10 + sum;
        \\}
    , 95, "");
}

test "the empty array literal adopts its contextual type" {
    try expectBuildsAndRuns("empty_array_literal",
        \\fn none<T>() -> &[T] {
        \\    return [];
        \\}
        \\fn main() -> i32 {
        \\    const e = &none<i64>();
        \\    var total = e.length() + 3;
        \\    for (e) |x| { total += x to u64; }
        \\    const bytes: &[u8] = [];
        \\    total += bytes.length();
        \\    return total to i32;
        \\}
    , 3, "");
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const mystery = [];
        \\    return 0;
        \\}
    , &.{"an empty array literal '[]' is only valid where a '&[T]' slice is expected"});
}

test "native executables materialize comptime struct arrays" {
    try expectBuildsAndRuns("comptime_struct_materialize",
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Pair = struct {
        \\    name: &[u8],
        \\    symbol: &[u8],
        \\    weight: u8
        \\};
        \\macro pairs() -> [Pair : 2] {
        \\    return [
        \\        Pair { .name = "plus", .symbol = "+", .weight = 1 },
        \\        Pair { .name = "arrow", .symbol = "->", .weight = 2 }
        \\    ];
        \\}
        \\fn main() -> i32 {
        \\    const table = #pairs();
        \\    var total: i32 = 0;
        \\    for (table) |&pair| {
        \\        printf("%.*s %.*s\n", pair.name.length() to i32, &pair.name, pair.symbol.length() to i32, &pair.symbol);
        \\        total += pair.weight to i32 * 10 + pair.symbol.length() to i32;
        \\    }
        \\    return total;
        \\}
    , 33, "plus +\narrow ->\n");
}

test "native executables monomorphize lambdas inside generics" {
    try expectBuildsAndRuns("generic_lambdas",
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn pick<T>(flag: bool, a: T, b: T) -> T {
        \\    const choose = |flag| (x: T, y: T) -> T {
        \\        if (flag) { return x; }
        \\        return y;
        \\    };
        \\    return choose(a, b);
        \\}
        \\fn main() -> i32 {
        \\    var total: i64 = pick(true, 40 to i64, 2 to i64);
        \\    total += pick(false, 1 to i64, 2 to i64);
        \\    var index = 0;
        \\    while (index < 3) {
        \\        var boxed: *var i64 = new (index to i64);
        \\        const grab = |move boxed| () -> i64 { return boxed; };
        \\        total += grab();
        \\        index += 1;
        \\    }
        \\    printf("total %d\n", total to i32);
        \\    return total to i32;
        \\}
    , 45, "total 45\n");
}

test "function values reject comparison, externs, and generics" {
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const a = (x: i64) -> i64 { return x; };
        \\    const b = a;
        \\    if (a == b) { return 1; }
        \\    return 0;
        \\}
    , &.{"function values cannot be compared (section 5.4)"});
    try expectCheckErrors(
        \\extern abs(x: i32) -> i32;
        \\fn main() -> i32 {
        \\    var f: (i32) -> i32 = abs;
        \\    return f(-1);
        \\}
    , &.{"an extern function cannot be used as a function value yet (section 6.3)"});
    try expectCheckErrors(
        \\fn pick<T>(a: T, b: T) -> T { return a; }
        \\fn main() -> i32 {
        \\    const f = pick;
        \\    return 0;
        \\}
    , &.{"a generic function cannot become a function value; its type parameters are unbound (section 5.4)"});
}

test "extensions win over function-typed fields in dynamic dispatch" {
    try expectRuns(
        \\interface Greeter {
        \\    fn id(self: &) -> i64;
        \\}
        \\type Robot : Greeter = struct { id: () -> i64 };
        \\fn id(self r: &Robot) -> i64 { return 100; }
        \\fn main() -> i32 {
        \\    var robot = Robot { .id = () -> i64 { return 7; } };
        \\    var viewed: &Greeter = &robot;
        \\    return viewed.id() to i32;
        \\}
    , 100, "");
}

test "move analysis rejects definite uses and accepts conditional ones" {
    try expectCheckErrors(
        \\type Box = struct { value: i32 };
        \\fn eat(b: *Box) -> i32 { return b.value; }
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 1 };
        \\    var first = eat(move p);
        \\    var second = eat(move p);
        \\    return first + second;
        \\}
    , &.{"'p' was already moved (section 5.2)"});
    try expectCheckErrors(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 1 };
        \\    var q: *var Box = move p;
        \\    p.value = 3;
        \\    return q.value;
        \\}
    , &.{"use of 'p' after 'move' (section 5.2)"});
    try expectChecks(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 1 };
        \\    var q: *var Box = move p;
        \\    p = new Box { .value = 2 };
        \\    return p.value + q.value;
        \\}
    );
    try expectChecks(
        \\type Box = struct { value: i32 };
        \\fn eat(b: *Box) -> i32 { return b.value; }
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 1 };
        \\    var flag = true;
        \\    var total = 0;
        \\    if (flag) {
        \\        total += eat(move p);
        \\    } else {
        \\        total += p.value;
        \\    }
        \\    return total;
        \\}
    );
    try expectChecks(
        \\type Box = struct { value: i32 };
        \\fn eat(b: *Box) -> i32 { return b.value; }
        \\fn main() -> i32 {
        \\    var items = [1, 2];
        \\    var p: *var Box = new Box { .value = 1 };
        \\    var total = 0;
        \\    for (items) |n| {
        \\        total += eat(move p) + n;
        \\        p = new Box { .value = 2 };
        \\    }
        \\    return total;
        \\}
    );
}

test "flow analysis requires every path to return or yield" {
    try expectCheckErrors(
        \\fn pick(flag: bool) -> i32 {
        \\    if (flag) { return 1; }
        \\}
    , &.{"control can fall off the end of 'pick', which must return i32 on every path (section 5.3)"});
    try expectCheckErrors(
        \\fn drain() -> i32 {
        \\    while (true) {
        \\        if (done()) { break; }
        \\    }
        \\}
        \\fn done() -> bool { return true; }
    , &.{"control can fall off the end of 'drain'"});
    try expectCheckErrors(
        \\fn label(flag: bool) -> i32 {
        \\    var chosen = if (flag) { yield 1; } else { };
        \\    return chosen;
        \\}
    , &.{"a branch of this value-yielding if can complete without 'yield' (section 5.3)"});
    try expectCheckErrors(
        \\fn grade(code: i32) -> i32 {
        \\    var value = match (code) {
        \\        0 { yield 1; }
        \\        else { }
        \\    };
        \\    return value;
        \\}
    , &.{"an arm of this value-yielding match can complete without 'yield'; add 'yield' to every arm or an external 'else' (section 5.3)"});
    try expectCheckErrors(
        \\fn count() -> i32 {
        \\    var total = 0;
        \\    var capped = while (total < 5) {
        \\        total += 1;
        \\        break total;
        \\    } else { };
        \\    return capped;
        \\}
    , &.{"the 'else' of this value-yielding loop can complete without 'yield' (section 5.3)"});
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const f = (flag: bool) -> i32 {
        \\        if (flag) { return 1; }
        \\    };
        \\    return f(true);
        \\}
    , &.{"control can fall off the end of this lambda, which must return i32 on every path (section 5.3)"});
    try expectChecks(
        \\fn spin() -> i32 {
        \\    while (true) {
        \\        poke();
        \\    }
        \\}
        \\fn poke() { }
    );
    try expectChecks(
        \\fn pick(flag: bool) -> i32 {
        \\    if (flag) { return 1; }
        \\    return 2;
        \\}
    );
}

test "break exits loops through ifs and yield reaches the value construct" {
    const source =
        \\fn main() -> i32 {
        \\    var total = 0;
        \\    for ([..10]) |n| {
        \\        total += n;
        \\        if (total > 6) { break; }
        \\    }
        \\    var label = if (total == 10) { yield 5; } else { yield 0; };
        \\    var crossed = if (true) {
        \\        for ([..5]) |k| {
        \\            if (k == 2) { yield 100; }
        \\        }
        \\        yield 1;
        \\    } else { yield 0; };
        \\    return total + label + crossed;
        \\}
    ;
    try expectRuns(source, 115, "");
    try expectBuildsAndRuns("break_yield_targets", source, 115, "");
}

test "native executables release loop temporaries and interrupted cursors" {
    const option_sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None, };" },
        .{ "std/iterable.alloy", "import std::option;\npub interface Iterator<T> { fn next(self: &var) -> Option<&T>; }\npub interface Iterable<T, It: Iterator<T>> { fn iterator(self: &) -> It; }" },
    });
    try expectBuildsAndRunsWith("loop_releases",
        \\import std::option;
        \\import std::iterable;
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Wallet = struct { cash: *var i64 };
        \\fn makeWallets() -> [Wallet : 2] {
        \\    return [Wallet { .cash = new 3 } : 2];
        \\}
        \\type Boxes : Iterable<i64, BoxCursor> = struct { limit: i64 };
        \\type BoxCursor : Iterator<i64> = struct { remaining: i64, current: i64, hoard: *var i64 };
        \\fn iterator(self b: &Boxes) -> BoxCursor {
        \\    return BoxCursor { .remaining = b.limit, .current = 0, .hoard = new 7 };
        \\}
        \\fn next(self it: &var BoxCursor) -> Option<&i64> {
        \\    if (it.remaining == 0) {
        \\        return ::None;
        \\    }
        \\    it.remaining -= 1;
        \\    it.current = it.remaining * 10;
        \\    return ::Some(&it.current);
        \\}
        \\fn main() -> i32 {
        \\    var total: i64 = 0;
        \\    for (makeWallets()) |w| {
        \\        total += w.cash;
        \\    }
        \\    const boxes = Boxes { .limit = 5 };
        \\    for (boxes) |value| {
        \\        total += value;
        \\        break;
        \\    }
        \\    const bump = (x: i64) -> i64 { return x + 1; };
        \\    total += bump(0);
        \\    printf("total %d\n", total to i32);
        \\    return total to i32;
        \\}
    , &option_sources, 47, "total 47\n");
}

test "native executables fill, move, and receive closures soundly" {
    try expectBuildsAndRuns("closure_soundness",
        \\extern printf(format: &[u8], ...) -> i32;
        \\interface Critter {
        \\    fn legs(self: &) -> i64;
        \\}
        \\type Spider : Critter = struct { hairs: *var i64 };
        \\fn legs(self s: &Spider) -> i64 { return 8; }
        \\fn makeAdder(base: i64) -> (i64) -> i64 {
        \\    return |base| (x: i64) -> i64 { return x + base; };
        \\}
        \\type Holder = struct { callback: (i64) -> i64 };
        \\fn pick() -> i64 {
        \\    printf("pick\n");
        \\    return 0;
        \\}
        \\type Counter = struct { step: (i64) -> i64 };
        \\fn makeCounter() -> Counter {
        \\    return Counter { .step = (x: i64) -> i64 { return x + 1; } };
        \\}
        \\fn main() -> i32 {
        \\    var total: i64 = 0;
        \\    const fixed = [makeAdder(1) : 4];
        \\    total += fixed[0](1) + fixed[3](2);
        \\    var heap: *[(i64) -> i64] = new [makeAdder(10) : 3];
        \\    total += heap[2](5);
        \\    var pet: *Critter = new Spider { .hairs = new 5 };
        \\    var second: *Critter = move pet;
        \\    total += second.legs();
        \\    var third: *Critter = new Spider { .hairs = new 7 };
        \\    const kill = |move third| () -> i64 { return third.legs(); };
        \\    total += kill();
        \\    var holders = [Holder { .callback = (x: i64) -> i64 { return x + 1; } }];
        \\    total += holders[pick()].callback(4);
        \\    total += makeCounter().step(41);
        \\    var text = "abc";
        \\    const sized = |&text| () -> i64 { return text.length() to i64; };
        \\    total += sized();
        \\    printf("total %d\n", total to i32);
        \\    return (total - 60) to i32;
        \\}
    , 26, "pick\ntotal 86\n");
}

test "heap arrays borrow as slices, subslice, and move on return" {
    try expectRuns(
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn sum(values: &[u8]) -> i64 {
        \\    var total: i64 = 0;
        \\    for (values) |value| {
        \\        total += value to i64;
        \\    }
        \\    return total;
        \\}
        \\fn make(length: u64) -> *var [u8] {
        \\    var buffer: *var [u8] = new [3 : length];
        \\    buffer[0] = 9;
        \\    return move buffer;
        \\}
        \\fn duplicate(values: &[u8]) -> *var [u8] {
        \\    var copied: *var [u8] = new [0 : values.length()];
        \\    var index: u64 = 0;
        \\    for (values) |value| {
        \\        copied[index] = value;
        \\        index += 1;
        \\    }
        \\    // a heap array leaves explicitly: move transfers it
        \\    return move copied;
        \\}
        \\fn main() -> i64 {
        \\    const owned = make(4);
        \\    const view: &[u8] = &owned;
        \\    const tail = &owned[2..4];
        \\    const doubled = duplicate(&view);
        \\    const text: &[u8] = "abc";
        \\    const middle = &text[1..2];
        \\    printf("%d %d %d %d %d\n", view.length(), sum(&view), sum(&tail), middle[0], sum(&doubled));
        \\    return sum(&view) + sum(&tail);
        \\}
    , 24, "4 18 6 98 18\n");
    try expectRunFault(
        \\fn main() -> i64 {
        \\    var buffer: *var [u8] = new [1 : 4];
        \\    const bad = &buffer[3..9];
        \\    return bad.length() to i64;
        \\}
    , "out of bounds");
}

test "native executables borrow, subslice, and return heap arrays" {
    try expectBuildsAndRuns("stdlib_views",
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn sum(values: &[u8]) -> i64 {
        \\    var total: i64 = 0;
        \\    for (values) |value| {
        \\        total += value to i64;
        \\    }
        \\    return total;
        \\}
        \\fn make(length: u64) -> *var [u8] {
        \\    var buffer: *var [u8] = new [3 : length];
        \\    buffer[0] = 9;
        \\    return move buffer;
        \\}
        \\fn duplicate(values: &[u8]) -> *var [u8] {
        \\    var copied: *var [u8] = new [0 : values.length()];
        \\    var index: u64 = 0;
        \\    for (values) |value| {
        \\        copied[index] = value;
        \\        index += 1;
        \\    }
        \\    // a heap array leaves explicitly: move transfers it
        \\    return move copied;
        \\}
        \\fn main() -> i32 {
        \\    const owned = make(4);
        \\    const view: &[u8] = &owned;
        \\    const tail = &owned[2..4];
        \\    const doubled = duplicate(&view);
        \\    const text: &[u8] = "abc";
        \\    const middle = &text[1..2];
        \\    printf("%d %d %d %d %d\n", view.length(), sum(&view), sum(&tail), middle[0], sum(&doubled));
        \\    return (sum(&view) + sum(&tail)) to i32;
        \\}
    , 24, "4 18 6 98 18\n");
}

test "borrow mutability is explicit: '&' is immutable, '&var' mutable" {
    // '&x' into a '&var T' parameter names the fix (section 5.2)
    try expectCheckErrors(
        \\fn bump(count: &var i32) { count += 1; }
        \\fn main() -> i32 {
        \\    var n: i32 = 0;
        \\    bump(&n);
        \\    return n;
        \\}
    , &.{"this borrow must be mutable here: write '&var' (section 5.2)"});
    // the same for a mutable slice parameter fed a heap array borrow
    try expectCheckErrors(
        \\fn fill(values: &var [u8]) { values[0] = 1; }
        \\fn main() -> i32 {
        \\    var buffer: *var [u8] = new [0 : 3];
        \\    fill(&buffer);
        \\    return 0;
        \\}
    , &.{"this borrow must be mutable here: write '&var' (section 5.2)"});
    // '&var' demands a mutable subject (section 5.2)
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const n: i32 = 3;
        \\    var r: &var i32 = &var n;
        \\    return r;
        \\}
    , &.{"a '&var' borrow requires a mutable subject (section 5.2)"});
    // re-borrowing through an immutable reference cannot add mutability
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    var n: i32 = 1;
        \\    const r: &i32 = &n;
        \\    var w: &var i32 = &var r;
        \\    return w;
        \\}
    , &.{"a '&var' borrow requires a mutable subject (section 5.2)"});
    // an assignment-position mismatch points at the form too
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    var n: i32 = 1;
        \\    var w: &var i32 = &n;
        \\    return w;
        \\}
    , &.{"expected &var i32: write '&var' to borrow mutably (section 5.2)"});
    // '&r' on a mutable source downgrades to the immutable view
    try expectCheckErrors(
        \\fn bump(count: &var i32) { count += 1; }
        \\fn main() -> i32 {
        \\    var n: i32 = 0;
        \\    var w: &var i32 = &var n;
        \\    bump(&w);
        \\    return n;
        \\}
    , &.{"this borrow must be mutable here: write '&var' (section 5.2)"});
    // a mutable borrow still fits an immutable target (section 4.3)
    try expectChecks(
        \\fn read(count: &i32) -> i32 { return count; }
        \\fn main() -> i32 {
        \\    var n: i32 = 1;
        \\    return read(&var n);
        \\}
    );
    // the engines agree on mutation through explicit '&var' borrows
    try expectBuildsAndRuns("explicit_borrows",
        \\fn bump(count: &var i32) { count += 1; }
        \\fn fill(values: &var [u8], value: u8) { values[0] = value; }
        \\fn main() -> i32 {
        \\    var n: i32 = 0;
        \\    bump(&var n);
        \\    bump(&var n);
        \\    var w: &var i32 = &var n;
        \\    bump(&var w);
        \\    var buffer: *var [u8] = new [3 : 2];
        \\    fill(&var buffer, 40);
        \\    return n + buffer[0] to i32 + buffer[1] to i32;
        \\}
    , 46, "");
}

test "a bare subslice is the unsized array value" {
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const text: &[u8] = "abcd";
        \\    const middle = text[1..3];
        \\    return middle.length() to i32;
        \\}
    , &.{"a subslice is the unsized array value: '&' borrows the view, '&var' borrows it mutably, 'new' copies it (section 3.1)"});
    // a mutable view needs a mutable subject
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const text: &[u8] = "abcd";
        \\    const window = &var text[1..3];
        \\    return window[0] to i32;
        \\}
    , &.{"a '&var' borrow requires a mutable subject (section 5.2)"});
    // '&'/'&var' keep the view, 'new' copies the range, and in-place
    // consumers (a 'for' subject, a receiver) still take it bare
    try expectBuildsAndRuns("subslice_views",
        \\fn main() -> i32 {
        \\    var buffer: *var [u8] = new [5 : 6];
        \\    const window = &var buffer[1..4];
        \\    window[0] = 9;
        \\    var total: u64 = 0;
        \\    for (buffer[..2]) |b| {
        \\        total += b to u64;
        \\    }
        \\    const copied = new buffer[2..5];
        \\    return (total + window.length() + copied.length() + copied[0] to u64) to i32;
        \\}
    , 25, "");
}

test "file io externs fault without a host filesystem" {
    try expectRunFault(
        \\extern fopen(path: &[u8], mode: &[u8]) -> i64;
        \\fn main() -> i64 {
        \\    return fopen("anything.txt", "rb");
        \\}
    , "file io is unavailable here");
}

test "the interpreter serves process arguments through the lang item" {
    var sources = TestSources.initComptime(.{
        .{ "std/process.alloy", "pub fn arguments() -> &[&[u8]] {\n    const empty: [&[u8] : 1] = [\"\"];\n    return &empty[..0];\n}" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::process;
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn main() -> i64 {
        \\    const all = &arguments();
        \\    for (all) |argument| {
        \\        printf("%s\n", &argument);
        \\    }
        \\    return all.length() to i64;
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const run_arguments = [_][]const u8{ "program.alloy", "alpha", "beta" };
    const exit_code = try compilation.interpretWithEnvironment(&output.writer, .{ .arguments = &run_arguments });
    try std.testing.expectEqualStrings("program.alloy\nalpha\nbeta\n", output.writer.buffered());
    try std.testing.expectEqual(@as(i64, 3), exit_code);
}

test "qualified functions reject a self receiver" {
    try expectCheckErrors(
        \\type Point = struct { x: i64 };
        \\fn Point::shift(self p: &var Point, amount: i64) {
        \\    p.x += amount;
        \\}
        \\fn main() -> i64 { return 0; }
    , &.{"'Point::shift' cannot take 'self': a qualified function is not an extension (section 6.4)"});
}

test "qualified functions live in their type's namespace" {
    var sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::option;
        \\type Point = struct { x: i64, y: i64 };
        \\fn Point::at(x: i64, y: i64) -> Point {
        \\    return Point { .x = x, .y = y };
        \\}
        \\fn Option::flag<T>(condition: bool, value: T) -> Option<T> {
        \\    if (condition) {
        \\        return ::Some(value);
        \\    }
        \\    return ::None;
        \\}
        \\fn main() -> i64 {
        \\    const moved = Point::at(3, 4);
        \\    var chosen: Option<i64> = Option::flag(true, 5);
        \\    var fallback: i64 = 0;
        \\    if (chosen is ::Some |value|) {
        \\        fallback = value;
        \\    }
        \\    return moved.x + moved.y + fallback;
        \\}
    );
    const loader: ModuleLoader = .{ .context = @constCast(@ptrCast(&sources)), .function = testLoader };
    try expectRunSucceeds(&compilation, loader);
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 12), exit_code);
}

// --- spec revision 2026-08-23 conformance ---------------------------------

test "inline layouts reflect as #Type in both engines" {
    try expectBuildsAndRuns("inline_layouts",
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn development() -> bool { return true; }
        \\type P = #if (development()) #struct { id: u32, tag: u8 } else #struct { id: u64 };
        \\type Mode = #enum { Fast, Slow: u8 };
        \\fn main() -> i32 {
        \\    const p = P { .id = 7, .tag = 2 };
        \\    var m: Mode = ::Slow(3);
        \\    var total: i32 = p.id to i32 + p.tag to i32;
        \\    if (m is ::Slow |s|) { total += s to i32; }
        \\    total += #struct { a: u8, b: u16 }.member_names().length() to i32;
        \\    total += #if (#struct { x: i32 }.is_struct()) 100 else 0;
        \\    printf("total %d\n", total);
        \\    return total;
        \\}
    , 114, "total 114\n");
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    var layout = #struct { id: u32 };
        \\    return 0;
        \\}
    , &.{"a '#Type' cannot be retained in a runtime declaration"});
}

test "value loops yield from the body, break a value, and yield from the else" {
    try expectBuildsAndRuns("value_loops",
        \\fn first_even(items: &[i64]) -> i64 {
        \\    return for (items) |x| {
        \\        if (x % 2 == 0) { yield x; }
        \\    } else {
        \\        yield -1;
        \\    };
        \\}
        \\fn count_to(limit: i64) -> i64 {
        \\    var n: i64 = 0;
        \\    const hit = while (n < limit) {
        \\        n += 1;
        \\        if (n == 3) { break n * 10; }
        \\    } else {
        \\        yield n;
        \\    };
        \\    return hit;
        \\}
        \\fn outer(flag: bool) -> i64 {
        \\    // a yield inside a statement loop passes out to the value if
        \\    return if (flag) {
        \\        var i = 0;
        \\        while (i < 5) {
        \\            i += 1;
        \\            if (i == 2) { yield 200; }
        \\        }
        \\        yield 0;
        \\    } else 7;
        \\}
        \\fn main() -> i32 {
        \\    var a: *var [i64] = new [0 : 3];
        \\    a[0] = 1;
        \\    a[1] = 3;
        \\    a[2] = 4;
        \\    var b: *var [i64] = new [1 : 2];
        \\    var total = first_even(&a) + first_even(&b);
        \\    total += count_to(10) + count_to(2);
        \\    total += outer(true) + outer(false);
        \\    return total to i32;
        \\}
    , 242, "");
    try expectCheckErrors(
        \\fn f() -> i32 {
        \\    var n = 0;
        \\    while (n < 3) {
        \\        n += 1;
        \\        break n;
        \\    }
        \\    return n;
        \\}
    , &.{"'break value' needs a loop used as a value"});
    try expectCheckErrors(
        \\fn f() -> i32 {
        \\    var n = 0;
        \\    const v = while (n < 3) {
        \\        n += 1;
        \\    } else {
        \\        n = 0;
        \\    };
        \\    return v;
        \\}
    , &.{
        "the 'else' of this value-yielding loop can complete without 'yield'",
        "this loop is used as a value but never yields one",
    });
}

test "bare-expression if branches and match arms yield implicitly" {
    try expectBuildsAndRuns("bare_branches",
        \\extern printf(format: &[u8], ...) -> i32;
        \\type State = enum { Idle, Busy: i32 };
        \\fn grade(score: i32) -> &[u8] {
        \\    return if (score > 90) "high" else "low";
        \\}
        \\fn load(s: State) -> i32 {
        \\    return match (s) {
        \\        ::Idle 0;
        \\        ::Busy |l| l;
        \\        else -1;
        \\    };
        \\}
        \\fn mixed(c: bool) -> i32 {
        \\    return if (c) 1 else { yield 2; };
        \\}
        \\fn chain(a: bool, b: bool) -> i32 {
        \\    return if (a) 1 else if (b) 2 else 3;
        \\}
        \\fn main() -> i32 {
        \\    printf("%s %s\n", &grade(95), &grade(10));
        \\    var total = load(::Idle) + load(::Busy(5));
        \\    total += mixed(true) + mixed(false);
        \\    total += chain(false, true) + chain(false, false);
        \\    return total;
        \\}
    , 13, "high low\n");
}

test "logical operators short-circuit, bitwise ones do not" {
    try expectBuildsAndRuns("short_circuit",
        \\fn tick(count: &var i32, value: bool) -> bool {
        \\    count += 1;
        \\    return value;
        \\}
        \\fn bit(count: &var i32, value: i32) -> i32 {
        \\    count += 1;
        \\    return value;
        \\}
        \\fn main() -> i32 {
        \\    var n: i32 = 0;
        \\    if (tick(&var n, false) && tick(&var n, true)) { n += 100; }
        \\    if (tick(&var n, true) || tick(&var n, true)) { }
        \\    if ((bit(&var n, 0) & bit(&var n, 1)) != 0) { n += 100; }
        \\    return n;
        \\}
    , 4, "");
}

test "interface functions declare their receiver indirection" {
    try expectBuildsAndRuns("interface_receivers",
        \\interface Counter {
        \\    fn bump(self: &var);
        \\    fn peek(self: &) -> i32;
        \\}
        \\type C : Counter = struct { n: i32 };
        \\fn bump(self c: &var C) { c.n += 1; }
        \\fn peek(self c: &C) -> i32 { return c.n; }
        \\fn main() -> i32 {
        \\    var c = C { .n = 1 };
        \\    var object: &var Counter = &var c;
        \\    object.bump();
        \\    object.bump();
        \\    const view: &Counter = &c;
        \\    return view.peek();
        \\}
    , 3, "");
    try expectCheckErrors(
        \\interface Counter {
        \\    fn bump(self: &var);
        \\    fn peek(self: &) -> i32;
        \\}
        \\type C : Counter = struct { n: i32 };
        \\fn bump(self c: &C) { }
        \\fn peek(self c: &C) -> i32 { return c.n; }
        \\fn poke(object: &Counter) { object.bump(); }
    , &.{
        "takes the wrong receiver: 'Counter' declares 'self: &var'",
        "'bump' is declared 'self: &var' in 'Counter', which a '&Counter' object cannot provide",
    });
}

test "interface-object captures state their form and may take ownership" {
    try expectBuildsAndRuns("interface_move_capture",
        \\interface Pet {
        \\    fn legs(self: &) -> i64;
        \\}
        \\type Spider : Pet = struct { hairs: *i64 };
        \\fn legs(self s: &Spider) -> i64 { return 8; }
        \\fn main() -> i32 {
        \\    var pet: *Pet = new Spider { .hairs = new 5 };
        \\    var total: i64 = 0;
        \\    if (pet is Spider |move s|) {
        \\        total += s.hairs + s.legs();
        \\    }
        \\    var second: *Pet = new Spider { .hairs = new 7 };
        \\    match (second) {
        \\        Spider |move sp| { total += sp.hairs; }
        \\        else { total += 1000; }
        \\    }
        \\    return total to i32;
        \\}
    , 20, "");
    try expectCheckErrors(
        \\interface Pet {
        \\    fn legs(self: &) -> i64;
        \\}
        \\type Spider : Pet = struct { hairs: i64 };
        \\fn legs(self s: &Spider) -> i64 { return 8; }
        \\fn f(pet: &Pet) {
        \\    if (pet is Spider |s|) { }
        \\    if (pet is Spider |&var s|) { }
        \\    if (pet is Spider |move s|) { }
        \\}
    , &.{
        "an interface-object capture must state its form",
        "needs a '&var' or '*var' interface object",
        "needs an owning '*' or '*var' interface object",
    });
}

test "extern functions never return owning pointers" {
    try expectCheckErrors(
        \\extern make() -> *u8;
        \\extern release(buffer: &var u8);
    , &.{"may not return the owning type *u8"});
}

test "named type transparency runs one way down the chain" {
    try expectChecks(
        \\type Meters = f32;
        \\type Distance = Meters;
        \\fn f(d: Distance) -> f32 {
        \\    const m: Meters = d;
        \\    const raw: f32 = m;
        \\    const literal: Meters = 1.5;
        \\    return raw + d + literal;
        \\}
    );
    try expectCheckErrors(
        \\type Meters = f32;
        \\type Feet = f32;
        \\fn up(raw: f32) {
        \\    const m: Meters = raw;
        \\}
        \\fn across(m: Meters) {
        \\    const f: Feet = m;
        \\}
    , &.{
        "expected Meters, found f32",
        "expected Feet, found Meters",
    });
}

test "statement-position constructs take no else and no bare bodies" {
    try expectCheckErrors(
        \\fn f(n: i32) {
        \\    match (n) {
        \\        0 { }
        \\    } else { }
        \\}
    , &.{"an external 'else' on a match is only valid when the match is used as a value"});
    try expectCheckErrors(
        \\fn f() -> i32 {
        \\    return 1;;
        \\}
    , &.{"this ';' terminates nothing"});
}

test "macros declare their result type and it governs the call" {
    // the declared type types the call without running the body, so a
    // body may reference definitions declared later (section 7.3)
    try expectBuildsAndRuns("macro_declared_types",
        \\macro twice(x: i64) -> i64 {
        \\    return helper(x) * 2;
        \\}
        \\macro names() -> [&[u8] : 2] {
        \\    return ["a", "bc"];
        \\}
        \\type Pair = struct { left: i64, right: i64 };
        \\macro pair() -> Pair {
        \\    return Pair { .left = 3, .right = 4 };
        \\}
        \\fn helper(x: i64) -> i64 { return x + 1; }
        \\fn main() -> i32 {
        \\    const doubled = #twice(20);
        \\    const table = #names();
        \\    const p = #pair();
        \\    return (doubled + table[1].length() to i64 + p.left + p.right) to i32;
        \\}
    , 51, "");
    try expectCheckErrors(
        \\macro mismatch() -> i32 {
        \\    return "text";
        \\}
        \\fn main() -> i32 {
        \\    const v = #mismatch();
        \\    return v;
        \\}
    , &.{
        "expected i32, found &[u8]",
        "this compile-time expression produced &[u8] but is declared i32",
    });
    try expectCheckErrors(
        \\macro layout() -> #Type;
        \\fn f(t: #Type) -> #Type { return t; }
        \\fn main() -> i32 { return 0; }
    , &.{
        "'#Type' may only appear in a macro signature",
        "'#Type' may only appear in a macro signature",
    });
}

test "a return unwinding through a loop leaves the caller's frames intact" {
    // 'return' inside a statement-if inside a 'for' unwinds on the error
    // channel; the loop and if frames must drop with it, or the caller's
    // own bindings vanish behind a stale barrier (the leaked-frame bug)
    try expectBuildsAndRuns("frame_unwind",
        \\type S = struct { count: i64 };
        \\fn find(a: &[u8]) -> bool {
        \\    for (a) |x| {
        \\        if (x == 'b') {
        \\            return true;
        \\        }
        \\    }
        \\    return false;
        \\}
        \\fn bump(state: &var S) {
        \\    const hit = find("abc");
        \\    state.count += 1;
        \\    if (hit) { state.count += 10; }
        \\}
        \\fn main() -> i32 {
        \\    var s = S { .count = 0 };
        \\    bump(&var s);
        \\    bump(&var s);
        \\    return s.count to i32;
        \\}
    , 22, "");
}

test "borrowing captures reach through owning payloads to the pointee" {
    // '|&x|' on a '*T' payload binds '&T', on a '*[T]' payload the slice
    // '&[T]' - never a reference to the pointer itself (section 3.1)
    try expectBuildsAndRuns("capture_pointee",
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Holder = enum { Data: *[u8], Boxed: *i64, Empty };
        \\fn view_len(text: &[u8]) -> u64 {
        \\    return text.length();
        \\}
        \\fn main() -> i32 {
        \\    var h: Holder = Holder::Data(new [65 : 5]);
        \\    var total: i64 = 0;
        \\    if (h is ::Data |&bytes|) {
        \\        total += view_len(&bytes) to i64;
        \\        total += bytes.length() to i64;
        \\    }
        \\    match (h) {
        \\        ::Data |&bytes| { total += bytes[0] to i64; }
        \\        else { total += 1000; }
        \\    }
        \\    var b: Holder = Holder::Boxed(new 7);
        \\    if (b is ::Boxed |&inner|) {
        \\        total += inner;
        \\    }
        \\    printf("total %d\n", total to i32);
        \\    return total to i32;
        \\}
    , 82, "total 82\n");
    // the bare use of such a capture means the unsized array value
    try expectCheckErrors(
        \\type Holder = enum { Data: *[u8], Empty };
        \\fn f(h: &Holder) -> &[u8] {
        \\    return match (h) {
        \\        ::Data |&bytes| bytes;
        \\        else { yield ""; }
        \\    };
        \\}
    , &.{"a '&[T]' variable used here means the array value, which is unsized"});
}

test "'new' on a slice copies the elements into an owned heap array" {
    // a string literal never allocates; 'new "text"' is the owned copy of
    // its BYTES, type *[u8], never a boxed view (sections 2.6, 5.2)
    try expectBuildsAndRuns("new_slice_copy",
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn main() -> i32 {
        \\    const a: *[u8] = new "hello";
        \\    const src = "abcd";
        \\    const c: *[u8] = new src;
        \\    var d: *var [u8] = new "abc";
        \\    d[0] = 'B';
        \\    printf("%d %d %c\n", a.length() to i32, c.length() to i32, d[0] to i32);
        \\    return (a.length() + c.length() + d.length()) to i32;
        \\}
    , 12, "5 4 B\n");
}

test "a bare '*[T]' place read is the unsized array value" {
    try expectCheckErrors(
        \\type Holder = enum { Data: *[u8], Empty };
        \\fn f(h: &var Holder) -> *[u8] {
        \\    return match (h) {
        \\        ::Data |move src| src;
        \\        else { yield new "x"; }
        \\    };
        \\}
    , &.{"a '*[T]' variable used here means the array value, which is unsized: 'move' transfers the allocation, '&' passes the view, 'new' copies it"});
    try expectBuildsAndRuns("heap_array_move_yield",
        \\type Holder = enum { Data: *[u8], Empty };
        \\fn take(h: &var Holder) -> *[u8] {
        \\    return match (h) {
        \\        ::Data |move src| move src;
        \\        else { yield new "x"; }
        \\    };
        \\}
        \\fn main() -> i32 {
        \\    var h: Holder = Holder::Data(new [66 : 4]);
        \\    const owned = take(&var h);
        \\    return (owned.length() + owned[0] to u64) to i32;
        \\}
    , 70, "");
}
