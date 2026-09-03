//! Staged compilation pipeline. Each Alloy source file is a module. Modules
//! run their per-module stages independently (tokenize, then parse), which
//! keeps them parallelizable, and a stage must complete without errors before
//! its output is handed to the next stage. Imports discovered while parsing
//! are loaded through the supplied loader and run the same per-module stages.
//! After the per-module stages the modules merge into one compilation unit
//! for the whole-program stages: name resolution, type checking, and then
//! either execution through the tree-walking interpreter ('alloyc run') or
//! native code generation ('alloyc build'). The conformance suite that
//! drives all of this lives in conformance_tests.zig.

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
        if (compilation.checker) |checked| {
            interpreter.pierced_results = &checked.pierced_results;
            interpreter.pierce_depths = &checked.pierce_depths;
            // per-instantiation '#' values and slice-cast layouts need the
            // checker at run time (sections 4.4, 4.5)
            interpreter.checker_hooks = checked.runtimeHooks();
        }
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
