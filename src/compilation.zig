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

/// Loads the source of an imported module given its relative file path
/// ('std/vec.alloy'). Returns null when the file does not exist; the source
/// must be allocated with the passed allocator.
pub const ModuleLoader = struct {
    context: ?*anyopaque,
    function: *const fn (context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8,
};

pub const Module = struct {
    path: []const u8,
    source: []const u8,
    // canonical import key ('std::option'), null for the entry module
    key: ?[]const u8 = null,
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

pub const Compilation = struct {
    allocator: std.mem.Allocator,
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
    // the checker outlives its stage: code generation queries it for
    // section 3.9 layouts instead of re-deriving them
    checker: ?*Checker = null,
    // the message of the last interpreter runtime fault
    fault: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator) Compilation {
        return .{
            .allocator = allocator,
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

            for (compilation.modules.items[batch_start..batch_end]) |*module| {
                try module.tokenize(compilation.allocator, &compilation.diagnostics);
            }
            if (compilation.diagnostics.items.len != 0) return false;

            for (compilation.modules.items[batch_start..batch_end]) |*module| {
                try module.parse(compilation.allocator, &compilation.diagnostics);
            }
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

    // resolves 'import a::b::c' to the file 'a/b/c.alloy' through the loader
    // and queues the loaded module for the next per-module batch
    fn loadImports(compilation: *Compilation, batch_start: usize, batch_end: usize, loader: ?ModuleLoader) !void {
        const arena = compilation.arena.allocator();

        const Request = struct {
            key: []const u8,
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
                for (import.path, 0..) |segment, segment_index| {
                    if (segment_index != 0) try key.appendSlice(arena, "::");
                    try key.appendSlice(arena, segment.slice(module.source));
                }
                const owned_key = try key.toOwnedSlice(arena);
                if (compilation.loaded_keys.contains(owned_key)) continue;
                var already_requested = false;
                for (requests.items) |request| {
                    if (std.mem.eql(u8, request.key, owned_key)) {
                        already_requested = true;
                        break;
                    }
                }
                if (already_requested) continue;
                try requests.append(compilation.allocator, .{
                    .key = owned_key,
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
            const file_path = try importFilePath(arena, request.key);
            const source: ?[]const u8 = if (loader) |active_loader|
                try active_loader.function(active_loader.context, arena, file_path)
            else
                null;
            if (source) |loaded_source| {
                _ = try compilation.addImportedModule(request.key, file_path, loaded_source);
            } else {
                try compilation.diagnostics.append(compilation.allocator, .{
                    .path = request.importer_path,
                    .source = request.importer_source,
                    .span = request.span,
                    .message = try std.fmt.allocPrint(arena, "module '{s}' not found (expected file '{s}')", .{ request.key, file_path }),
                });
            }
        }
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
            &compilation.merged.?.globals,
            &compilation.diagnostics,
            compilation.allocator,
        );
        try checker.run();
        compilation.checker = checker;
        // the side tables live in the compilation arena and feed stage 5
        compilation.expression_types = checker.expression_types;
        compilation.call_targets = checker.call_targets;
        compilation.call_type_bindings = checker.call_type_bindings;
        compilation.comptime_values = checker.comptime_values;
        compilation.cast_shapes = checker.cast_shapes;
    }

    /// Stage 5: executes 'main' through the tree-walking interpreter.
    /// On error.RuntimeFault the message is available in 'fault'.
    pub fn interpret(compilation: *Compilation, output: *std.Io.Writer) !i64 {
        var interpreter = Interpreter.init(
            compilation.arena.allocator(),
            compilation.views,
            &compilation.merged.?.globals,
            &compilation.expression_types,
            &compilation.call_targets,
            &compilation.call_type_bindings,
            &compilation.comptime_values,
            &compilation.cast_shapes,
            output,
        );
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
            &compilation.merged.?.globals,
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

/// The section 5.4 import search order. A relative module path 'a/b/c.alloy'
/// (from importFilePath) is looked up under, in order: the current directory,
/// the compiler-executable's directory, then $ALLOY_STDLIB. Returns the
/// ordered candidate paths to try; the current-directory candidate is the bare
/// relative path, and an absent (null or empty) root is skipped. The loader
/// reads each in turn and takes the first that exists.
pub fn importSearchPaths(
    allocator: std.mem.Allocator,
    relative_path: []const u8,
    exe_dir: ?[]const u8,
    stdlib_dir: ?[]const u8,
) ![]const []const u8 {
    var paths: std.ArrayList([]const u8) = .empty;
    // 1. the current directory: the relative path as-is
    try paths.append(allocator, try allocator.dupe(u8, relative_path));
    // 2. the compiler-executable's directory, then 3. $ALLOY_STDLIB
    for ([_]?[]const u8{ exe_dir, stdlib_dir }) |root| {
        const dir = root orelse continue;
        if (dir.len == 0) continue;
        try paths.append(allocator, try std.fs.path.join(allocator, &.{ dir, relative_path }));
    }
    return paths.toOwnedSlice(allocator);
}

fn expectRuns(source: []const u8, expected_exit: i64, expected_output: []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    try std.testing.expect(try compilation.run(null));
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
    try std.testing.expect(try compilation.run(null));
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
        \\        ::Idle { break 0; }
        \\        ::Busy |load| { break load; }
        \\    };
        \\    if (st is ::Busy) |amount| {
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
    try expectRunFault(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 5 };
        \\    var q: *var Box = move p;
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
    try expectRunFault(
        \\type Box = struct { value: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .value = 5 };
        \\    const get = |p: *| () -> i32 { return p.value; };
        \\    return get() + p.value;
        \\}
    , "moved-from pointer");
}

test "the interpreter drives custom iterables through the cursor protocol" {
    var sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None };" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::option;
        \\type Range = struct { limit: u64 };
        \\type RangeCursor = struct { current: u64, limit: u64 };
        \\fn iterator(self r: &Range) -> RangeCursor {
        \\    return RangeCursor { .current = 0, .limit = r.limit };
        \\}
        \\fn next(self c: &var RangeCursor) -> Option<u64> {
        \\    if (c.current == c.limit) {
        \\        return Option::None;
        \\    };
        \\    c.current += 1;
        \\    return Option::Some(c.current);
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
    try std.testing.expect(try compilation.run(loader));
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    const exit_code = try compilation.interpret(&output.writer);
    try std.testing.expectEqual(@as(i64, 10), exit_code);
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
        \\    const choice = #if (base > 4) break 50 else break 100;
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
    , &.{"cannot carry a '&T' or '*T' indirection"});
}

test "macros synthesise types and reflect at compile time" {
    try expectRuns(
        \\macro vector2() {
        \\    var t = #struct_type();
        \\    t.add_member("x", #f32);
        \\    t.add_member("y", #f32);
        \\    return t;
        \\}
        \\macro widthOf(wide: bool) {
        \\    if (wide) {
        \\        return #u64;
        \\    }
        \\    return #u32;
        \\}
        \\macro answer() { return 42; }
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
        \\macro signal() {
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
        \\        Signal::Idle { total += 1 }
        \\        Signal::Busy |load| { total += load to i64 }
        \\    }
        \\    if (busy is Signal::Busy) |load| {
        \\        total += load to i64;
        \\    }
        \\    var word = busy as u64;
        \\    var round = word as Signal;
        \\    if (round is Signal::Busy) |load| {
        \\        total += load to i64;
        \\    }
        \\    printf("members %d\n", #Signal.member_names().length());
        \\    return total;
        \\}
    , 15, "members 2\n");
    // a synthesised enum compares structurally with a matching inline enum
    // (section 3.3 rule 7)
    try expectChecks(
        \\macro signal() {
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
        \\macro answer() { return 42; }
        \\fn main() -> i32 { return answer(); }
    , &.{"a macro call must be invoked with '#'"});
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const t = #struct_type();
        \\    return 0;
        \\}
    , &.{"cannot be retained in a runtime declaration"});
}

test "extension receivers handle temporaries and ownership" {
    // a temporary receiver materializes for an '&' self (section 4.5)
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
    // a bare owning place does not (section 4.2)
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
        \\    fn area() -> i32;
        \\    fn sides() -> i32;
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
        \\        Square |sq| { total += sq.side; }
        \\        else {}
        \\    }
        \\    return total;
        \\}
    , 20, "");
}

test "clean module passes all stages" {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    const module = try compilation.addModule("main.alloy", "fn main() -> i32 { return 0 }");
    try std.testing.expect(try compilation.run(null));
    try std.testing.expectEqual(@as(usize, 0), compilation.diagnostics.items.len);
    // 10 tokens plus the end_of_file marker
    try std.testing.expectEqual(@as(usize, 11), module.tokens.items.len);
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
    try std.testing.expect(try compilation.run(loader));
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
        \\    if (holder is Holder::Boxed) |inner| {
        \\        consume(inner);
        \\    }
        \\    match (holder) {
        \\        Holder::Boxed |payload| { consume(payload); }
        \\        else { }
        \\    }
        \\    for ([1, 2, 3]) |element: &| {
        \\        consume(element);
        \\    }
        \\    return value;
        \\}
        \\fn consume(anything: u32) { }
        \\fn consume(anything: i32) { }
    );
    try std.testing.expect(try compilation.run(null));
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

test "references and pointers obey section 4.2 assignment rules" {
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

test "casts follow section 3.5" {
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
    // is a 1-byte tag padded plus a u32 payload (8 bytes) - section 3.9
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
        \\    if (round is Status::Busy) |load| {
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
        \\        0 { break "zero"; }
        \\        else { break "more"; }
        \\    };
        \\    var capped = while (counter < 10) {
        \\        counter += 1;
        \\    } else {
        \\        break counter;
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
    , &.{"only permitted when the loop is used as an expression"});
}

test "enum payload captures type the payload" {
    try expectChecks(
        \\type Holder = enum { Boxed: *u32, Empty };
        \\fn f() {
        \\    var h: Holder = Holder::Boxed(new 5);
        \\    if (h is Holder::Boxed) |taken: *| {
        \\        var inner: u32 = taken;
        \\    }
        \\    if (h is Holder::Empty) { }
        \\}
    );
    try expectCheckErrors(
        \\type Holder = enum { Boxed: u32, Empty };
        \\fn f() {
        \\    const h: Holder = Holder::Boxed(1);
        \\    if (h is Holder::Boxed) |taken: *| { }
        \\}
    , &.{"owning capture requires a pointer-typed value"});
}

test "string literals are static u8 slices" {
    try expectChecks(
        \\fn consume(text: &[u8]) -> u64 { return text.length(); }
        \\fn f() {
        \\    var greeting: &[u8] = "hello";
        \\    var size = consume(greeting);
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
        \\    fn area() -> f32;
        \\    fn name() -> &[u8];
        \\}
        \\fn name(self s: &Shape) -> &[u8] { return "shape"; }
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius * c.radius; }
    );
}

test "interface verification reports missing and mismatched extensions" {
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area() -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
    , &.{"'Circle' does not implement 'Shape': no extension function 'area'"});
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area() -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> u64 { return 1; }
    , &.{"the extension 'area' for 'Circle' does not match the signature"});
}

test "a concrete extension overrides an interface default" {
    try expectChecks(
        \\interface Shape {
        \\    fn name() -> &[u8];
        \\}
        \\fn name(self s: &Shape) -> &[u8] { return "shape"; }
        \\type Circle : Shape = struct { radius: f32 };
        \\fn name(self c: &Circle) -> &[u8] { return "circle"; }
        \\fn f() {
        \\    const circle = Circle { .radius = 1.0 };
        \\    const label: &[u8] = circle.name();
        \\}
    );
}

test "interface objects convert and dispatch dynamically" {
    try expectChecks(
        \\interface Shape {
        \\    fn area() -> f32;
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
        \\    fn area() -> f32;
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
        \\    fn area() -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn measure(shape: &Shape) -> f32 {
        \\    if (shape is Circle) |c| {
        \\        return c.area();
        \\    }
        \\    return 0.0;
        \\}
        \\fn grow(shape: *var Shape) {
        \\    if (shape is Circle) |c| {
        \\        c.radius += 1.0;
        \\    }
        \\}
    );
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area() -> f32;
        \\}
        \\type Square = struct { side: f32 };
        \\fn measure(shape: &Shape) -> f32 {
        \\    if (shape is Square) |s| {
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
        \\    fn area() -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\type Square : Shape = struct { side: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn area(self s: &Square) -> f32 { return s.side; }
        \\fn measure(shape: &Shape) -> f32 {
        \\    return match (shape) {
        \\        Circle |c| { break c.area(); }
        \\        Square |s| { break s.area(); }
        \\        else { break 0.0; }
        \\    };
        \\}
    );
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area() -> f32;
        \\}
        \\type Circle : Shape = struct { radius: f32 };
        \\type Blob = struct { x: f32 };
        \\fn area(self c: &Circle) -> f32 { return c.radius; }
        \\fn measure(shape: &Shape) -> f32 {
        \\    return match (shape) {
        \\        Blob |b| { break b.x; }
        \\        5 { break 1.0; }
        \\        Circle |c: &| { break c.area(); }
        \\        else { break 0.0; }
        \\    };
        \\}
    , &.{
        "'Blob' does not implement 'Shape', so this arm can never match",
        "must name a concrete type implementing 'Shape'",
        "a downcast capture mirrors the subject's indirection",
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
        \\    if (s is ::Busy) |load| {
        \\        return load;
        \\    }
        \\    return match (s) {
        \\        ::Idle { break 0; }
        \\        ::Busy |load| { break load; }
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
        \\        ::Ok { break 0; }
        \\        ::Err |code| { break code; }
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

test "matches must be exhaustive" {
    try expectChecks(
        \\type State = enum { Idle, Busy: u32 };
        \\fn f(s: State) -> u32 {
        \\    return match (s) {
        \\        State::Idle { break 0; }
        \\        State::Busy |load| { break load; }
        \\    } else {
        \\        break 0;
        \\    };
        \\}
        \\fn g(s: State) -> u32 {
        \\    return match (s) {
        \\        State::Busy |load| { break load; }
        \\        else { break 0; }
        \\    };
        \\}
    );
    try expectCheckErrors(
        \\type State = enum { Idle, Busy: u32, Done };
        \\fn f(s: State) {
        \\    match (s) {
        \\        State::Idle { }
        \\    }
        \\}
        \\fn g(n: u32) {
        \\    match (n) {
        \\        0 { }
        \\        1 { }
        \\    }
        \\}
    , &.{
        "this match does not cover variants 'Busy', 'Done' of 'State'",
        "a match over this subject can never cover every value: add an 'else' arm",
    });
}

test "a bare interface type needs an indirection" {
    try expectCheckErrors(
        \\interface Shape {
        \\    fn area() -> f32;
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
        \\    fn area() -> f32;
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
        .{ "std/iterable.alloy", "pub interface Iterable {\n    fn length() -> u64;\n}" },
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::number;
        \\import std::iterable;
        \\fn double<T: Number>(value: T) -> T { return value + value; }
        \\fn count<T: Iterable>(subject: T) -> u64 { return 0; }
        \\fn f() {
        \\    const doubled: u32 = double(7 as u32);
        \\    const fill = [1 : 4];
        \\    const total = count(fill);
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
    });
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy",
        \\import std::option;
        \\type Range = struct { limit: u64 };
        \\type RangeCursor = struct { current: u64, limit: u64 };
        \\fn iterator(self r: &Range) -> RangeCursor {
        \\    return RangeCursor { .current = 0, .limit = r.limit };
        \\}
        \\fn next(self c: &var RangeCursor) -> Option<u64> {
        \\    if (c.current == c.limit) {
        \\        return Option::None;
        \\    };
        \\    c.current += 1;
        \\    return Option::Some(c.current);
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
    try std.testing.expect(try compilation.run(loader));
    // entry plus std::vec plus the transitive std::option
    try std.testing.expectEqual(@as(usize, 3), compilation.modules.items.len);
}

fn expectGenerates(source: []const u8, release_mode: bool, present: []const []const u8, absent: []const []const u8) !void {
    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    try std.testing.expect(try compilation.run(null));
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
    try std.testing.expect(try compilation.run(null));
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
    return expectBuildsAndRunsWith(name, source, null, expected_exit, expected_output);
}

fn expectBuildsAndRunsWith(name: []const u8, source: []const u8, extra_sources: ?*const TestSources, expected_exit: u8, expected_output: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const clang = testClang(arena) orelse return error.SkipZigTest;

    var compilation = Compilation.init(std.testing.allocator);
    defer compilation.deinit();
    _ = try compilation.addModule("main.alloy", source);
    const loader: ?ModuleLoader = if (extra_sources) |sources|
        .{ .context = @constCast(@ptrCast(sources)), .function = testLoader }
    else
        null;
    try std.testing.expect(try compilation.run(loader));
    const ir_text = (try compilation.generate(false)) orelse {
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

    const link_result = try std.process.run(arena, io, .{
        .argv = &.{ clang, ir_path, "-o", executable_path, "-O0", "-Wno-override-module" },
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
        "define i32 @main()",
        "llvm.smul.with.overflow.i32",
        "@\"alloy.fault\"",
        "call void @llvm.trap()",
    }, &.{});
    // release builds wrap arithmetic instead of trapping (section 4.2)
    try expectGenerates(
        \\fn main() -> i32 {
        \\    var total = 40;
        \\    total += 2;
        \\    return total;
        \\}
    , true, &.{"add i32"}, &.{"with.overflow"});
}

test "native codegen reports unsupported constructs" {
    try expectGenerateErrors(
        \\interface Greeter {
        \\    fn id() -> i32;
        \\}
        \\type Robot : Greeter = struct { tag: i32 };
        \\fn id(self r: &Robot) -> i32 { return r.tag; }
        \\fn main() -> i32 {
        \\    var owned: *Greeter = new Robot { .tag = 1 };
        \\    return 0;
        \\}
    , &.{"owning interface objects ('*I') are not yet supported by native code generation"});
    try expectGenerateErrors(
        \\fn main() -> i32 {
        \\    const add = (a: i32, b: i32) -> i32 { return a + b; };
        \\    return add(1, 2);
        \\}
    , &.{"function values are not yet supported by native code generation"});
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
        \\    } else { break (countdown to i32) + 41; };
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
        \\    fn area() -> i32;
        \\    fn sides() -> i32;
        \\}
        \\type Square : Shape = struct { width: i32 };
        \\type Circle : Shape = struct { radius: i32 };
        \\fn area(self s: &Square) -> i32 { return s.width * s.width; }
        \\fn sides(self s: &Square) -> i32 { return 4; }
        \\fn area(self c: &Circle) -> i32 { return 3 * c.radius * c.radius; }
        \\fn sides(self s: &Shape) -> i32 { return 0; }
        \\fn describe(s: &Shape) -> i32 {
        \\    if (s is Square) |sq| {
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
        \\        Square { break 1; }
        \\        Circle { break 2; }
        \\        else { break 0; }
        \\    };
        \\    return total + label;
        \\}
    , 43, "square 5\n");
}

test "native executables match strings and run cursors" {
    const cursor_sources = TestSources.initComptime(.{
        .{ "std/option.alloy", "pub type Option<T> = enum { Some: T, None, };" },
    });
    try expectBuildsAndRunsWith("strings_cursors",
        \\import std::option;
        \\extern printf(format: &[u8], ...) -> i32;
        \\type Countdown = struct { current: i32 };
        \\type CountdownCursor = struct { remaining: i32 };
        \\fn iterator(self c: &Countdown) -> CountdownCursor {
        \\    return CountdownCursor { .remaining = c.current };
        \\}
        \\fn next(self it: &var CountdownCursor) -> Option<i32> {
        \\    if (it.remaining == 0) {
        \\        return ::None;
        \\    }
        \\    it.remaining -= 1;
        \\    return ::Some(it.remaining);
        \\}
        \\fn nameOf(code: i32) -> &[u8] {
        \\    var name = match (code) {
        \\        1 { break "one"; }
        \\        2 { break "two"; }
        \\        else { break "many"; }
        \\    };
        \\    return name;
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

// ===========================================================================
// Spec-conformance corpus (from the LANGUAGE_SPEC.md audit)
//
// Each test pins one spec rule to a concrete program. Two kinds:
//
//   * "spec ok: ..."  — the compiler already matches the spec. These PASS now
//     and guard against regression.
//   * "spec gap: ..." — the compiler currently deviates from the spec. The
//     assertion states the SPEC-CORRECT target and is gated by `pendingGap()`
//     so it reports as *skipped*, not failed (the suite stays green and every
//     gap is visible in the skip list). To start implementing one: delete its
//     `try pendingGap();` line; the test then must pass. Fragment strings in
//     gap tests are the intended messages — adjust to the final wording.
// ===========================================================================

// Marks a spec-conformance test as not-yet-implemented: reported skipped, not
// failed. Delete the `try pendingGap();` line in a test to activate it.
fn pendingGap() error{SkipZigTest}!void {
    return error.SkipZigTest;
}

test "spec ok: §1.2 block comments nest" {
    try expectChecks(
        \\/* outer /* nested */ still open */
        \\fn main() -> i32 { return 0; }
    );
}

test "spec ok: §1.6 a character literal over 8 bytes is rejected" {
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    var big: u64 = 'abcdefghi';
        \\    return 0;
        \\}
    , &.{"exceeds 8 bytes"});
}

test "spec ok: §4.2 'move' requires a pointer operand" {
    try expectCheckErrors(
        \\type Box = struct { v: i32 };
        \\fn main() -> i32 {
        \\    var b = Box { .v = 1 };
        \\    var c = move b;
        \\    return 0;
        \\}
    , &.{"'move' requires a pointer"});
}

test "spec ok: §3.6 extern declarations do not overload" {
    try expectCheckErrors(
        \\extern puts(s: &[u8]) -> i32;
        \\extern puts(s: &[u8]) -> i32;
        \\fn main() -> i32 { return 0; }
    , &.{"redeclaration of 'puts'"});
}

test "spec ok: §5.4 a function name reused for a non-function is a redeclaration" {
    try expectCheckErrors(
        \\fn thing() -> i32 { return 0; }
        \\type thing = u32;
        \\fn main() -> i32 { return 0; }
    , &.{"redeclaration of 'thing'"});
}

test "spec ok: §4.2 a bare owning pointer cannot fill a '*T' parameter" {
    try expectCheckErrors(
        \\type Box = struct { v: i32 };
        \\fn take(p: *Box) -> i32 { return p.v; }
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .v = 1 };
        \\    return take(p);
        \\}
    , &.{"no overload of 'take'"});
}

test "spec ok: §3.3 rule 6 a wider named struct coerces to an anonymous layout" {
    try expectChecks(
        \\type Big = struct { a: u32, b: u32, c: u32 };
        \\fn take(s: struct { a: u32, b: u32 }) -> u32 { return s.a; }
        \\fn main() -> i32 {
        \\    var big = Big { .a = 1, .b = 2, .c = 3 };
        \\    return take(big) to i32;
        \\}
    );
}

test "spec ok: §3.5 'as' on a value rejects a width mismatch" {
    // 'p' pierces to Box (4 bytes), so 'p as u64' is a width error, not an
    // address reinterpretation (pointee transparency, §4.2)
    try expectCheckErrors(
        \\type Box = struct { v: i32 };
        \\fn main() -> i32 {
        \\    var p: *var Box = new Box { .v = 1 };
        \\    var bits = p as u64;
        \\    return 0;
        \\}
    , &.{"Box is 4 byte(s) but u64 is 8 byte(s)"});
}

test "spec ok: §3.5 codegen lowers a reference reinterpretation cast" {
    // native code generation handles '&S as &T'; the interpreter does not yet
    // (see the matching gap test below)
    try expectBuildsAndRuns("spec_ref_as",
        \\type Pair = struct { low: u32, high: u32 };
        \\fn main() -> i32 {
        \\    var pair = Pair { .low = 3, .high = 0 };
        \\    var viewed: &u64 = &pair as &u64;
        \\    return viewed to i32;
        \\}
    , 3, "");
}

test "spec ok: §1.6 'new \"text\"' yields an owned *[u8] heap copy" {
    // section 1.6: a string literal is a static '&[u8]'; 'new' copies its
    // bytes into a fresh owned '*[u8]' heap array (not a '*&[u8]' header)
    try expectChecks(
        \\fn main() -> i32 {
        \\    var p: *[u8] = new "hello";
        \\    return 0;
        \\}
    );
    // the copy is real heap bytes: indexing reads them back ('i' == 105)
    try expectRuns(
        \\fn main() -> i32 {
        \\    var p: *[u8] = new "hi";
        \\    return p[1] to i32;
        \\}
    , 105, "");
    try expectBuildsAndRuns("spec_new_string",
        \\fn main() -> i32 {
        \\    var p: *[u8] = new "hi";
        \\    return p[1] to i32;
        \\}
    , 105, "");
}

test "spec ok: §4.5 'self' is only valid on the first parameter" {
    // section 4.5: 'self' marks an extension receiver; on any later parameter
    // it is a definition-time error
    try expectCheckErrors(
        \\fn f(a: u32, self b: u32) -> u32 { return a; }
        \\fn main() -> i32 { return 0; }
    , &.{"first parameter"});
    // the first parameter may still be 'self'
    try expectChecks(
        \\type V = struct { n: u32 };
        \\fn get(self v: &V, k: u32) -> u32 { return v.n + k; }
        \\fn main() -> i32 { return 0; }
    );
}

test "spec ok: §3.6 two functions with an identical parameter list are a redeclaration" {
    // section 3.6: same name + identical parameter type list is a
    // redeclaration, not an overload
    try expectCheckErrors(
        \\fn dup(a: u32) -> u32 { return a; }
        \\fn dup(a: u32) -> u32 { return a + 1; }
        \\fn main() -> i32 { return 0; }
    , &.{"redeclaration"});
    // differing parameter type lists still overload cleanly
    try expectChecks(
        \\fn dup(a: u32) -> u32 { return a; }
        \\fn dup(a: u32, b: u32) -> u32 { return a + b; }
        \\fn dup(a: bool) -> u32 { return 1; }
        \\fn main() -> i32 { return 0; }
    );
}

test "spec ok: §2.1 a const array-fill count makes a stack array" {
    // section 2.1: a stack '[value : count]' count is any compile-time
    // evaluatable expression, not only an integer-literal token
    try expectChecks(
        \\fn main() -> i32 {
        \\    const n = 4;
        \\    var a = [0 : n];
        \\    return 0;
        \\}
    );
    // the const folds to a real fixed array: index 2 of '[7 : 4]' is 7
    try expectRuns(
        \\fn main() -> i32 {
        \\    const n = 4;
        \\    var a = [7 : n];
        \\    return a[2];
        \\}
    , 7, "");
    try expectBuildsAndRuns("spec_const_fill",
        \\fn main() -> i32 {
        \\    const n = 4;
        \\    var a = [7 : n];
        \\    return a[2];
        \\}
    , 7, "");
}

test "spec ok: §2.1 a comptime array-fill count makes a stack array" {
    // a '#' count folds the same way; '[7 : #side()]' is a 3-element array
    try expectChecks(
        \\fn side() -> i32 { return 3; }
        \\fn main() -> i32 {
        \\    var a = [0 : #side()];
        \\    return 0;
        \\}
    );
    try expectRuns(
        \\fn side() -> i32 { return 3; }
        \\fn main() -> i32 {
        \\    var a = [7 : #side()];
        \\    return a[1] + a[2];
        \\}
    , 14, "");
    try expectBuildsAndRuns("spec_comptime_fill",
        \\fn side() -> i32 { return 3; }
        \\fn main() -> i32 {
        \\    var a = [7 : #side()];
        \\    return a[1] + a[2];
        \\}
    , 14, "");
}

test "spec ok: §3.2 a fixed-array TYPE length must be an integer literal" {
    // decided 2026-06-16: §3.2's array-type length stays integer-literal only.
    // (Only array-FILL counts fold compile-time consts, §2.1; the type
    // annotation does not.)
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    const n = 4;
        \\    var a: [u8 : n] = [1 : n];
        \\    return 0;
        \\}
    , &.{"integer length"});
    // a literal length is accepted
    try expectChecks(
        \\fn main() -> i32 {
        \\    var a: [u8 : 4] = [1 : 4];
        \\    return 0;
        \\}
    );
}

test "spec ok: §3.5 the interpreter reinterprets '&S as &T'" {
    // section 3.5: '&S as &T' views the same memory as the new pointee; the
    // interpreter now reinterprets the pointee on read, matching codegen
    try expectRuns(
        \\type Pair = struct { low: u32, high: u32 };
        \\fn main() -> i32 {
        \\    var pair = Pair { .low = 3, .high = 0 };
        \\    var viewed: &u64 = &pair as &u64;
        \\    return viewed to i32;
        \\}
    , 3, "");
    try expectBuildsAndRuns("spec_ref_reinterpret",
        \\type Pair = struct { low: u32, high: u32 };
        \\fn main() -> i32 {
        \\    var pair = Pair { .low = 3, .high = 0 };
        \\    var viewed: &u64 = &pair as &u64;
        \\    return viewed to i32;
        \\}
    , 3, "");
}

test "spec gap: §4.4 native code generation lowers lambdas" {
    // DEFERRED 2026-06-16: codegen rejects lambda expressions and function
    // values; full closure lowering (capture lists, owning-capture moves, the
    // closure calling convention) is a large change left for later. The
    // interpreter runs lambdas; only native codegen is missing.
    try pendingGap();
    try expectBuildsAndRuns("spec_lambda",
        \\fn main() -> i32 {
        \\    const add = (a: i32, b: i32) -> i32 { return a + b; };
        \\    return add(1, 2);
        \\}
    , 3, "");
}

test "spec ok: §1.6 uppercase backslash-X / backslash-U escapes are rejected" {
    // decided 2026-06-16: only lowercase '\xHH' and '\u{...}' decode; the
    // uppercase forms are unrecognized escape sequences (section 1.6)
    try expectCheckErrors(
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn main() -> i32 {
        \\    printf("\X41\n");
        \\    return 0;
        \\}
    , &.{"unrecognized escape sequence"});
    try expectCheckErrors(
        \\fn main() -> i32 {
        \\    var s = "\U{1F600}";
        \\    return 0;
        \\}
    , &.{"unrecognized escape sequence"});
    // the lowercase forms still decode
    try expectChecks(
        \\extern printf(format: &[u8], ...) -> i32;
        \\fn main() -> i32 {
        \\    printf("\x41\n");
        \\    return 0;
        \\}
    );
}

test "spec ok: §3.3/§3.6 an untyped literal does not disambiguate overloads" {
    // decided 2026-06-16: an untyped integer literal stays ambiguous between
    // i32 and i64 overloads (it does NOT prefer its i32 default); an explicit
    // type is required (sections 3.3, 3.6)
    try expectCheckErrors(
        \\fn f(x: i32) -> i32 { return x; }
        \\fn f(x: i64) -> i64 { return x; }
        \\fn main() -> i32 { return f(5); }
    , &.{"ambiguous"});
    // an explicitly typed argument selects one overload
    try expectRuns(
        \\fn f(x: i32) -> i32 { return x; }
        \\fn f(x: i64) -> i64 { return x; }
        \\fn main() -> i32 {
        \\    var n: i32 = 5;
        \\    return f(n);
        \\}
    , 5, "");
}

test "spec ok: §3.7 surplus explicit type arguments are rejected" {
    // section 3.7: explicit type arguments bind left-to-right; supplying more
    // than the function declares is an error (decided 2026-06-16)
    try expectCheckErrors(
        \\fn id<T>(x: T) -> T { return x; }
        \\fn main() -> i32 { return id<i32, u64>(5); }
    , &.{"type argument"});
    // the exact count still binds and checks cleanly
    try expectChecks(
        \\fn id<T>(x: T) -> T { return x; }
        \\fn main() -> i32 { return id<i32>(5); }
    );
}

test "spec ok: §5.4 imports resolve via cwd, the exe dir, then $ALLOY_STDLIB" {
    // section 5.4: 'a/b/c.alloy' is searched under the current directory, the
    // compiler-executable's directory, then $ALLOY_STDLIB, in that order
    const allocator = std.testing.allocator;
    const paths = try importSearchPaths(allocator, "std/vec.alloy", "exe_root", "stdlib_root");
    defer {
        for (paths) |p| allocator.free(p);
        allocator.free(paths);
    }
    try std.testing.expectEqual(@as(usize, 3), paths.len);
    // 1. current directory: the bare relative path
    try std.testing.expectEqualStrings("std/vec.alloy", paths[0]);
    // 2. the compiler-executable's directory, 3. $ALLOY_STDLIB; each is the
    // root joined to the relative path (separator is platform-specific)
    try std.testing.expect(std.mem.startsWith(u8, paths[1], "exe_root"));
    try std.testing.expect(std.mem.endsWith(u8, paths[1], "vec.alloy"));
    try std.testing.expect(std.mem.startsWith(u8, paths[2], "stdlib_root"));
    try std.testing.expect(std.mem.endsWith(u8, paths[2], "vec.alloy"));

    // absent roots drop out, leaving only the current-directory candidate
    const only_cwd = try importSearchPaths(allocator, "std/vec.alloy", null, null);
    defer {
        for (only_cwd) |p| allocator.free(p);
        allocator.free(only_cwd);
    }
    try std.testing.expectEqual(@as(usize, 1), only_cwd.len);
    try std.testing.expectEqualStrings("std/vec.alloy", only_cwd[0]);
}