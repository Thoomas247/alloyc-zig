//! Native code generation, the execution stage behind 'alloyc build'. It
//! lowers the checked merged unit into textual LLVM IR which an external
//! clang turns into an executable. Like the interpreter it consumes the
//! checker's side tables, and it queries the live checker for section 3.9
//! layouts so byte offsets have a single source of truth.
//!
//! Lowering model: aggregates (structs, enums, fixed arrays, slices) are
//! byte blobs addressed with 'getelementptr i8' at checker-computed offsets
//! (opaque pointers, no LLVM struct types); locals live in allocas; value
//! yielding control flow writes through result slots instead of phi nodes.
//! Checked builds fault on integer overflow, division by zero, out-of-range
//! shifts, index bounds, and lockstep mismatches (section 4.2) by printing
//! the fault and executing 'llvm.trap'; release builds wrap arithmetic and
//! skip the checks, keeping only division-by-zero faults. Enum tags count
//! variants in declaration order, matching the 'as' serialization shapes.
//!
//! Ownership (section 4.2) is lowered through per-type helper functions:
//! 'alloy.copy.N' deep-copies a value (cloning every owned allocation) and
//! 'alloy.drop.N' recursively frees what a value owns. Operands track
//! freshness: a temporary transfers its bits to its consumer, a place-backed
//! value clones. Owning locals drop on every exit path (scope end, 'break',
//! 'return'), reassignment drops the old value first, and a '*[T]' carries
//! its length in a malloc prefix at user_ptr - 8.
//!
//! Generic functions monomorphize: each distinct set of resolved type
//! bindings (section 3.7) becomes one IR function, and every recorded type
//! substitutes through the active instance's bindings before any layout
//! question.
//!
//! Interface objects (section 5.2) are fat pairs of a data pointer and a
//! per-type identity global; calls without a static target dispatch through
//! a closed-world chain over the interface's implementers, falling back to
//! the default implementation, which receives the fat pair itself. 'is'
//! tests, downcasts, and match arms compare type identities. An owning
//! '*I' drops and clones virtually: the identity selects the concrete
//! helper over the same closed world. Custom iterables drive the cursor
//! protocol (section 4.3), and string subjects match through memcmp.
//!
//! Closures (section 4.4) are heap blocks headed by call, drop, and copy
//! function pointers with the captured environment behind them; a function
//! value is one pointer everywhere. Named functions used as values share a
//! constant block whose null drop and copy slots mark it static. Calls
//! with no static target load the block's call slot and pass the block as
//! the leading argument.

const std = @import("std");
const tokenizer_module = @import("tokenizer.zig");
const Token = tokenizer_module.Token;
const ast = @import("ast.zig");
const types = @import("types.zig");
const Type = types.Type;
const resolution = @import("resolution.zig");
const Diagnostic = @import("diagnostic.zig").Diagnostic;
const Checker = @import("checker.zig").Checker;
const interpreter_module = @import("interpreter.zig");
const Interpreter = interpreter_module.Interpreter;

const empty_type_environment: Checker.TypeEnvironment = .empty;

pub const Codegen = struct {
    arena: std.mem.Allocator,
    views: []const resolution.ModuleView,
    globals: *const std.StringHashMapUnmanaged(resolution.SymbolList),
    injected: []const resolution.InjectedMap,
    checker: *Checker,
    expression_types: *const std.AutoHashMapUnmanaged(*const ast.Expression, *const Type),
    call_targets: *const std.AutoHashMapUnmanaged(*const ast.Expression, resolution.Symbol),
    call_type_bindings: *const std.AutoHashMapUnmanaged(*const ast.Expression, []const Type.Binding),
    // per-instance targets for constraint-dispatched calls (section 5.2),
    // filled during emission by constraintCallTarget
    constraint_call_bindings: std.AutoHashMapUnmanaged(*const ast.Expression, []const Type.Binding) = .empty,
    comptime_values: *const std.AutoHashMapUnmanaged(*const ast.Expression, Interpreter.Value),
    cast_shapes: *const std.AutoHashMapUnmanaged(*const ast.Expression, types.CastShapes),
    diagnostics: *std.ArrayList(Diagnostic),
    diagnostics_allocator: std.mem.Allocator,
    release_mode: bool,

    constants: std.Io.Writer.Allocating,
    functions: std.Io.Writer.Allocating,
    extern_declarations: std.StringHashMapUnmanaged([]const u8),
    intrinsic_declarations: std.StringHashMapUnmanaged(void),
    byte_globals: std.StringHashMapUnmanaged(ByteGlobal),
    slice_globals: std.StringHashMapUnmanaged([]const u8),
    // one entry per function instance: generics monomorphize, so the key
    // pairs the definition with its canonical type-binding key
    function_infos: std.StringHashMapUnmanaged(*FunctionInfo),
    extern_infos: std.AutoHashMapUnmanaged(*const ast.Definition, *ExternInfo),
    queue: std.ArrayList(QueuedFunction),
    // deep-copy and drop helper functions, memoized by canonical type key
    helper_names: std.StringHashMapUnmanaged([]const u8),
    // runtime type identity globals, one per concrete declared type; the
    // address is the identity an interface object carries (section 5.2)
    type_descriptors: std.AutoHashMapUnmanaged(*const ast.Definition, []const u8),
    // per-instantiation identities for generic types behind generic
    // interface objects, keyed by definition pointer + canonical type key
    instance_descriptors: std.StringHashMapUnmanaged([]const u8) = .empty,
    // implementer lists per generic-interface instantiation
    instance_implementer_lists: std.StringHashMapUnmanaged([]const Implementer) = .empty,
    // constant closure blocks for named functions used as values (section
    // 4.4), memoized by the instance name
    static_closures: std.StringHashMapUnmanaged([]const u8),
    // the closed world of implementers per interface, memoized per
    // interface definition
    implementer_lists: std.AutoHashMapUnmanaged(*const ast.Definition, []const Implementer),
    fault_helper_emitted: bool,
    global_counter: usize,

    current_view: usize,
    allocas: std.Io.Writer.Allocating,
    body: std.Io.Writer.Allocating,
    temp_counter: usize,
    terminated: bool,
    scopes: std.ArrayList(Frame),
    // 'break' targets the innermost loop, 'yield' the innermost value
    // if or match; separate stacks make each transparent to the other
    // (section 4.3)
    break_targets: std.ArrayList(BreakTarget),
    yield_targets: std.ArrayList(BreakTarget),
    return_type: *const Type,
    return_slot: ?[]const u8,
    // the active instance's resolved type parameters (section 3.7); every
    // recorded type substitutes through these before any layout question
    current_bindings: ?*const Checker.TypeEnvironment,
    // debug info (checked builds only): DWARF metadata nodes accumulate
    // here and print at the module tail; ids come from metadata_counter
    debug_metadata: std.Io.Writer.Allocating,
    metadata_counter: usize,
    debug_compile_unit: ?usize,
    debug_subroutine_type: ?usize,
    // view index to DIFile id
    debug_files: std.AutoHashMapUnmanaged(usize, usize),
    // the DISubprogram of the function currently emitting, when it is a
    // user function or lambda; helpers carry no debug scope
    debug_subprogram: ?usize,
    debug_location: ?usize,
    // (subprogram, line, column) to DILocation id
    debug_location_memo: std.AutoHashMapUnmanaged(u128, usize),
    // type key to DWARF type id, reserved before members emit so
    // self-referential types terminate
    debug_types: std.StringHashMapUnmanaged(usize),
    debug_file_id: ?usize,
    debug_line: usize,
    debug_column: usize,
    // innermost-last lexical blocks under the active subprogram
    debug_scopes: std.ArrayList(usize),

    pub const Error = error{ OutOfMemory, Unsupported };

    const ByteGlobal = struct {
        name: []const u8,
        length: usize,
    };

    const QueuedFunction = struct {
        symbol: resolution.Symbol,
        info: *FunctionInfo,
    };

    const FunctionInfo = struct {
        name: []const u8,
        parameter_types: []const *const Type,
        return_type: *const Type,
        aggregate_return: bool,
        bindings_environment: *const Checker.TypeEnvironment,
    };

    const ExternInfo = struct {
        name: []const u8,
        parameter_types: []const *const Type,
        variadic: bool,
        return_type: *const Type,
    };

    const Frame = struct {
        locals: std.ArrayList(Local),
        // whether this frame opened a DILexicalBlock to pop with it
        debug_scope_pushed: bool = false,
    };

    const Local = struct {
        name: []const u8,
        pointer: []const u8,
        declared_type: *const Type,
        // a pinned local is storage the frame does not own: a closure
        // capture lives in the environment block (dropped by the closure's
        // drop function), a downcast pointer borrows the subject's data
        pinned: bool = false,
    };

    const BreakTarget = struct {
        exit_label: []const u8,
        slot: ?Slot,
        // scope depth at construct entry: 'break' drops every owning local
        // in frames deeper than this before branching out (section 4.2)
        frame_depth: usize,
    };

    const Slot = struct {
        pointer: []const u8,
        value_type: *const Type,
    };

    const Place = struct {
        pointer: []const u8,
        value_type: *const Type,
    };

    const Operand = union(enum) {
        none,
        scalar: Scalar,
        memory: Memory,
    };

    const Scalar = struct {
        text: []const u8,
        llvm: []const u8,
    };

    const Memory = struct {
        pointer: []const u8,
        layout: Checker.Layout,
        // a fresh operand is a temporary that owns its bits: consuming it
        // transfers ownership; consuming a place-backed operand must deep
        // copy any heap it owns (section 4.2)
        fresh: bool = false,
    };

    const Class = union(enum) {
        void_class,
        scalar: []const u8,
        aggregate: Checker.Layout,
    };

    pub fn init(
        arena: std.mem.Allocator,
        views: []const resolution.ModuleView,
        unit: *const resolution.MergedUnit,
        checker: *Checker,
        expression_types: *const std.AutoHashMapUnmanaged(*const ast.Expression, *const Type),
        call_targets: *const std.AutoHashMapUnmanaged(*const ast.Expression, resolution.Symbol),
        call_type_bindings: *const std.AutoHashMapUnmanaged(*const ast.Expression, []const Type.Binding),
        comptime_values: *const std.AutoHashMapUnmanaged(*const ast.Expression, Interpreter.Value),
        cast_shapes: *const std.AutoHashMapUnmanaged(*const ast.Expression, types.CastShapes),
        diagnostics: *std.ArrayList(Diagnostic),
        diagnostics_allocator: std.mem.Allocator,
        release_mode: bool,
    ) Codegen {
        return .{
            .arena = arena,
            .views = views,
            .globals = &unit.globals,
            .injected = unit.injected,
            .checker = checker,
            .expression_types = expression_types,
            .call_targets = call_targets,
            .call_type_bindings = call_type_bindings,
            .comptime_values = comptime_values,
            .cast_shapes = cast_shapes,
            .diagnostics = diagnostics,
            .diagnostics_allocator = diagnostics_allocator,
            .release_mode = release_mode,
            .constants = .init(arena),
            .functions = .init(arena),
            .extern_declarations = .empty,
            .intrinsic_declarations = .empty,
            .byte_globals = .empty,
            .slice_globals = .empty,
            .function_infos = .empty,
            .extern_infos = .empty,
            .queue = .empty,
            .helper_names = .empty,
            .type_descriptors = .empty,
            .static_closures = .empty,
            .implementer_lists = .empty,
            .fault_helper_emitted = false,
            .global_counter = 0,
            .current_view = 0,
            .allocas = .init(arena),
            .body = .init(arena),
            .temp_counter = 0,
            .terminated = false,
            .scopes = .empty,
            .break_targets = .empty,
            .yield_targets = .empty,
            .return_type = &void_type,
            .return_slot = null,
            .current_bindings = null,
            .debug_metadata = .init(arena),
            .metadata_counter = 0,
            .debug_compile_unit = null,
            .debug_subroutine_type = null,
            .debug_files = .empty,
            .debug_subprogram = null,
            .debug_location = null,
            .debug_location_memo = .empty,
            .debug_types = .empty,
            .debug_file_id = null,
            .debug_line = 1,
            .debug_column = 1,
            .debug_scopes = .empty,
        };
    }

    // a recorded type inside a generic body may name type parameters; the
    // active instance's bindings make it concrete (section 3.7)
    fn substituted(self: *Codegen, candidate: *const Type) Error!*const Type {
        const bindings = self.current_bindings orelse return candidate;
        return self.checker.substitute(candidate, bindings);
    }

    fn layoutQuery(self: *Codegen, candidate: *const Type, depth: usize) Error!?Checker.Layout {
        return self.checker.layoutOf(try self.substituted(candidate), depth);
    }

    fn fieldSlotsQuery(self: *Codegen, candidate: *const Type) Error!?[]const Checker.FieldSlot {
        return self.checker.fieldSlots(try self.substituted(candidate));
    }

    fn enumFrameQuery(self: *Codegen, candidate: *const Type) Error!?Checker.EnumFrame {
        return self.checker.enumFrame(try self.substituted(candidate));
    }

    /// Lowers the unit into one LLVM IR module. On error.Unsupported a
    /// diagnostic has been reported; the caller treats it as a failed stage.
    pub fn run(self: *Codegen) Error![]const u8 {
        const main_symbol = self.findMain() orelse {
            try self.diagnostics.append(self.diagnostics_allocator, .{
                .path = self.views[0].path,
                .source = self.views[0].source,
                .span = .{ .start = 0, .end = 0 },
                .message = "no 'main' function to build",
            });
            return error.Unsupported;
        };
        const main_definition = main_symbol.definition.kind.fn_def;
        if (main_definition.function.parameters.len != 0) {
            return self.report(main_definition.name.location, "'main' takes no parameters", .{});
        }
        const main_info = try self.functionInfo(main_symbol, main_definition.name.location, &.{});
        while (self.queue.pop()) |queued| {
            try self.emitFunction(queued.symbol, queued.info);
        }
        try self.emitMainWrapper(main_info);

        var module: std.Io.Writer.Allocating = .init(self.arena);
        const writer = &module.writer;
        writer.print("; generated by alloyc\n", .{}) catch return error.OutOfMemory;
        var extern_iterator = self.extern_declarations.valueIterator();
        while (extern_iterator.next()) |line| {
            writer.print("{s}\n", .{line.*}) catch return error.OutOfMemory;
        }
        var intrinsic_iterator = self.intrinsic_declarations.keyIterator();
        while (intrinsic_iterator.next()) |line| {
            writer.print("{s}\n", .{line.*}) catch return error.OutOfMemory;
        }
        // argv captured by the main wrapper for std::process::arguments
        writer.print("@\"alloy.process.argument_count\" = internal global i64 0\n", .{}) catch return error.OutOfMemory;
        writer.print("@\"alloy.process.argument_data\" = internal global ptr null\n", .{}) catch return error.OutOfMemory;
        writer.print("{s}", .{self.constants.writer.buffered()}) catch return error.OutOfMemory;
        writer.print("{s}", .{self.functions.writer.buffered()}) catch return error.OutOfMemory;
        if (self.debug_compile_unit) |unit_id| {
            const version_flag = self.nextMetadata();
            const dwarf_flag = self.nextMetadata();
            writer.print("!llvm.dbg.cu = !{{!{d}}}\n", .{unit_id}) catch return error.OutOfMemory;
            writer.print("!llvm.module.flags = !{{!{d}, !{d}}}\n", .{ version_flag, dwarf_flag }) catch return error.OutOfMemory;
            writer.print("!{d} = !{{i32 2, !\"Debug Info Version\", i32 3}}\n", .{version_flag}) catch return error.OutOfMemory;
            writer.print("!{d} = !{{i32 7, !\"Dwarf Version\", i32 4}}\n", .{dwarf_flag}) catch return error.OutOfMemory;
            writer.print("{s}", .{self.debug_metadata.writer.buffered()}) catch return error.OutOfMemory;
        }
        return module.writer.buffered();
    }

    // the std::process::arguments lang item (section 5.1a), recognized by
    // its canonical module key and name like the checker's lang items
    fn processArgumentsLangItem(self: *const Codegen, symbol: resolution.Symbol) bool {
        const key = self.views[symbol.view_index].key orelse return false;
        if (!std.mem.eql(u8, key, "std::process")) return false;
        const fn_def = symbol.definition.kind.fn_def;
        return std.mem.eql(u8, fn_def.name.slice(self.views[symbol.view_index].source), "arguments");
    }

    fn findMain(self: *Codegen) ?resolution.Symbol {
        const symbols = self.globals.get("main") orelse return null;
        for (symbols.items) |candidate| {
            if (candidate.definition.kind == .fn_def) return candidate;
        }
        return null;
    }

    // the process entry adapts the Alloy 'main' result to the C 'int' and
    // captures argv as slices for std::process::arguments (section 5.1a)
    fn emitMainWrapper(self: *Codegen, info: *FunctionInfo) Error!void {
        const writer = &self.functions.writer;
        try self.declareMalloc();
        if (!self.extern_declarations.contains("strlen")) {
            try self.extern_declarations.put(self.arena, "strlen", "declare i64 @\"strlen\"(ptr)");
        }
        writer.print("define i32 @main(i32 %argc, ptr %argv) {{\nentry:\n", .{}) catch return error.OutOfMemory;
        writer.print(
            \\  %count = sext i32 %argc to i64
            \\  store i64 %count, ptr @"alloy.process.argument_count"
            \\  %bytes = mul i64 %count, 16
            \\  %slices = call ptr @"malloc"(i64 %bytes)
            \\  store ptr %slices, ptr @"alloy.process.argument_data"
            \\  br label %capture.head
            \\capture.head:
            \\  %index = phi i64 [ 0, %entry ], [ %index.next, %capture.body ]
            \\  %done = icmp uge i64 %index, %count
            \\  br i1 %done, label %capture.done, label %capture.body
            \\capture.body:
            \\  %argument.slot = getelementptr inbounds ptr, ptr %argv, i64 %index
            \\  %argument = load ptr, ptr %argument.slot
            \\  %argument.length = call i64 @"strlen"(ptr %argument)
            \\  %pair.offset = mul i64 %index, 16
            \\  %pair = getelementptr inbounds i8, ptr %slices, i64 %pair.offset
            \\  store ptr %argument, ptr %pair
            \\  %pair.length = getelementptr inbounds i8, ptr %pair, i64 8
            \\  store i64 %argument.length, ptr %pair.length
            \\  %index.next = add i64 %index, 1
            \\  br label %capture.head
            \\capture.done:
            \\
        , .{}) catch return error.OutOfMemory;
        const resolved = try self.resolvedOf(info.return_type);
        if (resolved.* == .primitive and !resolved.primitive.isFloat() and resolved.primitive != .bool) {
            const llvm = scalarTypeText(resolved.primitive);
            writer.print("  %result = call {s} @\"{s}\"()\n", .{ llvm, info.name }) catch return error.OutOfMemory;
            if (std.mem.eql(u8, llvm, "i32")) {
                writer.print("  ret i32 %result\n}}\n", .{}) catch return error.OutOfMemory;
            } else if (std.mem.eql(u8, llvm, "i64")) {
                writer.print("  %exit = trunc i64 %result to i32\n  ret i32 %exit\n}}\n", .{}) catch return error.OutOfMemory;
            } else {
                const extend: []const u8 = if (resolved.primitive.isSigned()) "sext" else "zext";
                writer.print("  %exit = {s} {s} %result to i32\n  ret i32 %exit\n}}\n", .{ extend, llvm }) catch return error.OutOfMemory;
            }
            return;
        }
        if (info.aggregate_return) {
            const layout = (try self.layoutQuery(info.return_type, 0)) orelse Checker.Layout{ .size = 8, .alignment = 8 };
            writer.print("  %slot = alloca [{d} x i8], align {d}\n  call void @\"{s}\"(ptr %slot)\n  ret i32 0\n}}\n", .{ layout.size, layout.alignment, info.name }) catch return error.OutOfMemory;
            return;
        }
        if (resolved.* == .void_type or resolved.* == .unknown) {
            writer.print("  call void @\"{s}\"()\n  ret i32 0\n}}\n", .{info.name}) catch return error.OutOfMemory;
            return;
        }
        const scalar = try self.classify(info.return_type, .{ .start = 0, .end = 0 });
        writer.print("  %result = call {s} @\"{s}\"()\n  ret i32 0\n}}\n", .{ scalar.scalar, info.name }) catch return error.OutOfMemory;
    }

    fn functionInfo(self: *Codegen, symbol: resolution.Symbol, location: Token.Location, bindings: []const Type.Binding) Error!*FunctionInfo {
        var instance_key: std.Io.Writer.Allocating = .init(self.arena);
        instance_key.writer.print("{d}", .{@intFromPtr(symbol.definition)}) catch return error.OutOfMemory;
        for (bindings) |binding| {
            instance_key.writer.print("|{s}={s}", .{ binding.name, try self.typeKey(binding.bound, 0) }) catch return error.OutOfMemory;
        }
        if (self.function_infos.get(instance_key.writer.buffered())) |existing| return existing;

        const fn_def = symbol.definition.kind.fn_def;
        const definition_source = self.views[symbol.view_index].source;
        const environment = try self.arena.create(Checker.TypeEnvironment);
        environment.* = .empty;
        for (bindings) |binding| {
            try environment.put(self.arena, binding.name, binding.bound);
        }
        var parameter_types: std.ArrayList(*const Type) = .empty;
        for (fn_def.function.parameters) |parameter| {
            const parameter_type = try self.checker.typeFromExpressionIn(parameter.parameter_type, environment, symbol.view_index);
            if (try self.unsupportedReason(parameter_type, 0)) |reason| {
                return self.report(parameter.name.location, "{s} are not yet supported by native code generation", .{reason});
            }
            try parameter_types.append(self.arena, parameter_type);
        }
        const return_type: *const Type = if (fn_def.function.return_type) |return_expression|
            try self.checker.typeFromExpressionIn(return_expression, environment, symbol.view_index)
        else
            &void_type;
        if (try self.unsupportedReason(return_type, 0)) |reason| {
            return self.report(fn_def.name.location, "{s} are not yet supported by native code generation", .{reason});
        }
        const info = try self.arena.create(FunctionInfo);
        const id = self.global_counter;
        self.global_counter += 1;
        info.* = .{
            .name = try std.fmt.allocPrint(self.arena, "alloy.{s}.{d}", .{ fn_def.name.slice(definition_source), id }),
            .parameter_types = try parameter_types.toOwnedSlice(self.arena),
            .return_type = return_type,
            .aggregate_return = switch (try self.classify(return_type, location)) {
                .aggregate => true,
                else => false,
            },
            .bindings_environment = environment,
        };
        try self.function_infos.put(self.arena, instance_key.writer.buffered(), info);
        try self.queue.append(self.arena, .{ .symbol = symbol, .info = info });
        return info;
    }

    fn externInfo(self: *Codegen, symbol: resolution.Symbol, location: Token.Location) Error!*ExternInfo {
        if (self.extern_infos.get(symbol.definition)) |existing| return existing;
        const extern_def = symbol.definition.kind.extern_def;
        const definition_source = self.views[symbol.view_index].source;
        const name = extern_def.name.slice(definition_source);
        var parameter_types: std.ArrayList(*const Type) = .empty;
        var declaration: std.Io.Writer.Allocating = .init(self.arena);
        const writer = &declaration.writer;
        const return_type: *const Type = if (extern_def.return_type) |return_expression|
            try self.checker.typeFromExpressionIn(return_expression, &empty_type_environment, symbol.view_index)
        else
            &void_type;
        writer.print("declare {s} @\"{s}\"(", .{ try self.externTypeText(return_type, location), name }) catch return error.OutOfMemory;
        for (extern_def.parameters, 0..) |parameter, index| {
            const parameter_type = try self.checker.typeFromExpressionIn(parameter.parameter_type, &empty_type_environment, symbol.view_index);
            try parameter_types.append(self.arena, parameter_type);
            if (index != 0) writer.print(", ", .{}) catch return error.OutOfMemory;
            writer.print("{s}", .{try self.externTypeText(parameter_type, parameter.name.location)}) catch return error.OutOfMemory;
        }
        if (extern_def.variadic) {
            if (extern_def.parameters.len != 0) writer.print(", ", .{}) catch return error.OutOfMemory;
            writer.print("...", .{}) catch return error.OutOfMemory;
        }
        writer.print(")", .{}) catch return error.OutOfMemory;
        const info = try self.arena.create(ExternInfo);
        info.* = .{
            .name = name,
            .parameter_types = try parameter_types.toOwnedSlice(self.arena),
            .variadic = extern_def.variadic,
            .return_type = return_type,
        };
        try self.extern_infos.put(self.arena, symbol.definition, info);
        if (!self.extern_declarations.contains(name)) {
            try self.extern_declarations.put(self.arena, name, declaration.writer.buffered());
        }
        return info;
    }

    // a C-facing type: slices decay to their data pointer (section 5.3)
    fn externTypeText(self: *Codegen, candidate: *const Type, location: Token.Location) Error![]const u8 {
        const resolved = try self.resolvedOf(candidate);
        return switch (resolved.*) {
            .void_type => "void",
            .primitive => |primitive| scalarTypeText(primitive),
            .reference, .pointer, .heap_array, .slice => "ptr",
            else => self.report(location, "this type cannot cross the extern boundary yet", .{}),
        };
    }

    const SignatureParameter = struct {
        register: []const u8,
        class: Class,
    };

    const Signature = struct {
        header: []const u8,
        return_text: []const u8,
        aggregate_return: bool,
        parameters: []const SignatureParameter,
    };

    // the one mapping from Alloy types to an LLVM function signature:
    // named functions, lambda bodies, and function-value thunks all share
    // it so their calling conventions can never drift apart
    fn functionSignature(self: *Codegen, name: []const u8, leading_closure: bool, parameter_types: []const *const Type, return_type: *const Type, span: Token.Location, subprogram: ?usize) Error!Signature {
        const return_class = try self.classify(return_type, span);
        const aggregate_return = return_class == .aggregate;
        const return_text: []const u8 = switch (return_class) {
            .void_class, .aggregate => "void",
            .scalar => |llvm| llvm,
        };
        var header: std.Io.Writer.Allocating = .init(self.arena);
        const writer = &header.writer;
        writer.print("define internal {s} @\"{s}\"(", .{ return_text, name }) catch return error.OutOfMemory;
        var first = true;
        if (leading_closure) {
            writer.print("ptr %closure", .{}) catch return error.OutOfMemory;
            first = false;
        }
        if (aggregate_return) {
            if (!first) writer.print(", ", .{}) catch return error.OutOfMemory;
            writer.print("ptr %return.slot", .{}) catch return error.OutOfMemory;
            first = false;
        }
        const parameters = try self.arena.alloc(SignatureParameter, parameter_types.len);
        for (parameter_types, parameters, 0..) |parameter_type, *parameter, index| {
            if (!first) writer.print(", ", .{}) catch return error.OutOfMemory;
            first = false;
            const register = try std.fmt.allocPrint(self.arena, "%argument.{d}", .{index});
            const class = try self.classify(parameter_type, span);
            switch (class) {
                .void_class => return self.report(span, "a parameter cannot have no runtime value", .{}),
                .scalar => |llvm| writer.print("{s} {s}", .{ llvm, register }) catch return error.OutOfMemory,
                .aggregate => writer.print("ptr {s}", .{register}) catch return error.OutOfMemory,
            }
            parameter.* = .{ .register = register, .class = class };
        }
        if (subprogram) |id| {
            writer.print(") !dbg !{d} {{\nentry:\n", .{id}) catch return error.OutOfMemory;
        } else {
            writer.print(") {{\nentry:\n", .{}) catch return error.OutOfMemory;
        }
        return .{
            .header = header.writer.buffered(),
            .return_text = return_text,
            .aggregate_return = aggregate_return,
            .parameters = parameters,
        };
    }

    fn bindParameters(self: *Codegen, names: []const []const u8, parameter_types: []const *const Type, parameters: []const SignatureParameter) Error!void {
        for (names, parameter_types, parameters) |name, parameter_type, parameter| {
            switch (parameter.class) {
                .void_class => unreachable,
                .scalar => |llvm| {
                    // a slot makes the parameter addressable like any local
                    const slot = try self.scalarSlot(llvm);
                    try self.storeScalar(slot, .{ .text = parameter.register, .llvm = llvm });
                    try self.bindLocal(name, slot, parameter_type);
                },
                .aggregate => try self.bindLocal(name, parameter.register, parameter_type),
            }
        }
    }

    fn emitFunction(self: *Codegen, symbol: resolution.Symbol, info: *FunctionInfo) Error!void {
        const fn_def = symbol.definition.kind.fn_def;
        self.current_view = symbol.view_index;
        self.allocas = .init(self.arena);
        self.body = .init(self.arena);
        self.temp_counter = 0;
        self.terminated = false;
        self.scopes = .empty;
        self.break_targets = .empty;
        self.yield_targets = .empty;
        self.return_type = info.return_type;
        self.return_slot = null;
        self.current_bindings = info.bindings_environment;
        try self.pushFrame();

        var subprogram: ?usize = null;
        if (!self.release_mode) {
            const view_source_name = fn_def.name.slice(self.views[symbol.view_index].source);
            subprogram = try self.beginDebugFunction(view_source_name, symbol.view_index, fn_def.name.location.start);
        }
        const signature = try self.functionSignature(info.name, false, info.parameter_types, info.return_type, fn_def.name.location, subprogram);
        if (signature.aggregate_return) self.return_slot = "%return.slot";

        const view_source = self.views[symbol.view_index].source;
        const names = try self.arena.alloc([]const u8, fn_def.function.parameters.len);
        for (fn_def.function.parameters, names) |parameter, *name| {
            name.* = parameter.name.slice(view_source);
        }
        try self.bindParameters(names, info.parameter_types, signature.parameters);

        try self.execStatement(fn_def.function.body);
        if (!self.terminated) try self.emitDefaultReturn();

        const out = &self.functions.writer;
        out.print("{s}", .{signature.header}) catch return error.OutOfMemory;
        out.print("{s}", .{self.allocas.writer.buffered()}) catch return error.OutOfMemory;
        out.print("{s}", .{self.body.writer.buffered()}) catch return error.OutOfMemory;
        out.print("}}\n", .{}) catch return error.OutOfMemory;
        self.debug_subprogram = null;
        self.debug_location = null;
    }

    // the checker's path-termination analysis rejects typed functions that
    // can fall through (section 4.3); this default return remains as the
    // structural safety net for void functions and conservative cases
    fn emitDefaultReturn(self: *Codegen) Error!void {
        try self.emitDropsDownTo(0, null);
        if (self.return_slot != null) {
            try self.instruction("ret void", .{});
            self.terminated = true;
            return;
        }
        switch (try self.classify(self.return_type, .{ .start = 0, .end = 0 })) {
            .void_class, .aggregate => try self.instruction("ret void", .{}),
            .scalar => |llvm| {
                if (std.mem.eql(u8, llvm, "float") or std.mem.eql(u8, llvm, "double")) {
                    try self.instruction("ret {s} 0.000000e+00", .{llvm});
                } else if (std.mem.eql(u8, llvm, "ptr")) {
                    try self.instruction("ret ptr null", .{});
                } else {
                    try self.instruction("ret {s} 0", .{llvm});
                }
            },
        }
        self.terminated = true;
    }

    fn execStatement(self: *Codegen, statement: *const ast.Statement) Error!void {
        // stepping granularity: one location per statement
        if (self.debug_subprogram != null) {
            try self.setDebugLocation(self.statementOffset(statement));
        }
        switch (statement.*) {
            .block => |statements| {
                try self.pushFrame();
                for (statements) |child| {
                    try self.execStatement(child);
                }
                try self.closeFrame();
            },
            .var_def => |var_def| try self.execVarDef(var_def),
            .assign => |assign| try self.execAssign(assign),
            .break_stmt => |break_stmt| {
                const target = if (self.break_targets.items.len != 0)
                    self.break_targets.items[self.break_targets.items.len - 1]
                else
                    return self.report(break_stmt.keyword.location, "'break' outside a loop", .{});
                if (break_stmt.value) |value_expression| {
                    const value = try self.evalExpression(value_expression);
                    if (target.slot) |slot| {
                        var coerced = try self.coerceOperand(value, try self.typeOf(value_expression), slot.value_type, break_stmt.keyword.location);
                        coerced = try self.copyOwnedScalarRead(value_expression, coerced, break_stmt.keyword.location);
                        try self.storeOperand(slot.pointer, coerced, slot.value_type, break_stmt.keyword.location);
                    } else {
                        try self.dropDiscarded(value, try self.typeOf(value_expression), break_stmt.keyword.location);
                    }
                }
                try self.emitDropsDownTo(target.frame_depth, null);
                try self.instruction("br label %{s}", .{target.exit_label});
                self.terminated = true;
            },
            .yield_stmt => |yield_stmt| {
                const target = if (self.yield_targets.items.len != 0)
                    self.yield_targets.items[self.yield_targets.items.len - 1]
                else
                    return self.report(yield_stmt.keyword.location, "'yield' outside an if or match used as a value", .{});
                const value = try self.evalExpression(yield_stmt.value);
                if (target.slot) |slot| {
                    var coerced = try self.coerceOperand(value, try self.typeOf(yield_stmt.value), slot.value_type, yield_stmt.keyword.location);
                    coerced = try self.copyOwnedScalarRead(yield_stmt.value, coerced, yield_stmt.keyword.location);
                    try self.storeOperand(slot.pointer, coerced, slot.value_type, yield_stmt.keyword.location);
                } else {
                    try self.dropDiscarded(value, try self.typeOf(yield_stmt.value), yield_stmt.keyword.location);
                }
                try self.emitDropsDownTo(target.frame_depth, null);
                try self.instruction("br label %{s}", .{target.exit_label});
                self.terminated = true;
            },
            .return_stmt => |return_stmt| {
                if (return_stmt.value) |value_expression| {
                    const value = try self.evalExpression(value_expression);
                    var coerced = try self.coerceOperand(value, try self.typeOf(value_expression), self.return_type, return_stmt.keyword.location);
                    // no implicit move (section 4.2): a bare read of an
                    // owning heap-array local returns by value, so its
                    // allocation clones before the scope-end drops below
                    // free the original; 'return move v' transfers instead
                    coerced = try self.copyOwnedScalarRead(value_expression, coerced, return_stmt.keyword.location);
                    if (self.return_slot) |slot| {
                        try self.storeOperand(slot, coerced, self.return_type, return_stmt.keyword.location);
                        try self.emitDropsDownTo(0, null);
                        try self.instruction("ret void", .{});
                    } else switch (coerced) {
                        .scalar => |scalar| {
                            try self.emitDropsDownTo(0, null);
                            try self.instruction("ret {s} {s}", .{ scalar.llvm, scalar.text });
                        },
                        else => {
                            try self.emitDropsDownTo(0, null);
                            try self.instruction("ret void", .{});
                        },
                    }
                } else {
                    try self.emitDropsDownTo(0, null);
                    try self.instruction("ret void", .{});
                }
                self.terminated = true;
            },
            .expression => |expression| {
                const value = try self.evalExpression(expression);
                if (self.expression_types.get(expression)) |recorded| {
                    try self.dropDiscarded(value, recorded, self.spanOf(expression));
                }
            },
        }
    }

    fn execVarDef(self: *Codegen, var_def: ast.VarDef) Error!void {
        const binding_type: *const Type = if (var_def.declared_type) |declared_expression|
            try self.checker.typeFromExpressionIn(declared_expression, self.current_bindings orelse &empty_type_environment, self.current_view)
        else
            try self.checker.defaulted(try self.substituted(try self.typeOf(var_def.value)));
        if (try self.unsupportedReason(binding_type, 0)) |reason| {
            return self.report(var_def.name.location, "{s} are not yet supported by native code generation", .{reason});
        }
        const value = try self.evalExpression(var_def.value);
        const coerced = try self.coerceOperand(value, try self.typeOf(var_def.value), binding_type, var_def.name.location);
        const slot = switch (try self.classify(binding_type, var_def.name.location)) {
            .void_class => return self.report(var_def.name.location, "a binding cannot have no runtime value", .{}),
            .scalar => |llvm| try self.scalarSlot(llvm),
            .aggregate => |layout| try self.aggregateSlot(layout),
        };
        try self.storeOperand(slot, coerced, binding_type, var_def.name.location);
        try self.bindLocal(var_def.name.slice(self.source()), slot, binding_type);
    }

    fn execAssign(self: *Codegen, assign: anytype) Error!void {
        const span = assign.operator.location;
        if (assign.operator.tag == .equal) {
            // plain '=' targets the raw place: a pointer or reference place
            // rebinds rather than writing the pointee (section 4.2)
            const place = (try self.evalPlaceRaw(assign.target)) orelse
                return self.report(span, "assignment to a non-assignable expression", .{});
            const value = try self.evalExpression(assign.value);
            var coerced = try self.coerceOperand(value, try self.typeOf(assign.value), place.value_type, span);
            if (try self.ownsHeap(place.value_type, 0)) {
                // free-on-reassign (section 4.2): the value is staged first
                // because it may read through the old one, then the old
                // allocation drops, then the staged bits transfer
                if (coerced == .memory and !coerced.memory.fresh) {
                    const layout = (try self.layoutQuery(place.value_type, 0)) orelse coerced.memory.layout;
                    const staged = try self.aggregateSlot(layout);
                    const helper = try self.copyHelper(place.value_type, span);
                    try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, staged, coerced.memory.pointer });
                    coerced = .{ .memory = .{ .pointer = staged, .layout = layout, .fresh = true } };
                }
                const drop = try self.dropHelper(place.value_type, span);
                try self.instruction("call void @\"{s}\"(ptr {s})", .{ drop, place.pointer });
            }
            try self.storeOperand(place.pointer, coerced, place.value_type, span);
            return;
        }
        // compound assignment writes through to the pointee (section 4.2)
        const place = (try self.evalPlace(assign.target)) orelse
            return self.report(span, "assignment to a non-assignable expression", .{});
        const operator: Token.Tag = switch (assign.operator.tag) {
            .plus_equal => .plus,
            .minus_equal => .minus,
            .asterisk_equal => .asterisk,
            .slash_equal => .slash,
            .percent_equal => .percent,
            .shift_left_equal => .shift_left,
            .shift_right_equal => .shift_right,
            .ampersand_equal => .ampersand,
            .pipe_equal => .pipe,
            .caret_equal => .caret,
            else => return self.report(assign.operator.location, "this assignment operator is not supported", .{}),
        };
        const llvm = switch (try self.classify(place.value_type, assign.operator.location)) {
            .scalar => |text| text,
            else => return self.report(assign.operator.location, "compound assignment needs a scalar place", .{}),
        };
        const left: Scalar = .{ .text = try self.loadScalar(place.pointer, llvm), .llvm = llvm };
        const value = try self.evalExpression(assign.value);
        const coerced = try self.coerceOperand(value, try self.typeOf(assign.value), place.value_type, assign.operator.location);
        const result = try self.applyArithmetic(operator, left, coerced.scalar, place.value_type, assign.operator.location);
        try self.storeScalar(place.pointer, result);
    }

    fn evalPlace(self: *Codegen, expression: *const ast.Expression) Error!?Place {
        const raw = (try self.evalPlaceRaw(expression)) orelse return null;
        return try self.piercePlace(raw);
    }

    // the place without the final pierce: 'move' reads the pointer slot
    // itself, and assignment targets store into the raw place (section 4.2)
    fn evalPlaceRaw(self: *Codegen, expression: *const ast.Expression) Error!?Place {
        switch (expression.*) {
            .grouped => |inner| return self.evalPlaceRaw(inner),
            .path => |path| {
                if (path.len != 1) return null;
                const local = self.lookupLocal(path[0].slice(self.source())) orelse return null;
                return .{ .pointer = local.pointer, .value_type = local.declared_type };
            },
            .member => |member| {
                const object = (try self.evalPlace(member.object)) orelse object: {
                    // a temporary is readable through its materialized storage
                    const value = try self.evalExpression(member.object);
                    const object_type = try self.typeOf(member.object);
                    const memory = try self.ensureMemory(value, object_type, member.name.location);
                    break :object try self.piercePlace(.{ .pointer = memory.pointer, .value_type = object_type });
                };
                const slots = (try self.fieldSlotsQuery(object.value_type)) orelse
                    return self.report(member.name.location, "this value has no fields to access", .{});
                const name = member.name.slice(self.source());
                const slot = for (slots) |candidate| {
                    if (std.mem.eql(u8, candidate.name, name)) break candidate;
                } else return self.report(member.name.location, "no field '{s}' here", .{name});
                const pointer = try self.byteOffset(object.pointer, slot.offset);
                return .{ .pointer = pointer, .value_type = slot.field_type };
            },
            .index => |index| {
                const object = (try self.evalPlace(index.object)) orelse object: {
                    const value = try self.evalExpression(index.object);
                    const object_type = try self.typeOf(index.object);
                    const memory = try self.ensureMemory(value, object_type, self.spanOf(expression));
                    break :object try self.piercePlace(.{ .pointer = memory.pointer, .value_type = object_type });
                };
                const resolved = try self.resolvedOf(object.value_type);
                const span = self.spanOf(expression);
                var element_type: *const Type = undefined;
                var data_pointer: []const u8 = undefined;
                var length_text: []const u8 = undefined;
                switch (resolved.*) {
                    .fixed_array => |array| {
                        element_type = array.element;
                        data_pointer = object.pointer;
                        length_text = try std.fmt.allocPrint(self.arena, "{d}", .{array.length});
                    },
                    .slice => |slice| {
                        element_type = slice.child;
                        data_pointer = try self.loadPointerField(object.pointer, 0);
                        length_text = try self.loadIntegerField(object.pointer, 8);
                    },
                    .heap_array => |heap| {
                        element_type = heap.child;
                        const heap_view = try self.heapArrayView(object.pointer);
                        data_pointer = heap_view.data;
                        length_text = heap_view.length;
                    },
                    else => return self.report(span, "indexing is not supported on this type yet", .{}),
                }
                const element_layout = (try self.layoutQuery(element_type, 0)) orelse
                    return self.report(span, "this element type has no defined layout", .{});
                const subscript = try self.evalExpression(index.subscript);
                const subscript_type = try self.resolvedOf(try self.typeOf(index.subscript));
                const wide = try self.widenToIndex(subscript.scalar, subscript_type);
                if (!self.release_mode) {
                    const in_bounds = try self.freshTemp();
                    try self.instruction("{s} = icmp ult i64 {s}, {s}", .{ in_bounds, wide, length_text });
                    try self.faultUnless(in_bounds, "runtime fault: index out of bounds (section 5.1)");
                }
                const scaled = try self.freshTemp();
                try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, wide, element_layout.size });
                const pointer = try self.freshTemp();
                try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ pointer, data_pointer, scaled });
                return .{ .pointer = pointer, .value_type = element_type };
            },
            else => return null,
        }
    }

    // 'arr[start..end]' borrows a slice fat pair viewing the subject's
    // element range in place (section 3.2); checked builds fault on
    // bounds that escape the subject
    fn evalSubslice(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const subslice = expression.subslice;
        const span = self.spanOf(expression);
        const object = (try self.evalPlace(subslice.object)) orelse object: {
            const value = try self.evalExpression(subslice.object);
            const object_type = try self.typeOf(subslice.object);
            const memory = try self.ensureMemory(value, object_type, span);
            break :object try self.piercePlace(.{ .pointer = memory.pointer, .value_type = object_type });
        };
        const resolved = try self.resolvedOf(object.value_type);
        var element_type: *const Type = undefined;
        var data_pointer: []const u8 = undefined;
        var length_text: []const u8 = undefined;
        switch (resolved.*) {
            .fixed_array => |array| {
                element_type = array.element;
                data_pointer = object.pointer;
                length_text = try std.fmt.allocPrint(self.arena, "{d}", .{array.length});
            },
            .slice => |slice| {
                element_type = slice.child;
                data_pointer = try self.loadPointerField(object.pointer, 0);
                length_text = try self.loadIntegerField(object.pointer, 8);
            },
            .heap_array => |heap| {
                element_type = heap.child;
                const heap_view = try self.heapArrayView(object.pointer);
                data_pointer = heap_view.data;
                length_text = heap_view.length;
            },
            else => return self.report(span, "subslicing is not supported on this type", .{}),
        }
        const element_layout = (try self.layoutQuery(element_type, 0)) orelse
            return self.report(span, "this element type has no defined layout", .{});
        var start_text: []const u8 = "0";
        if (subslice.start) |start_expression| {
            const start = try self.evalExpression(start_expression);
            const start_type = try self.resolvedOf(try self.typeOf(start_expression));
            start_text = try self.widenToIndex(start.scalar, start_type);
        }
        const end = try self.evalExpression(subslice.end);
        const end_type = try self.resolvedOf(try self.typeOf(subslice.end));
        const end_text = try self.widenToIndex(end.scalar, end_type);
        if (!self.release_mode) {
            const ordered = try self.freshTemp();
            try self.instruction("{s} = icmp ule i64 {s}, {s}", .{ ordered, start_text, end_text });
            try self.faultUnless(ordered, "runtime fault: subslice bounds out of order (section 3.2)");
            const within = try self.freshTemp();
            try self.instruction("{s} = icmp ule i64 {s}, {s}", .{ within, end_text, length_text });
            try self.faultUnless(within, "runtime fault: subslice out of bounds (section 3.2)");
        }
        const scaled = try self.freshTemp();
        try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, start_text, element_layout.size });
        const view_data = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ view_data, data_pointer, scaled });
        const view_length = try self.freshTemp();
        try self.instruction("{s} = sub i64 {s}, {s}", .{ view_length, end_text, start_text });
        const pair = try self.aggregateSlot(.{ .size = 16, .alignment = 8 });
        try self.instruction("store ptr {s}, ptr {s}", .{ view_data, pair });
        const length_slot = try self.byteOffset(pair, 8);
        try self.instruction("store i64 {s}, ptr {s}", .{ view_length, length_slot });
        return .{ .memory = .{ .pointer = pair, .layout = .{ .size = 16, .alignment = 8 } } };
    }

    const HeapArrayView = struct {
        data: []const u8,
        length: []const u8,
    };

    // loads a heap array's data pointer from its place and the length from
    // the prefix at data - 8 (section 4.2); checked builds reject null
    fn heapArrayView(self: *Codegen, place_pointer: []const u8) Error!HeapArrayView {
        const data = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr {s}", .{ data, place_pointer });
        if (!self.release_mode) {
            const live = try self.freshTemp();
            try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, data });
            try self.faultUnless(live, "runtime fault: use of a moved-from array (section 4.2)");
        }
        const base = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 -8", .{ base, data });
        const length = try self.freshTemp();
        try self.instruction("{s} = load i64, ptr {s}", .{ length, base });
        return .{ .data = data, .length = length };
    }

    // pointee transparency (section 4.2): a reference or pointer place
    // reads through to its pointee; checked builds verify a pierced pointer
    // is not null (a use after move)
    fn piercePlace(self: *Codegen, place: Place) Error!Place {
        var current = place;
        var depth: usize = 0;
        while (depth < 16) : (depth += 1) {
            const resolved = try self.resolvedOf(current.value_type);
            switch (resolved.*) {
                .reference => |indirection| {
                    // an interface object is the fat pair itself, not a
                    // pointer to pierce through (section 3.2)
                    if ((try self.resolvedOf(indirection.child)).* == .interface) return current;
                    const loaded = try self.freshTemp();
                    try self.instruction("{s} = load ptr, ptr {s}", .{ loaded, current.pointer });
                    current = .{ .pointer = loaded, .value_type = indirection.child };
                },
                .pointer => |indirection| {
                    if ((try self.resolvedOf(indirection.child)).* == .interface) return current;
                    const loaded = try self.freshTemp();
                    try self.instruction("{s} = load ptr, ptr {s}", .{ loaded, current.pointer });
                    if (!self.release_mode) {
                        const live = try self.freshTemp();
                        try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
                        try self.faultUnless(live, "runtime fault: null pointer dereference, a use after 'move' (section 4.2)");
                    }
                    current = .{ .pointer = loaded, .value_type = indirection.child };
                },
                else => return current,
            }
        }
        return current;
    }

    fn evalExpression(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        switch (expression.*) {
            .grouped => |inner| return self.evalExpression(inner),
            .integer_literal => |token| return self.integerLiteralOperand(expression, token),
            .float_literal => |token| {
                const value = std.fmt.parseFloat(f64, token.slice(self.source())) catch 0;
                const primitive = self.primitiveOf(expression) orelse .f32;
                return .{ .scalar = try self.floatConstant(value, primitive) };
            },
            .bool_literal => |literal| return .{ .scalar = .{ .text = if (literal.value) "true" else "false", .llvm = "i1" } },
            .character_literal => |token| {
                const text = token.slice(self.source());
                const bytes = try tokenizer_module.unescape(self.arena, text[1 .. text.len - 1]);
                var value: i128 = 0;
                for (bytes) |byte| value = (value << 8) | byte;
                const primitive = self.primitiveOf(expression) orelse .i32;
                return .{ .scalar = .{
                    .text = try std.fmt.allocPrint(self.arena, "{d}", .{value}),
                    .llvm = scalarTypeText(primitive),
                } };
            },
            .string_literal => |token| {
                const text = token.slice(self.source());
                const bytes = try tokenizer_module.unescape(self.arena, text[1 .. text.len - 1]);
                const slice_global = try self.sliceGlobal(bytes);
                return .{ .memory = .{ .pointer = slice_global, .layout = .{ .size = 16, .alignment = 8 } } };
            },
            .path => {
                if (try self.evalPlace(expression)) |place| return self.loadPlace(place, self.spanOf(expression));
                // a payload-less variant written 'Enum::Variant'
                const recorded = try self.resolvedOf(try self.typeOf(expression));
                if ((try self.enumFrameQuery(recorded)) != null and expression.path.len >= 2) {
                    return self.enumConstruction(expression, expression.path[expression.path.len - 1].slice(self.source()), &.{});
                }
                if (recorded.* == .function) return self.functionValue(expression);
                return self.report(self.spanOf(expression), "this name has no runtime value here", .{});
            },
            .implied_variant => |token| return self.enumConstruction(expression, token.slice(self.source()), &.{}),
            .member, .index => {
                const place = (try self.evalPlace(expression)) orelse
                    return self.report(self.spanOf(expression), "cannot read this expression", .{});
                return self.loadPlace(place, self.spanOf(expression));
            },
            .subslice => return self.evalSubslice(expression),
            .comptime_expr => |inner| {
                if (self.comptime_values.get(expression)) |value| {
                    return self.constantFromValue(value, expression);
                }
                return self.evalExpression(inner);
            },
            .unary => return self.evalUnary(expression),
            .binary => return self.evalBinary(expression),
            .cast => return self.evalCast(expression),
            .call => {
                const result = try self.evalCall(expression);
                // a bare '&T' result at a use site pierces to a copy of the
                // pointee (section 4.2): scalars load, aggregates hand back
                // a place-backed operand that consumers deep-copy
                if (self.checker.pierced_results.contains(expression)) {
                    // the recorded type is already the pointee: the callee
                    // returned a pointer, so load or wrap it
                    const pointee = try self.resolvedOf(try self.typeOf(expression));
                    switch (try self.classify(pointee, self.spanOf(expression))) {
                        .scalar => |llvm| {
                            const loaded = try self.freshTemp();
                            try self.instruction("{s} = load {s}, ptr {s}", .{ loaded, llvm, result.scalar.text });
                            return .{ .scalar = .{ .text = loaded, .llvm = llvm } };
                        },
                        .aggregate => |layout| return .{ .memory = .{ .pointer = result.scalar.text, .layout = layout } },
                        .void_class => {},
                    }
                }
                return result;
            },
            .struct_init => return self.evalStructInit(expression),
            .array_literal => return self.evalArrayLiteral(expression),
            .array_fill => return self.evalArrayFill(expression),
            .array_range => return self.evalArrayRange(expression),
            .if_expr => |if_expr| return self.evalIf(expression, if_expr),
            .while_expr => |while_expr| return self.evalWhile(expression, while_expr),
            .for_expr => |for_expr| return self.evalFor(expression, for_expr),
            .match_expr => |match_expr| return self.evalMatch(expression, match_expr),
            .lambda => return self.evalLambda(expression),
        }
    }

    fn integerLiteralOperand(self: *Codegen, expression: *const ast.Expression, token: Token) Error!Operand {
        const text = token.slice(self.source());
        const value = interpreter_module.parseIntegerText(text) catch
            return self.report(token.location, "invalid integer literal '{s}'", .{text});
        const primitive = self.primitiveOf(expression) orelse .i32;
        if (primitive.isFloat()) {
            return .{ .scalar = try self.floatConstant(@floatFromInt(value), primitive) };
        }
        return .{ .scalar = .{
            .text = try std.fmt.allocPrint(self.arena, "{d}", .{value}),
            .llvm = scalarTypeText(primitive),
        } };
    }

    // LLVM requires exact float constants; the hexadecimal form of the
    // double (rounded through f32 first when the type is f32) is exact
    fn floatConstant(self: *Codegen, value: f64, primitive: types.Primitive) Error!Scalar {
        const exact: f64 = if (primitive == .f32) @as(f64, @floatCast(@as(f32, @floatCast(value)))) else value;
        const bits: u64 = @bitCast(exact);
        return .{
            .text = try std.fmt.allocPrint(self.arena, "0x{X:0>16}", .{bits}),
            .llvm = scalarTypeText(primitive),
        };
    }

    fn constantFromValue(self: *Codegen, value: Interpreter.Value, expression: *const ast.Expression) Error!Operand {
        switch (value) {
            .integer => |integer| {
                const primitive = integer.primitive orelse (self.primitiveOf(expression) orelse .i32);
                if (primitive.isFloat()) return .{ .scalar = try self.floatConstant(@floatFromInt(integer.value), primitive) };
                return .{ .scalar = .{
                    .text = try std.fmt.allocPrint(self.arena, "{d}", .{integer.value}),
                    .llvm = scalarTypeText(primitive),
                } };
            },
            .float => |float| return .{ .scalar = try self.floatConstant(float.value, float.primitive orelse .f64) },
            .bool_value => |truth| return .{ .scalar = .{ .text = if (truth) "true" else "false", .llvm = "i1" } },
            .slice, .array, .heap_array => {
                // materialization (section 6.2): the value becomes typed
                // static program data shaped by the checker's recorded type
                const resolved = try self.resolvedOf(try self.typeOf(expression));
                const rendered = (try self.staticConstant(value, resolved)) orelse
                    return self.report(self.spanOf(expression), "this compile-time value cannot be lowered to native code yet", .{});
                const layout = (try self.layoutQuery(resolved, 0)) orelse
                    return self.report(self.spanOf(expression), "this compile-time value has no native layout", .{});
                const name = try std.fmt.allocPrint(self.arena, "@\"comptime.{d}\"", .{self.global_counter});
                self.global_counter += 1;
                self.constants.writer.print(
                    "{s} = private unnamed_addr constant {s} {s}, align {d}\n",
                    .{ name, rendered.type_text, rendered.value_text, layout.alignment },
                ) catch return error.OutOfMemory;
                return .{ .memory = .{ .pointer = name, .layout = layout } };
            },
            else => return self.report(self.spanOf(expression), "this compile-time value cannot be lowered to native code yet", .{}),
        }
    }

    const RenderedConstant = struct {
        type_text: []const u8,
        value_text: []const u8,
    };

    // renders a comptime value as a typed LLVM constant matching the
    // checker's layout for the given resolved type; null when the shape
    // has no static lowering yet (structs, enums, pointers)
    fn staticConstant(self: *Codegen, value: Interpreter.Value, resolved: *const Type) Error!?RenderedConstant {
        switch (resolved.*) {
            .primitive => |primitive| {
                switch (value) {
                    .integer => |integer| {
                        if (primitive.isFloat()) {
                            const scalar = try self.floatConstant(@floatFromInt(integer.value), primitive);
                            return .{ .type_text = scalar.llvm, .value_text = scalar.text };
                        }
                        return .{
                            .type_text = scalarTypeText(primitive),
                            .value_text = try std.fmt.allocPrint(self.arena, "{d}", .{integer.value}),
                        };
                    },
                    .float => |float| {
                        const scalar = try self.floatConstant(float.value, primitive);
                        return .{ .type_text = scalar.llvm, .value_text = scalar.text };
                    },
                    .bool_value => |truth| return .{ .type_text = "i1", .value_text = if (truth) "true" else "false" },
                    else => return null,
                }
            },
            .slice => |slice| {
                const instance = staticElements(value) orelse return null;
                const element_type = try self.resolvedOf(slice.child);
                // strings dedupe through the byte-global cache
                if (element_type.* == .primitive and (element_type.primitive == .u8 or element_type.primitive == .i8)) {
                    var bytes: std.ArrayList(u8) = .empty;
                    for (instance) |element| {
                        if (element != .integer) return null;
                        try bytes.append(self.arena, @truncate(@as(u128, @bitCast(element.integer.value))));
                    }
                    const data = try self.byteGlobal(bytes.items);
                    return .{
                        .type_text = "{ ptr, i64 }",
                        .value_text = try std.fmt.allocPrint(self.arena, "{{ ptr @\"{s}\", i64 {d} }}", .{ data.name, data.length }),
                    };
                }
                const data = (try self.staticArrayConstant(instance, element_type)) orelse return null;
                const name = try std.fmt.allocPrint(self.arena, "@\"comptime.data.{d}\"", .{self.global_counter});
                self.global_counter += 1;
                const element_layout = (try self.layoutQuery(element_type, 0)) orelse return null;
                self.constants.writer.print(
                    "{s} = private unnamed_addr constant {s} {s}, align {d}\n",
                    .{ name, data.type_text, data.value_text, element_layout.alignment },
                ) catch return error.OutOfMemory;
                return .{
                    .type_text = "{ ptr, i64 }",
                    .value_text = try std.fmt.allocPrint(self.arena, "{{ ptr {s}, i64 {d} }}", .{ name, instance.len }),
                };
            },
            .fixed_array => |array| {
                const instance = staticElements(value) orelse return null;
                return self.staticArrayConstant(instance, try self.resolvedOf(array.element));
            },
            else => return null,
        }
    }

    fn staticArrayConstant(self: *Codegen, elements: []const Interpreter.Value, element_type: *const Type) Error!?RenderedConstant {
        var values: std.ArrayList(u8) = .empty;
        var type_text: ?[]const u8 = null;
        try values.appendSlice(self.arena, "[");
        for (elements, 0..) |element, index| {
            const rendered = (try self.staticConstant(element, element_type)) orelse return null;
            type_text = rendered.type_text;
            if (index != 0) try values.appendSlice(self.arena, ", ");
            try values.appendSlice(self.arena, rendered.type_text);
            try values.appendSlice(self.arena, " ");
            try values.appendSlice(self.arena, rendered.value_text);
        }
        try values.appendSlice(self.arena, "]");
        // an empty array still needs its element type in the array type
        const element_text = type_text orelse text: {
            const rendered = (try self.staticConstant(.void_value, element_type)) orelse {
                if (element_type.* == .slice) break :text "{ ptr, i64 }";
                if (element_type.* == .primitive) break :text scalarTypeText(element_type.primitive);
                return null;
            };
            break :text rendered.type_text;
        };
        return .{
            .type_text = try std.fmt.allocPrint(self.arena, "[{d} x {s}]", .{ elements.len, element_text }),
            .value_text = try values.toOwnedSlice(self.arena),
        };
    }

    fn staticElements(value: Interpreter.Value) ?[]const Interpreter.Value {
        return switch (value) {
            .slice => |instance| instance.elements,
            .array => |instance| instance.elements,
            .heap_array => |instance| if (instance) |live| live.elements else null,
            else => null,
        };
    }

    fn evalUnary(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const unary = expression.unary;
        const span = unary.operator.location;
        switch (unary.operator.tag) {
            .minus => {
                const operand = try self.evalExpression(unary.operand);
                const operand_type = try self.typeOf(expression);
                const scalar = operand.scalar;
                if (std.mem.eql(u8, scalar.llvm, "float") or std.mem.eql(u8, scalar.llvm, "double")) {
                    const result = try self.freshTemp();
                    try self.instruction("{s} = fneg {s} {s}", .{ result, scalar.llvm, scalar.text });
                    return .{ .scalar = .{ .text = result, .llvm = scalar.llvm } };
                }
                const zero: Scalar = .{ .text = "0", .llvm = scalar.llvm };
                return .{ .scalar = try self.applyArithmetic(.minus, zero, scalar, operand_type, span) };
            },
            .bang => {
                const operand = try self.evalExpression(unary.operand);
                const result = try self.freshTemp();
                try self.instruction("{s} = xor i1 {s}, true", .{ result, operand.scalar.text });
                return .{ .scalar = .{ .text = result, .llvm = "i1" } };
            },
            .tilde => {
                const operand = try self.evalExpression(unary.operand);
                const result = try self.freshTemp();
                try self.instruction("{s} = xor {s} {s}, -1", .{ result, operand.scalar.llvm, operand.scalar.text });
                return .{ .scalar = .{ .text = result, .llvm = operand.scalar.llvm } };
            },
            .ampersand => {
                // '&' on a reference-typed call result keeps the borrow
                // (section 4.2): the operand already carries the pointer
                if (unwrapGrouped(unary.operand).* == .call) {
                    const operand_type = try self.resolvedOf(try self.typeOf(unary.operand));
                    if (operand_type.* == .reference or operand_type.* == .slice) {
                        return self.evalExpression(unary.operand);
                    }
                }
                const place = (try self.evalPlace(unary.operand)) orelse
                    return self.report(span, "'&' needs an addressable operand", .{});
                // borrowing a heap array yields a slice fat pair viewing
                // its elements in place (section 4.2)
                if ((try self.resolvedOf(place.value_type)).* == .heap_array) {
                    const view = try self.heapArrayView(place.pointer);
                    const pair = try self.aggregateSlot(.{ .size = 16, .alignment = 8 });
                    try self.instruction("store ptr {s}, ptr {s}", .{ view.data, pair });
                    const length_slot = try self.byteOffset(pair, 8);
                    try self.instruction("store i64 {s}, ptr {s}", .{ view.length, length_slot });
                    return .{ .memory = .{ .pointer = pair, .layout = .{ .size = 16, .alignment = 8 } } };
                }
                // re-borrowing a slice-typed place yields the same view:
                // the fat pair itself is the value (section 4.2)
                if ((try self.resolvedOf(place.value_type)).* == .slice) {
                    return .{ .memory = .{ .pointer = place.pointer, .layout = .{ .size = 16, .alignment = 8 } } };
                }
                return .{ .scalar = .{ .text = place.pointer, .llvm = "ptr" } };
            },
            .keyword_new => return self.evalNew(expression),
            .keyword_move => {
                // 'move' reads the pointer slot itself and clears the
                // source, transferring ownership (section 4.2)
                const place = (try self.evalPlaceRaw(unary.operand)) orelse
                    return self.report(span, "'move' needs an addressable operand", .{});
                // an owning interface object is a 16-byte fat pair: the
                // whole pair transfers, and nulling the data half marks
                // the source moved-from (section 5.2)
                if ((try self.classify(place.value_type, span)) == .aggregate) {
                    const layout = (try self.layoutQuery(place.value_type, 0)) orelse
                        return self.report(span, "this type has no defined runtime layout", .{});
                    const pair = try self.aggregateSlot(layout);
                    try self.copyBytes(pair, place.pointer, layout.size);
                    try self.instruction("store ptr null, ptr {s}", .{place.pointer});
                    return .{ .memory = .{ .pointer = pair, .layout = layout, .fresh = true } };
                }
                const taken = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr {s}", .{ taken, place.pointer });
                try self.instruction("store ptr null, ptr {s}", .{place.pointer});
                return .{ .scalar = .{ .text = taken, .llvm = "ptr" } };
            },
            else => return self.report(span, "this operator is not yet supported by native code generation", .{}),
        }
    }

    fn evalNew(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const operand = expression.unary.operand;
        const span = self.spanOf(expression);
        const result_type = try self.resolvedOf(try self.typeOf(expression));
        try self.declareMalloc();
        switch (result_type.*) {
            .pointer => |indirection| {
                // 'new' under an interface-typed expectation allocates the
                // concrete operand and yields an owning interface object:
                // the fat pair carries the identity for the virtual drop
                // (section 5.2)
                if ((try self.resolvedOf(indirection.child)).* == .interface) {
                    const interface = (try self.resolvedOf(indirection.child)).interface;
                    const concrete = try self.resolvedOf(try self.typeOf(operand));
                    if (concrete.* != .declared) {
                        return self.report(span, "only named types convert to interface objects (section 5.2)", .{});
                    }
                    const concrete_layout = (try self.layoutQuery(concrete, 0)) orelse
                        return self.report(span, "this pointee type has no defined layout", .{});
                    const value = try self.evalExpression(operand);
                    const coerced = try self.coerceOperand(value, try self.typeOf(operand), concrete, span);
                    const allocation = try self.allocateFixed(concrete_layout.size);
                    try self.zeroFill(allocation, concrete_layout.size);
                    try self.storeOperand(allocation, coerced, concrete, span);
                    return self.interfacePair(allocation, concrete, interface.arguments.len != 0);
                }
                const pointee_layout = (try self.layoutQuery(indirection.child, 0)) orelse
                    return self.report(span, "this pointee type has no defined layout", .{});
                const value = try self.evalExpression(operand);
                const coerced = try self.coerceOperand(value, try self.typeOf(operand), indirection.child, span);
                const allocation = try self.allocateFixed(pointee_layout.size);
                try self.zeroFill(allocation, pointee_layout.size);
                try self.storeOperand(allocation, coerced, indirection.child, span);
                return .{ .scalar = .{ .text = allocation, .llvm = "ptr" } };
            },
            .heap_array => |indirection| return self.evalNewArray(operand, indirection.child, span),
            else => return self.report(span, "'new' produced no pointer type here", .{}),
        }
    }

    // 'new [value : count]' and friends: the allocation carries its length
    // in a prefix at user_ptr - 8 (section 4.2)
    fn evalNewArray(self: *Codegen, operand: *const ast.Expression, element_type: *const Type, span: Token.Location) Error!Operand {
        const element_layout = (try self.layoutQuery(element_type, 0)) orelse
            return self.report(span, "this element type has no defined layout", .{});
        switch (operand.*) {
            .array_fill => |array_fill| {
                const count = try self.evalExpression(array_fill.count);
                const count_type = try self.resolvedOf(try self.typeOf(array_fill.count));
                const wide = try self.widenToIndex(count.scalar, count_type);
                if (!self.release_mode) {
                    const non_negative = try self.freshTemp();
                    try self.instruction("{s} = icmp sge i64 {s}, 0", .{ non_negative, wide });
                    try self.faultUnless(non_negative, "runtime fault: array fill count is negative (section 4.2)");
                }
                const data = try self.allocateHeapArray(wide, element_layout.size);
                const fill_value = try self.evalExpression(array_fill.value);
                const fill_type = try self.typeOf(array_fill.value);
                const coerced = try self.coerceOperand(fill_value, fill_type, element_type, span);
                const fill = try self.fillSource(coerced, element_type, span);
                try self.emitRuntimeFill(data, wide, fill.operand, element_type, element_layout.size, span);
                try self.dropFillSource(fill, element_type, span);
                return .{ .scalar = .{ .text = data, .llvm = "ptr" } };
            },
            .array_range => |array_range| {
                const start: Scalar = if (array_range.start) |start_expression|
                    (try self.evalExpression(start_expression)).scalar
                else
                    .{ .text = "0", .llvm = "i64" };
                const start_wide = if (array_range.start) |start_expression|
                    try self.widenToIndex(start, try self.resolvedOf(try self.typeOf(start_expression)))
                else
                    start.text;
                const end = (try self.evalExpression(array_range.end)).scalar;
                const end_wide = try self.widenToIndex(end, try self.resolvedOf(try self.typeOf(array_range.end)));
                if (!self.release_mode) {
                    const ordered = try self.freshTemp();
                    try self.instruction("{s} = icmp sle i64 {s}, {s}", .{ ordered, start_wide, end_wide });
                    try self.faultUnless(ordered, "runtime fault: the range end is below its start (section 4.3)");
                }
                const length = try self.freshTemp();
                try self.instruction("{s} = sub i64 {s}, {s}", .{ length, end_wide, start_wide });
                const data = try self.allocateHeapArray(length, element_layout.size);
                const llvm = switch (try self.classify(element_type, span)) {
                    .scalar => |text| text,
                    else => return self.report(span, "a range yields scalar elements", .{}),
                };
                const index_slot = try self.scalarSlot("i64");
                try self.storeScalar(index_slot, .{ .text = "0", .llvm = "i64" });
                const header = try self.freshLabel("range.header");
                const body = try self.freshLabel("range.body");
                const exit = try self.freshLabel("range.exit");
                try self.startBlock(header);
                const index = try self.loadScalar(index_slot, "i64");
                const continues = try self.freshTemp();
                try self.instruction("{s} = icmp ult i64 {s}, {s}", .{ continues, index, length });
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ continues, body, exit });
                self.terminated = true;
                try self.startBlock(body);
                const counter = try self.freshTemp();
                try self.instruction("{s} = add i64 {s}, {s}", .{ counter, start_wide, index });
                const narrowed = try self.narrowFromIndex(counter, llvm);
                const scaled = try self.freshTemp();
                try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, index, element_layout.size });
                const pointer = try self.freshTemp();
                try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ pointer, data, scaled });
                try self.storeScalar(pointer, .{ .text = narrowed, .llvm = llvm });
                const next = try self.freshTemp();
                try self.instruction("{s} = add i64 {s}, 1", .{ next, index });
                try self.storeScalar(index_slot, .{ .text = next, .llvm = "i64" });
                try self.instruction("br label %{s}", .{header});
                self.terminated = true;
                try self.startBlock(exit);
                return .{ .scalar = .{ .text = data, .llvm = "ptr" } };
            },
            else => {
                // 'new [a, b, c]': the literal materializes on the stack
                // and its bits transfer into the allocation
                const value = try self.evalExpression(operand);
                const value_type = try self.resolvedOf(try self.typeOf(operand));
                if (value_type.* != .fixed_array) {
                    return self.report(span, "'new' on this operand is not yet supported by native code generation", .{});
                }
                const memory = try self.ensureMemory(value, value_type, span);
                const length = try std.fmt.allocPrint(self.arena, "{d}", .{value_type.fixed_array.length});
                const data = try self.allocateHeapArray(length, element_layout.size);
                const data_size = element_layout.size * value_type.fixed_array.length;
                if (!memory.fresh and try self.ownsHeap(element_type, 0)) {
                    const helper = try self.copyHelper(element_type, span);
                    try self.emitHelperElementLoop(data, length, element_layout.size, helper, memory.pointer);
                } else {
                    try self.copyBytes(data, memory.pointer, data_size);
                }
                return .{ .scalar = .{ .text = data, .llvm = "ptr" } };
            },
        }
    }

    // malloc(8 + length * stride): the length lands in the prefix and the
    // returned pointer addresses the first element (section 4.2)
    fn allocateHeapArray(self: *Codegen, length: []const u8, stride: u64) Error![]const u8 {
        const data_size = try self.freshTemp();
        try self.instruction("{s} = mul i64 {s}, {d}", .{ data_size, length, stride });
        const total = try self.freshTemp();
        try self.instruction("{s} = add i64 {s}, 8", .{ total, data_size });
        const base = try self.allocateDynamic(total);
        try self.instruction("store i64 {s}, ptr {s}", .{ length, base });
        const data = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 8", .{ data, base });
        return data;
    }

    fn emitRuntimeFill(self: *Codegen, data: []const u8, length: []const u8, value: Operand, element_type: *const Type, stride: u64, span: Token.Location) Error!void {
        const index_slot = try self.scalarSlot("i64");
        try self.storeScalar(index_slot, .{ .text = "0", .llvm = "i64" });
        const header = try self.freshLabel("fill.header");
        const body = try self.freshLabel("fill.body");
        const exit = try self.freshLabel("fill.exit");
        try self.startBlock(header);
        const index = try self.loadScalar(index_slot, "i64");
        const continues = try self.freshTemp();
        try self.instruction("{s} = icmp ult i64 {s}, {s}", .{ continues, index, length });
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ continues, body, exit });
        self.terminated = true;
        try self.startBlock(body);
        const scaled = try self.freshTemp();
        try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, index, stride });
        const pointer = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ pointer, data, scaled });
        // every element gets its own deep copy of the fill value
        switch (value) {
            .scalar => |scalar| try self.storeScalar(pointer, scalar),
            .memory => |memory| {
                if (try self.ownsHeap(element_type, 0)) {
                    const helper = try self.copyHelper(element_type, span);
                    try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, pointer, memory.pointer });
                } else {
                    try self.copyBytes(pointer, memory.pointer, memory.layout.size);
                }
            },
            .none => {},
        }
        const next = try self.freshTemp();
        try self.instruction("{s} = add i64 {s}, 1", .{ next, index });
        try self.storeScalar(index_slot, .{ .text = next, .llvm = "i64" });
        try self.instruction("br label %{s}", .{header});
        self.terminated = true;
        try self.startBlock(exit);
    }

    fn evalBinary(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const binary = expression.binary;
        const span = binary.operator.location;
        if (binary.operator.tag == .ampersand_ampersand or binary.operator.tag == .pipe_pipe) {
            return self.evalShortCircuit(binary);
        }
        const left_recorded = try self.typeOf(binary.left);
        const right_recorded = try self.typeOf(binary.right);
        const left_type = try self.resolvedOf(left_recorded);
        const right_type = try self.resolvedOf(right_recorded);
        // an untyped literal adopts the typed side (section 3.3 rule 2)
        const operand_type = operand: {
            if (isUntyped(left_recorded) and !isUntyped(right_recorded)) break :operand right_type;
            if (isUntyped(right_recorded) and !isUntyped(left_recorded)) break :operand left_type;
            break :operand try self.widerOf(left_type, right_type, span);
        };
        const left_raw = try self.evalExpression(binary.left);
        const right_raw = try self.evalExpression(binary.right);
        const left = (try self.coerceOperand(left_raw, left_type, operand_type, span)).scalar;
        const right = (try self.coerceOperand(right_raw, right_type, operand_type, span)).scalar;
        switch (binary.operator.tag) {
            .plus, .minus, .asterisk, .slash, .percent, .shift_left, .shift_right, .ampersand, .pipe, .caret => {
                const result_type = try self.typeOf(expression);
                return .{ .scalar = try self.applyArithmetic(binary.operator.tag, left, right, result_type, span) };
            },
            .equal_equal, .bang_equal, .angle_left, .angle_left_equal, .angle_right, .angle_right_equal => {
                return .{ .scalar = try self.applyComparison(binary.operator.tag, left, right, operand_type) };
            },
            else => return self.report(span, "this operator is not yet supported by native code generation", .{}),
        }
    }

    // logical operators short-circuit (section 4.1) through a result slot
    fn evalShortCircuit(self: *Codegen, binary: anytype) Error!Operand {
        const slot = try self.scalarSlot("i1");
        const left = try self.evalExpression(binary.left);
        try self.storeScalar(slot, left.scalar);
        const right_label = try self.freshLabel("logic.right");
        const exit_label = try self.freshLabel("logic.exit");
        if (binary.operator.tag == .ampersand_ampersand) {
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ left.scalar.text, right_label, exit_label });
        } else {
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ left.scalar.text, exit_label, right_label });
        }
        self.terminated = true;
        try self.startBlock(right_label);
        const right = try self.evalExpression(binary.right);
        try self.storeScalar(slot, right.scalar);
        try self.startBlock(exit_label);
        const result = try self.loadScalar(slot, "i1");
        return .{ .scalar = .{ .text = result, .llvm = "i1" } };
    }

    fn applyComparison(self: *Codegen, operator: Token.Tag, left: Scalar, right: Scalar, operand_type: *const Type) Error!Scalar {
        const is_float = std.mem.eql(u8, left.llvm, "float") or std.mem.eql(u8, left.llvm, "double");
        const result = try self.freshTemp();
        if (is_float) {
            const condition: []const u8 = switch (operator) {
                .equal_equal => "oeq",
                .bang_equal => "une",
                .angle_left => "olt",
                .angle_left_equal => "ole",
                .angle_right => "ogt",
                .angle_right_equal => "oge",
                else => unreachable,
            };
            try self.instruction("{s} = fcmp {s} {s} {s}, {s}", .{ result, condition, left.llvm, left.text, right.text });
            return .{ .text = result, .llvm = "i1" };
        }
        const signed = operand_type.* == .primitive and operand_type.primitive.isSigned();
        const condition: []const u8 = switch (operator) {
            .equal_equal => "eq",
            .bang_equal => "ne",
            .angle_left => if (signed) "slt" else "ult",
            .angle_left_equal => if (signed) "sle" else "ule",
            .angle_right => if (signed) "sgt" else "ugt",
            .angle_right_equal => if (signed) "sge" else "uge",
            else => unreachable,
        };
        try self.instruction("{s} = icmp {s} {s} {s}, {s}", .{ result, condition, left.llvm, left.text, right.text });
        return .{ .text = result, .llvm = "i1" };
    }

    fn applyArithmetic(self: *Codegen, operator: Token.Tag, left: Scalar, right: Scalar, result_type: *const Type, span: Token.Location) Error!Scalar {
        const resolved = try self.resolvedOf(result_type);
        const primitive: types.Primitive = if (resolved.* == .primitive) resolved.primitive else .i32;
        if (primitive.isFloat()) {
            const opcode: []const u8 = switch (operator) {
                .plus => "fadd",
                .minus => "fsub",
                .asterisk => "fmul",
                .slash => "fdiv",
                .percent => "frem",
                else => return self.report(span, "this operator needs integer operands", .{}),
            };
            const result = try self.freshTemp();
            try self.instruction("{s} = {s} {s} {s}, {s}", .{ result, opcode, left.llvm, left.text, right.text });
            return .{ .text = result, .llvm = left.llvm };
        }
        const signed = primitive.isSigned();
        const llvm = left.llvm;
        switch (operator) {
            .plus, .minus, .asterisk => {
                if (self.release_mode) {
                    const opcode: []const u8 = switch (operator) {
                        .plus => "add",
                        .minus => "sub",
                        else => "mul",
                    };
                    const result = try self.freshTemp();
                    try self.instruction("{s} = {s} {s} {s}, {s}", .{ result, opcode, llvm, left.text, right.text });
                    return .{ .text = result, .llvm = llvm };
                }
                const intrinsic_name: []const u8 = switch (operator) {
                    .plus => if (signed) "sadd" else "uadd",
                    .minus => if (signed) "ssub" else "usub",
                    else => if (signed) "smul" else "umul",
                };
                const full = try std.fmt.allocPrint(self.arena, "llvm.{s}.with.overflow.{s}", .{ intrinsic_name, llvm });
                try self.declareIntrinsic(try std.fmt.allocPrint(self.arena, "declare {{ {s}, i1 }} @{s}({s}, {s})", .{ llvm, full, llvm, llvm }));
                const pair = try self.freshTemp();
                try self.instruction("{s} = call {{ {s}, i1 }} @{s}({s} {s}, {s} {s})", .{ pair, llvm, full, llvm, left.text, llvm, right.text });
                const value = try self.freshTemp();
                try self.instruction("{s} = extractvalue {{ {s}, i1 }} {s}, 0", .{ value, llvm, pair });
                const overflowed = try self.freshTemp();
                try self.instruction("{s} = extractvalue {{ {s}, i1 }} {s}, 1", .{ overflowed, llvm, pair });
                const safe = try self.freshTemp();
                try self.instruction("{s} = xor i1 {s}, true", .{ safe, overflowed });
                try self.faultUnless(safe, "runtime fault: integer overflow (section 4.2)");
                return .{ .text = value, .llvm = llvm };
            },
            .slash, .percent => {
                // division by zero faults in every build mode (section 4.2)
                const nonzero = try self.freshTemp();
                try self.instruction("{s} = icmp ne {s} {s}, 0", .{ nonzero, llvm, right.text });
                try self.faultUnless(nonzero, "runtime fault: division by zero (section 4.2)");
                if (signed and !self.release_mode) {
                    const minimum = try self.freshTemp();
                    try self.instruction("{s} = icmp eq {s} {s}, {d}", .{ minimum, llvm, left.text, signedMinimum(primitive) });
                    const negative_one = try self.freshTemp();
                    try self.instruction("{s} = icmp eq {s} {s}, -1", .{ negative_one, llvm, right.text });
                    const overflow = try self.freshTemp();
                    try self.instruction("{s} = and i1 {s}, {s}", .{ overflow, minimum, negative_one });
                    const safe = try self.freshTemp();
                    try self.instruction("{s} = xor i1 {s}, true", .{ safe, overflow });
                    try self.faultUnless(safe, "runtime fault: integer overflow (section 4.2)");
                }
                const opcode: []const u8 = if (operator == .slash)
                    (if (signed) "sdiv" else "udiv")
                else
                    (if (signed) "srem" else "urem");
                const result = try self.freshTemp();
                try self.instruction("{s} = {s} {s} {s}, {s}", .{ result, opcode, llvm, left.text, right.text });
                return .{ .text = result, .llvm = llvm };
            },
            .shift_left, .shift_right => {
                if (!self.release_mode) {
                    const in_range = try self.freshTemp();
                    try self.instruction("{s} = icmp ult {s} {s}, {d}", .{ in_range, llvm, right.text, primitive.width() * 8 });
                    try self.faultUnless(in_range, "runtime fault: shift amount out of range (section 4.2)");
                }
                const opcode: []const u8 = if (operator == .shift_left)
                    "shl"
                else
                    (if (signed) "ashr" else "lshr");
                const result = try self.freshTemp();
                try self.instruction("{s} = {s} {s} {s}, {s}", .{ result, opcode, llvm, left.text, right.text });
                return .{ .text = result, .llvm = llvm };
            },
            .ampersand, .pipe, .caret => {
                const opcode: []const u8 = switch (operator) {
                    .ampersand => "and",
                    .pipe => "or",
                    else => "xor",
                };
                const result = try self.freshTemp();
                try self.instruction("{s} = {s} {s} {s}, {s}", .{ result, opcode, llvm, left.text, right.text });
                return .{ .text = result, .llvm = llvm };
            },
            else => return self.report(span, "this operator is not yet supported by native code generation", .{}),
        }
    }

    fn evalCast(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const cast = expression.cast;
        const span = cast.operator.location;
        switch (cast.operator.tag) {
            .keyword_to => {
                const operand = try self.evalExpression(cast.operand);
                const source_resolved = try self.resolvedOf(try self.typeOf(cast.operand));
                const target_resolved = try self.resolvedOf(try self.typeOf(expression));
                if (source_resolved.* != .primitive or target_resolved.* != .primitive) {
                    return self.report(span, "'to' needs numeric operands", .{});
                }
                return .{ .scalar = try self.convertNumeric(operand.scalar, source_resolved.primitive, target_resolved.primitive) };
            },
            .keyword_as => {
                if (self.cast_shapes.get(expression)) |shapes| {
                    return self.reinterpretShaped(expression, cast.operand, shapes, span);
                }
                const operand = try self.evalExpression(cast.operand);
                const source_resolved = try self.resolvedOf(try self.typeOf(cast.operand));
                const target_resolved = try self.resolvedOf(try self.typeOf(expression));
                if (source_resolved.* == .reference and target_resolved.* == .reference) {
                    // same memory viewed as another pointee; nothing moves
                    return operand;
                }
                if (source_resolved.* != .primitive or target_resolved.* != .primitive) {
                    return self.report(span, "'as' on this type is not yet supported by native code generation", .{});
                }
                return .{ .scalar = try self.reinterpretPrimitive(operand.scalar, source_resolved.primitive, target_resolved.primitive) };
            },
            .keyword_is => {
                const operand_type = try self.typeOf(cast.operand);
                const place = (try self.evalPlace(cast.operand)) orelse place: {
                    const value = try self.evalExpression(cast.operand);
                    const memory = try self.ensureMemory(value, operand_type, span);
                    break :place Place{ .pointer = memory.pointer, .value_type = operand_type };
                };
                // an interface-object subject tests its concrete type
                // identity (section 3.2)
                if (try self.interfaceOfPlace(place)) |_| {
                    const descriptor = try self.downcastDescriptor(cast.target, span);
                    const type_id = try self.loadPointerField(place.pointer, 8);
                    const result = try self.freshTemp();
                    try self.instruction("{s} = icmp eq ptr {s}, @\"{s}\"", .{ result, type_id, descriptor.name });
                    return .{ .scalar = .{ .text = result, .llvm = "i1" } };
                }
                const frame = (try self.enumFrameQuery(place.value_type)) orelse
                    return self.report(span, "'is' needs an enum or interface-object subject", .{});
                const target_token = cast.target.named.path[cast.target.named.path.len - 1];
                const variant_index = try self.variantIndex(frame, target_token.slice(self.source()), span);
                const tag = try self.loadTag(place.pointer, frame);
                const result = try self.freshTemp();
                try self.instruction("{s} = icmp eq {s} {s}, {d}", .{ result, tagTypeText(frame.tag_size), tag, variant_index });
                return .{ .scalar = .{ .text = result, .llvm = "i1" } };
            },
            else => return self.report(span, "this cast is not yet supported by native code generation", .{}),
        }
    }

    fn convertNumeric(self: *Codegen, operand: Scalar, origin: types.Primitive, target: types.Primitive) Error!Scalar {
        const target_llvm = scalarTypeText(target);
        if (origin == target) return operand;
        const result = try self.freshTemp();
        if (origin.isFloat() and target.isFloat()) {
            const opcode: []const u8 = if (target.width() > origin.width()) "fpext" else "fptrunc";
            try self.instruction("{s} = {s} {s} {s} to {s}", .{ result, opcode, operand.llvm, operand.text, target_llvm });
            return .{ .text = result, .llvm = target_llvm };
        }
        if (origin.isFloat()) {
            const opcode: []const u8 = if (target.isSigned()) "fptosi" else "fptoui";
            try self.instruction("{s} = {s} {s} {s} to {s}", .{ result, opcode, operand.llvm, operand.text, target_llvm });
            return .{ .text = result, .llvm = target_llvm };
        }
        if (target.isFloat()) {
            const opcode: []const u8 = if (origin.isSigned()) "sitofp" else "uitofp";
            try self.instruction("{s} = {s} {s} {s} to {s}", .{ result, opcode, operand.llvm, operand.text, target_llvm });
            return .{ .text = result, .llvm = target_llvm };
        }
        if (target.width() == origin.width()) return .{ .text = operand.text, .llvm = target_llvm };
        if (target.width() < origin.width()) {
            try self.instruction("{s} = trunc {s} {s} to {s}", .{ result, operand.llvm, operand.text, target_llvm });
            return .{ .text = result, .llvm = target_llvm };
        }
        const opcode: []const u8 = if (origin.isSigned()) "sext" else "zext";
        try self.instruction("{s} = {s} {s} {s} to {s}", .{ result, opcode, operand.llvm, operand.text, target_llvm });
        return .{ .text = result, .llvm = target_llvm };
    }

    // 'as' between same-width primitives reuses the bits (section 3.5)
    fn reinterpretPrimitive(self: *Codegen, operand: Scalar, origin: types.Primitive, target: types.Primitive) Error!Scalar {
        const target_llvm = scalarTypeText(target);
        if (std.mem.eql(u8, operand.llvm, target_llvm)) return .{ .text = operand.text, .llvm = target_llvm };
        const result = try self.freshTemp();
        if (origin == .bool) {
            try self.instruction("{s} = zext i1 {s} to {s}", .{ result, operand.text, target_llvm });
            return .{ .text = result, .llvm = target_llvm };
        }
        if (target == .bool) {
            try self.instruction("{s} = icmp ne {s} {s}, 0", .{ result, operand.llvm, operand.text });
            return .{ .text = result, .llvm = "i1" };
        }
        try self.instruction("{s} = bitcast {s} {s} to {s}", .{ result, operand.llvm, operand.text, target_llvm });
        return .{ .text = result, .llvm = target_llvm };
    }

    // a shaped 'as' (section 3.5) reads the same bytes as the target type;
    // checked builds verify a reinterpreted enum tag still names a variant
    fn reinterpretShaped(self: *Codegen, expression: *const ast.Expression, operand_expression: *const ast.Expression, shapes: types.CastShapes, span: Token.Location) Error!Operand {
        const operand = try self.evalExpression(operand_expression);
        const operand_type = try self.typeOf(operand_expression);
        const memory = try self.ensureMemory(operand, operand_type, span);
        const target_type = try self.typeOf(expression);
        if (!self.release_mode and shapes.target.* == .tagged) {
            const frame = (try self.enumFrameQuery(target_type)) orelse
                return self.report(span, "'as' target has no enum frame", .{});
            const tag = try self.loadTag(memory.pointer, frame);
            const valid = try self.freshTemp();
            try self.instruction("{s} = icmp ult {s} {s}, {d}", .{ valid, tagTypeText(frame.tag_size), tag, frame.variants.len });
            try self.faultUnless(valid, "runtime fault: 'as' produced a tag that names no variant (section 3.5)");
        }
        switch (try self.classify(target_type, span)) {
            .void_class => return .none,
            .scalar => |llvm| {
                const loaded = try self.loadScalar(memory.pointer, llvm);
                return .{ .scalar = .{ .text = loaded, .llvm = llvm } };
            },
            .aggregate => |layout| return .{ .memory = .{ .pointer = memory.pointer, .layout = layout, .fresh = memory.fresh } },
        }
    }

    fn evalCall(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const call = expression.call;
        const callee = unwrapGrouped(call.callee);
        const span = self.spanOf(expression);
        if (callee.* == .implied_variant) {
            return self.enumConstruction(expression, callee.implied_variant.slice(self.source()), call.arguments);
        }
        if (self.call_targets.get(expression)) |symbol| {
            // std::process::arguments is compiler-provided (section 5.1a):
            // the main wrapper captured argv as slices at startup
            if (symbol.definition.kind == .fn_def and self.processArgumentsLangItem(symbol)) {
                const data = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr @\"alloy.process.argument_data\"", .{data});
                const count = try self.freshTemp();
                try self.instruction("{s} = load i64, ptr @\"alloy.process.argument_count\"", .{count});
                const pair = try self.aggregateSlot(.{ .size = 16, .alignment = 8 });
                try self.instruction("store ptr {s}, ptr {s}", .{ data, pair });
                const length_slot = try self.byteOffset(pair, 8);
                try self.instruction("store i64 {s}, ptr {s}", .{ count, length_slot });
                return .{ .memory = .{ .pointer = pair, .layout = .{ .size = 16, .alignment = 8 } } };
            }
            return switch (symbol.definition.kind) {
                .extern_def => self.callExtern(expression, symbol),
                .fn_def => self.callFunction(expression, symbol, callee),
                else => self.report(span, "this callee is not callable", .{}),
            };
        }
        if (callee.* == .path and callee.path.len >= 2) {
            return self.enumConstruction(expression, callee.path[callee.path.len - 1].slice(self.source()), call.arguments);
        }
        if (callee.* == .member) {
            const name = callee.member.name.slice(self.source());
            if (std.mem.eql(u8, name, "length") and call.arguments.len == 0) {
                return self.lengthCall(callee.member.object, span);
            }
            // a call through a generic constraint has no static target
            // (the checker typed it against the interface, section 5.2);
            // the monomorphized receiver resolves the extension here
            if (try self.constraintCallTarget(expression, callee.member)) |symbol| {
                return self.callFunction(expression, symbol, callee);
            }
            // the receiver place evaluates exactly once: a second walk
            // would re-run subscript side effects; a temporary receiver
            // materializes and drops after the call (section 4.5)
            var receiver_cleanup: ?Place = null;
            const object_place = (try self.evalPlace(callee.member.object)) orelse object: {
                const value = try self.evalExpression(callee.member.object);
                const object_type = try self.typeOf(callee.member.object);
                const memory = try self.ensureMemory(value, object_type, span);
                if (memory.fresh and try self.ownsHeap(object_type, 0)) {
                    receiver_cleanup = .{ .pointer = memory.pointer, .value_type = object_type };
                }
                break :object try self.piercePlace(.{ .pointer = memory.pointer, .value_type = object_type });
            };
            // a call with no static target dispatches at runtime through
            // the interface object's type identity (section 5.2), or calls
            // through a function-typed field (section 4.4)
            const result = if (try self.interfaceOfPlace(object_place)) |interface|
                try self.interfaceDispatch(expression, callee.member, object_place, interface)
            else result: {
                const slots = (try self.fieldSlotsQuery(object_place.value_type)) orelse
                    return self.report(span, "this callee is not callable", .{});
                const slot = for (slots) |candidate| {
                    if (std.mem.eql(u8, candidate.name, name)) break candidate;
                } else return self.report(callee.member.name.location, "no field '{s}' here", .{name});
                const field_place = try self.piercePlace(.{
                    .pointer = try self.byteOffset(object_place.pointer, slot.offset),
                    .value_type = slot.field_type,
                });
                if ((try self.resolvedOf(field_place.value_type)).* != .function) {
                    return self.report(span, "this callee is not callable", .{});
                }
                break :result try self.indirectCall(expression, field_place);
            };
            if (receiver_cleanup) |cleanup| {
                const helper = try self.dropHelper(cleanup.value_type, span);
                try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, cleanup.pointer });
            }
            return result;
        }
        // a function-typed place (a local) calls indirectly through its
        // closure block (section 4.4)
        if (try self.evalPlace(callee)) |place| {
            if ((try self.resolvedOf(place.value_type)).* == .function) {
                return self.indirectCall(expression, place);
            }
            return self.report(span, "this callee is not callable", .{});
        }
        const callee_type = try self.resolvedOf(try self.typeOf(callee));
        if (callee_type.* == .function) {
            // an immediately invoked function value: a fresh closure drops
            // once the call returns (section 4.2)
            const operand = try self.evalExpression(callee);
            const memory = try self.ensureMemory(operand, try self.typeOf(callee), span);
            const result = try self.indirectCall(expression, .{ .pointer = memory.pointer, .value_type = try self.typeOf(callee) });
            if (memory.fresh) {
                const helper = try self.dropHelper(try self.typeOf(callee), span);
                try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, memory.pointer });
            }
            return result;
        }
        return self.report(span, "this call form is not yet supported by native code generation", .{});
    }

    // '.length()' on the built-in array forms (section 5.1); the checker
    // resolves this receiver without recording its type, so the place's
    // binding type is the source of truth
    fn lengthCall(self: *Codegen, object: *const ast.Expression, span: Token.Location) Error!Operand {
        const place = (try self.evalPlace(object)) orelse place: {
            const value = try self.evalExpression(object);
            const object_type = try self.typeOf(object);
            const memory = try self.ensureMemory(value, object_type, span);
            break :place Place{ .pointer = memory.pointer, .value_type = object_type };
        };
        const resolved = try self.resolvedOf(place.value_type);
        switch (resolved.*) {
            .fixed_array => |array| return .{ .scalar = .{
                .text = try std.fmt.allocPrint(self.arena, "{d}", .{array.length}),
                .llvm = "i64",
            } },
            .slice => {
                const length = try self.loadIntegerField(place.pointer, 8);
                return .{ .scalar = .{ .text = length, .llvm = "i64" } };
            },
            .heap_array => {
                const heap_view = try self.heapArrayView(place.pointer);
                return .{ .scalar = .{ .text = heap_view.length, .llvm = "i64" } };
            },
            else => return self.report(span, "'.length()' on this type is not yet supported by native code generation", .{}),
        }
    }

    fn enumConstruction(self: *Codegen, expression: *const ast.Expression, variant_name: []const u8, arguments: []const *const ast.Expression) Error!Operand {
        const span = self.spanOf(expression);
        const result_type = try self.typeOf(expression);
        const frame = (try self.enumFrameQuery(result_type)) orelse
            return self.report(span, "this variant has no enum type here", .{});
        const variant_index = try self.variantIndex(frame, variant_name, span);
        const slot = try self.aggregateSlot(frame.layout);
        try self.zeroFill(slot, frame.layout.size);
        try self.instruction("store {s} {d}, ptr {s}", .{ tagTypeText(frame.tag_size), variant_index, slot });
        if (arguments.len != 0) {
            const payload_type = frame.variants[variant_index].payload orelse
                return self.report(span, "'{s}' carries no payload", .{variant_name});
            const value = try self.evalExpression(arguments[0]);
            const coerced = try self.coerceOperand(value, try self.typeOf(arguments[0]), payload_type, span);
            const payload_pointer = try self.byteOffset(slot, frame.payload_offset);
            try self.storeOperand(payload_pointer, coerced, payload_type, span);
        }
        return .{ .memory = .{ .pointer = slot, .layout = frame.layout, .fresh = true } };
    }

    fn variantIndex(self: *Codegen, frame: Checker.EnumFrame, name: []const u8, span: Token.Location) Error!usize {
        for (frame.variants, 0..) |variant, index| {
            if (std.mem.eql(u8, variant.name, name)) return index;
        }
        return self.report(span, "no variant '{s}' here", .{name});
    }

    // resolves a member call whose receiver the checker typed as a
    // constrained type parameter: the target only exists per instance,
    // where the current bindings make the receiver concrete (section 5.2)
    fn constraintCallTarget(self: *Codegen, expression: *const ast.Expression, member: anytype) Error!?resolution.Symbol {
        const object = unwrapGrouped(member.object);
        const recorded = self.checker.expression_types.get(object) orelse local: {
            // the receiver of a checked member call has no recorded
            // expression type when it is a plain local read
            if (object.* != .path or object.path.len != 1) return null;
            const local = self.lookupLocal(object.path[0].slice(self.source())) orelse return null;
            break :local local.declared_type;
        };
        // interface objects dispatch through their identity instead
        if ((try self.resolvedOf(try self.checker.pierce(recorded))).* == .interface) return null;
        const call = expression.call;
        const receiver_type = try self.substituted(recorded);
        const argument_types = try self.arena.alloc(*const Type, call.arguments.len);
        for (call.arguments, 0..) |argument, index| {
            argument_types[index] = try self.substituted(try self.typeOf(argument));
        }
        const name = member.name.slice(self.source());
        const candidate = (try self.checker.resolveInstanceMethod(receiver_type, name, argument_types, true)) orelse return null;
        try self.constraint_call_bindings.put(self.arena, expression, candidate.type_bindings);
        return candidate.symbol;
    }

    fn callFunction(self: *Codegen, expression: *const ast.Expression, symbol: resolution.Symbol, callee: *const ast.Expression) Error!Operand {
        const call = expression.call;
        const span = self.spanOf(expression);
        // the call site's inferred bindings (section 3.7) substitute through
        // the caller's own bindings, so nested generics chain concretely
        // a constraint-dispatched call resolved per instance overrides the
        // checker's recorded (generic) bindings; instances emit one at a
        // time, so the freshest entry is always the current instance's
        const raw_bindings = self.constraint_call_bindings.get(expression) orelse
            (self.call_type_bindings.get(expression) orelse &.{});
        var bindings: std.ArrayList(Type.Binding) = .empty;
        for (raw_bindings) |binding| {
            try bindings.append(self.arena, .{ .name = binding.name, .bound = try self.substituted(binding.bound) });
        }
        const info = try self.functionInfo(symbol, span, bindings.items);
        const fn_def = symbol.definition.kind.fn_def;
        const has_self = fn_def.function.parameters.len != 0 and fn_def.function.parameters[0].is_self;

        var lowered: std.ArrayList([]const u8) = .empty;
        var result_slot: ?[]const u8 = null;
        if (info.aggregate_return) {
            const layout = (try self.layoutQuery(info.return_type, 0)) orelse
                return self.report(span, "this return type has no defined layout", .{});
            result_slot = try self.aggregateSlot(layout);
            try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{result_slot.?}));
        }

        var argument_index: usize = 0;
        var receiver_cleanup: ?Place = null;
        if (has_self and callee.* == .member) {
            const self_type = try self.resolvedOf(info.parameter_types[0]);
            if (self_type.* == .reference) {
                const place = (try self.evalPlace(callee.member.object)) orelse place: {
                    // a temporary receiver materializes for the call's
                    // duration and drops afterwards (section 4.5)
                    const value = try self.evalExpression(callee.member.object);
                    const object_type = try self.typeOf(callee.member.object);
                    const memory = try self.ensureMemory(value, object_type, span);
                    if (memory.fresh and try self.ownsHeap(object_type, 0)) {
                        receiver_cleanup = .{ .pointer = memory.pointer, .value_type = object_type };
                    }
                    break :place Place{ .pointer = memory.pointer, .value_type = object_type };
                };
                try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{place.pointer}));
            } else {
                // a by-value or pointer 'self' follows parameter semantics
                // (section 4.5): the callee owns it
                const value = try self.evalExpression(callee.member.object);
                const coerced = try self.coerceOperand(value, try self.typeOf(callee.member.object), info.parameter_types[0], span);
                try lowered.append(self.arena, try self.lowerArgument(coerced, info.parameter_types[0], span));
            }
            argument_index = 1;
        }
        for (call.arguments) |argument| {
            if (argument_index >= info.parameter_types.len) break;
            const parameter_type = info.parameter_types[argument_index];
            argument_index += 1;
            const value = try self.evalExpression(argument);
            const coerced = try self.coerceOperand(value, try self.typeOf(argument), parameter_type, self.spanOf(argument));
            try lowered.append(self.arena, try self.lowerArgument(coerced, parameter_type, self.spanOf(argument)));
        }

        const joined = try self.joinArguments(lowered.items);
        var result: Operand = .none;
        if (info.aggregate_return) {
            try self.instruction("call void @\"{s}\"({s})", .{ info.name, joined });
            const layout = (try self.layoutQuery(info.return_type, 0)).?;
            result = .{ .memory = .{ .pointer = result_slot.?, .layout = layout, .fresh = true } };
        } else switch (try self.classify(info.return_type, span)) {
            .void_class => {
                try self.instruction("call void @\"{s}\"({s})", .{ info.name, joined });
            },
            .scalar => |llvm| {
                const value = try self.freshTemp();
                try self.instruction("{s} = call {s} @\"{s}\"({s})", .{ value, llvm, info.name, joined });
                result = .{ .scalar = .{ .text = value, .llvm = llvm } };
            },
            .aggregate => unreachable,
        }
        if (receiver_cleanup) |cleanup| {
            const helper = try self.dropHelper(cleanup.value_type, span);
            try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, cleanup.pointer });
        }
        return result;
    }

    fn lowerArgument(self: *Codegen, operand: Operand, parameter_type: *const Type, span: Token.Location) Error![]const u8 {
        switch (try self.classify(parameter_type, span)) {
            .void_class => return self.report(span, "an argument cannot have no runtime value", .{}),
            .scalar => |llvm| {
                if (operand != .scalar) {
                    return self.report(span, "internal: this scalar argument has no scalar operand", .{});
                }
                return std.fmt.allocPrint(self.arena, "{s} {s}", .{ llvm, operand.scalar.text });
            },
            .aggregate => |layout| {
                const memory = try self.ensureMemory(operand, parameter_type, span);
                // a by-value parameter owns its copy and drops it at the
                // callee's scope end (section 4.2): a place-backed owning
                // argument clones, a temporary transfers
                if (!memory.fresh and try self.ownsHeap(parameter_type, 0)) {
                    const slot = try self.aggregateSlot(layout);
                    const helper = try self.copyHelper(parameter_type, span);
                    try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, slot, memory.pointer });
                    return std.fmt.allocPrint(self.arena, "ptr {s}", .{slot});
                }
                return std.fmt.allocPrint(self.arena, "ptr {s}", .{memory.pointer});
            },
        }
    }

    fn callExtern(self: *Codegen, expression: *const ast.Expression, symbol: resolution.Symbol) Error!Operand {
        const call = expression.call;
        const span = self.spanOf(expression);
        const info = try self.externInfo(symbol, span);
        var lowered: std.ArrayList([]const u8) = .empty;
        for (call.arguments, 0..) |argument, index| {
            const value = try self.evalExpression(argument);
            const argument_type = try self.typeOf(argument);
            if (index < info.parameter_types.len) {
                const coerced = try self.coerceOperand(value, argument_type, info.parameter_types[index], self.spanOf(argument));
                try lowered.append(self.arena, try self.lowerExternArgument(coerced, info.parameter_types[index], self.spanOf(argument)));
            } else {
                // the variadic tail follows the C default promotions
                try lowered.append(self.arena, try self.promoteVariadic(value, argument_type, self.spanOf(argument)));
            }
        }
        const joined = try self.joinArguments(lowered.items);
        const return_resolved = try self.resolvedOf(info.return_type);
        const return_text = try self.externTypeText(info.return_type, span);
        var prototype: []const u8 = return_text;
        if (info.variadic) {
            var signature: std.Io.Writer.Allocating = .init(self.arena);
            signature.writer.print("{s} (", .{return_text}) catch return error.OutOfMemory;
            for (info.parameter_types, 0..) |parameter_type, index| {
                if (index != 0) signature.writer.print(", ", .{}) catch return error.OutOfMemory;
                signature.writer.print("{s}", .{try self.externTypeText(parameter_type, span)}) catch return error.OutOfMemory;
            }
            if (info.parameter_types.len != 0) signature.writer.print(", ", .{}) catch return error.OutOfMemory;
            signature.writer.print("...)", .{}) catch return error.OutOfMemory;
            prototype = signature.writer.buffered();
        }
        if (return_resolved.* == .void_type) {
            try self.instruction("call {s} @\"{s}\"({s})", .{ prototype, info.name, joined });
            return .none;
        }
        const result = try self.freshTemp();
        try self.instruction("{s} = call {s} @\"{s}\"({s})", .{ result, prototype, info.name, joined });
        return .{ .scalar = .{ .text = result, .llvm = return_text } };
    }

    // a slice argument decays to its data pointer at the extern boundary
    // (section 5.3)
    fn lowerExternArgument(self: *Codegen, operand: Operand, parameter_type: *const Type, span: Token.Location) Error![]const u8 {
        const resolved = try self.resolvedOf(parameter_type);
        if (resolved.* == .slice) {
            const memory = try self.ensureMemory(operand, parameter_type, span);
            const data = try self.loadPointerField(memory.pointer, 0);
            return std.fmt.allocPrint(self.arena, "ptr {s}", .{data});
        }
        switch (operand) {
            .scalar => |scalar| return std.fmt.allocPrint(self.arena, "{s} {s}", .{ scalar.llvm, scalar.text }),
            else => return self.report(span, "this argument cannot cross the extern boundary yet", .{}),
        }
    }

    fn promoteVariadic(self: *Codegen, operand: Operand, argument_type: *const Type, span: Token.Location) Error![]const u8 {
        const resolved = try self.resolvedOf(argument_type);
        if (resolved.* == .slice) {
            const memory = try self.ensureMemory(operand, argument_type, span);
            const data = try self.loadPointerField(memory.pointer, 0);
            return std.fmt.allocPrint(self.arena, "ptr {s}", .{data});
        }
        switch (operand) {
            .scalar => |scalar| {
                if (std.mem.eql(u8, scalar.llvm, "float")) {
                    const widened = try self.freshTemp();
                    try self.instruction("{s} = fpext float {s} to double", .{ widened, scalar.text });
                    return std.fmt.allocPrint(self.arena, "double {s}", .{widened});
                }
                if (std.mem.eql(u8, scalar.llvm, "i1") or std.mem.eql(u8, scalar.llvm, "i8") or std.mem.eql(u8, scalar.llvm, "i16")) {
                    const signed = resolved.* == .primitive and resolved.primitive.isSigned();
                    const widened = try self.freshTemp();
                    try self.instruction("{s} = {s} {s} {s} to i32", .{ widened, if (signed) "sext" else "zext", scalar.llvm, scalar.text });
                    return std.fmt.allocPrint(self.arena, "i32 {s}", .{widened});
                }
                return std.fmt.allocPrint(self.arena, "{s} {s}", .{ scalar.llvm, scalar.text });
            },
            else => return self.report(span, "this argument cannot cross the extern boundary yet", .{}),
        }
    }

    fn joinArguments(self: *Codegen, arguments: []const []const u8) Error![]const u8 {
        var joined: std.Io.Writer.Allocating = .init(self.arena);
        for (arguments, 0..) |argument, index| {
            if (index != 0) joined.writer.print(", ", .{}) catch return error.OutOfMemory;
            joined.writer.print("{s}", .{argument}) catch return error.OutOfMemory;
        }
        return joined.writer.buffered();
    }

    // a closure is a heap block: the call, drop, and copy function pointers
    // head the block, the captured environment follows (section 4.4)
    const closure_header_size: u64 = 24;

    const ClosureCapture = struct {
        capture: ast.Capture,
        // the checker's capture typing, already substituted through the
        // active instance's bindings
        binding_type: *const Type,
        mode: CaptureMode,
        offset: u64,
        layout: Checker.Layout,
    };

    const ClosureShape = struct {
        captures: []const ClosureCapture,
        size: u64,
    };

    fn closureShape(self: *Codegen, expression: *const ast.Expression, lambda: ast.Lambda, span: Token.Location) Error!ClosureShape {
        const bindings = self.checker.lambda_captures.get(expression) orelse &.{};
        if (bindings.len != lambda.captures.len) {
            return self.report(span, "internal: no capture typing recorded for this lambda", .{});
        }
        const captures = try self.arena.alloc(ClosureCapture, bindings.len);
        var cursor: u64 = closure_header_size;
        for (lambda.captures, bindings, 0..) |capture, binding, index| {
            const binding_type = try self.substituted(binding.binding_type);
            if (try self.unsupportedReason(binding_type, 0)) |reason| {
                return self.report(capture.name.location, "{s} are not yet supported by native code generation", .{reason});
            }
            // the mode mirrors the checker's captureBinding (section 2.1):
            // a prefix modifier borrows or takes ownership; an annotation
            // borrows only when the checker typed it as a reference, and
            // otherwise deep-copies (an annotation like '&[u8]' names a
            // slice VALUE, not a borrow)
            const mode: CaptureMode = if (capture.modifier) |modifier| switch (modifier) {
                .reference, .reference_var => .reference,
                .pointer, .pointer_var => .owning,
            } else if ((try self.resolvedOf(binding_type)).* == .reference) .reference else .copy;
            const layout: Checker.Layout = switch (mode) {
                .reference => .{ .size = 8, .alignment = 8 },
                // an owning capture moves the whole owning value: a thin
                // pointer is 8 bytes, an interface object's fat pair is 16
                .owning, .copy => (try self.layoutQuery(binding_type, 0)) orelse
                    return self.report(capture.name.location, "this capture type has no defined runtime layout", .{}),
            };
            cursor = Checker.alignForward(cursor, layout.alignment);
            captures[index] = .{ .capture = capture, .binding_type = binding_type, .mode = mode, .offset = cursor, .layout = layout };
            cursor += layout.size;
        }
        return .{ .captures = captures, .size = Checker.alignForward(cursor, 8) };
    }

    fn evalLambda(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const lambda = expression.lambda;
        const span = self.spanOf(expression);
        const fn_resolved = try self.resolvedOf(try self.typeOf(expression));
        if (fn_resolved.* != .function) {
            return self.report(span, "internal: a lambda without a function type", .{});
        }
        var parameter_types: std.ArrayList(*const Type) = .empty;
        for (fn_resolved.function.parameter_types) |parameter_type| {
            const concrete = try self.substituted(parameter_type);
            if (try self.unsupportedReason(concrete, 0)) |reason| {
                return self.report(span, "{s} are not yet supported by native code generation", .{reason});
            }
            try parameter_types.append(self.arena, concrete);
        }
        const return_type = try self.substituted(fn_resolved.function.return_type);
        if (try self.unsupportedReason(return_type, 0)) |reason| {
            return self.report(span, "{s} are not yet supported by native code generation", .{reason});
        }
        const shape = try self.closureShape(expression, lambda, span);

        const id = self.global_counter;
        self.global_counter += 1;
        const body_name = try std.fmt.allocPrint(self.arena, "alloy.lambda.{d}", .{id});
        const drop_name = try std.fmt.allocPrint(self.arena, "alloy.lambda.drop.{d}", .{id});
        const copy_name = try std.fmt.allocPrint(self.arena, "alloy.lambda.copy.{d}", .{id});
        try self.emitLambdaBody(body_name, lambda, parameter_types.items, return_type, shape.captures);
        // a capture-less lambda carries no state: a constant block with
        // null drop and copy slots costs nothing to share or release
        if (shape.captures.len == 0) {
            const block_name = try std.fmt.allocPrint(self.arena, "alloy.closure.{d}", .{id});
            self.constants.writer.print("@\"{s}\" = internal constant {{ ptr, ptr, ptr }} {{ ptr @\"{s}\", ptr null, ptr null }}\n", .{ block_name, body_name }) catch return error.OutOfMemory;
            const slot = try self.scalarSlot("ptr");
            try self.instruction("store ptr @\"{s}\", ptr {s}", .{ block_name, slot });
            return .{ .memory = .{ .pointer = slot, .layout = .{ .size = 8, .alignment = 8 }, .fresh = true } };
        }
        try self.emitLambdaDrop(drop_name, shape.captures, span);
        try self.emitLambdaCopy(copy_name, shape.captures, shape.size, span);

        const block = try self.allocateFixed(shape.size);
        try self.instruction("store ptr @\"{s}\", ptr {s}", .{ body_name, block });
        const drop_slot = try self.byteOffset(block, 8);
        try self.instruction("store ptr @\"{s}\", ptr {s}", .{ drop_name, drop_slot });
        const copy_slot = try self.byteOffset(block, 16);
        try self.instruction("store ptr @\"{s}\", ptr {s}", .{ copy_name, copy_slot });

        // capture values come from the enclosing scope at construction
        // (section 4.4): copies clone, references alias the place, owning
        // captures take the owning value and null its source
        for (shape.captures) |closure_capture| {
            const capture_name = closure_capture.capture.name.slice(self.source());
            const local = self.lookupLocal(capture_name) orelse
                return self.report(closure_capture.capture.name.location, "a capture must name a local variable (section 4.4)", .{});
            const raw = Place{ .pointer = local.pointer, .value_type = local.declared_type };
            const environment_slot = try self.byteOffset(block, closure_capture.offset);
            switch (closure_capture.mode) {
                .copy => {
                    const pierced = try self.piercePlace(raw);
                    const value = try self.loadPlace(pierced, span);
                    const coerced = try self.coerceOperand(value, pierced.value_type, closure_capture.binding_type, span);
                    try self.storeOperand(environment_slot, coerced, closure_capture.binding_type, span);
                },
                .reference => {
                    const pierced = try self.piercePlace(raw);
                    try self.instruction("store ptr {s}, ptr {s}", .{ pierced.pointer, environment_slot });
                },
                .owning => {
                    // the whole owning value transfers (a thin pointer or
                    // an interface fat pair); nulling the leading pointer
                    // marks the source moved-from (section 4.2)
                    try self.copyBytes(environment_slot, raw.pointer, closure_capture.layout.size);
                    try self.instruction("store ptr null, ptr {s}", .{raw.pointer});
                },
            }
        }
        const slot = try self.scalarSlot("ptr");
        try self.instruction("store ptr {s}, ptr {s}", .{ block, slot });
        return .{ .memory = .{ .pointer = slot, .layout = .{ .size = 8, .alignment = 8 }, .fresh = true } };
    }

    // the lambda's body compiles like a named function with a leading
    // closure parameter; captures bind to environment slots, pinned so the
    // frame never drops storage the closure owns
    fn emitLambdaBody(self: *Codegen, name: []const u8, lambda: ast.Lambda, parameter_types: []const *const Type, return_type: *const Type, captures: []const ClosureCapture) Error!void {
        const zero_span = Token.Location{ .start = 0, .end = 0 };
        const saved = self.beginHelperFunction();
        const saved_scopes = self.scopes;
        const saved_break_targets = self.break_targets;
        const saved_yield_targets = self.yield_targets;
        const saved_return_type = self.return_type;
        const saved_return_slot = self.return_slot;
        self.scopes = .empty;
        self.break_targets = .empty;
        self.yield_targets = .empty;
        self.return_type = return_type;
        self.return_slot = null;

        var subprogram: ?usize = null;
        if (!self.release_mode) {
            subprogram = try self.beginDebugFunction("lambda", self.current_view, self.statementOffset(lambda.function.body));
        }
        const signature = try self.functionSignature(name, true, parameter_types, return_type, zero_span, subprogram);
        if (signature.aggregate_return) self.return_slot = "%return.slot";

        try self.pushFrame();
        for (captures) |closure_capture| {
            const capture_name = closure_capture.capture.name.slice(self.source());
            const environment_slot = try self.byteOffset("%closure", closure_capture.offset);
            try self.bindPinned(capture_name, environment_slot, closure_capture.binding_type);
        }
        const names = try self.arena.alloc([]const u8, lambda.function.parameters.len);
        for (lambda.function.parameters, names) |parameter, *parameter_name| {
            parameter_name.* = parameter.name.slice(self.source());
        }
        try self.bindParameters(names, parameter_types, signature.parameters);

        try self.execStatement(lambda.function.body);
        if (!self.terminated) try self.emitDefaultReturn();
        try self.finishHelperFunction(saved, signature.header);

        self.scopes = saved_scopes;
        self.break_targets = saved_break_targets;
        self.yield_targets = saved_yield_targets;
        self.return_type = saved_return_type;
        self.return_slot = saved_return_slot;
    }

    fn capturedOwnership(self: *Codegen, closure_capture: ClosureCapture) Error!bool {
        return switch (closure_capture.mode) {
            .reference => false,
            .copy, .owning => self.ownsHeap(closure_capture.binding_type, 0),
        };
    }

    fn emitLambdaDrop(self: *Codegen, name: []const u8, captures: []const ClosureCapture, span: Token.Location) Error!void {
        try self.declareFree();
        const saved = self.beginHelperFunction();
        for (captures) |closure_capture| {
            if (!try self.capturedOwnership(closure_capture)) continue;
            const environment_slot = try self.byteOffset("%closure", closure_capture.offset);
            const helper = try self.dropHelper(closure_capture.binding_type, span);
            try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, environment_slot });
        }
        try self.instruction("call void @\"free\"(ptr %closure)", .{});
        try self.instruction("ret void", .{});
        self.terminated = true;
        const header = try std.fmt.allocPrint(self.arena, "define internal void @\"{s}\"(ptr %closure) {{\nentry:\n", .{name});
        try self.finishHelperFunction(saved, header);
    }

    fn emitLambdaCopy(self: *Codegen, name: []const u8, captures: []const ClosureCapture, size: u64, span: Token.Location) Error!void {
        const saved = self.beginHelperFunction();
        const fresh_block = try self.allocateFixed(size);
        try self.copyBytes(fresh_block, "%closure", size);
        for (captures) |closure_capture| {
            if (!try self.capturedOwnership(closure_capture)) continue;
            const destination = try self.byteOffset(fresh_block, closure_capture.offset);
            const origin = try self.byteOffset("%closure", closure_capture.offset);
            const helper = try self.copyHelper(closure_capture.binding_type, span);
            try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, destination, origin });
        }
        try self.instruction("ret ptr {s}", .{fresh_block});
        self.terminated = true;
        const header = try std.fmt.allocPrint(self.arena, "define internal ptr @\"{s}\"(ptr %closure) {{\nentry:\n", .{name});
        try self.finishHelperFunction(saved, header);
    }

    // a named function used as a value: a constant block whose null drop
    // and copy slots mark it static, with a thunk absorbing the closure
    // argument (section 4.4); the checker recorded exactly which symbol
    // this path resolved to
    fn functionValue(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const span = self.spanOf(expression);
        const symbol = self.checker.value_targets.get(expression) orelse
            return self.report(span, "internal: no recorded function for this function value", .{});
        const info = try self.functionInfo(symbol, span, &.{});
        const block = try self.staticClosure(info, span);
        const slot = try self.scalarSlot("ptr");
        try self.instruction("store ptr @\"{s}\", ptr {s}", .{ block, slot });
        return .{ .memory = .{ .pointer = slot, .layout = .{ .size = 8, .alignment = 8 }, .fresh = true } };
    }

    fn staticClosure(self: *Codegen, info: *FunctionInfo, span: Token.Location) Error![]const u8 {
        if (self.static_closures.get(info.name)) |existing| return existing;
        const id = self.global_counter;
        self.global_counter += 1;
        const thunk_name = try std.fmt.allocPrint(self.arena, "alloy.fnvalue.{d}", .{id});
        const block_name = try std.fmt.allocPrint(self.arena, "alloy.closure.{d}", .{id});
        try self.static_closures.put(self.arena, info.name, block_name);

        const saved = self.beginHelperFunction();
        const signature = try self.functionSignature(thunk_name, true, info.parameter_types, info.return_type, span, null);
        var forwarded: std.ArrayList([]const u8) = .empty;
        if (signature.aggregate_return) {
            try forwarded.append(self.arena, "ptr %return.slot");
        }
        for (signature.parameters) |parameter| {
            const text = switch (parameter.class) {
                .void_class => unreachable,
                .scalar => |llvm| try std.fmt.allocPrint(self.arena, "{s} {s}", .{ llvm, parameter.register }),
                .aggregate => try std.fmt.allocPrint(self.arena, "ptr {s}", .{parameter.register}),
            };
            try forwarded.append(self.arena, text);
        }
        const joined = try self.joinArguments(forwarded.items);
        if (std.mem.eql(u8, signature.return_text, "void")) {
            try self.instruction("call void @\"{s}\"({s})", .{ info.name, joined });
            try self.instruction("ret void", .{});
        } else {
            const result = try self.freshTemp();
            try self.instruction("{s} = call {s} @\"{s}\"({s})", .{ result, signature.return_text, info.name, joined });
            try self.instruction("ret {s} {s}", .{ signature.return_text, result });
        }
        self.terminated = true;
        try self.finishHelperFunction(saved, signature.header);

        self.constants.writer.print("@\"{s}\" = internal constant {{ ptr, ptr, ptr }} {{ ptr @\"{s}\", ptr null, ptr null }}\n", .{ block_name, thunk_name }) catch return error.OutOfMemory;
        return block_name;
    }

    // a call through a function value: the block's first slot is the call
    // target, which receives the block itself as its leading argument
    fn indirectCall(self: *Codegen, expression: *const ast.Expression, place: Place) Error!Operand {
        const call = expression.call;
        const span = self.spanOf(expression);
        const fn_resolved = try self.resolvedOf(place.value_type);
        const fn_type = fn_resolved.function;
        const return_type = try self.substituted(fn_type.return_type);
        const return_class = try self.classify(return_type, span);
        var lowered: std.ArrayList([]const u8) = .empty;
        var result_slot: ?[]const u8 = null;
        if (return_class == .aggregate) {
            result_slot = try self.aggregateSlot(return_class.aggregate);
            try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{result_slot.?}));
        }
        for (call.arguments, 0..) |argument, index| {
            if (index >= fn_type.parameter_types.len) break;
            const parameter_type = try self.substituted(fn_type.parameter_types[index]);
            const value = try self.evalExpression(argument);
            const coerced = try self.coerceOperand(value, try self.typeOf(argument), parameter_type, self.spanOf(argument));
            try lowered.append(self.arena, try self.lowerArgument(coerced, parameter_type, self.spanOf(argument)));
        }
        // the block loads AFTER the arguments: an argument may reassign the
        // callee place, and the call must go through the live block
        const block = try self.loadScalar(place.pointer, "ptr");
        if (!self.release_mode) {
            const live = try self.freshTemp();
            try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, block });
            try self.faultUnless(live, "runtime fault: call through an absent function value (section 4.4)");
        }
        const call_pointer = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr {s}", .{ call_pointer, block });
        try lowered.insert(self.arena, 0, try std.fmt.allocPrint(self.arena, "ptr {s}", .{block}));
        const joined = try self.joinArguments(lowered.items);
        switch (return_class) {
            .aggregate => |layout| {
                try self.instruction("call void {s}({s})", .{ call_pointer, joined });
                return .{ .memory = .{ .pointer = result_slot.?, .layout = layout, .fresh = true } };
            },
            .void_class => {
                try self.instruction("call void {s}({s})", .{ call_pointer, joined });
                return .none;
            },
            .scalar => |llvm| {
                const result = try self.freshTemp();
                try self.instruction("{s} = call {s} {s}({s})", .{ result, llvm, call_pointer, joined });
                return .{ .scalar = .{ .text = result, .llvm = llvm } };
            },
        }
    }

    fn evalStructInit(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const struct_init = expression.struct_init;
        const span = self.spanOf(expression);
        const result_type = try self.typeOf(expression);
        const slots = (try self.fieldSlotsQuery(result_type)) orelse
            return self.report(span, "this construction has no struct layout", .{});
        const layout = (try self.layoutQuery(result_type, 0)) orelse
            return self.report(span, "this type has no defined layout", .{});
        const storage = try self.aggregateSlot(layout);
        // zeroed padding keeps 'as' reinterpretation deterministic
        try self.zeroFill(storage, layout.size);
        for (struct_init.members) |member| {
            const name = member.name.slice(self.source());
            const slot = for (slots) |candidate| {
                if (std.mem.eql(u8, candidate.name, name)) break candidate;
            } else return self.report(member.name.location, "no field '{s}' here", .{name});
            const value = try self.evalExpression(member.value);
            const coerced = try self.coerceOperand(value, try self.typeOf(member.value), slot.field_type, member.name.location);
            const pointer = try self.byteOffset(storage, slot.offset);
            try self.storeOperand(pointer, coerced, slot.field_type, member.name.location);
        }
        return .{ .memory = .{ .pointer = storage, .layout = layout, .fresh = true } };
    }

    fn evalArrayLiteral(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const elements = expression.array_literal;
        const span = self.spanOf(expression);
        const resolved = try self.resolvedOf(try self.typeOf(expression));
        if (resolved.* != .fixed_array) return self.report(span, "this array literal has no fixed layout", .{});
        const element_type = resolved.fixed_array.element;
        const element_layout = (try self.layoutQuery(element_type, 0)) orelse
            return self.report(span, "this element type has no defined layout", .{});
        const layout = (try self.layoutQuery(resolved, 0)) orelse
            return self.report(span, "this array has no defined layout", .{});
        const storage = try self.aggregateSlot(layout);
        try self.zeroFill(storage, layout.size);
        for (elements, 0..) |element, index| {
            const value = try self.evalExpression(element);
            const coerced = try self.coerceOperand(value, try self.typeOf(element), element_type, self.spanOf(element));
            const pointer = try self.byteOffset(storage, element_layout.size * index);
            try self.storeOperand(pointer, coerced, element_type, self.spanOf(element));
        }
        return .{ .memory = .{ .pointer = storage, .layout = layout, .fresh = true } };
    }

    fn evalArrayFill(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const array_fill = expression.array_fill;
        const span = self.spanOf(expression);
        const resolved = try self.resolvedOf(try self.typeOf(expression));
        if (resolved.* != .fixed_array) {
            return self.report(span, "heap arrays ('*[T]') are not yet supported by native code generation", .{});
        }
        const element_type = resolved.fixed_array.element;
        const element_layout = (try self.layoutQuery(element_type, 0)) orelse
            return self.report(span, "this element type has no defined layout", .{});
        const layout = (try self.layoutQuery(resolved, 0)) orelse
            return self.report(span, "this array has no defined layout", .{});
        const storage = try self.aggregateSlot(layout);
        try self.zeroFill(storage, layout.size);
        const value = try self.evalExpression(array_fill.value);
        const coerced = try self.coerceOperand(value, try self.typeOf(array_fill.value), element_type, span);
        const fill = try self.fillSource(coerced, element_type, span);
        try self.emitFillLoop(storage, fill.operand, element_type, element_layout.size, resolved.fixed_array.length, span);
        try self.dropFillSource(fill, element_type, span);
        return .{ .memory = .{ .pointer = storage, .layout = layout, .fresh = true } };
    }

    const FillSource = struct {
        operand: Operand,
        // a fresh owning fill value transfers into the loop as clones, so
        // the original drops once after the last element copies
        drop_after: bool,
    };

    // every element of a fill owns its own deep copy (section 5.1): an
    // owning fill value is pinned as a non-fresh memory operand so each
    // element store clones instead of aliasing one allocation
    fn fillSource(self: *Codegen, operand: Operand, element_type: *const Type, span: Token.Location) Error!FillSource {
        if (!try self.ownsHeap(element_type, 0)) return .{ .operand = operand, .drop_after = false };
        var memory = try self.ensureMemory(operand, element_type, span);
        const owned_here = memory.fresh;
        memory.fresh = false;
        return .{ .operand = .{ .memory = memory }, .drop_after = owned_here };
    }

    fn dropFillSource(self: *Codegen, fill: FillSource, element_type: *const Type, span: Token.Location) Error!void {
        if (!fill.drop_after) return;
        const helper = try self.dropHelper(element_type, span);
        try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, fill.operand.memory.pointer });
    }

    fn evalArrayRange(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const array_range = expression.array_range;
        const span = self.spanOf(expression);
        const resolved = try self.resolvedOf(try self.typeOf(expression));
        if (resolved.* != .fixed_array) {
            return self.report(span, "a runtime-sized range only materializes on the heap, which native code generation does not support yet", .{});
        }
        const element_type = resolved.fixed_array.element;
        const element_layout = (try self.layoutQuery(element_type, 0)) orelse
            return self.report(span, "this element type has no defined layout", .{});
        const layout = (try self.layoutQuery(resolved, 0)) orelse
            return self.report(span, "this array has no defined layout", .{});
        const storage = try self.aggregateSlot(layout);
        try self.zeroFill(storage, layout.size);
        const start: Scalar = if (array_range.start) |start_expression| start: {
            const value = try self.evalExpression(start_expression);
            break :start value.scalar;
        } else .{ .text = "0", .llvm = "i64" };
        const start_wide = try self.widenToIndex(start, try self.resolvedOf(try self.typeOf(array_range.start orelse expression)));
        const llvm = switch (try self.classify(element_type, span)) {
            .scalar => |text| text,
            else => return self.report(span, "a range yields scalar elements", .{}),
        };
        const index_slot = try self.scalarSlot("i64");
        try self.storeScalar(index_slot, .{ .text = "0", .llvm = "i64" });
        const header = try self.freshLabel("range.header");
        const body = try self.freshLabel("range.body");
        const exit = try self.freshLabel("range.exit");
        try self.startBlock(header);
        const index = try self.loadScalar(index_slot, "i64");
        const continues = try self.freshTemp();
        try self.instruction("{s} = icmp ult i64 {s}, {d}", .{ continues, index, resolved.fixed_array.length });
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ continues, body, exit });
        self.terminated = true;
        try self.startBlock(body);
        const counter = try self.freshTemp();
        try self.instruction("{s} = add i64 {s}, {s}", .{ counter, start_wide, index });
        const narrowed = try self.narrowFromIndex(counter, llvm);
        const scaled = try self.freshTemp();
        try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, index, element_layout.size });
        const pointer = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ pointer, storage, scaled });
        try self.storeScalar(pointer, .{ .text = narrowed, .llvm = llvm });
        const next = try self.freshTemp();
        try self.instruction("{s} = add i64 {s}, 1", .{ next, index });
        try self.storeScalar(index_slot, .{ .text = next, .llvm = "i64" });
        try self.instruction("br label %{s}", .{header});
        self.terminated = true;
        try self.startBlock(exit);
        return .{ .memory = .{ .pointer = storage, .layout = layout, .fresh = true } };
    }

    fn emitFillLoop(self: *Codegen, storage: []const u8, value: Operand, element_type: *const Type, stride: u64, count: u64, span: Token.Location) Error!void {
        const index_slot = try self.scalarSlot("i64");
        try self.storeScalar(index_slot, .{ .text = "0", .llvm = "i64" });
        const header = try self.freshLabel("fill.header");
        const body = try self.freshLabel("fill.body");
        const exit = try self.freshLabel("fill.exit");
        try self.startBlock(header);
        const index = try self.loadScalar(index_slot, "i64");
        const continues = try self.freshTemp();
        try self.instruction("{s} = icmp ult i64 {s}, {d}", .{ continues, index, count });
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ continues, body, exit });
        self.terminated = true;
        try self.startBlock(body);
        const scaled = try self.freshTemp();
        try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, index, stride });
        const pointer = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ pointer, storage, scaled });
        try self.storeOperand(pointer, value, element_type, span);
        const next = try self.freshTemp();
        try self.instruction("{s} = add i64 {s}, 1", .{ next, index });
        try self.storeScalar(index_slot, .{ .text = next, .llvm = "i64" });
        try self.instruction("br label %{s}", .{header});
        self.terminated = true;
        try self.startBlock(exit);
    }

    fn resultSlot(self: *Codegen, expression: *const ast.Expression) Error!?Slot {
        const recorded = self.expression_types.get(expression) orelse return null;
        const resolved = try self.resolvedOf(recorded);
        if (resolved.* == .void_type or resolved.* == .unknown) return null;
        const value_type = try self.checker.defaulted(recorded);
        // zeroed so a path that never breaks a value still drops safely
        const pointer = switch (try self.classify(value_type, self.spanOf(expression))) {
            .void_class => return null,
            .scalar => |llvm| slot: {
                const slot = try self.scalarSlot(llvm);
                if (std.mem.eql(u8, llvm, "ptr")) {
                    try self.instruction("store ptr null, ptr {s}", .{slot});
                }
                break :slot slot;
            },
            .aggregate => |layout| slot: {
                const slot = try self.aggregateSlot(layout);
                try self.zeroFill(slot, layout.size);
                break :slot slot;
            },
        };
        return .{ .pointer = pointer, .value_type = value_type };
    }

    fn slotOperand(self: *Codegen, slot: ?Slot, span: Token.Location) Error!Operand {
        const present = slot orelse return .none;
        switch (try self.classify(present.value_type, span)) {
            .void_class => return .none,
            .scalar => |llvm| return .{ .scalar = .{ .text = try self.loadScalar(present.pointer, llvm), .llvm = llvm } },
            .aggregate => |layout| return .{ .memory = .{ .pointer = present.pointer, .layout = layout, .fresh = true } },
        }
    }

    fn evalIf(self: *Codegen, expression: *const ast.Expression, if_expr: ast.IfExpression) Error!Operand {
        const span = self.spanOf(expression);
        const slot = try self.resultSlot(expression);
        const then_label = try self.freshLabel("if.then");
        const else_label = if (if_expr.else_branch != null) try self.freshLabel("if.else") else null;
        const exit_label = try self.freshLabel("if.exit");

        var capture_binding: ?CaptureBinding = null;
        if (if_expr.capture) |capture| {
            capture_binding = try self.isConditionCapture(if_expr.condition, capture, then_label, else_label orelse exit_label);
        } else {
            const condition = try self.evalExpression(if_expr.condition);
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ condition.scalar.text, then_label, else_label orelse exit_label });
            self.terminated = true;
        }

        // only a value-position if receives 'yield'; a statement-position
        // if is transparent so yields reach the enclosing value construct
        if (slot != null) {
            try self.yield_targets.append(self.arena, .{ .exit_label = exit_label, .slot = slot, .frame_depth = self.scopes.items.len });
        }
        try self.startBlock(then_label);
        try self.pushFrame();
        if (capture_binding) |binding| {
            if (binding.borrowed) {
                try self.bindBorrowed(binding.capture.name.slice(self.source()), binding.payload_pointer, binding.payload_type);
            } else {
                try self.bindCapture(binding.capture, binding.payload_pointer, binding.payload_type, span);
            }
        }
        try self.execStatement(if_expr.then_branch);
        try self.closeFrame();
        if (!self.terminated) {
            try self.instruction("br label %{s}", .{exit_label});
            self.terminated = true;
        }
        if (if_expr.else_branch) |else_branch| {
            try self.startBlock(else_label.?);
            try self.pushFrame();
            try self.execStatement(else_branch);
            try self.closeFrame();
            if (!self.terminated) {
                try self.instruction("br label %{s}", .{exit_label});
                self.terminated = true;
            }
        }
        if (slot != null) _ = self.yield_targets.pop();
        try self.startBlock(exit_label);
        return self.slotOperand(slot, span);
    }

    const CaptureBinding = struct {
        capture: ast.Capture,
        payload_pointer: []const u8,
        payload_type: *const Type,
        // a downcast capture borrows the concrete value behind the data
        // pointer instead of copying a payload (section 3.2)
        borrowed: bool = false,
    };

    const Descriptor = struct {
        name: []const u8,
        concrete: *const Type,
    };

    // resolves the named target of a downcast ('is Dog', a match arm) to
    // its type identity global and concrete type
    fn downcastDescriptor(self: *Codegen, target: *const ast.TypeExpression, span: Token.Location) Error!Descriptor {
        const target_type = try self.checker.typeFromExpressionIn(target, self.current_bindings orelse &empty_type_environment, self.current_view);
        const resolved = try self.resolvedOf(target_type);
        if (resolved.* != .declared) {
            return self.report(span, "a downcast target must be a named type (section 3.2)", .{});
        }
        return .{
            .name = try self.typeDescriptor(resolved.declared.definition),
            .concrete = resolved,
        };
    }

    fn bindBorrowed(self: *Codegen, name: []const u8, pointer_value: []const u8, child_type: *const Type) Error!void {
        const slot = try self.scalarSlot("ptr");
        try self.storeScalar(slot, .{ .text = pointer_value, .llvm = "ptr" });
        const reference_type = try self.arena.create(Type);
        reference_type.* = .{ .reference = .{ .mutable = false, .child = child_type } };
        try self.bindLocal(name, slot, reference_type);
    }

    // 'if (subject is ::Variant) |payload|' binds the payload; with an
    // interface-object subject 'if (s is Dog) |d|' borrows the concrete
    // value on a type identity match (section 3.2)
    fn isConditionCapture(self: *Codegen, condition: *const ast.Expression, capture: ast.Capture, then_label: []const u8, miss_label: []const u8) Error!CaptureBinding {
        const unwrapped = unwrapGrouped(condition);
        const span = self.spanOf(condition);
        if (unwrapped.* != .cast or unwrapped.cast.operator.tag != .keyword_is) {
            return self.report(span, "an 'if' capture needs an 'is' condition", .{});
        }
        const cast = unwrapped.cast;
        const operand_type = try self.typeOf(cast.operand);
        const place = (try self.evalPlace(cast.operand)) orelse place: {
            const value = try self.evalExpression(cast.operand);
            const memory = try self.ensureMemory(value, operand_type, span);
            break :place Place{ .pointer = memory.pointer, .value_type = operand_type };
        };
        if (try self.interfaceOfPlace(place)) |_| {
            const descriptor = try self.downcastDescriptor(cast.target, span);
            const data = try self.loadPointerField(place.pointer, 0);
            const type_id = try self.loadPointerField(place.pointer, 8);
            const matches = try self.freshTemp();
            try self.instruction("{s} = icmp eq ptr {s}, @\"{s}\"", .{ matches, type_id, descriptor.name });
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, then_label, miss_label });
            self.terminated = true;
            return .{ .capture = capture, .payload_pointer = data, .payload_type = descriptor.concrete, .borrowed = true };
        }
        const frame = (try self.enumFrameQuery(place.value_type)) orelse
            return self.report(span, "'is' needs an enum or interface-object subject", .{});
        const target_token = cast.target.named.path[cast.target.named.path.len - 1];
        const variant_name = target_token.slice(self.source());
        const variant_index = try self.variantIndex(frame, variant_name, span);
        const tag = try self.loadTag(place.pointer, frame);
        const matches = try self.freshTemp();
        try self.instruction("{s} = icmp eq {s} {s}, {d}", .{ matches, tagTypeText(frame.tag_size), tag, variant_index });
        // the payload address computes before the branch terminates this
        // block; afterwards it would land in an unreachable block and fail
        // to dominate the capture binding in the then block
        const payload_type = frame.variants[variant_index].payload;
        const payload_pointer = if (payload_type != null)
            try self.byteOffset(place.pointer, frame.payload_offset)
        else
            place.pointer;
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, then_label, miss_label });
        self.terminated = true;
        return .{
            .capture = capture,
            .payload_pointer = payload_pointer,
            .payload_type = payload_type orelse &void_type,
        };
    }

    fn evalWhile(self: *Codegen, expression: *const ast.Expression, while_expr: ast.WhileExpression) Error!Operand {
        const span = self.spanOf(expression);
        const slot = try self.resultSlot(expression);
        const header = try self.freshLabel("while.header");
        const body = try self.freshLabel("while.body");
        const after = try self.freshLabel("while.after");
        const exit = try self.freshLabel("while.exit");
        try self.startBlock(header);
        const condition = try self.evalExpression(while_expr.condition);
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ condition.scalar.text, body, after });
        self.terminated = true;
        try self.break_targets.append(self.arena, .{ .exit_label = exit, .slot = slot, .frame_depth = self.scopes.items.len });
        try self.startBlock(body);
        try self.pushFrame();
        try self.execStatement(while_expr.body);
        try self.closeFrame();
        if (!self.terminated) {
            try self.instruction("br label %{s}", .{header});
            self.terminated = true;
        }
        try self.startBlock(after);
        if (while_expr.else_branch) |else_branch| {
            try self.pushFrame();
            try self.execStatement(else_branch);
            try self.closeFrame();
        }
        _ = self.break_targets.pop();
        try self.startBlock(exit);
        return self.slotOperand(slot, span);
    }

    const ForSubject = struct {
        kind: union(enum) {
            counter: []const u8,
            elements: struct {
                data_pointer: []const u8,
                element_type: *const Type,
                stride: u64,
            },
        },
        length: []const u8,
        capture_type: *const Type,
    };

    // array forms and ranges lower to a counting loop; any other single
    // subject drives the cursor protocol (section 4.3)
    fn evalFor(self: *Codegen, expression: *const ast.Expression, for_expr: ast.ForExpression) Error!Operand {
        const span = self.spanOf(expression);
        if (for_expr.subjects.len == 1 and for_expr.subjects[0].* != .array_range) {
            const subject_type = try self.substituted(try self.typeOf(for_expr.subjects[0]));
            const resolved = try self.resolvedOf(subject_type);
            switch (resolved.*) {
                .slice, .fixed_array, .heap_array, .unknown => {},
                else => return self.evalForCursor(expression, for_expr, subject_type),
            }
        }
        const slot = try self.resultSlot(expression);
        // the loop frame owns materialized subject temporaries: 'break'
        // leaves it intact and it drops once at the shared exit block
        try self.pushFrame();
        var subjects: std.ArrayList(ForSubject) = .empty;
        for (for_expr.subjects) |subject_expression| {
            try subjects.append(self.arena, try self.forSubject(subject_expression));
        }
        if (!self.release_mode and subjects.items.len > 1) {
            for (subjects.items[1..]) |subject| {
                const equal = try self.freshTemp();
                try self.instruction("{s} = icmp eq i64 {s}, {s}", .{ equal, subjects.items[0].length, subject.length });
                try self.faultUnless(equal, "runtime fault: for subjects disagree on length (section 4.3)");
            }
        }
        const index_slot = try self.scalarSlot("i64");
        try self.storeScalar(index_slot, .{ .text = "0", .llvm = "i64" });
        const header = try self.freshLabel("for.header");
        const body = try self.freshLabel("for.body");
        const latch = try self.freshLabel("for.latch");
        const after = try self.freshLabel("for.after");
        const exit = try self.freshLabel("for.exit");
        try self.startBlock(header);
        const index = try self.loadScalar(index_slot, "i64");
        const continues = try self.freshTemp();
        try self.instruction("{s} = icmp slt i64 {s}, {s}", .{ continues, index, subjects.items[0].length });
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ continues, body, after });
        self.terminated = true;
        try self.break_targets.append(self.arena, .{ .exit_label = exit, .slot = slot, .frame_depth = self.scopes.items.len });
        try self.startBlock(body);
        try self.pushFrame();
        const capture_count = @min(for_expr.captures.len, subjects.items.len);
        for (for_expr.captures[0..capture_count], subjects.items[0..capture_count]) |capture, subject| {
            switch (subject.kind) {
                .counter => |start| {
                    const counter = try self.freshTemp();
                    try self.instruction("{s} = add i64 {s}, {s}", .{ counter, start, index });
                    const llvm = switch (try self.classify(subject.capture_type, span)) {
                        .scalar => |text| text,
                        else => return self.report(span, "a range capture is a scalar", .{}),
                    };
                    const narrowed = try self.narrowFromIndex(counter, llvm);
                    const capture_slot = try self.scalarSlot(llvm);
                    try self.storeScalar(capture_slot, .{ .text = narrowed, .llvm = llvm });
                    try self.bindLocal(capture.name.slice(self.source()), capture_slot, subject.capture_type);
                },
                .elements => |elements| {
                    const scaled = try self.freshTemp();
                    try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, index, elements.stride });
                    const pointer = try self.freshTemp();
                    try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ pointer, elements.data_pointer, scaled });
                    try self.bindCapture(capture, pointer, elements.element_type, span);
                },
            }
        }
        try self.execStatement(for_expr.body);
        try self.closeFrame();
        if (!self.terminated) {
            try self.instruction("br label %{s}", .{latch});
            self.terminated = true;
        }
        try self.startBlock(latch);
        const advanced = try self.freshTemp();
        const reloaded = try self.loadScalar(index_slot, "i64");
        try self.instruction("{s} = add i64 {s}, 1", .{ advanced, reloaded });
        try self.storeScalar(index_slot, .{ .text = advanced, .llvm = "i64" });
        try self.instruction("br label %{s}", .{header});
        self.terminated = true;
        try self.startBlock(after);
        if (for_expr.else_branch) |else_branch| {
            try self.pushFrame();
            try self.execStatement(else_branch);
            try self.closeFrame();
        }
        _ = self.break_targets.pop();
        try self.startBlock(exit);
        // both the break path and the normal path pass through the exit
        // block, so the loop frame drops exactly once here
        try self.closeFrame();
        return self.slotOperand(slot, span);
    }

    // the cursor protocol (section 4.3): 'subject.iterator()' yields a
    // cursor advanced by 'next()' until it reports 'None'
    fn evalForCursor(self: *Codegen, expression: *const ast.Expression, for_expr: ast.ForExpression, subject_type: *const Type) Error!Operand {
        const span = self.spanOf(expression);
        const slot = try self.resultSlot(expression);
        const protocol = (try self.checker.cursorProtocolOf(subject_type)) orelse
            return self.report(span, "this subject is not iterable: provide 'iterator()' and 'next()' extension functions (section 4.3)", .{});
        // the loop frame owns the cursor state and any materialized
        // subject temporary: 'break' leaves it intact and it drops once
        // at the shared exit block; 'return' unwinds it like any frame
        try self.pushFrame();
        const subject_expression = for_expr.subjects[0];
        const place = (try self.evalPlace(subject_expression)) orelse place: {
            const value = try self.evalExpression(subject_expression);
            const memory = try self.ensureMemory(value, subject_type, span);
            if (memory.fresh and try self.ownsHeap(subject_type, 0)) {
                try self.bindLocal("", memory.pointer, subject_type);
            }
            break :place Place{ .pointer = memory.pointer, .value_type = subject_type };
        };

        // the subject type is concrete here, so the protocol's inferred
        // bindings are already concrete
        const iterator_info = try self.functionInfo(protocol.iterator, span, protocol.iterator_bindings);
        const cursor_type = iterator_info.return_type;
        const cursor_slot = switch (try self.classify(cursor_type, span)) {
            .void_class => return self.report(span, "'iterator()' must yield a cursor value (section 4.3)", .{}),
            .scalar => |llvm| try self.scalarSlot(llvm),
            .aggregate => |layout| try self.aggregateSlot(layout),
        };
        {
            const receiver_resolved = try self.resolvedOf(iterator_info.parameter_types[0]);
            var lowered: std.ArrayList([]const u8) = .empty;
            if (iterator_info.aggregate_return) {
                try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{cursor_slot}));
            }
            if (receiver_resolved.* == .reference) {
                try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{place.pointer}));
            } else {
                const receiver = try self.loadPlace(place, span);
                const coerced = try self.coerceOperand(receiver, place.value_type, iterator_info.parameter_types[0], span);
                try lowered.append(self.arena, try self.lowerArgument(coerced, iterator_info.parameter_types[0], span));
            }
            const joined = try self.joinArguments(lowered.items);
            if (iterator_info.aggregate_return) {
                try self.instruction("call void @\"{s}\"({s})", .{ iterator_info.name, joined });
            } else switch (try self.classify(cursor_type, span)) {
                .scalar => |llvm| {
                    const value = try self.freshTemp();
                    try self.instruction("{s} = call {s} @\"{s}\"({s})", .{ value, llvm, iterator_info.name, joined });
                    try self.storeScalar(cursor_slot, .{ .text = value, .llvm = llvm });
                },
                else => try self.instruction("call void @\"{s}\"({s})", .{ iterator_info.name, joined }),
            }
        }

        const next_info = try self.functionInfo(protocol.next, span, protocol.next_bindings);
        if ((try self.resolvedOf(next_info.parameter_types[0])).* != .reference) {
            return self.report(span, "'next()' must take its cursor by reference (section 4.3)", .{});
        }
        const option_type = next_info.return_type;
        const frame = (try self.enumFrameQuery(option_type)) orelse
            return self.report(span, "'next()' must yield an 'Option' value (section 4.3)", .{});
        const some_index = try self.variantIndex(frame, "Some", span);
        const option_slot = try self.aggregateSlot(frame.layout);
        const option_owns = try self.ownsHeap(option_type, 0);
        // a break can leave a live 'Some' payload in the option slot, and
        // the cursor itself may own heap: the loop frame releases both
        try self.zeroFill(option_slot, frame.layout.size);
        if (option_owns) try self.bindLocal("", option_slot, option_type);
        if (try self.ownsHeap(cursor_type, 0)) try self.bindLocal("", cursor_slot, cursor_type);

        const header = try self.freshLabel("cursor.header");
        const body = try self.freshLabel("cursor.body");
        const after = try self.freshLabel("cursor.after");
        const exit = try self.freshLabel("cursor.exit");
        try self.break_targets.append(self.arena, .{ .exit_label = exit, .slot = slot, .frame_depth = self.scopes.items.len });
        try self.startBlock(header);
        try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ next_info.name, option_slot, cursor_slot });
        const tag = try self.loadTag(option_slot, frame);
        const produced = try self.freshTemp();
        try self.instruction("{s} = icmp eq {s} {s}, {d}", .{ produced, tagTypeText(frame.tag_size), tag, some_index });
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ produced, body, after });
        self.terminated = true;
        try self.startBlock(body);
        try self.pushFrame();
        if (for_expr.captures.len != 0) {
            if (frame.variants[some_index].payload) |payload_type| {
                const payload_pointer = try self.byteOffset(option_slot, frame.payload_offset);
                try self.bindCapture(for_expr.captures[0], payload_pointer, payload_type, span);
            }
        }
        try self.execStatement(for_expr.body);
        try self.closeFrame();
        if (!self.terminated) {
            if (option_owns) {
                // the capture copied the payload out; the option still owns
                // its original, which drops before the next advance
                const helper = try self.dropHelper(option_type, span);
                try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, option_slot });
                try self.zeroFill(option_slot, frame.layout.size);
            }
            try self.instruction("br label %{s}", .{header});
            self.terminated = true;
        }
        try self.startBlock(after);
        if (for_expr.else_branch) |else_branch| {
            try self.pushFrame();
            try self.execStatement(else_branch);
            try self.closeFrame();
        }
        _ = self.break_targets.pop();
        try self.startBlock(exit);
        // both the break path and the normal path pass through the exit
        // block, so the loop frame drops exactly once here
        try self.closeFrame();
        return self.slotOperand(slot, span);
    }

    fn forSubject(self: *Codegen, subject_expression: *const ast.Expression) Error!ForSubject {
        const span = self.spanOf(subject_expression);
        if (subject_expression.* == .array_range) {
            const array_range = subject_expression.array_range;
            const start: Scalar = if (array_range.start) |start_expression|
                (try self.evalExpression(start_expression)).scalar
            else
                .{ .text = "0", .llvm = "i64" };
            const start_wide = if (array_range.start) |start_expression|
                try self.widenToIndex(start, try self.resolvedOf(try self.typeOf(start_expression)))
            else
                start.text;
            const end = (try self.evalExpression(array_range.end)).scalar;
            const end_wide = try self.widenToIndex(end, try self.resolvedOf(try self.typeOf(array_range.end)));
            if (!self.release_mode) {
                const ordered = try self.freshTemp();
                try self.instruction("{s} = icmp sle i64 {s}, {s}", .{ ordered, start_wide, end_wide });
                try self.faultUnless(ordered, "runtime fault: the range end is below its start (section 4.3)");
            }
            const length = try self.freshTemp();
            try self.instruction("{s} = sub i64 {s}, {s}", .{ length, end_wide, start_wide });
            // a range subject never materializes (section 4.3), so the
            // checker records no type for it; the counter is an i32
            const capture_type: *const Type = capture: {
                const recorded = self.expression_types.get(subject_expression) orelse break :capture &integer_type;
                const resolved = try self.resolvedOf(recorded);
                if (resolved.* == .fixed_array) break :capture try self.checker.defaulted(resolved.fixed_array.element);
                break :capture &integer_type;
            };
            return .{ .kind = .{ .counter = start_wide }, .length = length, .capture_type = capture_type };
        }
        const subject_type = try self.resolvedOf(try self.typeOf(subject_expression));
        const place = (try self.evalPlace(subject_expression)) orelse place: {
            const value = try self.evalExpression(subject_expression);
            const memory = try self.ensureMemory(value, subject_type, span);
            // a fresh owning temporary joins the loop frame, which drops
            // it on every exit path (section 4.2)
            if (memory.fresh and try self.ownsHeap(subject_type, 0)) {
                try self.bindLocal("", memory.pointer, subject_type);
            }
            break :place Place{ .pointer = memory.pointer, .value_type = subject_type };
        };
        const resolved = try self.resolvedOf(place.value_type);
        switch (resolved.*) {
            .fixed_array => |array| {
                const element_layout = (try self.layoutQuery(array.element, 0)) orelse
                    return self.report(span, "this element type has no defined layout", .{});
                return .{
                    .kind = .{ .elements = .{
                        .data_pointer = place.pointer,
                        .element_type = array.element,
                        .stride = element_layout.size,
                    } },
                    .length = try std.fmt.allocPrint(self.arena, "{d}", .{array.length}),
                    .capture_type = array.element,
                };
            },
            .slice => |slice| {
                const element_layout = (try self.layoutQuery(slice.child, 0)) orelse
                    return self.report(span, "this element type has no defined layout", .{});
                return .{
                    .kind = .{ .elements = .{
                        .data_pointer = try self.loadPointerField(place.pointer, 0),
                        .element_type = slice.child,
                        .stride = element_layout.size,
                    } },
                    .length = try self.loadIntegerField(place.pointer, 8),
                    .capture_type = slice.child,
                };
            },
            .heap_array => |heap| {
                const element_layout = (try self.layoutQuery(heap.child, 0)) orelse
                    return self.report(span, "this element type has no defined layout", .{});
                const heap_view = try self.heapArrayView(place.pointer);
                return .{
                    .kind = .{ .elements = .{
                        .data_pointer = heap_view.data,
                        .element_type = heap.child,
                        .stride = element_layout.size,
                    } },
                    .length = heap_view.length,
                    .capture_type = heap.child,
                };
            },
            else => return self.report(span, "the cursor protocol is not yet supported by native code generation", .{}),
        }
    }

    const MatchSubject = union(enum) {
        tagged: struct { place: Place, frame: Checker.EnumFrame, tag: []const u8 },
        integer: Scalar,
        bytes: ByteView,
        // an interface object matches arms naming its concrete type
        interface_object: struct { data: []const u8, type_id: []const u8 },
    };

    fn evalMatch(self: *Codegen, expression: *const ast.Expression, match_expr: ast.MatchExpression) Error!Operand {
        const span = self.spanOf(expression);
        const slot = try self.resultSlot(expression);
        const subject_type = try self.typeOf(match_expr.subject);
        const subject_resolved = try self.resolvedOf(subject_type);
        const exit = try self.freshLabel("match.exit");
        const external_else: ?[]const u8 = if (match_expr.else_branch != null) try self.freshLabel("match.else") else null;
        const arm_complete = external_else orelse exit;

        var subject_cleanup: ?Place = null;
        const subject: MatchSubject = subject: {
            if (try self.enumFrameQuery(subject_resolved)) |frame| {
                const place = (try self.evalPlace(match_expr.subject)) orelse place: {
                    const value = try self.evalExpression(match_expr.subject);
                    const memory = try self.ensureMemory(value, subject_type, span);
                    if (memory.fresh and try self.ownsHeap(subject_type, 0)) {
                        subject_cleanup = .{ .pointer = memory.pointer, .value_type = subject_type };
                    }
                    break :place Place{ .pointer = memory.pointer, .value_type = subject_type };
                };
                break :subject .{ .tagged = .{ .place = place, .frame = frame, .tag = try self.loadTag(place.pointer, frame) } };
            }
            if (subject_resolved.* == .primitive and !subject_resolved.primitive.isFloat()) {
                break :subject .{ .integer = (try self.evalExpression(match_expr.subject)).scalar };
            }
            switch (subject_resolved.*) {
                .slice, .fixed_array, .heap_array => {
                    break :subject .{ .bytes = try self.byteViewOf(match_expr.subject) };
                },
                // a read of an interface object records as the pierced bare
                // interface; the place still holds the fat pair
                .reference, .pointer, .interface => {
                    const place = (try self.evalPlace(match_expr.subject)) orelse
                        return self.report(span, "matching on this subject is not yet supported by native code generation", .{});
                    if (try self.interfaceOfPlace(place)) |_| {
                        break :subject .{ .interface_object = .{
                            .data = try self.loadPointerField(place.pointer, 0),
                            .type_id = try self.loadPointerField(place.pointer, 8),
                        } };
                    }
                    return self.report(span, "matching on this subject is not yet supported by native code generation", .{});
                },
                else => return self.report(span, "matching on this subject is not yet supported by native code generation", .{}),
            }
        };

        // only a value-position match receives 'yield' (section 4.3)
        if (slot != null) {
            try self.yield_targets.append(self.arena, .{ .exit_label = exit, .slot = slot, .frame_depth = self.scopes.items.len });
        }
        var arm_labels: std.ArrayList([]const u8) = .empty;
        for (match_expr.arms) |_| {
            try arm_labels.append(self.arena, try self.freshLabel("match.arm"));
        }
        for (match_expr.arms, arm_labels.items, 0..) |arm, arm_label, index| {
            const miss_label: []const u8 = if (index + 1 < match_expr.arms.len)
                try self.freshLabel("match.test")
            else
                exit;
            if (arm.pattern) |pattern| {
                const matches: []const u8 = switch (subject) {
                    .tagged => |tagged| matches: {
                        const variant_name = try self.patternVariantName(pattern);
                        const variant_index = try self.variantIndex(tagged.frame, variant_name, self.spanOf(pattern));
                        const matches = try self.freshTemp();
                        try self.instruction("{s} = icmp eq {s} {s}, {d}", .{ matches, tagTypeText(tagged.frame.tag_size), tagged.tag, variant_index });
                        break :matches matches;
                    },
                    .integer => |scalar| matches: {
                        const pattern_value = try self.evalExpression(pattern);
                        const coerced = try self.coerceOperand(pattern_value, try self.typeOf(pattern), subject_type, self.spanOf(pattern));
                        const matches = try self.freshTemp();
                        try self.instruction("{s} = icmp eq {s} {s}, {s}", .{ matches, scalar.llvm, scalar.text, coerced.scalar.text });
                        break :matches matches;
                    },
                    .bytes => |view| try self.bytesMatch(view, pattern),
                    .interface_object => |object| matches: {
                        const descriptor = try self.patternDescriptor(pattern);
                        const matches = try self.freshTemp();
                        try self.instruction("{s} = icmp eq ptr {s}, @\"{s}\"", .{ matches, object.type_id, descriptor.name });
                        break :matches matches;
                    },
                };
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, arm_label, miss_label });
                self.terminated = true;
            } else {
                try self.instruction("br label %{s}", .{arm_label});
                self.terminated = true;
            }
            if (index + 1 < match_expr.arms.len) {
                try self.startBlock(miss_label);
            }
        }
        if (match_expr.arms.len == 0) {
            try self.instruction("br label %{s}", .{exit});
            self.terminated = true;
        }

        for (match_expr.arms, arm_labels.items) |arm, arm_label| {
            try self.startBlock(arm_label);
            try self.pushFrame();
            if (arm.capture) |capture| {
                switch (subject) {
                    .tagged => |tagged| {
                        if (arm.pattern) |pattern| {
                            const variant_name = try self.patternVariantName(pattern);
                            const variant_index = try self.variantIndex(tagged.frame, variant_name, self.spanOf(pattern));
                            if (tagged.frame.variants[variant_index].payload) |payload_type| {
                                const payload_pointer = try self.byteOffset(tagged.place.pointer, tagged.frame.payload_offset);
                                try self.bindCapture(capture, payload_pointer, payload_type, span);
                            }
                        }
                    },
                    .interface_object => |object| {
                        // the arm borrows the concrete value (section 4.3)
                        if (arm.pattern) |pattern| {
                            const descriptor = try self.patternDescriptor(pattern);
                            try self.bindBorrowed(capture.name.slice(self.source()), object.data, descriptor.concrete);
                        }
                    },
                    else => return self.report(span, "a capture needs an enum or interface-object subject", .{}),
                }
            }
            try self.execStatement(arm.body);
            try self.closeFrame();
            if (!self.terminated) {
                // an arm that completes without 'break' runs the external
                // else (section 4.3)
                try self.instruction("br label %{s}", .{arm_complete});
                self.terminated = true;
            }
        }
        if (match_expr.else_branch) |else_branch| {
            try self.startBlock(external_else.?);
            try self.pushFrame();
            try self.execStatement(else_branch);
            try self.closeFrame();
            if (!self.terminated) {
                try self.instruction("br label %{s}", .{exit});
                self.terminated = true;
            }
        }
        if (slot != null) _ = self.yield_targets.pop();
        try self.startBlock(exit);
        if (subject_cleanup) |cleanup| {
            // a materialized owning subject drops once the match is done;
            // copy captures cloned out of it, so it still owns its payload
            const helper = try self.dropHelper(cleanup.value_type, span);
            try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, cleanup.pointer });
        }
        return self.slotOperand(slot, span);
    }

    fn patternVariantName(self: *Codegen, pattern: *const ast.Expression) Error![]const u8 {
        const unwrapped = unwrapGrouped(pattern);
        return switch (unwrapped.*) {
            .path => |path| path[path.len - 1].slice(self.source()),
            .implied_variant => |token| token.slice(self.source()),
            else => self.report(self.spanOf(pattern), "this pattern does not name an enum variant", .{}),
        };
    }

    // unqualified visibility from one view (section 5.4): own library plus
    // the 'exp' symbols of libraries the view imported without an alias,
    // mirroring the checker's rule
    fn firstVisible(self: *const Codegen, name: []const u8, view_index: usize) ?resolution.Symbol {
        const symbols = self.globals.get(name) orelse return null;
        for (symbols.items) |symbol| {
            if (resolution.sameLibrary(self.views[view_index].library, self.views[symbol.view_index].library)) return symbol;
            if (symbol.visibility != .exported) continue;
            const library = self.views[symbol.view_index].library orelse continue;
            if (self.injected[view_index].contains(library)) return symbol;
        }
        return null;
    }

    // an interface-object arm names a concrete type (section 4.3); the
    // checker recorded which definition the pattern resolved to
    fn patternDescriptor(self: *Codegen, pattern: *const ast.Expression) Error!Descriptor {
        const span = self.spanOf(pattern);
        const unwrapped = unwrapGrouped(pattern);
        if (unwrapped.* != .path) {
            return self.report(span, "this arm must name a concrete type (section 4.3)", .{});
        }
        const name = unwrapped.path[unwrapped.path.len - 1].slice(self.source());
        if (self.checker.type_targets.get(pattern)) |target| {
            const concrete = try self.implementerType(.{
                .definition = target.definition,
                .view_index = target.view_index,
                .name = name,
            });
            return .{ .name = try self.typeDescriptor(target.definition), .concrete = concrete };
        }
        const symbol = self.firstVisible(name, self.current_view) orelse
            return self.report(span, "'{s}' names no type here", .{name});
        if (symbol.definition.kind != .type_def) {
            return self.report(span, "'{s}' is not a type", .{name});
        }
        const concrete = try self.implementerType(.{
            .definition = symbol.definition,
            .view_index = symbol.view_index,
            .name = name,
        });
        return .{ .name = try self.typeDescriptor(symbol.definition), .concrete = concrete };
    }

    const ByteView = HeapArrayView;

    // the byte sequence behind any string-like subject (section 4.3)
    fn byteViewOf(self: *Codegen, expression: *const ast.Expression) Error!ByteView {
        const span = self.spanOf(expression);
        const value_type = try self.typeOf(expression);
        const place = (try self.evalPlace(expression)) orelse place: {
            const value = try self.evalExpression(expression);
            const memory = try self.ensureMemory(value, value_type, span);
            break :place Place{ .pointer = memory.pointer, .value_type = value_type };
        };
        const resolved = try self.resolvedOf(place.value_type);
        switch (resolved.*) {
            .slice => return .{
                .data = try self.loadPointerField(place.pointer, 0),
                .length = try self.loadIntegerField(place.pointer, 8),
            },
            .fixed_array => |array| return .{
                .data = place.pointer,
                .length = try std.fmt.allocPrint(self.arena, "{d}", .{array.length}),
            },
            .heap_array => return self.heapArrayView(place.pointer),
            else => return self.report(span, "this value has no byte sequence to match", .{}),
        }
    }

    // strings compare by length, then content through memcmp (section 4.3)
    fn bytesMatch(self: *Codegen, subject: ByteView, pattern: *const ast.Expression) Error![]const u8 {
        if (!self.extern_declarations.contains("memcmp")) {
            try self.extern_declarations.put(self.arena, "memcmp", "declare i32 @\"memcmp\"(ptr, ptr, i64)");
        }
        const pattern_view = try self.byteViewOf(pattern);
        const slot = try self.scalarSlot("i1");
        try self.storeScalar(slot, .{ .text = "false", .llvm = "i1" });
        const lengths_equal = try self.freshTemp();
        try self.instruction("{s} = icmp eq i64 {s}, {s}", .{ lengths_equal, subject.length, pattern_view.length });
        const compare_label = try self.freshLabel("bytes.compare");
        const done_label = try self.freshLabel("bytes.done");
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ lengths_equal, compare_label, done_label });
        self.terminated = true;
        try self.startBlock(compare_label);
        const difference = try self.freshTemp();
        try self.instruction("{s} = call i32 @\"memcmp\"(ptr {s}, ptr {s}, i64 {s})", .{ difference, subject.data, pattern_view.data, subject.length });
        const equal = try self.freshTemp();
        try self.instruction("{s} = icmp eq i32 {s}, 0", .{ equal, difference });
        try self.storeScalar(slot, .{ .text = equal, .llvm = "i1" });
        try self.startBlock(done_label);
        return self.loadScalar(slot, "i1");
    }

    // capture typing (section 2.1): deep copy by default, '&' borrows the
    // payload in place; owning captures need heap support
    fn bindCapture(self: *Codegen, capture: ast.Capture, payload_pointer: []const u8, payload_type: *const Type, span: Token.Location) Error!void {
        const name = capture.name.slice(self.source());
        switch (captureMode(capture)) {
            .copy => {
                // a copy capture deep-copies an owning payload, so the
                // binding and the subject own separate heap (section 2.1)
                const owns = try self.ownsHeap(payload_type, 0);
                const slot = switch (try self.classify(payload_type, span)) {
                    .void_class => return,
                    .scalar => |llvm| slot: {
                        const slot = try self.scalarSlot(llvm);
                        if (owns) break :slot slot;
                        const loaded = try self.loadScalar(payload_pointer, llvm);
                        try self.storeScalar(slot, .{ .text = loaded, .llvm = llvm });
                        break :slot slot;
                    },
                    .aggregate => |layout| slot: {
                        const slot = try self.aggregateSlot(layout);
                        if (owns) break :slot slot;
                        try self.copyBytes(slot, payload_pointer, layout.size);
                        break :slot slot;
                    },
                };
                if (owns) {
                    const helper = try self.copyHelper(payload_type, span);
                    try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, slot, payload_pointer });
                }
                try self.bindLocal(name, slot, payload_type);
            },
            .reference => try self.bindBorrowed(name, payload_pointer, payload_type),
            .owning => {
                // an owning capture takes the owning payload out, leaving
                // the source moved-from (section 2.1); an interface fat
                // pair moves whole, its data half nulled (section 5.2)
                const resolved = try self.resolvedOf(payload_type);
                if (resolved.* != .pointer and resolved.* != .heap_array) {
                    return self.report(span, "an owning capture needs a pointer payload (section 2.1)", .{});
                }
                const layout = (try self.layoutQuery(payload_type, 0)) orelse
                    return self.report(span, "this payload type has no defined layout", .{});
                const slot = try self.aggregateSlot(layout);
                try self.copyBytes(slot, payload_pointer, layout.size);
                try self.instruction("store ptr null, ptr {s}", .{payload_pointer});
                try self.bindLocal(name, slot, payload_type);
            },
        }
    }

    fn loadTag(self: *Codegen, pointer: []const u8, frame: Checker.EnumFrame) Error![]const u8 {
        const result = try self.freshTemp();
        try self.instruction("{s} = load {s}, ptr {s}", .{ result, tagTypeText(frame.tag_size), pointer });
        return result;
    }

    fn typeOf(self: *Codegen, expression: *const ast.Expression) Error!*const Type {
        return self.expression_types.get(expression) orelse
            self.report(self.spanOf(expression), "internal: no recorded type for this expression", .{});
    }

    fn primitiveOf(self: *Codegen, expression: *const ast.Expression) ?types.Primitive {
        const recorded = self.expression_types.get(expression) orelse return null;
        const resolved = self.resolvedOf(recorded) catch return null;
        return if (resolved.* == .primitive) resolved.primitive else null;
    }

    fn resolvedOf(self: *Codegen, candidate: *const Type) Error!*const Type {
        return self.checker.resolveAlias(try self.checker.defaulted(try self.substituted(candidate)));
    }

    fn classify(self: *Codegen, candidate: *const Type, span: Token.Location) Error!Class {
        const resolved = try self.resolvedOf(candidate);
        return switch (resolved.*) {
            .void_type, .unknown => .void_class,
            .primitive => |primitive| .{ .scalar = scalarTypeText(primitive) },
            .reference, .pointer => |indirection| {
                // an interface object is the fat pair itself (section 3.2)
                if ((try self.resolvedOf(indirection.child)).* == .interface) {
                    return .{ .aggregate = .{ .size = 16, .alignment = 8 } };
                }
                return .{ .scalar = "ptr" };
            },
            .heap_array => .{ .scalar = "ptr" },
            // a function value is one block pointer, but it OWNS the block:
            // classifying it as an 8-byte aggregate routes every consumer
            // through the memory machinery, whose clone-versus-transfer
            // rules already handle ownership (section 4.2)
            .function => .{ .aggregate = .{ .size = 8, .alignment = 8 } },
            else => {
                const layout = (try self.layoutQuery(resolved, 0)) orelse
                    return self.report(span, "this type has no defined runtime layout", .{});
                return .{ .aggregate = layout };
            },
        };
    }

    fn unsupportedReason(self: *Codegen, candidate: *const Type, depth: usize) Error!?[]const u8 {
        if (depth > 8) return null;
        const resolved = try self.resolvedOf(candidate);
        switch (resolved.*) {
            .pointer => |indirection| return self.unsupportedReason(indirection.child, depth + 1),
            .heap_array => |indirection| return self.unsupportedReason(indirection.child, depth + 1),
            .type_parameter => return "generics",
            .reference => return null,
            .slice => |indirection| return self.unsupportedReason(indirection.child, depth + 1),
            .fixed_array => |array| return self.unsupportedReason(array.element, depth + 1),
            .structural, .declared, .inline_enum, .structural_enum => {
                if (try self.enumFrameQuery(resolved)) |frame| {
                    for (frame.variants) |variant| {
                        const payload = variant.payload orelse continue;
                        if (try self.unsupportedReason(payload, depth + 1)) |reason| return reason;
                    }
                    return null;
                }
                const fields = (try self.checker.structuralFieldsOf(resolved)) orelse return null;
                for (fields) |field| {
                    if (try self.unsupportedReason(field.field_type, depth + 1)) |reason| return reason;
                }
                return null;
            },
            else => return null,
        }
    }

    // a value owns heap when it is a pointer or heap array, or transitively
    // contains an owning member, element, or variant payload (section 4.2)
    fn ownsHeap(self: *Codegen, candidate: *const Type, depth: usize) Error!bool {
        if (depth > 16) return false;
        const resolved = try self.resolvedOf(candidate);
        switch (resolved.*) {
            .pointer, .heap_array => return true,
            // a closure owns its captured environment (section 4.2)
            .function => return true,
            .fixed_array => |array| return self.ownsHeap(array.element, depth + 1),
            .structural, .declared, .inline_enum, .structural_enum => {
                if (try self.enumFrameQuery(resolved)) |frame| {
                    for (frame.variants) |variant| {
                        const payload = variant.payload orelse continue;
                        if (try self.ownsHeap(payload, depth + 1)) return true;
                    }
                    return false;
                }
                const fields = (try self.checker.structuralFieldsOf(resolved)) orelse return false;
                for (fields) |field| {
                    if (try self.ownsHeap(field.field_type, depth + 1)) return true;
                }
                return false;
            },
            else => return false,
        }
    }

    // a canonical structural key for helper memoization; Type.render is not
    // enough because every inline enum renders identically
    fn typeKey(self: *Codegen, candidate: *const Type, depth: usize) Error![]const u8 {
        if (depth > 16) return "...";
        const resolved = try self.resolvedOf(candidate);
        switch (resolved.*) {
            .primitive => |primitive| return @tagName(primitive),
            .reference => return "&",
            .slice => return "&[]",
            // every closure shares one runtime shape: a block pointer whose
            // header carries its own drop and copy functions
            .function => return "fn",
            .interface => |interface| return std.fmt.allocPrint(self.arena, "interface.{d}", .{@intFromPtr(interface.definition)}),
            .pointer => |indirection| return std.fmt.allocPrint(self.arena, "*({s})", .{try self.typeKey(indirection.child, depth + 1)}),
            .heap_array => |indirection| return std.fmt.allocPrint(self.arena, "*[{s}]", .{try self.typeKey(indirection.child, depth + 1)}),
            .fixed_array => |array| return std.fmt.allocPrint(self.arena, "[{s}:{d}]", .{ try self.typeKey(array.element, depth + 1), array.length }),
            .structural, .declared, .inline_enum, .structural_enum => {
                var key: std.Io.Writer.Allocating = .init(self.arena);
                if (try self.enumFrameQuery(resolved)) |frame| {
                    key.writer.print("enum(", .{}) catch return error.OutOfMemory;
                    for (frame.variants, 0..) |variant, index| {
                        if (index != 0) key.writer.print(",", .{}) catch return error.OutOfMemory;
                        key.writer.print("{s}", .{variant.name}) catch return error.OutOfMemory;
                        if (variant.payload) |payload| {
                            key.writer.print(":{s}", .{try self.typeKey(payload, depth + 1)}) catch return error.OutOfMemory;
                        }
                    }
                    key.writer.print(")", .{}) catch return error.OutOfMemory;
                    return key.writer.buffered();
                }
                const fields = (try self.checker.structuralFieldsOf(resolved)) orelse return "?";
                key.writer.print("struct(", .{}) catch return error.OutOfMemory;
                for (fields, 0..) |field, index| {
                    if (index != 0) key.writer.print(",", .{}) catch return error.OutOfMemory;
                    key.writer.print("{s}:{s}", .{ field.name, try self.typeKey(field.field_type, depth + 1) }) catch return error.OutOfMemory;
                }
                key.writer.print(")", .{}) catch return error.OutOfMemory;
                return key.writer.buffered();
            },
            else => return "?",
        }
    }

    const SavedFunction = struct {
        allocas: std.Io.Writer.Allocating,
        body: std.Io.Writer.Allocating,
        temp_counter: usize,
        terminated: bool,
        // helpers emit without a debug scope: a location pointing into the
        // enclosing user function would fail the verifier
        debug_subprogram: ?usize,
        debug_location: ?usize,
        debug_scopes: std.ArrayList(usize),
    };

    // helper functions generate while another function is mid-emission, so
    // the per-function buffers swap out and back
    fn beginHelperFunction(self: *Codegen) SavedFunction {
        const saved: SavedFunction = .{
            .allocas = self.allocas,
            .body = self.body,
            .temp_counter = self.temp_counter,
            .terminated = self.terminated,
            .debug_subprogram = self.debug_subprogram,
            .debug_location = self.debug_location,
            .debug_scopes = self.debug_scopes,
        };
        self.allocas = .init(self.arena);
        self.body = .init(self.arena);
        self.temp_counter = 0;
        self.terminated = false;
        self.debug_subprogram = null;
        self.debug_location = null;
        self.debug_scopes = .empty;
        return saved;
    }

    fn finishHelperFunction(self: *Codegen, saved: SavedFunction, header: []const u8) Error!void {
        const out = &self.functions.writer;
        out.print("{s}", .{header}) catch return error.OutOfMemory;
        out.print("{s}", .{self.allocas.writer.buffered()}) catch return error.OutOfMemory;
        out.print("{s}", .{self.body.writer.buffered()}) catch return error.OutOfMemory;
        out.print("}}\n", .{}) catch return error.OutOfMemory;
        self.allocas = saved.allocas;
        self.body = saved.body;
        self.temp_counter = saved.temp_counter;
        self.terminated = saved.terminated;
        self.debug_subprogram = saved.debug_subprogram;
        self.debug_location = saved.debug_location;
        self.debug_scopes = saved.debug_scopes;
    }

    fn declareMalloc(self: *Codegen) Error!void {
        if (!self.extern_declarations.contains("malloc")) {
            try self.extern_declarations.put(self.arena, "malloc", "declare ptr @\"malloc\"(i64)");
        }
    }

    // every allocation goes through here: checked builds fault when the
    // allocator is exhausted instead of piercing a null block
    fn allocateDynamic(self: *Codegen, size: []const u8) Error![]const u8 {
        try self.declareMalloc();
        const allocation = try self.freshTemp();
        try self.instruction("{s} = call ptr @\"malloc\"(i64 {s})", .{ allocation, size });
        if (!self.release_mode) {
            const live = try self.freshTemp();
            try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, allocation });
            try self.faultUnless(live, "runtime fault: out of memory");
        }
        return allocation;
    }

    fn allocateFixed(self: *Codegen, size: u64) Error![]const u8 {
        return self.allocateDynamic(try std.fmt.allocPrint(self.arena, "{d}", .{size}));
    }

    fn declareFree(self: *Codegen) Error!void {
        if (!self.extern_declarations.contains("free")) {
            try self.extern_declarations.put(self.arena, "free", "declare void @\"free\"(ptr)");
        }
    }

    /// The drop helper for a type: recursively frees the heap owned by the
    /// value at the given place (section 4.2). Memoized; self-referential
    /// types terminate through the memo entry and runtime null checks.
    fn dropHelper(self: *Codegen, candidate: *const Type, span: Token.Location) Error![]const u8 {
        const resolved = try self.resolvedOf(candidate);
        const key = try std.fmt.allocPrint(self.arena, "drop:{s}", .{try self.typeKey(resolved, 0)});
        if (self.helper_names.get(key)) |existing| return existing;
        const name = try std.fmt.allocPrint(self.arena, "alloy.drop.{d}", .{self.global_counter});
        self.global_counter += 1;
        try self.helper_names.put(self.arena, key, name);
        try self.declareFree();

        const saved = self.beginHelperFunction();
        switch (resolved.*) {
            .pointer => |indirection| {
                const child_resolved = try self.resolvedOf(indirection.child);
                if (child_resolved.* == .interface) {
                    try self.emitInterfaceDrop(child_resolved.interface, span);
                } else {
                    const loaded = try self.freshTemp();
                    try self.instruction("{s} = load ptr, ptr %value", .{loaded});
                    const live = try self.freshTemp();
                    try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
                    const live_label = try self.freshLabel("live");
                    const done_label = try self.freshLabel("done");
                    try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                    self.terminated = true;
                    try self.startBlock(live_label);
                    if (try self.ownsHeap(indirection.child, 0)) {
                        const child_drop = try self.dropHelper(indirection.child, span);
                        try self.instruction("call void @\"{s}\"(ptr {s})", .{ child_drop, loaded });
                    }
                    try self.instruction("call void @\"free\"(ptr {s})", .{loaded});
                    try self.startBlock(done_label);
                }
            },
            .function => {
                // the block's own drop function releases captures and the
                // block; a null drop pointer marks a static block (a named
                // function value), a null block a transferred-away value
                const loaded = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr %value", .{loaded});
                const live = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
                const live_label = try self.freshLabel("live");
                const done_label = try self.freshLabel("done");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                self.terminated = true;
                try self.startBlock(live_label);
                const drop_slot = try self.byteOffset(loaded, 8);
                const drop_pointer = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr {s}", .{ drop_pointer, drop_slot });
                const owned = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ owned, drop_pointer });
                const owned_label = try self.freshLabel("owned");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ owned, owned_label, done_label });
                self.terminated = true;
                try self.startBlock(owned_label);
                try self.instruction("call void {s}(ptr {s})", .{ drop_pointer, loaded });
                try self.startBlock(done_label);
            },
            .heap_array => |indirection| {
                const loaded = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr %value", .{loaded});
                const live = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
                const live_label = try self.freshLabel("live");
                const done_label = try self.freshLabel("done");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                self.terminated = true;
                try self.startBlock(live_label);
                const base = try self.freshTemp();
                try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 -8", .{ base, loaded });
                if (try self.ownsHeap(indirection.child, 0)) {
                    const element_layout = (try self.layoutQuery(indirection.child, 0)) orelse
                        return self.report(span, "this element type has no defined layout", .{});
                    const child_drop = try self.dropHelper(indirection.child, span);
                    const length = try self.freshTemp();
                    try self.instruction("{s} = load i64, ptr {s}", .{ length, base });
                    try self.emitHelperElementLoop(loaded, length, element_layout.size, child_drop, null);
                }
                try self.instruction("call void @\"free\"(ptr {s})", .{base});
                try self.startBlock(done_label);
            },
            .fixed_array => |array| {
                const element_layout = (try self.layoutQuery(array.element, 0)) orelse
                    return self.report(span, "this element type has no defined layout", .{});
                const child_drop = try self.dropHelper(array.element, span);
                const length = try std.fmt.allocPrint(self.arena, "{d}", .{array.length});
                try self.emitHelperElementLoop("%value", length, element_layout.size, child_drop, null);
            },
            else => {
                if (try self.enumFrameQuery(resolved)) |frame| {
                    try self.emitEnumPayloadDispatch(frame, "drop", span);
                } else if (try self.fieldSlotsQuery(resolved)) |slots| {
                    for (slots) |slot| {
                        if (!try self.ownsHeap(slot.field_type, 0)) continue;
                        const field_drop = try self.dropHelper(slot.field_type, span);
                        const pointer = try self.byteOffset("%value", slot.offset);
                        try self.instruction("call void @\"{s}\"(ptr {s})", .{ field_drop, pointer });
                    }
                }
            },
        }
        try self.instruction("ret void", .{});
        self.terminated = true;
        const header = try std.fmt.allocPrint(self.arena, "define internal void @\"{s}\"(ptr %value) {{\nentry:\n", .{name});
        try self.finishHelperFunction(saved, header);
        return name;
    }

    /// The deep-copy helper for a type: copies the bytes, then clones every
    /// owned allocation so the copy owns fresh heap (section 4.2).
    fn copyHelper(self: *Codegen, candidate: *const Type, span: Token.Location) Error![]const u8 {
        const resolved = try self.resolvedOf(candidate);
        const key = try std.fmt.allocPrint(self.arena, "copy:{s}", .{try self.typeKey(resolved, 0)});
        if (self.helper_names.get(key)) |existing| return existing;
        const name = try std.fmt.allocPrint(self.arena, "alloy.copy.{d}", .{self.global_counter});
        self.global_counter += 1;
        try self.helper_names.put(self.arena, key, name);
        try self.declareMalloc();

        const layout = (try self.layoutQuery(resolved, 0)) orelse
            return self.report(span, "this type has no defined runtime layout", .{});
        const saved = self.beginHelperFunction();
        switch (resolved.*) {
            .pointer => |indirection| {
                const child_resolved = try self.resolvedOf(indirection.child);
                if (child_resolved.* == .interface) {
                    try self.emitInterfaceCopy(child_resolved.interface, span);
                } else {
                    const loaded = try self.freshTemp();
                    try self.instruction("{s} = load ptr, ptr %origin", .{loaded});
                    try self.instruction("store ptr {s}, ptr %destination", .{loaded});
                    const live = try self.freshTemp();
                    try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
                    const live_label = try self.freshLabel("live");
                    const done_label = try self.freshLabel("done");
                    try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                    self.terminated = true;
                    try self.startBlock(live_label);
                    const pointee_layout = (try self.layoutQuery(indirection.child, 0)) orelse
                        return self.report(span, "this pointee type has no defined layout", .{});
                    const allocation = try self.allocateFixed(pointee_layout.size);
                    if (try self.ownsHeap(indirection.child, 0)) {
                        const child_copy = try self.copyHelper(indirection.child, span);
                        try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ child_copy, allocation, loaded });
                    } else {
                        try self.copyBytes(allocation, loaded, pointee_layout.size);
                    }
                    try self.instruction("store ptr {s}, ptr %destination", .{allocation});
                    try self.startBlock(done_label);
                }
            },
            .function => {
                // the block's own copy function clones captures into a
                // fresh block; a null copy pointer marks a static block,
                // shared rather than cloned
                const loaded = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr %origin", .{loaded});
                try self.instruction("store ptr {s}, ptr %destination", .{loaded});
                const live = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
                const live_label = try self.freshLabel("live");
                const done_label = try self.freshLabel("done");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                self.terminated = true;
                try self.startBlock(live_label);
                const copy_slot = try self.byteOffset(loaded, 16);
                const copy_pointer = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr {s}", .{ copy_pointer, copy_slot });
                const owned = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ owned, copy_pointer });
                const owned_label = try self.freshLabel("owned");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ owned, owned_label, done_label });
                self.terminated = true;
                try self.startBlock(owned_label);
                const cloned = try self.freshTemp();
                try self.instruction("{s} = call ptr {s}(ptr {s})", .{ cloned, copy_pointer, loaded });
                try self.instruction("store ptr {s}, ptr %destination", .{cloned});
                try self.startBlock(done_label);
            },
            .heap_array => |indirection| {
                const loaded = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr %origin", .{loaded});
                try self.instruction("store ptr {s}, ptr %destination", .{loaded});
                const live = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
                const live_label = try self.freshLabel("live");
                const done_label = try self.freshLabel("done");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                self.terminated = true;
                try self.startBlock(live_label);
                const element_layout = (try self.layoutQuery(indirection.child, 0)) orelse
                    return self.report(span, "this element type has no defined layout", .{});
                const old_base = try self.freshTemp();
                try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 -8", .{ old_base, loaded });
                const length = try self.freshTemp();
                try self.instruction("{s} = load i64, ptr {s}", .{ length, old_base });
                const data_size = try self.freshTemp();
                try self.instruction("{s} = mul i64 {s}, {d}", .{ data_size, length, element_layout.size });
                const total = try self.freshTemp();
                try self.instruction("{s} = add i64 {s}, 8", .{ total, data_size });
                const new_base = try self.allocateDynamic(total);
                try self.instruction("store i64 {s}, ptr {s}", .{ length, new_base });
                const new_data = try self.freshTemp();
                try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 8", .{ new_data, new_base });
                if (try self.ownsHeap(indirection.child, 0)) {
                    const child_copy = try self.copyHelper(indirection.child, span);
                    try self.emitHelperElementLoop(new_data, length, element_layout.size, child_copy, loaded);
                } else {
                    try self.declareIntrinsic("declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)");
                    try self.instruction("call void @llvm.memcpy.p0.p0.i64(ptr {s}, ptr {s}, i64 {s}, i1 false)", .{ new_data, loaded, data_size });
                }
                try self.instruction("store ptr {s}, ptr %destination", .{new_data});
                try self.startBlock(done_label);
            },
            .fixed_array => |array| {
                try self.copyBytes("%destination", "%origin", layout.size);
                const element_layout = (try self.layoutQuery(array.element, 0)) orelse
                    return self.report(span, "this element type has no defined layout", .{});
                const child_copy = try self.copyHelper(array.element, span);
                const length = try std.fmt.allocPrint(self.arena, "{d}", .{array.length});
                try self.emitHelperElementLoop("%destination", length, element_layout.size, child_copy, "%origin");
            },
            else => {
                try self.copyBytes("%destination", "%origin", layout.size);
                if (try self.enumFrameQuery(resolved)) |frame| {
                    try self.emitEnumPayloadDispatch(frame, "copy", span);
                } else if (try self.fieldSlotsQuery(resolved)) |slots| {
                    for (slots) |slot| {
                        if (!try self.ownsHeap(slot.field_type, 0)) continue;
                        const field_copy = try self.copyHelper(slot.field_type, span);
                        const destination = try self.byteOffset("%destination", slot.offset);
                        const origin = try self.byteOffset("%origin", slot.offset);
                        try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ field_copy, destination, origin });
                    }
                }
            },
        }
        try self.instruction("ret void", .{});
        self.terminated = true;
        const header = try std.fmt.allocPrint(self.arena, "define internal void @\"{s}\"(ptr %destination, ptr %origin) {{\nentry:\n", .{name});
        try self.finishHelperFunction(saved, header);
        return name;
    }

    // a per-element helper invocation loop; with an origin base the helper
    // takes (destination, origin) element pairs, otherwise one element
    fn emitHelperElementLoop(self: *Codegen, base: []const u8, length: []const u8, stride: u64, helper: []const u8, origin_base: ?[]const u8) Error!void {
        const index_slot = try self.scalarSlot("i64");
        try self.storeScalar(index_slot, .{ .text = "0", .llvm = "i64" });
        const header = try self.freshLabel("each.header");
        const body = try self.freshLabel("each.body");
        const exit = try self.freshLabel("each.exit");
        try self.startBlock(header);
        const index = try self.loadScalar(index_slot, "i64");
        const continues = try self.freshTemp();
        try self.instruction("{s} = icmp ult i64 {s}, {s}", .{ continues, index, length });
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ continues, body, exit });
        self.terminated = true;
        try self.startBlock(body);
        const scaled = try self.freshTemp();
        try self.instruction("{s} = mul i64 {s}, {d}", .{ scaled, index, stride });
        const element = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ element, base, scaled });
        if (origin_base) |origin| {
            const origin_element = try self.freshTemp();
            try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {s}", .{ origin_element, origin, scaled });
            try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, element, origin_element });
        } else {
            try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, element });
        }
        const next = try self.freshTemp();
        try self.instruction("{s} = add i64 {s}, 1", .{ next, index });
        try self.storeScalar(index_slot, .{ .text = next, .llvm = "i64" });
        try self.instruction("br label %{s}", .{header});
        self.terminated = true;
        try self.startBlock(exit);
    }

    // tag dispatch over the variants whose payloads own heap; '%value' (or
    // the destination/origin pair for copy) addresses the enum in place
    fn emitEnumPayloadDispatch(self: *Codegen, frame: Checker.EnumFrame, comptime kind: []const u8, span: Token.Location) Error!void {
        const is_copy = comptime std.mem.eql(u8, kind, "copy");
        const value_base: []const u8 = if (is_copy) "%destination" else "%value";
        const tag = try self.freshTemp();
        try self.instruction("{s} = load {s}, ptr {s}", .{ tag, tagTypeText(frame.tag_size), value_base });
        const exit = try self.freshLabel("dispatch.exit");
        for (frame.variants, 0..) |variant, index| {
            const payload = variant.payload orelse continue;
            if (!try self.ownsHeap(payload, 0)) continue;
            const arm = try self.freshLabel("dispatch.arm");
            const miss = try self.freshLabel("dispatch.miss");
            const matches = try self.freshTemp();
            try self.instruction("{s} = icmp eq {s} {s}, {d}", .{ matches, tagTypeText(frame.tag_size), tag, index });
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, arm, miss });
            self.terminated = true;
            try self.startBlock(arm);
            if (is_copy) {
                const helper = try self.copyHelper(payload, span);
                const destination = try self.byteOffset("%destination", frame.payload_offset);
                const origin = try self.byteOffset("%origin", frame.payload_offset);
                try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, destination, origin });
            } else {
                const helper = try self.dropHelper(payload, span);
                const pointer = try self.byteOffset("%value", frame.payload_offset);
                try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, pointer });
            }
            try self.instruction("br label %{s}", .{exit});
            self.terminated = true;
            try self.startBlock(miss);
        }
        try self.startBlock(exit);
    }

    fn typeDescriptor(self: *Codegen, definition: *const ast.Definition) Error![]const u8 {
        if (self.type_descriptors.get(definition)) |existing| return existing;
        // the type name rides in the symbol so debuggers resolving the
        // identity pointer show which concrete type an interface holds
        const type_name = switch (definition.kind) {
            .type_def => |type_def| name: {
                const view = for (self.views) |view| {
                    if (definitionBelongsTo(view.module, definition)) break view;
                } else self.views[0];
                break :name type_def.name.slice(view.source);
            },
            else => "type",
        };
        const name = try std.fmt.allocPrint(self.arena, "alloy.type.{s}.{d}", .{ type_name, self.global_counter });
        self.global_counter += 1;
        // internal (not private) linkage keeps the symbol visible to
        // debuggers, which resolve the identity pointer through it
        self.constants.writer.print("@\"{s}\" = internal constant i8 0\n", .{name}) catch return error.OutOfMemory;
        try self.type_descriptors.put(self.arena, definition, name);
        return name;
    }

    // whether a place holds an interface object: a reference (or pointer)
    // whose pointee is an interface (section 5.2)
    fn interfaceOfPlace(self: *Codegen, place: Place) Error!?Type.Interface {
        const resolved = try self.resolvedOf(place.value_type);
        const child = switch (resolved.*) {
            .reference => |indirection| indirection.child,
            .pointer => |indirection| indirection.child,
            else => return null,
        };
        const pointee = try self.resolvedOf(child);
        return if (pointee.* == .interface) pointee.interface else null;
    }

    const Implementer = struct {
        definition: *const ast.Definition,
        view_index: usize,
        name: []const u8,
        // the implementer's own instantiation behind a generic interface
        // object ('VectorCursor<E> : Iterator<E>' behind '&Iterator<u64>'
        // carries '[u64]'); empty for non-generic implementers
        arguments: []const *const Type = &.{},
    };

    // the merged unit is the whole program, so the implementers of an
    // interface form a closed world the dispatch chain can enumerate;
    // memoized because dispatch sites and the virtual drop and copy
    // helpers all enumerate the same list
    fn interfaceImplementers(self: *Codegen, interface: Type.Interface) Error![]const Implementer {
        // an instantiated generic interface enumerates per instantiation:
        // each conforming type's marker unifies against the required
        // arguments to derive that implementer's own bindings
        if (interface.arguments.len != 0) {
            const memo_key = try self.instantiationKey(interface);
            if (self.instance_implementer_lists.get(memo_key)) |existing| return existing;
            var implementers: std.ArrayList(Implementer) = .empty;
            for (self.views, 0..) |view, view_index| {
                for (view.module.definitions) |*definition| {
                    if (definition.kind != .type_def) continue;
                    const arguments = (try self.checker.implementerArguments(definition, view_index, interface)) orelse continue;
                    try implementers.append(self.arena, .{
                        .definition = definition,
                        .view_index = view_index,
                        .name = definition.kind.type_def.name.slice(view.source),
                        .arguments = arguments,
                    });
                }
            }
            const list = try implementers.toOwnedSlice(self.arena);
            try self.instance_implementer_lists.put(self.arena, memo_key, list);
            return list;
        }
        if (self.implementer_lists.get(interface.definition)) |existing| return existing;
        var implementers: std.ArrayList(Implementer) = .empty;
        for (self.views, 0..) |view, view_index| {
            for (view.module.definitions) |*definition| {
                if (definition.kind != .type_def) continue;
                const type_def = definition.kind.type_def;
                for (type_def.interfaces) |marker| {
                    // the marker resolves where the type is declared
                    const symbol = self.firstVisible(marker.name.slice(view.source), view_index) orelse continue;
                    if (symbol.definition != interface.definition) continue;
                    try implementers.append(self.arena, .{
                        .definition = definition,
                        .view_index = view_index,
                        .name = type_def.name.slice(view.source),
                    });
                    break;
                }
            }
        }
        const list = try implementers.toOwnedSlice(self.arena);
        try self.implementer_lists.put(self.arena, interface.definition, list);
        return list;
    }

    fn instantiationKey(self: *Codegen, interface: Type.Interface) Error![]const u8 {
        var text: std.Io.Writer.Allocating = .init(self.arena);
        defer text.deinit();
        text.writer.print("{d}", .{@intFromPtr(interface.definition)}) catch return error.OutOfMemory;
        for (interface.arguments) |argument| {
            text.writer.print("|{s}", .{try self.typeKey(argument, 0)}) catch return error.OutOfMemory;
        }
        return self.arena.dupe(u8, text.writer.buffered());
    }

    // the interface-object fat pair: the data pointer at offset 0 and the
    // per-type identity at offset 8 (section 5.2); a generic interface's
    // identity is per instantiation, a non-generic one stays per
    // definition so downcasts keep matching
    fn interfacePair(self: *Codegen, data_pointer: []const u8, concrete: *const Type, generic_interface: bool) Error!Operand {
        const resolved = try self.resolvedOf(concrete);
        const descriptor = if (generic_interface)
            try self.typeDescriptorFor(concrete)
        else
            try self.typeDescriptor(resolved.declared.definition);
        const pair = try self.aggregateSlot(.{ .size = 16, .alignment = 8 });
        try self.instruction("store ptr {s}, ptr {s}", .{ data_pointer, pair });
        const identity_pointer = try self.byteOffset(pair, 8);
        try self.instruction("store ptr @\"{s}\", ptr {s}", .{ descriptor, identity_pointer });
        return .{ .memory = .{ .pointer = pair, .layout = .{ .size = 16, .alignment = 8 }, .fresh = true } };
    }

    fn implementerType(self: *Codegen, implementer: Implementer) Error!*const Type {
        const concrete = try self.arena.create(Type);
        concrete.* = .{ .declared = .{
            .definition = implementer.definition,
            .view_index = implementer.view_index,
            .name = implementer.name,
            .arguments = implementer.arguments,
        } };
        return concrete;
    }

    // the identity for one concrete instantiation: a generic type gets a
    // descriptor per instantiation, so 'Cursor<u64>' and 'Cursor<u8>'
    // behind '&Iterator<...>' objects never confuse dispatch (section 5.2)
    fn typeDescriptorFor(self: *Codegen, concrete: *const Type) Error![]const u8 {
        const resolved = try self.resolvedOf(concrete);
        if (resolved.* != .declared) return self.report(.{ .start = 0, .end = 0 }, "internal: an interface object needs a declared concrete type", .{});
        if (resolved.declared.arguments.len == 0) return self.typeDescriptor(resolved.declared.definition);
        const key = try std.fmt.allocPrint(self.arena, "{d}|{s}", .{ @intFromPtr(resolved.declared.definition), try self.typeKey(resolved, 0) });
        if (self.instance_descriptors.get(key)) |existing| return existing;
        const name = try std.fmt.allocPrint(self.arena, "alloy.type.{s}.{d}", .{ resolved.declared.name, self.global_counter });
        self.global_counter += 1;
        self.constants.writer.print("@\"{s}\" = internal constant i8 0\n", .{name}) catch return error.OutOfMemory;
        try self.instance_descriptors.put(self.arena, key, name);
        return name;
    }

    // the drop body for an owning interface object '*I' (section 5.2):
    // the type identity selects the concrete drop before the data frees,
    // a virtual drop over the closed world of implementers
    fn emitInterfaceDrop(self: *Codegen, interface: Type.Interface, span: Token.Location) Error!void {
        const loaded = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr %value", .{loaded});
        const live = try self.freshTemp();
        try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
        const live_label = try self.freshLabel("live");
        const done_label = try self.freshLabel("done");
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
        self.terminated = true;
        try self.startBlock(live_label);
        const identity_slot = try self.byteOffset("%value", 8);
        const identity = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr {s}", .{ identity, identity_slot });
        const free_label = try self.freshLabel("free");
        for (try self.interfaceImplementers(interface)) |implementer| {
            const concrete = try self.implementerType(implementer);
            // implementers without owned heap free the allocation alone
            if (!try self.ownsHeap(concrete, 0)) continue;
            const descriptor = try self.typeDescriptorFor(concrete);
            const arm = try self.freshLabel("drop.arm");
            const miss = try self.freshLabel("drop.miss");
            const matches = try self.freshTemp();
            try self.instruction("{s} = icmp eq ptr {s}, @\"{s}\"", .{ matches, identity, descriptor });
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, arm, miss });
            self.terminated = true;
            try self.startBlock(arm);
            const concrete_drop = try self.dropHelper(concrete, span);
            try self.instruction("call void @\"{s}\"(ptr {s})", .{ concrete_drop, loaded });
            try self.instruction("br label %{s}", .{free_label});
            self.terminated = true;
            try self.startBlock(miss);
        }
        try self.startBlock(free_label);
        try self.instruction("call void @\"free\"(ptr {s})", .{loaded});
        try self.startBlock(done_label);
    }

    // the copy body for '*I': the identity selects the concrete clone, so
    // the copy owns a fresh allocation of the right concrete type
    fn emitInterfaceCopy(self: *Codegen, interface: Type.Interface, span: Token.Location) Error!void {
        try self.copyBytes("%destination", "%origin", 16);
        const loaded = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr %origin", .{loaded});
        const live = try self.freshTemp();
        try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, loaded });
        const live_label = try self.freshLabel("live");
        const done_label = try self.freshLabel("done");
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
        self.terminated = true;
        try self.startBlock(live_label);
        const identity_slot = try self.byteOffset("%origin", 8);
        const identity = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr {s}", .{ identity, identity_slot });
        for (try self.interfaceImplementers(interface)) |implementer| {
            const concrete = try self.implementerType(implementer);
            const descriptor = try self.typeDescriptorFor(concrete);
            const arm = try self.freshLabel("copy.arm");
            const miss = try self.freshLabel("copy.miss");
            const matches = try self.freshTemp();
            try self.instruction("{s} = icmp eq ptr {s}, @\"{s}\"", .{ matches, identity, descriptor });
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, arm, miss });
            self.terminated = true;
            try self.startBlock(arm);
            const layout = (try self.layoutQuery(concrete, 0)) orelse
                return self.report(span, "this implementer has no defined layout", .{});
            const allocation = try self.allocateFixed(layout.size);
            if (try self.ownsHeap(concrete, 0)) {
                const concrete_copy = try self.copyHelper(concrete, span);
                try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ concrete_copy, allocation, loaded });
            } else {
                try self.copyBytes(allocation, loaded, layout.size);
            }
            try self.instruction("store ptr {s}, ptr %destination", .{allocation});
            try self.instruction("br label %{s}", .{done_label});
            self.terminated = true;
            try self.startBlock(miss);
        }
        // closed world: an unmatched identity cannot alias the original
        try self.instruction("store ptr null, ptr %destination", .{});
        try self.startBlock(done_label);
    }

    // the extension implementing 'name' for the concrete type, mirroring
    // the interpreter's runtime dispatch by receiver type (section 5.2)
    fn findTypeExtension(self: *Codegen, type_definition: *const ast.Definition, name: []const u8) Error!?resolution.Symbol {
        const symbols = self.globals.get(name) orelse return null;
        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            if (fn_def.function.parameters.len == 0 or !fn_def.function.parameters[0].is_self) continue;
            const self_type = try self.candidateSelfType(fn_def, symbol.view_index);
            const pierced = try self.checker.pierce(self_type);
            const resolved = try self.checker.resolveAlias(pierced);
            if (resolved.* == .declared and resolved.declared.definition == type_definition) return symbol;
        }
        return null;
    }

    // the receiver type in the candidate's OWN parameter environment: a
    // generic candidate's 'Cursor<E>' must not report 'E' as undeclared
    fn candidateSelfType(self: *Codegen, fn_def: ast.FnDef, view_index: usize) Error!*const Type {
        const environment = try self.checker.typeParameterEnvironment(fn_def.type_parameters, view_index);
        return self.checker.typeFromExpressionIn(fn_def.function.parameters[0].parameter_type, environment, view_index);
    }

    // the default implementation: an extension whose receiver is the
    // interface itself (section 5.2)
    fn findInterfaceDefault(self: *Codegen, interface: Type.Interface, name: []const u8) Error!?resolution.Symbol {
        const symbols = self.globals.get(name) orelse return null;
        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            if (fn_def.function.parameters.len == 0 or !fn_def.function.parameters[0].is_self) continue;
            const self_type = try self.candidateSelfType(fn_def, symbol.view_index);
            const pierced = try self.checker.pierce(self_type);
            if (pierced.* == .interface and pierced.interface.definition == interface.definition) return symbol;
        }
        return null;
    }

    fn emitFault(self: *Codegen, message: []const u8) Error!void {
        const global = try self.byteGlobal(message);
        try self.emitFaultHelper();
        try self.instruction("call void @\"alloy.fault\"(ptr @\"{s}\")", .{global.name});
        try self.instruction("unreachable", .{});
        self.terminated = true;
    }

    // an interface method call: compare the object's type identity against
    // every implementer, calling the concrete extension on a match; the
    // default implementation receives the interface object itself
    fn interfaceDispatch(self: *Codegen, expression: *const ast.Expression, member: anytype, place: Place, interface: Type.Interface) Error!Operand {
        const call = expression.call;
        const span = self.spanOf(expression);
        const name = member.name.slice(self.source());
        const data = try self.loadPointerField(place.pointer, 0);
        const type_id = try self.loadPointerField(place.pointer, 8);

        var argument_operands: std.ArrayList(Operand) = .empty;
        var argument_types: std.ArrayList(*const Type) = .empty;
        for (call.arguments) |argument| {
            try argument_operands.append(self.arena, try self.evalExpression(argument));
            try argument_types.append(self.arena, try self.typeOf(argument));
        }

        const return_type = try self.substituted(try self.typeOf(expression));
        const slot: ?Slot = switch (try self.classify(return_type, span)) {
            .void_class => null,
            .scalar => |llvm| .{ .pointer = try self.scalarSlot(llvm), .value_type = return_type },
            .aggregate => |layout| .{ .pointer = try self.aggregateSlot(layout), .value_type = return_type },
        };
        const exit = try self.freshLabel("dispatch.exit");

        const implementers = try self.interfaceImplementers(interface);
        for (implementers) |implementer| {
            // the concrete instantiation resolves the extension (and its
            // type bindings, for generic implementers) like the checker
            // would at a static call site
            const concrete = try self.implementerType(implementer);
            var substituted_arguments: std.ArrayList(*const Type) = .empty;
            for (argument_types.items) |argument_type| {
                try substituted_arguments.append(self.arena, try self.substituted(argument_type));
            }
            const target = (try self.checker.resolveInstanceMethod(concrete, name, substituted_arguments.items, true)) orelse continue;
            const descriptor = try self.typeDescriptorFor(concrete);
            const arm = try self.freshLabel("dispatch.arm");
            const miss = try self.freshLabel("dispatch.miss");
            const matches = try self.freshTemp();
            try self.instruction("{s} = icmp eq ptr {s}, @\"{s}\"", .{ matches, type_id, descriptor });
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, arm, miss });
            self.terminated = true;
            try self.startBlock(arm);
            try self.emitDispatchArm(target.symbol, target.type_bindings, "ptr", data, argument_operands.items, argument_types.items, slot, span);
            try self.instruction("br label %{s}", .{exit});
            self.terminated = true;
            try self.startBlock(miss);
        }
        if (try self.findInterfaceDefault(interface, name)) |default_symbol| {
            // the default implementation's receiver is the interface
            // object: the fat pair passes whole (section 5.2)
            try self.emitDispatchArm(default_symbol, &.{}, "ptr", place.pointer, argument_operands.items, argument_types.items, slot, span);
            try self.instruction("br label %{s}", .{exit});
            self.terminated = true;
        } else {
            try self.emitFault("runtime fault: no implementation reachable for this interface call (section 5.2)");
        }
        try self.startBlock(exit);
        return self.slotOperand(slot, span);
    }

    fn emitDispatchArm(self: *Codegen, symbol: resolution.Symbol, type_bindings: []const Type.Binding, receiver_llvm: []const u8, receiver_text: []const u8, argument_operands: []const Operand, argument_types: []const *const Type, slot: ?Slot, span: Token.Location) Error!void {
        const info = try self.functionInfo(symbol, span, type_bindings);
        var lowered: std.ArrayList([]const u8) = .empty;
        var result_pointer: ?[]const u8 = null;
        if (info.aggregate_return) {
            result_pointer = slot.?.pointer;
            try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{result_pointer.?}));
        }
        try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "{s} {s}", .{ receiver_llvm, receiver_text }));
        for (argument_operands, argument_types, 0..) |operand, argument_type, index| {
            if (index + 1 >= info.parameter_types.len) break;
            const parameter_type = info.parameter_types[index + 1];
            const coerced = try self.coerceOperand(operand, argument_type, parameter_type, span);
            try lowered.append(self.arena, try self.lowerArgument(coerced, parameter_type, span));
        }
        const joined = try self.joinArguments(lowered.items);
        if (info.aggregate_return) {
            try self.instruction("call void @\"{s}\"({s})", .{ info.name, joined });
            return;
        }
        switch (try self.classify(info.return_type, span)) {
            .void_class => try self.instruction("call void @\"{s}\"({s})", .{ info.name, joined }),
            .scalar => |llvm| {
                const value = try self.freshTemp();
                try self.instruction("{s} = call {s} @\"{s}\"({s})", .{ value, llvm, info.name, joined });
                if (slot) |present| try self.storeScalar(present.pointer, .{ .text = value, .llvm = llvm });
            },
            .aggregate => unreachable,
        }
    }

    fn loadPlace(self: *Codegen, place: Place, span: Token.Location) Error!Operand {
        switch (try self.classify(place.value_type, span)) {
            .void_class => return .none,
            .scalar => |llvm| return .{ .scalar = .{ .text = try self.loadScalar(place.pointer, llvm), .llvm = llvm } },
            .aggregate => |layout| return .{ .memory = .{ .pointer = place.pointer, .layout = layout } },
        }
    }

    fn coerceOperand(self: *Codegen, operand: Operand, from: *const Type, to: *const Type, span: Token.Location) Error!Operand {
        const from_resolved = try self.resolvedOf(from);
        const to_resolved = try self.resolvedOf(to);
        if (from_resolved.eql(to_resolved)) return operand;
        if (from_resolved.* == .primitive and to_resolved.* == .primitive) {
            if (operand != .scalar) return self.report(span, "internal: a primitive operand is not a scalar", .{});
            return .{ .scalar = try self.convertNumeric(operand.scalar, from_resolved.primitive, to_resolved.primitive) };
        }
        // a reference or pointer to a concrete implementer converts to an
        // interface object: the fat pair carries the data pointer and the
        // type identity; an owning '*T' yields an owning '*I' (section 5.2)
        const to_child: ?*const Type = switch (to_resolved.*) {
            .reference => |indirection| indirection.child,
            .pointer => |indirection| indirection.child,
            else => null,
        };
        if (to_child != null and (try self.resolvedOf(to_child.?)).* == .interface) {
            const from_child: ?*const Type = switch (from_resolved.*) {
                .reference => |indirection| indirection.child,
                .pointer => |indirection| indirection.child,
                else => null,
            };
            if (from_child) |child| {
                const pointee = try self.resolvedOf(child);
                if (pointee.* == .interface) return operand;
                if (pointee.* != .declared) {
                    return self.report(span, "only named types convert to interface objects (section 5.2)", .{});
                }
                if (operand != .scalar) {
                    return self.report(span, "internal: this indirection operand has no scalar pointer", .{});
                }
                const interface = (try self.resolvedOf(to_child.?)).interface;
                return self.interfacePair(operand.scalar.text, pointee, interface.arguments.len != 0);
            }
        }
        switch (operand) {
            .scalar, .none => return operand,
            .memory => |memory| {
                // structurally compatible aggregates (rules 6 and 7) share
                // their layout when field lists agree in order and type
                const target_layout = (try self.layoutQuery(to_resolved, 0)) orelse return operand;
                if (target_layout.size == memory.layout.size) return operand;
                return self.report(span, "this conversion changes layout and is not yet supported by native code generation", .{});
            },
        }
    }

    fn widerOf(self: *Codegen, left: *const Type, right: *const Type, span: Token.Location) Error!*const Type {
        if (left.* != .primitive) return right;
        if (right.* != .primitive) return left;
        if (left.primitive == right.primitive) return left;
        if (left.primitive.isFloat() != right.primitive.isFloat()) {
            return if (left.primitive.isFloat()) left else right;
        }
        if (left.primitive.width() == right.primitive.width()) {
            if (left.primitive.isSigned() != right.primitive.isSigned()) {
                return self.report(span, "mixed-sign operands are not supported here", .{});
            }
            return left;
        }
        return if (left.primitive.width() > right.primitive.width()) left else right;
    }

    fn widenToIndex(self: *Codegen, scalar: Scalar, scalar_type: *const Type) Error![]const u8 {
        if (std.mem.eql(u8, scalar.llvm, "i64")) return scalar.text;
        const signed = scalar_type.* == .primitive and scalar_type.primitive.isSigned();
        const result = try self.freshTemp();
        try self.instruction("{s} = {s} {s} {s} to i64", .{ result, if (signed) "sext" else "zext", scalar.llvm, scalar.text });
        return result;
    }

    fn narrowFromIndex(self: *Codegen, text: []const u8, llvm: []const u8) Error![]const u8 {
        if (std.mem.eql(u8, llvm, "i64")) return text;
        const result = try self.freshTemp();
        try self.instruction("{s} = trunc i64 {s} to {s}", .{ result, text, llvm });
        return result;
    }

    fn ensureMemory(self: *Codegen, operand: Operand, value_type: *const Type, span: Token.Location) Error!Memory {
        switch (operand) {
            .memory => |memory| return memory,
            .scalar => |scalar| {
                const slot = try self.scalarSlot(scalar.llvm);
                try self.storeScalar(slot, scalar);
                const layout = (try self.layoutQuery(value_type, 0)) orelse Checker.Layout{ .size = 8, .alignment = 8 };
                return .{ .pointer = slot, .layout = layout, .fresh = true };
            },
            .none => return self.report(span, "this value has no storage", .{}),
        }
    }

    fn storeOperand(self: *Codegen, pointer: []const u8, operand: Operand, value_type: *const Type, span: Token.Location) Error!void {
        switch (operand) {
            .none => {},
            .scalar => |scalar| try self.storeScalar(pointer, scalar),
            .memory => |memory| {
                if (std.mem.eql(u8, pointer, memory.pointer)) return;
                // a temporary transfers its bits; a place-backed value deep
                // copies the heap it owns (section 4.2)
                if (!memory.fresh and try self.ownsHeap(value_type, 0)) {
                    const helper = try self.copyHelper(value_type, span);
                    try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, pointer, memory.pointer });
                    return;
                }
                const layout = (try self.layoutQuery(value_type, 0)) orelse memory.layout;
                try self.copyBytes(pointer, memory.pointer, layout.size);
            },
        }
    }

    // booleans are 'i1' in registers and one byte in memory (section 3.9)
    fn loadScalar(self: *Codegen, pointer: []const u8, llvm: []const u8) Error![]const u8 {
        if (std.mem.eql(u8, llvm, "i1")) {
            const wide = try self.freshTemp();
            try self.instruction("{s} = load i8, ptr {s}", .{ wide, pointer });
            const result = try self.freshTemp();
            try self.instruction("{s} = trunc i8 {s} to i1", .{ result, wide });
            return result;
        }
        const result = try self.freshTemp();
        try self.instruction("{s} = load {s}, ptr {s}", .{ result, llvm, pointer });
        return result;
    }

    fn storeScalar(self: *Codegen, pointer: []const u8, scalar: Scalar) Error!void {
        if (std.mem.eql(u8, scalar.llvm, "i1")) {
            const wide = try self.freshTemp();
            try self.instruction("{s} = zext i1 {s} to i8", .{ wide, scalar.text });
            try self.instruction("store i8 {s}, ptr {s}", .{ wide, pointer });
            return;
        }
        try self.instruction("store {s} {s}, ptr {s}", .{ scalar.llvm, scalar.text, pointer });
    }

    fn loadPointerField(self: *Codegen, base: []const u8, offset: u64) Error![]const u8 {
        const pointer = try self.byteOffset(base, offset);
        const result = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr {s}", .{ result, pointer });
        return result;
    }

    fn loadIntegerField(self: *Codegen, base: []const u8, offset: u64) Error![]const u8 {
        const pointer = try self.byteOffset(base, offset);
        const result = try self.freshTemp();
        try self.instruction("{s} = load i64, ptr {s}", .{ result, pointer });
        return result;
    }

    fn byteOffset(self: *Codegen, base: []const u8, offset: u64) Error![]const u8 {
        if (offset == 0) return base;
        const result = try self.freshTemp();
        try self.instruction("{s} = getelementptr inbounds i8, ptr {s}, i64 {d}", .{ result, base, offset });
        return result;
    }

    fn copyBytes(self: *Codegen, destination: []const u8, origin: []const u8, size: u64) Error!void {
        try self.declareIntrinsic("declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)");
        try self.instruction("call void @llvm.memcpy.p0.p0.i64(ptr {s}, ptr {s}, i64 {d}, i1 false)", .{ destination, origin, size });
    }

    fn zeroFill(self: *Codegen, pointer: []const u8, size: u64) Error!void {
        try self.declareIntrinsic("declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)");
        try self.instruction("call void @llvm.memset.p0.i64(ptr {s}, i8 0, i64 {d}, i1 false)", .{ pointer, size });
    }

    fn scalarSlot(self: *Codegen, llvm: []const u8) Error![]const u8 {
        const name = try std.fmt.allocPrint(self.arena, "%slot.{d}", .{self.temp_counter});
        self.temp_counter += 1;
        const stored: []const u8 = if (std.mem.eql(u8, llvm, "i1")) "i8" else llvm;
        self.allocas.writer.print("  {s} = alloca {s}\n", .{ name, stored }) catch return error.OutOfMemory;
        return name;
    }

    fn aggregateSlot(self: *Codegen, layout: Checker.Layout) Error![]const u8 {
        const name = try std.fmt.allocPrint(self.arena, "%slot.{d}", .{self.temp_counter});
        self.temp_counter += 1;
        self.allocas.writer.print("  {s} = alloca [{d} x i8], align {d}\n", .{ name, layout.size, layout.alignment }) catch return error.OutOfMemory;
        return name;
    }

    fn freshTemp(self: *Codegen) Error![]const u8 {
        const name = try std.fmt.allocPrint(self.arena, "%t{d}", .{self.temp_counter});
        self.temp_counter += 1;
        return name;
    }

    fn freshLabel(self: *Codegen, prefix: []const u8) Error![]const u8 {
        const name = try std.fmt.allocPrint(self.arena, "{s}.{d}", .{ prefix, self.temp_counter });
        self.temp_counter += 1;
        return name;
    }

    fn instruction(self: *Codegen, comptime format: []const u8, arguments: anytype) Error!void {
        if (self.terminated) {
            // statements after a terminator still need a block to live in
            const label = try self.freshLabel("dead");
            self.body.writer.print("{s}:\n", .{label}) catch return error.OutOfMemory;
            self.terminated = false;
        }
        self.body.writer.print("  " ++ format, arguments) catch return error.OutOfMemory;
        self.appendDebugLocation();
        self.body.writer.print("\n", .{}) catch return error.OutOfMemory;
    }

    fn startBlock(self: *Codegen, label: []const u8) Error!void {
        if (!self.terminated) {
            self.body.writer.print("  br label %{s}", .{label}) catch return error.OutOfMemory;
            self.appendDebugLocation();
            self.body.writer.print("\n", .{}) catch return error.OutOfMemory;
        }
        self.body.writer.print("{s}:\n", .{label}) catch return error.OutOfMemory;
        self.terminated = false;
    }

    fn appendDebugLocation(self: *Codegen) void {
        if (self.debug_subprogram == null) return;
        const location = self.debug_location orelse return;
        self.body.writer.print(", !dbg !{d}", .{location}) catch {};
    }

    fn nextMetadata(self: *Codegen) usize {
        const id = self.metadata_counter;
        self.metadata_counter += 1;
        return id;
    }

    // DWARF strings use forward slashes; a backslash would need escaping
    fn printDebugPath(self: *Codegen, path: []const u8) void {
        for (path) |byte| {
            const printable = if (byte == '\\') '/' else byte;
            if (printable == '"') continue;
            self.debug_metadata.writer.print("{c}", .{printable}) catch {};
        }
    }

    fn debugFile(self: *Codegen, view_index: usize) Error!usize {
        if (self.debug_files.get(view_index)) |existing| return existing;
        const id = self.nextMetadata();
        const path = self.views[view_index].path;
        const separator = std.mem.lastIndexOfAny(u8, path, "/\\");
        const base = if (separator) |index| path[index + 1 ..] else path;
        const directory = if (separator) |index| path[0..index] else ".";
        self.debug_metadata.writer.print("!{d} = !DIFile(filename: \"", .{id}) catch return error.OutOfMemory;
        self.printDebugPath(base);
        self.debug_metadata.writer.print("\", directory: \"", .{}) catch return error.OutOfMemory;
        self.printDebugPath(directory);
        self.debug_metadata.writer.print("\")\n", .{}) catch return error.OutOfMemory;
        try self.debug_files.put(self.arena, view_index, id);
        return id;
    }

    fn debugCompileUnit(self: *Codegen, view_index: usize) Error!usize {
        if (self.debug_compile_unit) |existing| return existing;
        const file = try self.debugFile(view_index);
        const id = self.nextMetadata();
        self.debug_metadata.writer.print(
            "!{d} = distinct !DICompileUnit(language: DW_LANG_C99, file: !{d}, producer: \"alloyc\", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)\n",
            .{ id, file },
        ) catch return error.OutOfMemory;
        self.debug_compile_unit = id;
        return id;
    }

    fn debugSubroutineType(self: *Codegen) Error!usize {
        if (self.debug_subroutine_type) |existing| return existing;
        const operands = self.nextMetadata();
        const id = self.nextMetadata();
        self.debug_metadata.writer.print("!{d} = !{{null}}\n!{d} = !DISubroutineType(types: !{d})\n", .{ operands, id, operands }) catch return error.OutOfMemory;
        self.debug_subroutine_type = id;
        return id;
    }

    // opens the debug scope of a user function or lambda: subsequent
    // statements attach line locations until the scope clears
    fn beginDebugFunction(self: *Codegen, display_name: []const u8, view_index: usize, offset: usize) Error!usize {
        const unit = try self.debugCompileUnit(view_index);
        const file = try self.debugFile(view_index);
        const subroutine_type = try self.debugSubroutineType();
        const position = lineColumnOf(self.views[view_index].source, offset);
        const id = self.nextMetadata();
        self.debug_metadata.writer.print(
            "!{d} = distinct !DISubprogram(name: \"{s}\", scope: !{d}, file: !{d}, line: {d}, type: !{d}, scopeLine: {d}, spFlags: DISPFlagDefinition, unit: !{d})\n",
            .{ id, display_name, file, file, position.line, subroutine_type, position.line, unit },
        ) catch return error.OutOfMemory;
        self.debug_subprogram = id;
        self.debug_file_id = file;
        self.debug_location = null;
        self.debug_scopes.clearRetainingCapacity();
        try self.setDebugLocation(offset);
        return id;
    }

    fn currentDebugScope(self: *const Codegen) usize {
        if (self.debug_scopes.items.len != 0) {
            return self.debug_scopes.items[self.debug_scopes.items.len - 1];
        }
        return self.debug_subprogram.?;
    }

    fn setDebugLocation(self: *Codegen, offset: usize) Error!void {
        if (self.debug_subprogram == null) return;
        const scope = self.currentDebugScope();
        const position = lineColumnOf(self.source(), offset);
        self.debug_line = position.line;
        self.debug_column = position.column;
        const key = (@as(u128, scope) << 64) | (@as(u128, position.line) << 32) | position.column;
        if (self.debug_location_memo.get(key)) |existing| {
            self.debug_location = existing;
            return;
        }
        const id = self.nextMetadata();
        self.debug_metadata.writer.print("!{d} = !DILocation(line: {d}, column: {d}, scope: !{d})\n", .{ id, position.line, position.column, scope }) catch return error.OutOfMemory;
        try self.debug_location_memo.put(self.arena, key, id);
        self.debug_location = id;
    }

    const LineColumn = struct {
        line: usize,
        column: usize,
    };

    // 1-based line and byte column for debug locations
    fn lineColumnOf(text: []const u8, offset: usize) LineColumn {
        const clamped = @min(offset, text.len);
        var line: usize = 1;
        var line_start: usize = 0;
        for (text[0..clamped], 0..) |byte, index| {
            if (byte == '\n') {
                line += 1;
                line_start = index + 1;
            }
        }
        return .{ .line = line, .column = clamped - line_start + 1 };
    }

    // renders a DWARF type for a checked type, mirroring the section 3.9
    // layouts; null when the type has no meaningful runtime description
    fn debugType(self: *Codegen, candidate: *const Type, depth: usize) Error!?usize {
        if (depth > 8) return null;
        const resolved = self.resolvedOf(candidate) catch return null;
        const key = self.typeKey(resolved, 0) catch return null;
        if (self.debug_types.get(key)) |existing| return existing;
        const out = &self.debug_metadata.writer;
        switch (resolved.*) {
            .primitive => |primitive| {
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                const encoding: []const u8 = if (primitive == .bool)
                    "DW_ATE_boolean"
                else if (primitive.isFloat())
                    "DW_ATE_float"
                else if (primitive.isSigned())
                    "DW_ATE_signed"
                else
                    "DW_ATE_unsigned";
                out.print("!{d} = !DIBasicType(name: \"{s}\", size: {d}, encoding: {s})\n", .{ id, @tagName(primitive), @as(u32, primitive.width()) * 8, encoding }) catch return error.OutOfMemory;
                return id;
            },
            .reference, .pointer => {
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                const child = switch (resolved.*) {
                    .reference => |reference| reference.child,
                    .pointer => |pointer| pointer.child,
                    else => unreachable,
                };
                if (try self.debugType(child, depth + 1)) |child_id| {
                    out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{d}, size: 64)\n", .{ id, child_id }) catch return error.OutOfMemory;
                } else {
                    out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)\n", .{id}) catch return error.OutOfMemory;
                }
                return id;
            },
            .heap_array => |heap| {
                // a NAMED typedef over the data pointer, so debugger
                // formatters can recognize heap arrays and read the
                // length prefix at pointer - 8
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                const pointer_id = self.nextMetadata();
                if (try self.debugType(heap.child, depth + 1)) |child_id| {
                    out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{d}, size: 64)\n", .{ pointer_id, child_id }) catch return error.OutOfMemory;
                } else {
                    out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)\n", .{pointer_id}) catch return error.OutOfMemory;
                }
                const element_name = heap.child.render(self.arena) catch "T";
                out.print("!{d} = !DIDerivedType(tag: DW_TAG_typedef, name: \"*[{s}]\", baseType: !{d})\n", .{ id, element_name, pointer_id }) catch return error.OutOfMemory;
                return id;
            },
            .function => {
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)\n", .{id}) catch return error.OutOfMemory;
                return id;
            },
            .slice => |slice| {
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                const data_pointer = self.nextMetadata();
                if (try self.debugType(slice.child, depth + 1)) |child_id| {
                    out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{d}, size: 64)\n", .{ data_pointer, child_id }) catch return error.OutOfMemory;
                } else {
                    out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)\n", .{data_pointer}) catch return error.OutOfMemory;
                }
                const length_type = (try self.debugType(&u64_type, depth + 1)).?;
                const data_member = self.nextMetadata();
                const length_member = self.nextMetadata();
                const elements = self.nextMetadata();
                out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"data\", baseType: !{d}, size: 64, offset: 0)\n", .{ data_member, data_pointer }) catch return error.OutOfMemory;
                out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"length\", baseType: !{d}, size: 64, offset: 64)\n", .{ length_member, length_type }) catch return error.OutOfMemory;
                out.print("!{d} = !{{!{d}, !{d}}}\n", .{ elements, data_member, length_member }) catch return error.OutOfMemory;
                out.print("!{d} = !DICompositeType(tag: DW_TAG_structure_type, name: \"slice\", size: 128, elements: !{d})\n", .{ id, elements }) catch return error.OutOfMemory;
                return id;
            },
            .interface => |interface| {
                // the fat pair as {data, identity}: resolving the identity
                // pointer to its symbol names the concrete type
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                const raw_pointer = self.nextMetadata();
                out.print("!{d} = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)\n", .{raw_pointer}) catch return error.OutOfMemory;
                const data_member = self.nextMetadata();
                const identity_member = self.nextMetadata();
                const elements = self.nextMetadata();
                out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"data\", baseType: !{d}, size: 64, offset: 0)\n", .{ data_member, raw_pointer }) catch return error.OutOfMemory;
                out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"identity\", baseType: !{d}, size: 64, offset: 64)\n", .{ identity_member, raw_pointer }) catch return error.OutOfMemory;
                out.print("!{d} = !{{!{d}, !{d}}}\n", .{ elements, data_member, identity_member }) catch return error.OutOfMemory;
                out.print("!{d} = !DICompositeType(tag: DW_TAG_structure_type, name: \"{s}\", size: 128, elements: !{d})\n", .{ id, interface.name, elements }) catch return error.OutOfMemory;
                return id;
            },
            .fixed_array => |array| {
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                const element = (try self.debugType(array.element, depth + 1)) orelse return null;
                const element_layout = (self.layoutQuery(array.element, 0) catch null) orelse return null;
                const subrange = self.nextMetadata();
                const elements = self.nextMetadata();
                out.print("!{d} = !DISubrange(count: {d})\n", .{ subrange, array.length }) catch return error.OutOfMemory;
                out.print("!{d} = !{{!{d}}}\n", .{ elements, subrange }) catch return error.OutOfMemory;
                out.print("!{d} = !DICompositeType(tag: DW_TAG_array_type, baseType: !{d}, size: {d}, elements: !{d})\n", .{ id, element, element_layout.size * array.length * 8, elements }) catch return error.OutOfMemory;
                return id;
            },
            .declared, .structural, .inline_enum, .structural_enum => {
                const layout = (self.layoutQuery(resolved, 0) catch null) orelse return null;
                const type_name: []const u8 = if (resolved.* == .declared) resolved.declared.name else "struct";
                const id = self.nextMetadata();
                try self.debug_types.put(self.arena, key, id);
                // enums render as tag + a union of variant payloads,
                // mirroring the section 3.9 frame
                if ((self.enumFrameQuery(resolved) catch null) != null) {
                    const frame = (self.enumFrameQuery(resolved) catch null).?;
                    try self.debugEnumType(id, type_name, layout, frame, depth);
                    return id;
                }
                const field_slots = (self.fieldSlotsQuery(resolved) catch null) orelse {
                    out.print("!{d} = !DICompositeType(tag: DW_TAG_structure_type, name: \"{s}\", size: {d})\n", .{ id, type_name, layout.size * 8 }) catch return error.OutOfMemory;
                    return id;
                };
                var members: std.ArrayList(usize) = .empty;
                for (field_slots) |slot| {
                    const member_type = (try self.debugType(slot.field_type, depth + 1)) orelse continue;
                    const member_layout = (self.layoutQuery(slot.field_type, 0) catch null) orelse continue;
                    const member = self.nextMetadata();
                    out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"{s}\", baseType: !{d}, size: {d}, offset: {d})\n", .{ member, slot.name, member_type, member_layout.size * 8, slot.offset * 8 }) catch return error.OutOfMemory;
                    try members.append(self.arena, member);
                }
                const elements = self.nextMetadata();
                out.print("!{d} = !{{", .{elements}) catch return error.OutOfMemory;
                for (members.items, 0..) |member, index| {
                    if (index != 0) out.print(", ", .{}) catch return error.OutOfMemory;
                    out.print("!{d}", .{member}) catch return error.OutOfMemory;
                }
                out.print("}}\n", .{}) catch return error.OutOfMemory;
                out.print("!{d} = !DICompositeType(tag: DW_TAG_structure_type, name: \"{s}\", size: {d}, elements: !{d})\n", .{ id, type_name, layout.size * 8, elements }) catch return error.OutOfMemory;
                return id;
            },
            else => return null,
        }
    }

    fn debugEnumType(self: *Codegen, id: usize, type_name: []const u8, layout: Checker.Layout, frame: Checker.EnumFrame, depth: usize) Error!void {
        const out = &self.debug_metadata.writer;
        const tag_type: *const Type = switch (frame.tag_size) {
            1 => &u8_type,
            2 => &u16_type,
            else => &u32_type,
        };
        const tag_type_id = (try self.debugType(tag_type, depth + 1)).?;
        const tag_member = self.nextMetadata();
        out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"tag\", baseType: !{d}, size: {d}, offset: 0)\n", .{ tag_member, tag_type_id, frame.tag_size * 8 }) catch return error.OutOfMemory;

        var union_members: std.ArrayList(usize) = .empty;
        for (frame.variants) |variant| {
            const payload_type = variant.payload orelse continue;
            const payload_id = (try self.debugType(payload_type, depth + 1)) orelse continue;
            const payload_layout = (self.layoutQuery(payload_type, 0) catch null) orelse continue;
            const member = self.nextMetadata();
            out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"{s}\", baseType: !{d}, size: {d}, offset: 0)\n", .{ member, variant.name, payload_id, payload_layout.size * 8 }) catch return error.OutOfMemory;
            try union_members.append(self.arena, member);
        }

        var payload_member: ?usize = null;
        if (union_members.items.len != 0) {
            const union_elements = self.nextMetadata();
            out.print("!{d} = !{{", .{union_elements}) catch return error.OutOfMemory;
            for (union_members.items, 0..) |member, index| {
                if (index != 0) out.print(", ", .{}) catch return error.OutOfMemory;
                out.print("!{d}", .{member}) catch return error.OutOfMemory;
            }
            out.print("}}\n", .{}) catch return error.OutOfMemory;
            const union_type = self.nextMetadata();
            const payload_bits = (layout.size - frame.payload_offset) * 8;
            out.print("!{d} = !DICompositeType(tag: DW_TAG_union_type, name: \"payload\", size: {d}, elements: !{d})\n", .{ union_type, payload_bits, union_elements }) catch return error.OutOfMemory;
            const member = self.nextMetadata();
            out.print("!{d} = !DIDerivedType(tag: DW_TAG_member, name: \"payload\", baseType: !{d}, size: {d}, offset: {d})\n", .{ member, union_type, payload_bits, frame.payload_offset * 8 }) catch return error.OutOfMemory;
            payload_member = member;
        }

        const elements = self.nextMetadata();
        if (payload_member) |member| {
            out.print("!{d} = !{{!{d}, !{d}}}\n", .{ elements, tag_member, member }) catch return error.OutOfMemory;
        } else {
            out.print("!{d} = !{{!{d}}}\n", .{ elements, tag_member }) catch return error.OutOfMemory;
        }
        out.print("!{d} = !DICompositeType(tag: DW_TAG_structure_type, name: \"{s}\", size: {d}, elements: !{d})\n", .{ id, type_name, layout.size * 8, elements }) catch return error.OutOfMemory;
    }

    // a named local becomes visible to debuggers: a DILocalVariable plus
    // an llvm.dbg.declare on its slot
    fn declareDebugVariable(self: *Codegen, name: []const u8, pointer: []const u8, declared_type: *const Type) Error!void {
        if (name.len == 0) return;
        if (self.debug_subprogram == null) return;
        const scope = self.currentDebugScope();
        const file = self.debug_file_id orelse return;
        const type_id = (try self.debugType(declared_type, 0)) orelse return;
        const variable = self.nextMetadata();
        self.debug_metadata.writer.print(
            "!{d} = !DILocalVariable(name: \"{s}\", scope: !{d}, file: !{d}, line: {d}, type: !{d})\n",
            .{ variable, name, scope, file, self.debug_line, type_id },
        ) catch return error.OutOfMemory;
        try self.intrinsic_declarations.put(self.arena, "declare void @llvm.dbg.declare(metadata, metadata, metadata)", {});
        try self.instruction("call void @llvm.dbg.declare(metadata ptr {s}, metadata !{d}, metadata !DIExpression())", .{ pointer, variable });
    }

    fn statementOffset(self: *const Codegen, statement: *const ast.Statement) usize {
        return switch (statement.*) {
            .block => |statements| if (statements.len != 0) self.statementOffset(statements[0]) else 0,
            .var_def => |var_def| var_def.name.location.start,
            .assign => |assign| self.checker.expressionSpan(assign.target).start,
            .expression => |expression| self.checker.expressionSpan(expression).start,
            .break_stmt => |break_stmt| break_stmt.keyword.location.start,
            .yield_stmt => |yield_stmt| yield_stmt.keyword.location.start,
            .return_stmt => |return_stmt| return_stmt.keyword.location.start,
        };
    }

    fn faultUnless(self: *Codegen, condition: []const u8, message: []const u8) Error!void {
        const continue_label = try self.freshLabel("ok");
        const fault_label = try self.freshLabel("fault");
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ condition, continue_label, fault_label });
        self.terminated = true;
        try self.startBlock(fault_label);
        const global = try self.byteGlobal(message);
        try self.emitFaultHelper();
        try self.instruction("call void @\"alloy.fault\"(ptr @\"{s}\")", .{global.name});
        try self.instruction("unreachable", .{});
        self.terminated = true;
        try self.startBlock(continue_label);
    }

    fn emitFaultHelper(self: *Codegen) Error!void {
        if (self.fault_helper_emitted) return;
        self.fault_helper_emitted = true;
        if (!self.extern_declarations.contains("puts")) {
            try self.extern_declarations.put(self.arena, "puts", "declare i32 @\"puts\"(ptr)");
        }
        if (!self.extern_declarations.contains("fflush")) {
            try self.extern_declarations.put(self.arena, "fflush", "declare i32 @\"fflush\"(ptr)");
        }
        try self.declareIntrinsic("declare void @llvm.trap()");
        // fflush(null) drains every stream so the message survives the trap
        self.functions.writer.print(
            "define internal void @\"alloy.fault\"(ptr %message) noreturn {{\nentry:\n  %ignored = call i32 @\"puts\"(ptr %message)\n  %flushed = call i32 @\"fflush\"(ptr null)\n  call void @llvm.trap()\n  unreachable\n}}\n",
            .{},
        ) catch return error.OutOfMemory;
    }

    fn declareIntrinsic(self: *Codegen, declaration: []const u8) Error!void {
        if (self.intrinsic_declarations.contains(declaration)) return;
        try self.intrinsic_declarations.put(self.arena, declaration, {});
    }

    fn byteGlobal(self: *Codegen, bytes: []const u8) Error!ByteGlobal {
        if (self.byte_globals.get(bytes)) |existing| return existing;
        const name = try std.fmt.allocPrint(self.arena, "bytes.{d}", .{self.global_counter});
        self.global_counter += 1;
        const writer = &self.constants.writer;
        writer.print("@\"{s}\" = private unnamed_addr constant [{d} x i8] c\"", .{ name, bytes.len + 1 }) catch return error.OutOfMemory;
        for (bytes) |byte| {
            if (byte >= 0x20 and byte < 0x7f and byte != '"' and byte != '\\') {
                writer.print("{c}", .{byte}) catch return error.OutOfMemory;
            } else {
                writer.print("\\{X:0>2}", .{byte}) catch return error.OutOfMemory;
            }
        }
        writer.print("\\00\"\n", .{}) catch return error.OutOfMemory;
        const global: ByteGlobal = .{ .name = name, .length = bytes.len };
        try self.byte_globals.put(self.arena, try self.arena.dupe(u8, bytes), global);
        return global;
    }

    // a string literal is a slice over static bytes (section 1.6): the
    // 16-byte slice header itself lives in a constant
    fn sliceGlobal(self: *Codegen, bytes: []const u8) Error![]const u8 {
        if (self.slice_globals.get(bytes)) |existing| return existing;
        const data = try self.byteGlobal(bytes);
        const name = try std.fmt.allocPrint(self.arena, "@\"{s}.slice\"", .{data.name});
        self.constants.writer.print(
            "{s} = private unnamed_addr constant {{ ptr, i64 }} {{ ptr @\"{s}\", i64 {d} }}, align 8\n",
            .{ name, data.name, data.length },
        ) catch return error.OutOfMemory;
        try self.slice_globals.put(self.arena, try self.arena.dupe(u8, bytes), name);
        return name;
    }

    fn pushFrame(self: *Codegen) Error!void {
        // each frame opens a lexical block so debuggers resolve shadowed
        // names to the innermost binding
        var scope_pushed = false;
        if (self.debug_subprogram != null) {
            const parent = self.currentDebugScope();
            const file = self.debug_file_id.?;
            const block = self.nextMetadata();
            self.debug_metadata.writer.print(
                "!{d} = distinct !DILexicalBlock(scope: !{d}, file: !{d}, line: {d}, column: {d})\n",
                .{ block, parent, file, self.debug_line, self.debug_column },
            ) catch return error.OutOfMemory;
            try self.debug_scopes.append(self.arena, block);
            scope_pushed = true;
        }
        try self.scopes.append(self.arena, .{ .locals = .empty, .debug_scope_pushed = scope_pushed });
    }

    fn popFrame(self: *Codegen) void {
        const frame = self.scopes.pop().?;
        if (frame.debug_scope_pushed) _ = self.debug_scopes.pop();
    }

    // reading an owning place yields a deep copy (section 4.2, no implicit
    // move): a bare heap-array local flowing out of 'return', 'break', or
    // 'yield' clones its allocation before the scope-end drops free the
    // original; 'move v' transfers explicitly and needs no copy
    fn copyOwnedScalarRead(self: *Codegen, value_expression: *const ast.Expression, operand: Operand, span: Token.Location) Error!Operand {
        if (operand != .scalar) return operand;
        const unwrapped = unwrapGrouped(value_expression);
        if (unwrapped.* != .path or unwrapped.path.len != 1) return operand;
        const local = self.lookupLocal(unwrapped.path[0].slice(self.source())) orelse return operand;
        if (!try self.ownsHeap(local.declared_type, 0)) return operand;
        if ((try self.classify(local.declared_type, span)) != .scalar) return operand;
        // a pierced read ('return p' yielding the pointee copy) already
        // copied: only a read of the owning slot itself clones here
        const read_type = try self.resolvedOf(try self.typeOf(value_expression));
        const local_type = try self.resolvedOf(local.declared_type);
        if (!read_type.eql(local_type)) return operand;
        const copied_slot = try self.scalarSlot("ptr");
        const helper = try self.copyHelper(local_type, span);
        try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ helper, copied_slot, local.pointer });
        return .{ .scalar = .{ .text = try self.loadScalar(copied_slot, "ptr"), .llvm = "ptr" } };
    }

    // scope-end drop (section 4.2): owning locals release their heap when
    // the frame closes normally; a terminated frame already dropped on its
    // break or return path
    fn closeFrame(self: *Codegen) Error!void {
        if (!self.terminated) {
            try self.emitFrameDrops(&self.scopes.items[self.scopes.items.len - 1], null);
        }
        const frame = self.scopes.pop().?;
        if (frame.debug_scope_pushed) _ = self.debug_scopes.pop();
    }

    fn emitFrameDrops(self: *Codegen, frame: *const Frame, skip_pointer: ?[]const u8) Error!void {
        var index = frame.locals.items.len;
        while (index > 0) {
            index -= 1;
            const local = frame.locals.items[index];
            if (local.pinned) continue;
            if (skip_pointer) |skip| {
                if (std.mem.eql(u8, skip, local.pointer)) continue;
            }
            if (!try self.ownsHeap(local.declared_type, 0)) continue;
            const helper = try self.dropHelper(local.declared_type, .{ .start = 0, .end = 0 });
            try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, local.pointer });
        }
    }

    // 'break' and 'return' leave several frames at once; every owning local
    // in the abandoned frames drops before the branch (section 4.2)
    fn emitDropsDownTo(self: *Codegen, depth: usize, skip_pointer: ?[]const u8) Error!void {
        var frame_index = self.scopes.items.len;
        while (frame_index > depth) {
            frame_index -= 1;
            try self.emitFrameDrops(&self.scopes.items[frame_index], skip_pointer);
        }
    }

    // a discarded temporary that owns heap drops immediately: an expression
    // statement or unconsumed receiver would otherwise leak (section 4.2)
    fn dropDiscarded(self: *Codegen, operand: Operand, value_type: *const Type, span: Token.Location) Error!void {
        if (!try self.ownsHeap(value_type, 0)) return;
        switch (operand) {
            .none => {},
            .scalar => |scalar| {
                const slot = try self.scalarSlot("ptr");
                try self.storeScalar(slot, scalar);
                const helper = try self.dropHelper(value_type, span);
                try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, slot });
            },
            .memory => |memory| {
                if (!memory.fresh) return;
                const helper = try self.dropHelper(value_type, span);
                try self.instruction("call void @\"{s}\"(ptr {s})", .{ helper, memory.pointer });
            },
        }
    }

    fn bindLocal(self: *Codegen, name: []const u8, pointer: []const u8, declared_type: *const Type) Error!void {
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        try frame.locals.append(self.arena, .{ .name = name, .pointer = pointer, .declared_type = declared_type });
        try self.declareDebugVariable(name, pointer, declared_type);
    }

    fn bindPinned(self: *Codegen, name: []const u8, pointer: []const u8, declared_type: *const Type) Error!void {
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        try frame.locals.append(self.arena, .{ .name = name, .pointer = pointer, .declared_type = declared_type, .pinned = true });
        try self.declareDebugVariable(name, pointer, declared_type);
    }

    fn lookupLocal(self: *Codegen, name: []const u8) ?Local {
        var frame_index = self.scopes.items.len;
        while (frame_index > 0) {
            frame_index -= 1;
            const frame = self.scopes.items[frame_index];
            var local_index = frame.locals.items.len;
            while (local_index > 0) {
                local_index -= 1;
                const local = frame.locals.items[local_index];
                if (std.mem.eql(u8, local.name, name)) return local;
            }
        }
        return null;
    }

    fn source(self: *const Codegen) []const u8 {
        return self.views[self.current_view].source;
    }

    fn spanOf(self: *const Codegen, expression: *const ast.Expression) Token.Location {
        return self.checker.expressionSpan(expression);
    }

    fn report(self: *Codegen, span: Token.Location, comptime format: []const u8, arguments: anytype) Error {
        const message = try std.fmt.allocPrint(self.arena, format, arguments);
        try self.diagnostics.append(self.diagnostics_allocator, .{
            .path = self.views[self.current_view].path,
            .source = self.source(),
            .span = span,
            .message = message,
        });
        return error.Unsupported;
    }
};

fn definitionBelongsTo(module: *const ast.Module, definition: *const ast.Definition) bool {
    for (module.definitions) |*candidate| {
        if (candidate == definition) return true;
    }
    return false;
}

const void_type: Type = .void_type;
const u8_type: Type = .{ .primitive = .u8 };
const u16_type: Type = .{ .primitive = .u16 };
const u32_type: Type = .{ .primitive = .u32 };
const u64_type: Type = .{ .primitive = .u64 };
const integer_type: Type = .{ .primitive = .i32 };

fn scalarTypeText(primitive: types.Primitive) []const u8 {
    return switch (primitive) {
        .u8, .i8 => "i8",
        .u16, .i16 => "i16",
        .u32, .i32 => "i32",
        .u64, .i64 => "i64",
        .f32 => "float",
        .f64 => "double",
        .bool => "i1",
    };
}

fn tagTypeText(tag_size: u64) []const u8 {
    return switch (tag_size) {
        1 => "i8",
        2 => "i16",
        else => "i32",
    };
}

fn signedMinimum(primitive: types.Primitive) i64 {
    return switch (primitive) {
        .i8 => std.math.minInt(i8),
        .i16 => std.math.minInt(i16),
        .i32 => std.math.minInt(i32),
        else => std.math.minInt(i64),
    };
}

const CaptureMode = enum { copy, reference, owning };

fn captureMode(capture: ast.Capture) CaptureMode {
    if (capture.modifier) |modifier| {
        return switch (modifier) {
            .reference, .reference_var => .reference,
            .pointer, .pointer_var => .owning,
        };
    }
    if (capture.annotation) |annotation| {
        if (annotation.* == .modified) {
            return switch (annotation.modified.modifier) {
                .reference, .reference_var => .reference,
                .pointer, .pointer_var => .owning,
            };
        }
    }
    return .copy;
}

fn isUntyped(candidate: *const Type) bool {
    return candidate.* == .untyped_integer or candidate.* == .untyped_float;
}

fn unwrapGrouped(expression: *const ast.Expression) *const ast.Expression {
    var current = expression;
    while (current.* == .grouped) current = current.grouped;
    return current;
}
