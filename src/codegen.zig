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
//! tests, downcasts, and match arms compare type identities. Custom
//! iterables drive the cursor protocol (section 4.3), and string subjects
//! match through memcmp.
//!
//! Not yet lowered (clean diagnostics): lambdas and owning interface
//! objects ('*I', which need a virtual drop).

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
    checker: *Checker,
    expression_types: *const std.AutoHashMapUnmanaged(*const ast.Expression, *const Type),
    call_targets: *const std.AutoHashMapUnmanaged(*const ast.Expression, resolution.Symbol),
    call_type_bindings: *const std.AutoHashMapUnmanaged(*const ast.Expression, []const Type.Binding),
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
    // lifted lambda functions (section 4.4), memoized like function_infos so a
    // lambda evaluated more than once emits its body only once
    lambda_infos: std.StringHashMapUnmanaged(*LambdaInfo),
    lambda_queue: std.ArrayList(*LambdaInfo),
    // deep-copy and drop helper functions, memoized by canonical type key
    helper_names: std.StringHashMapUnmanaged([]const u8),
    // runtime type identity globals, one per concrete declared type; the
    // address is the identity an interface object carries (section 5.2)
    type_descriptors: std.AutoHashMapUnmanaged(*const ast.Definition, []const u8),
    fault_helper_emitted: bool,
    global_counter: usize,

    current_view: usize,
    allocas: std.Io.Writer.Allocating,
    body: std.Io.Writer.Allocating,
    temp_counter: usize,
    terminated: bool,
    scopes: std.ArrayList(Frame),
    break_targets: std.ArrayList(BreakTarget),
    return_type: *const Type,
    return_slot: ?[]const u8,
    // the active instance's resolved type parameters (section 3.7); every
    // recorded type substitutes through these before any layout question
    current_bindings: ?*const Checker.TypeEnvironment,

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

    // a lifted lambda (section 4.4). A lambda value is an 8-byte aggregate
    // holding a pointer to a heap environment laid out as a 24-byte header
    // [ fn_ptr@0 | refcount(i64)@8 | drop_ptr@16 ] followed by the captures.
    // The value is an aggregate (not a bare ptr) so it flows through the
    // owning-value machinery: a copy bumps the refcount, a scope-end drop
    // releases it (freeing the environment and its owning captures at zero).
    const ENV_HEADER: u64 = 24;
    const LambdaInfo = struct {
        name: []const u8,
        // the per-lambda environment destructor (drops owning captures, frees)
        drop_name: []const u8,
        lambda: *const ast.Lambda,
        view_index: usize,
        parameter_types: []const *const Type,
        return_type: *const Type,
        aggregate_return: bool,
        captures: []const CaptureSlot,
        env_size: u64,
        env_alignment: u64,
        bindings_environment: ?*const Checker.TypeEnvironment,
    };

    const CaptureSlot = struct {
        name: []const u8,
        // the type stored in the env slot: the captured value for a copy, a
        // reference for '&'/'&var', the owning pointer for '*'/'*var'
        slot_type: *const Type,
        offset: u64,
        mode: CaptureMode,
    };

    const Frame = struct {
        locals: std.ArrayList(Local),
    };

    const Local = struct {
        name: []const u8,
        pointer: []const u8,
        declared_type: *const Type,
        // a borrowed local is not dropped at scope end: a lambda's captures
        // are owned by its environment, not by each invocation (section 4.4)
        borrowed: bool = false,
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
        globals: *const std.StringHashMapUnmanaged(resolution.SymbolList),
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
            .globals = globals,
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
            .lambda_infos = .empty,
            .lambda_queue = .empty,
            .helper_names = .empty,
            .type_descriptors = .empty,
            .fault_helper_emitted = false,
            .global_counter = 0,
            .current_view = 0,
            .allocas = .init(arena),
            .body = .init(arena),
            .temp_counter = 0,
            .terminated = false,
            .scopes = .empty,
            .break_targets = .empty,
            .return_type = &void_type,
            .return_slot = null,
            .current_bindings = null,
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
        // both queues grow as bodies are emitted (calls queue functions,
        // lambda expressions queue lifted functions); drain until settled
        while (self.queue.items.len != 0 or self.lambda_queue.items.len != 0) {
            if (self.queue.pop()) |queued| {
                try self.emitFunction(queued.symbol, queued.info);
            } else if (self.lambda_queue.pop()) |info| {
                try self.emitLambda(info);
            }
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
        writer.print("{s}", .{self.constants.writer.buffered()}) catch return error.OutOfMemory;
        writer.print("{s}", .{self.functions.writer.buffered()}) catch return error.OutOfMemory;
        return module.writer.buffered();
    }

    fn findMain(self: *Codegen) ?resolution.Symbol {
        const symbols = self.globals.get("main") orelse return null;
        for (symbols.items) |candidate| {
            if (candidate.definition.kind == .fn_def) return candidate;
        }
        return null;
    }

    // the process entry adapts the Alloy 'main' result to the C 'int'
    fn emitMainWrapper(self: *Codegen, info: *FunctionInfo) Error!void {
        const writer = &self.functions.writer;
        writer.print("define i32 @main() {{\nentry:\n", .{}) catch return error.OutOfMemory;
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

    fn emitFunction(self: *Codegen, symbol: resolution.Symbol, info: *FunctionInfo) Error!void {
        const fn_def = symbol.definition.kind.fn_def;
        self.current_view = symbol.view_index;
        self.allocas = .init(self.arena);
        self.body = .init(self.arena);
        self.temp_counter = 0;
        self.terminated = false;
        self.scopes = .empty;
        self.break_targets = .empty;
        self.return_type = info.return_type;
        self.return_slot = null;
        self.current_bindings = info.bindings_environment;
        try self.pushFrame();

        var header: std.Io.Writer.Allocating = .init(self.arena);
        const writer = &header.writer;
        const return_text: []const u8 = if (info.aggregate_return)
            "void"
        else switch (try self.classify(info.return_type, fn_def.name.location)) {
            .void_class => "void",
            .scalar => |llvm| llvm,
            .aggregate => "void",
        };
        writer.print("define internal {s} @\"{s}\"(", .{ return_text, info.name }) catch return error.OutOfMemory;
        var first = true;
        if (info.aggregate_return) {
            writer.print("ptr %return.slot", .{}) catch return error.OutOfMemory;
            self.return_slot = "%return.slot";
            first = false;
        }
        for (fn_def.function.parameters, info.parameter_types, 0..) |parameter, parameter_type, index| {
            if (!first) writer.print(", ", .{}) catch return error.OutOfMemory;
            first = false;
            const register = try std.fmt.allocPrint(self.arena, "%argument.{d}", .{index});
            switch (try self.classify(parameter_type, parameter.name.location)) {
                .void_class => return self.report(parameter.name.location, "a parameter cannot have no runtime value", .{}),
                .scalar => |llvm| writer.print("{s} {s}", .{ llvm, register }) catch return error.OutOfMemory,
                .aggregate => writer.print("ptr {s}", .{register}) catch return error.OutOfMemory,
            }
        }
        writer.print(") {{\nentry:\n", .{}) catch return error.OutOfMemory;

        const view_source = self.views[symbol.view_index].source;
        for (fn_def.function.parameters, info.parameter_types, 0..) |parameter, parameter_type, index| {
            const register = try std.fmt.allocPrint(self.arena, "%argument.{d}", .{index});
            const name = parameter.name.slice(view_source);
            switch (try self.classify(parameter_type, parameter.name.location)) {
                .void_class => unreachable,
                .scalar => |llvm| {
                    // a slot makes the parameter addressable like any local
                    const slot = try self.scalarSlot(llvm);
                    try self.storeScalar(slot, .{ .text = register, .llvm = llvm });
                    try self.bindLocal(name, slot, parameter_type);
                },
                .aggregate => try self.bindLocal(name, register, parameter_type),
            }
        }

        try self.execStatement(fn_def.function.body);
        if (!self.terminated) try self.emitDefaultReturn();

        const out = &self.functions.writer;
        out.print("{s}", .{header.writer.buffered()}) catch return error.OutOfMemory;
        out.print("{s}", .{self.allocas.writer.buffered()}) catch return error.OutOfMemory;
        out.print("{s}", .{self.body.writer.buffered()}) catch return error.OutOfMemory;
        out.print("}}\n", .{}) catch return error.OutOfMemory;
    }

    // flow analysis is deferred, so a typed function may still fall through;
    // the fall-through yields the type's zero value
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
                    return self.report(break_stmt.keyword.location, "'break' outside a breakable construct", .{});
                if (break_stmt.value) |value_expression| {
                    const value = try self.evalExpression(value_expression);
                    if (target.slot) |slot| {
                        const coerced = try self.coerceOperand(value, try self.typeOf(value_expression), slot.value_type, break_stmt.keyword.location);
                        try self.storeOperand(slot.pointer, coerced, slot.value_type, break_stmt.keyword.location);
                    } else {
                        try self.dropDiscarded(value, try self.typeOf(value_expression), break_stmt.keyword.location);
                    }
                }
                try self.emitDropsDownTo(target.frame_depth, null);
                try self.instruction("br label %{s}", .{target.exit_label});
                self.terminated = true;
            },
            .return_stmt => |return_stmt| {
                if (return_stmt.value) |value_expression| {
                    const value = try self.evalExpression(value_expression);
                    const coerced = try self.coerceOperand(value, try self.typeOf(value_expression), self.return_type, return_stmt.keyword.location);
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
                return self.report(self.spanOf(expression), "this name has no runtime value here (function values are not yet supported by native code generation)", .{});
            },
            .implied_variant => |token| return self.enumConstruction(expression, token.slice(self.source()), &.{}),
            .member, .index => {
                const place = (try self.evalPlace(expression)) orelse
                    return self.report(self.spanOf(expression), "cannot read this expression", .{});
                return self.loadPlace(place, self.spanOf(expression));
            },
            .comptime_expr => |inner| {
                if (self.comptime_values.get(expression)) |value| {
                    return self.constantFromValue(value, expression);
                }
                return self.evalExpression(inner);
            },
            .unary => return self.evalUnary(expression),
            .binary => return self.evalBinary(expression),
            .cast => return self.evalCast(expression),
            .call => return self.evalCall(expression),
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
            .slice, .array => {
                const instance = if (value == .slice) value.slice else value.array;
                var bytes: std.ArrayList(u8) = .empty;
                for (instance.elements) |element| {
                    if (element != .integer) return self.report(self.spanOf(expression), "this compile-time value cannot be lowered to native code yet", .{});
                    try bytes.append(self.arena, @intCast(@as(u8, @truncate(@as(u128, @bitCast(element.integer.value))))));
                }
                const slice_global = try self.sliceGlobal(bytes.items);
                return .{ .memory = .{ .pointer = slice_global, .layout = .{ .size = 16, .alignment = 8 } } };
            },
            else => return self.report(self.spanOf(expression), "this compile-time value cannot be lowered to native code yet", .{}),
        }
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
                const place = (try self.evalPlace(unary.operand)) orelse
                    return self.report(span, "'&' needs an addressable operand", .{});
                return .{ .scalar = .{ .text = place.pointer, .llvm = "ptr" } };
            },
            .keyword_new => return self.evalNew(expression),
            .keyword_move => {
                // 'move' reads the pointer slot itself and clears the
                // source, transferring ownership (section 4.2)
                const place = (try self.evalPlaceRaw(unary.operand)) orelse
                    return self.report(span, "'move' needs an addressable operand", .{});
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
                const pointee_layout = (try self.layoutQuery(indirection.child, 0)) orelse
                    return self.report(span, "this pointee type has no defined layout", .{});
                const value = try self.evalExpression(operand);
                const coerced = try self.coerceOperand(value, try self.typeOf(operand), indirection.child, span);
                const allocation = try self.freshTemp();
                try self.instruction("{s} = call ptr @\"malloc\"(i64 {d})", .{ allocation, pointee_layout.size });
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
                try self.emitRuntimeFill(data, wide, coerced, element_type, element_layout.size, span);
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
            .string_literal => |token| {
                // 'new "text"': the literal's bytes copy into a fresh '*[u8]'
                // allocation (section 1.6); the literal alone is a static slice
                const text = token.slice(self.source());
                const bytes = try tokenizer_module.unescape(self.arena, text[1 .. text.len - 1]);
                const length = try std.fmt.allocPrint(self.arena, "{d}", .{bytes.len});
                const data = try self.allocateHeapArray(length, element_layout.size);
                const global = try self.byteGlobal(bytes);
                const origin = try std.fmt.allocPrint(self.arena, "@\"{s}\"", .{global.name});
                try self.copyBytes(data, origin, bytes.len * element_layout.size);
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
        try self.declareMalloc();
        const data_size = try self.freshTemp();
        try self.instruction("{s} = mul i64 {s}, {d}", .{ data_size, length, stride });
        const total = try self.freshTemp();
        try self.instruction("{s} = add i64 {s}, 8", .{ total, data_size });
        const base = try self.freshTemp();
        try self.instruction("{s} = call ptr @\"malloc\"(i64 {s})", .{ base, total });
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
                const source_resolved = try self.resolvedOf(try self.typeOf(cast.operand));
                const target_resolved = try self.resolvedOf(try self.typeOf(expression));
                // '&S as &T' views the same memory as another pointee; the
                // pointer is unchanged, so this precedes the shaped path (the
                // shapes recorded for the interpreter do not apply here)
                if (source_resolved.* == .reference and target_resolved.* == .reference) {
                    return try self.evalExpression(cast.operand);
                }
                if (self.cast_shapes.get(expression)) |shapes| {
                    return self.reinterpretShaped(expression, cast.operand, shapes, span);
                }
                const operand = try self.evalExpression(cast.operand);
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
            return switch (symbol.definition.kind) {
                .extern_def => self.callExtern(expression, symbol),
                .fn_def => self.callFunction(expression, symbol, callee),
                else => self.report(span, "this callee is not callable", .{}),
            };
        }
        // a callee with no static target but a function type is a closure
        // value: call indirectly through its pointer (section 4.4)
        if (try self.calleeFunctionType(callee)) |fn_type| {
            return self.callClosure(expression, callee, fn_type, span);
        }
        if (callee.* == .path and callee.path.len >= 2) {
            return self.enumConstruction(expression, callee.path[callee.path.len - 1].slice(self.source()), call.arguments);
        }
        if (callee.* == .member) {
            const name = callee.member.name.slice(self.source());
            if (std.mem.eql(u8, name, "length") and call.arguments.len == 0) {
                return self.lengthCall(callee.member.object, span);
            }
            // a call with no static target dispatches at runtime through
            // the interface object's type identity (section 5.2)
            if (try self.evalPlace(callee.member.object)) |place| {
                if (try self.interfaceOfPlace(place)) |interface| {
                    return self.interfaceDispatch(expression, callee.member, place, interface);
                }
            }
            return self.report(span, "this dynamic call is not yet supported by native code generation", .{});
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

    fn callFunction(self: *Codegen, expression: *const ast.Expression, symbol: resolution.Symbol, callee: *const ast.Expression) Error!Operand {
        const call = expression.call;
        const span = self.spanOf(expression);
        // the call site's inferred bindings (section 3.7) substitute through
        // the caller's own bindings, so nested generics chain concretely
        const raw_bindings = self.call_type_bindings.get(expression) orelse &.{};
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
            .scalar => |llvm| return std.fmt.allocPrint(self.arena, "{s} {s}", .{ llvm, operand.scalar.text }),
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

    fn alignForward(value: u64, alignment: u64) u64 {
        return if (alignment <= 1) value else (value + alignment - 1) / alignment * alignment;
    }

    // the value beneath an indirection, for a copy capture's stored type
    fn pierceValue(self: *Codegen, candidate: *const Type) Error!*const Type {
        const resolved = try self.resolvedOf(candidate);
        return switch (resolved.*) {
            .reference, .pointer => |indirection| indirection.child,
            else => candidate,
        };
    }

    const ResolvedCapture = struct { name: []const u8, slot_type: *const Type, mode: CaptureMode };

    // mirrors checker capture typing (section 2.1): copy by default, '&'/'&var'
    // store a reference to the outer cell, '*'/'*var' move the owning pointer
    fn captureSlot(self: *Codegen, capture: ast.Capture, span: Token.Location) Error!ResolvedCapture {
        const name = capture.name.slice(self.source());
        const outer = self.lookupLocal(name) orelse
            return self.report(span, "the capture '{s}' is not a local here", .{name});
        const mode = captureMode(capture);
        const slot_type: *const Type = switch (mode) {
            // an annotation spells the reference type directly ('|a: &T|');
            // a bare '&'/'&var' modifier references the outer value in place
            .reference => reference: {
                if (capture.annotation) |annotation| {
                    break :reference try self.substituted(try self.checker.typeFromExpressionIn(annotation, self.current_bindings orelse &empty_type_environment, self.current_view));
                }
                const value = try self.pierceValue(outer.declared_type);
                const node = try self.arena.create(Type);
                node.* = .{ .reference = .{ .mutable = capture.modifier != null and capture.modifier.? == .reference_var, .child = value } };
                break :reference node;
            },
            // an owning capture stores the moved pointer itself
            .owning => outer.declared_type,
            // a copy stores the pierced value
            .copy => try self.pierceValue(outer.declared_type),
        };
        return .{ .name = name, .slot_type = slot_type, .mode = mode };
    }

    // a lambda's runtime value is a pointer to a heap environment laid out as
    // [ fn_ptr @ 0 | capture_0 | capture_1 ... ] (section 4.4)
    fn lambdaInfo(self: *Codegen, expression: *const ast.Expression) Error!*LambdaInfo {
        const lambda = &expression.lambda;
        const span = self.spanOf(expression);
        var key: std.Io.Writer.Allocating = .init(self.arena);
        key.writer.print("{d}", .{@intFromPtr(lambda)}) catch return error.OutOfMemory;
        if (self.current_bindings) |bindings| {
            var it = bindings.iterator();
            while (it.next()) |entry| {
                key.writer.print("|{s}={s}", .{ entry.key_ptr.*, try self.typeKey(entry.value_ptr.*, 0) }) catch return error.OutOfMemory;
            }
        }
        if (self.lambda_infos.get(key.writer.buffered())) |existing| return existing;

        const resolved = try self.resolvedOf(try self.typeOf(expression));
        if (resolved.* != .function) return self.report(span, "this lambda has no function type here", .{});
        const fn_type = resolved.function;
        var parameter_types: std.ArrayList(*const Type) = .empty;
        for (fn_type.parameter_types) |parameter_type| {
            const concrete = try self.substituted(parameter_type);
            if (try self.unsupportedReason(concrete, 0)) |reason| {
                return self.report(span, "{s} are not yet supported by native code generation", .{reason});
            }
            try parameter_types.append(self.arena, concrete);
        }
        const return_type = try self.substituted(fn_type.return_type);
        if (try self.unsupportedReason(return_type, 0)) |reason| {
            return self.report(span, "{s} are not yet supported by native code generation", .{reason});
        }

        var captures: std.ArrayList(CaptureSlot) = .empty;
        var offset: u64 = ENV_HEADER;
        var alignment: u64 = 8;
        for (lambda.captures) |capture| {
            const resolved_capture = try self.captureSlot(capture, span);
            const layout = (try self.layoutQuery(resolved_capture.slot_type, 0)) orelse
                return self.report(span, "this capture has no defined layout", .{});
            offset = alignForward(offset, layout.alignment);
            if (layout.alignment > alignment) alignment = layout.alignment;
            try captures.append(self.arena, .{
                .name = resolved_capture.name,
                .slot_type = resolved_capture.slot_type,
                .offset = offset,
                .mode = resolved_capture.mode,
            });
            offset += layout.size;
        }

        const info = try self.arena.create(LambdaInfo);
        const id = self.global_counter;
        self.global_counter += 1;
        info.* = .{
            .name = try std.fmt.allocPrint(self.arena, "alloy.lambda.{d}", .{id}),
            .drop_name = try std.fmt.allocPrint(self.arena, "alloy.lambda.{d}.drop", .{id}),
            .lambda = lambda,
            .view_index = self.current_view,
            .parameter_types = try parameter_types.toOwnedSlice(self.arena),
            .return_type = return_type,
            .aggregate_return = switch (try self.classify(return_type, span)) {
                .aggregate => true,
                else => false,
            },
            .captures = try captures.toOwnedSlice(self.arena),
            .env_size = alignForward(offset, alignment),
            .env_alignment = alignment,
            .bindings_environment = self.current_bindings,
        };
        try self.lambda_infos.put(self.arena, try self.arena.dupe(u8, key.writer.buffered()), info);
        try self.lambda_queue.append(self.arena, info);
        return info;
    }

    fn evalLambda(self: *Codegen, expression: *const ast.Expression) Error!Operand {
        const info = try self.lambdaInfo(expression);
        try self.declareMalloc();
        // the environment is heap-allocated so the closure may outlive the
        // scope that built it (section 4.4); it is reference counted and freed
        // when the last referencing closure value drops (section 4.2)
        const env = try self.freshTemp();
        try self.instruction("{s} = call ptr @\"malloc\"(i64 {d})", .{ env, info.env_size });
        // header: function pointer, refcount = 1, environment destructor
        try self.instruction("store ptr @\"{s}\", ptr {s}", .{ info.name, env });
        const refcount = try self.byteOffset(env, 8);
        try self.instruction("store i64 1, ptr {s}", .{refcount});
        const drop_slot = try self.byteOffset(env, 16);
        try self.instruction("store ptr @\"{s}\", ptr {s}", .{ info.drop_name, drop_slot });
        try self.storeCaptures(info, env);
        // the closure value is an 8-byte aggregate holding the env pointer
        const slot = try self.aggregateSlot(.{ .size = 8, .alignment = 8 });
        try self.instruction("store ptr {s}, ptr {s}", .{ env, slot });
        return .{ .memory = .{ .pointer = slot, .layout = .{ .size = 8, .alignment = 8 }, .fresh = true } };
    }

    // fills the environment at closure-creation time (section 4.4)
    fn storeCaptures(self: *Codegen, info: *LambdaInfo, env: []const u8) Error!void {
        const zero: Token.Location = .{ .start = 0, .end = 0 };
        for (info.captures) |capture| {
            const outer = self.lookupLocal(capture.name) orelse
                return self.report(zero, "the capture '{s}' is not a local here", .{capture.name});
            const dest = try self.byteOffset(env, capture.offset);
            switch (capture.mode) {
                .reference => try self.instruction("store ptr {s}, ptr {s}", .{ outer.pointer, dest }),
                .owning => {
                    // move the pointer in and leave the outer moved-from (null)
                    const moved = try self.loadScalar(outer.pointer, "ptr");
                    try self.instruction("store ptr {s}, ptr {s}", .{ moved, dest });
                    try self.instruction("store ptr null, ptr {s}", .{outer.pointer});
                },
                .copy => {
                    const outer_resolved = try self.resolvedOf(outer.declared_type);
                    // a copy through a pointer/reference captures the pointee
                    // by value (pointee transparency, section 4.2): read the
                    // address, then deep-copy the pointee into the env slot
                    const value: Operand = switch (outer_resolved.*) {
                        .pointer, .reference => |indirection| value: {
                            const address = try self.loadScalar(outer.pointer, "ptr");
                            const layout = (try self.layoutQuery(indirection.child, 0)) orelse Checker.Layout{ .size = 8, .alignment = 8 };
                            break :value Operand{ .memory = .{ .pointer = address, .layout = layout, .fresh = false } };
                        },
                        else => try self.loadPlace(.{ .pointer = outer.pointer, .value_type = outer.declared_type }, zero),
                    };
                    // storeOperand deep-copies a non-fresh owning value (4.2)
                    try self.storeOperand(dest, value, capture.slot_type, zero);
                },
            }
        }
    }

    fn emitLambda(self: *Codegen, info: *LambdaInfo) Error!void {
        const lambda = info.lambda;
        self.current_view = info.view_index;
        self.allocas = .init(self.arena);
        self.body = .init(self.arena);
        self.temp_counter = 0;
        self.terminated = false;
        self.scopes = .empty;
        self.break_targets = .empty;
        self.return_type = info.return_type;
        self.return_slot = null;
        self.current_bindings = info.bindings_environment;
        try self.pushFrame();

        const span: Token.Location = if (lambda.function.parameters.len != 0)
            lambda.function.parameters[0].name.location
        else
            .{ .start = 0, .end = 0 };

        var header: std.Io.Writer.Allocating = .init(self.arena);
        const writer = &header.writer;
        const return_text: []const u8 = if (info.aggregate_return)
            "void"
        else switch (try self.classify(info.return_type, span)) {
            .void_class => "void",
            .scalar => |llvm| llvm,
            .aggregate => "void",
        };
        writer.print("define internal {s} @\"{s}\"(", .{ return_text, info.name }) catch return error.OutOfMemory;
        if (info.aggregate_return) {
            writer.print("ptr %return.slot, ", .{}) catch return error.OutOfMemory;
            self.return_slot = "%return.slot";
        }
        // the environment pointer is the first ordinary parameter
        writer.print("ptr %env", .{}) catch return error.OutOfMemory;
        for (lambda.function.parameters, info.parameter_types, 0..) |parameter, parameter_type, index| {
            writer.print(", ", .{}) catch return error.OutOfMemory;
            const register = try std.fmt.allocPrint(self.arena, "%argument.{d}", .{index});
            switch (try self.classify(parameter_type, parameter.name.location)) {
                .void_class => return self.report(parameter.name.location, "a parameter cannot have no runtime value", .{}),
                .scalar => |llvm| writer.print("{s} {s}", .{ llvm, register }) catch return error.OutOfMemory,
                .aggregate => writer.print("ptr {s}", .{register}) catch return error.OutOfMemory,
            }
        }
        writer.print(") {{\nentry:\n", .{}) catch return error.OutOfMemory;

        try self.bindCaptures(info);

        const view_source = self.views[info.view_index].source;
        for (lambda.function.parameters, info.parameter_types, 0..) |parameter, parameter_type, index| {
            const register = try std.fmt.allocPrint(self.arena, "%argument.{d}", .{index});
            const name = parameter.name.slice(view_source);
            switch (try self.classify(parameter_type, parameter.name.location)) {
                .void_class => unreachable,
                .scalar => |llvm| {
                    const slot = try self.scalarSlot(llvm);
                    try self.storeScalar(slot, .{ .text = register, .llvm = llvm });
                    try self.bindLocal(name, slot, parameter_type);
                },
                .aggregate => try self.bindLocal(name, register, parameter_type),
            }
        }

        try self.execStatement(lambda.function.body);
        if (!self.terminated) try self.emitDefaultReturn();

        const out = &self.functions.writer;
        out.print("{s}", .{header.writer.buffered()}) catch return error.OutOfMemory;
        out.print("{s}", .{self.allocas.writer.buffered()}) catch return error.OutOfMemory;
        out.print("{s}", .{self.body.writer.buffered()}) catch return error.OutOfMemory;
        out.print("}}\n", .{}) catch return error.OutOfMemory;

        try self.emitLambdaDrop(info);
    }

    // the per-lambda environment destructor: drops each owning capture, then
    // frees the environment. Invoked by the generic closure drop when the
    // refcount reaches zero (section 4.2).
    fn emitLambdaDrop(self: *Codegen, info: *LambdaInfo) Error!void {
        try self.declareFree();
        const span: Token.Location = .{ .start = 0, .end = 0 };
        const saved = self.beginHelperFunction();
        for (info.captures) |capture| {
            if (!try self.ownsHeap(capture.slot_type, 0)) continue;
            const child_drop = try self.dropHelper(capture.slot_type, span);
            const pointer = try self.byteOffset("%env", capture.offset);
            try self.instruction("call void @\"{s}\"(ptr {s})", .{ child_drop, pointer });
        }
        try self.instruction("call void @\"free\"(ptr %env)", .{});
        try self.instruction("ret void", .{});
        self.terminated = true;
        const header = try std.fmt.allocPrint(self.arena, "define internal void @\"{s}\"(ptr %env) {{\nentry:\n", .{info.drop_name});
        try self.finishHelperFunction(saved, header);
    }

    // binds each capture from the environment at the lifted function's entry.
    // Captures are borrowed: the environment owns them and drops them once, so
    // the invocation's scope-end must not drop them (section 4.4).
    fn bindCaptures(self: *Codegen, info: *LambdaInfo) Error!void {
        for (info.captures) |capture| {
            const src = try self.byteOffset("%env", capture.offset);
            switch (capture.mode) {
                .reference, .owning => {
                    // the slot holds a pointer; load it into an addressable slot
                    const slot = try self.scalarSlot("ptr");
                    const loaded = try self.loadScalar(src, "ptr");
                    try self.storeScalar(slot, .{ .text = loaded, .llvm = "ptr" });
                    try self.bindBorrowedLocal(capture.name, slot, capture.slot_type);
                },
                // a copy lives inline in the env: its slot address is its place
                .copy => try self.bindBorrowedLocal(capture.name, src, capture.slot_type),
            }
        }
    }

    // the function type of a closure-valued callee. A bare local of function
    // type is not recorded by the checker (it calls callFunctionValue without
    // re-checking the callee), so it is looked up directly; every other callee
    // form has its type recorded.
    fn calleeFunctionType(self: *Codegen, callee: *const ast.Expression) Error!?Type.Function {
        if (callee.* == .path and callee.path.len == 1) {
            if (self.lookupLocal(callee.path[0].slice(self.source()))) |local| {
                const resolved = try self.resolvedOf(local.declared_type);
                if (resolved.* == .function) return resolved.function;
            }
        }
        if (self.expression_types.get(callee)) |recorded| {
            const resolved = try self.resolvedOf(try self.substituted(recorded));
            if (resolved.* == .function) return resolved.function;
        }
        return null;
    }

    // calls a closure value indirectly through the function pointer stored at
    // the head of its environment (section 4.4)
    fn callClosure(self: *Codegen, expression: *const ast.Expression, callee: *const ast.Expression, fn_type: Type.Function, span: Token.Location) Error!Operand {
        const call = expression.call;
        // the closure value is an 8-byte aggregate holding the environment
        // pointer; load the environment, then the function pointer at its head
        const closure_value = try self.evalExpression(callee);
        const closure_slot = switch (closure_value) {
            .memory => |memory| memory.pointer,
            else => return self.report(span, "internal: a closure callee is not an aggregate", .{}),
        };
        const env = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr {s}", .{ env, closure_slot });
        const fn_ptr = try self.freshTemp();
        try self.instruction("{s} = load ptr, ptr {s}", .{ fn_ptr, env });

        const return_type = try self.substituted(fn_type.return_type);
        const aggregate_return = switch (try self.classify(return_type, span)) {
            .aggregate => true,
            else => false,
        };

        var lowered: std.ArrayList([]const u8) = .empty;
        var result_slot: ?[]const u8 = null;
        if (aggregate_return) {
            const layout = (try self.layoutQuery(return_type, 0)) orelse
                return self.report(span, "this return type has no defined layout", .{});
            result_slot = try self.aggregateSlot(layout);
            try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{result_slot.?}));
        }
        // the environment is the first ordinary argument
        try lowered.append(self.arena, try std.fmt.allocPrint(self.arena, "ptr {s}", .{env}));
        for (call.arguments, 0..) |argument, index| {
            const parameter_type = if (index < fn_type.parameter_types.len)
                try self.substituted(fn_type.parameter_types[index])
            else
                try self.typeOf(argument);
            const value = try self.evalExpression(argument);
            const coerced = try self.coerceOperand(value, try self.typeOf(argument), parameter_type, self.spanOf(argument));
            try lowered.append(self.arena, try self.lowerArgument(coerced, parameter_type, self.spanOf(argument)));
        }
        const joined = try self.joinArguments(lowered.items);
        if (aggregate_return) {
            try self.instruction("call void {s}({s})", .{ fn_ptr, joined });
            const layout = (try self.layoutQuery(return_type, 0)).?;
            return .{ .memory = .{ .pointer = result_slot.?, .layout = layout, .fresh = true } };
        }
        switch (try self.classify(return_type, span)) {
            .void_class => {
                try self.instruction("call void {s}({s})", .{ fn_ptr, joined });
                return .none;
            },
            .scalar => |llvm| {
                const result = try self.freshTemp();
                try self.instruction("{s} = call {s} {s}({s})", .{ result, llvm, fn_ptr, joined });
                return .{ .scalar = .{ .text = result, .llvm = llvm } };
            },
            .aggregate => unreachable,
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
        try self.emitFillLoop(storage, coerced, element_type, element_layout.size, resolved.fixed_array.length, span);
        return .{ .memory = .{ .pointer = storage, .layout = layout, .fresh = true } };
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

        try self.break_targets.append(self.arena, .{ .exit_label = exit_label, .slot = slot, .frame_depth = self.scopes.items.len });
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
        _ = self.break_targets.pop();
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
        try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, then_label, miss_label });
        self.terminated = true;
        const payload_type = frame.variants[variant_index].payload orelse
            return .{ .capture = capture, .payload_pointer = place.pointer, .payload_type = &void_type };
        return .{
            .capture = capture,
            .payload_pointer = try self.byteOffset(place.pointer, frame.payload_offset),
            .payload_type = payload_type,
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
        return self.slotOperand(slot, span);
    }

    // the cursor protocol (section 4.3): 'subject.iterator()' yields a
    // cursor advanced by 'next()' until it reports 'None'
    fn evalForCursor(self: *Codegen, expression: *const ast.Expression, for_expr: ast.ForExpression, subject_type: *const Type) Error!Operand {
        const span = self.spanOf(expression);
        const slot = try self.resultSlot(expression);
        const protocol = (try self.checker.cursorProtocolOf(subject_type)) orelse
            return self.report(span, "this subject is not iterable: provide 'iterator()' and 'next()' extension functions (section 4.3)", .{});
        const subject_expression = for_expr.subjects[0];
        const place = (try self.evalPlace(subject_expression)) orelse place: {
            const value = try self.evalExpression(subject_expression);
            const memory = try self.ensureMemory(value, subject_type, span);
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

        try self.break_targets.append(self.arena, .{ .exit_label = exit, .slot = slot, .frame_depth = self.scopes.items.len });
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
        _ = self.break_targets.pop();
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

    // an interface-object arm names a concrete type (section 4.3)
    fn patternDescriptor(self: *Codegen, pattern: *const ast.Expression) Error!Descriptor {
        const span = self.spanOf(pattern);
        const unwrapped = unwrapGrouped(pattern);
        if (unwrapped.* != .path or unwrapped.path.len != 1) {
            return self.report(span, "this arm must name a concrete type (section 4.3)", .{});
        }
        const name = unwrapped.path[0].slice(self.source());
        const symbols = self.globals.get(name) orelse
            return self.report(span, "'{s}' names no type here", .{name});
        const symbol = symbols.items[0];
        if (symbol.definition.kind != .type_def) {
            return self.report(span, "'{s}' is not a type", .{name});
        }
        const concrete = try self.arena.create(Type);
        concrete.* = .{ .declared = .{
            .definition = symbol.definition,
            .view_index = symbol.view_index,
            .name = name,
            .arguments = &.{},
        } };
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
                // an owning capture takes the pointer payload out, leaving
                // the source moved-from (section 2.1)
                const resolved = try self.resolvedOf(payload_type);
                if (resolved.* != .pointer and resolved.* != .heap_array) {
                    return self.report(span, "an owning capture needs a pointer payload (section 2.1)", .{});
                }
                const slot = try self.scalarSlot("ptr");
                const taken = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr {s}", .{ taken, payload_pointer });
                try self.instruction("store ptr null, ptr {s}", .{payload_pointer});
                try self.storeScalar(slot, .{ .text = taken, .llvm = "ptr" });
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
            // a closure value is an 8-byte aggregate (a pointer to its heap
            // environment) so it flows through the owning-value machinery
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
            // a function value is a closure pointer (section 4.4), supported
            .function => return null,
            .type_parameter => return "generics",
            // reachable through '*I' (the pointer arm recurses): dropping
            // through an interface needs a virtual drop, deferred for now
            .interface => return "owning interface objects ('*I')",
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
            // a closure owns its reference-counted heap environment (4.4)
            .pointer, .heap_array, .function => return true,
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
            .function => return "fn",
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
    };

    // helper functions generate while another function is mid-emission, so
    // the per-function buffers swap out and back
    fn beginHelperFunction(self: *Codegen) SavedFunction {
        const saved: SavedFunction = .{
            .allocas = self.allocas,
            .body = self.body,
            .temp_counter = self.temp_counter,
            .terminated = self.terminated,
        };
        self.allocas = .init(self.arena);
        self.body = .init(self.arena);
        self.temp_counter = 0;
        self.terminated = false;
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
    }

    fn declareMalloc(self: *Codegen) Error!void {
        if (!self.extern_declarations.contains("malloc")) {
            try self.extern_declarations.put(self.arena, "malloc", "declare ptr @\"malloc\"(i64)");
        }
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
            .function => {
                // a closure holds its environment pointer; decrement the
                // environment's refcount and run its destructor at zero (4.4)
                const env = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr %value", .{env});
                const live = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, env });
                const live_label = try self.freshLabel("live");
                const done_label = try self.freshLabel("done");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                self.terminated = true;
                try self.startBlock(live_label);
                const rc_slot = try self.byteOffset(env, 8);
                const rc = try self.freshTemp();
                try self.instruction("{s} = load i64, ptr {s}", .{ rc, rc_slot });
                const dec = try self.freshTemp();
                try self.instruction("{s} = sub i64 {s}, 1", .{ dec, rc });
                try self.instruction("store i64 {s}, ptr {s}", .{ dec, rc_slot });
                const dead = try self.freshTemp();
                try self.instruction("{s} = icmp eq i64 {s}, 0", .{ dead, dec });
                const free_label = try self.freshLabel("free");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ dead, free_label, done_label });
                self.terminated = true;
                try self.startBlock(free_label);
                const drop_ptr = try self.freshTemp();
                const drop_slot = try self.byteOffset(env, 16);
                try self.instruction("{s} = load ptr, ptr {s}", .{ drop_ptr, drop_slot });
                try self.instruction("call void {s}(ptr {s})", .{ drop_ptr, env });
                try self.startBlock(done_label);
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
                const allocation = try self.freshTemp();
                try self.instruction("{s} = call ptr @\"malloc\"(i64 {d})", .{ allocation, pointee_layout.size });
                if (try self.ownsHeap(indirection.child, 0)) {
                    const child_copy = try self.copyHelper(indirection.child, span);
                    try self.instruction("call void @\"{s}\"(ptr {s}, ptr {s})", .{ child_copy, allocation, loaded });
                } else {
                    try self.copyBytes(allocation, loaded, pointee_layout.size);
                }
                try self.instruction("store ptr {s}, ptr %destination", .{allocation});
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
                const new_base = try self.freshTemp();
                try self.instruction("{s} = call ptr @\"malloc\"(i64 {s})", .{ new_base, total });
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
            .function => {
                // a closure copy shares the environment and bumps its refcount
                // (section 4.2); the destructor frees it once, at the last drop
                const env = try self.freshTemp();
                try self.instruction("{s} = load ptr, ptr %origin", .{env});
                try self.instruction("store ptr {s}, ptr %destination", .{env});
                const live = try self.freshTemp();
                try self.instruction("{s} = icmp ne ptr {s}, null", .{ live, env });
                const live_label = try self.freshLabel("live");
                const done_label = try self.freshLabel("done");
                try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ live, live_label, done_label });
                self.terminated = true;
                try self.startBlock(live_label);
                const rc_slot = try self.byteOffset(env, 8);
                const rc = try self.freshTemp();
                try self.instruction("{s} = load i64, ptr {s}", .{ rc, rc_slot });
                const inc = try self.freshTemp();
                try self.instruction("{s} = add i64 {s}, 1", .{ inc, rc });
                try self.instruction("store i64 {s}, ptr {s}", .{ inc, rc_slot });
                try self.startBlock(done_label);
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
        const name = try std.fmt.allocPrint(self.arena, "alloy.type.{d}", .{self.global_counter});
        self.global_counter += 1;
        self.constants.writer.print("@\"{s}\" = private constant i8 0\n", .{name}) catch return error.OutOfMemory;
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
    };

    // the merged unit is the whole program, so the implementers of an
    // interface form a closed world the dispatch chain can enumerate
    fn interfaceImplementers(self: *Codegen, interface: Type.Interface) Error![]const Implementer {
        var implementers: std.ArrayList(Implementer) = .empty;
        for (self.views, 0..) |view, view_index| {
            for (view.module.definitions) |*definition| {
                if (definition.kind != .type_def) continue;
                const type_def = definition.kind.type_def;
                for (type_def.interfaces) |marker| {
                    const symbols = self.globals.get(marker.slice(view.source)) orelse continue;
                    if (symbols.items[0].definition != interface.definition) continue;
                    try implementers.append(self.arena, .{
                        .definition = definition,
                        .view_index = view_index,
                        .name = type_def.name.slice(view.source),
                    });
                    break;
                }
            }
        }
        return implementers.toOwnedSlice(self.arena);
    }

    // the extension implementing 'name' for the concrete type, mirroring
    // the interpreter's runtime dispatch by receiver type (section 5.2)
    fn findTypeExtension(self: *Codegen, type_definition: *const ast.Definition, name: []const u8) Error!?resolution.Symbol {
        const symbols = self.globals.get(name) orelse return null;
        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            if (fn_def.function.parameters.len == 0 or !fn_def.function.parameters[0].is_self) continue;
            const self_type = try self.checker.typeFromExpressionIn(fn_def.function.parameters[0].parameter_type, &empty_type_environment, symbol.view_index);
            const pierced = try self.checker.pierce(self_type);
            const resolved = try self.checker.resolveAlias(pierced);
            if (resolved.* == .declared and resolved.declared.definition == type_definition) return symbol;
        }
        return null;
    }

    // the default implementation: an extension whose receiver is the
    // interface itself (section 5.2)
    fn findInterfaceDefault(self: *Codegen, interface: Type.Interface, name: []const u8) Error!?resolution.Symbol {
        const symbols = self.globals.get(name) orelse return null;
        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            if (fn_def.function.parameters.len == 0 or !fn_def.function.parameters[0].is_self) continue;
            const self_type = try self.checker.typeFromExpressionIn(fn_def.function.parameters[0].parameter_type, &empty_type_environment, symbol.view_index);
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
            const target = (try self.findTypeExtension(implementer.definition, name)) orelse continue;
            const descriptor = try self.typeDescriptor(implementer.definition);
            const arm = try self.freshLabel("dispatch.arm");
            const miss = try self.freshLabel("dispatch.miss");
            const matches = try self.freshTemp();
            try self.instruction("{s} = icmp eq ptr {s}, @\"{s}\"", .{ matches, type_id, descriptor });
            try self.instruction("br i1 {s}, label %{s}, label %{s}", .{ matches, arm, miss });
            self.terminated = true;
            try self.startBlock(arm);
            try self.emitDispatchArm(target, "ptr", data, argument_operands.items, argument_types.items, slot, span);
            try self.instruction("br label %{s}", .{exit});
            self.terminated = true;
            try self.startBlock(miss);
        }
        if (try self.findInterfaceDefault(interface, name)) |default_symbol| {
            // the default implementation's receiver is the interface
            // object: the fat pair passes whole (section 5.2)
            try self.emitDispatchArm(default_symbol, "ptr", place.pointer, argument_operands.items, argument_types.items, slot, span);
            try self.instruction("br label %{s}", .{exit});
            self.terminated = true;
        } else {
            try self.emitFault("runtime fault: no implementation reachable for this interface call (section 5.2)");
        }
        try self.startBlock(exit);
        return self.slotOperand(slot, span);
    }

    fn emitDispatchArm(self: *Codegen, symbol: resolution.Symbol, receiver_llvm: []const u8, receiver_text: []const u8, argument_operands: []const Operand, argument_types: []const *const Type, slot: ?Slot, span: Token.Location) Error!void {
        const info = try self.functionInfo(symbol, span, &.{});
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
        // a reference to a concrete implementer converts to an interface
        // object: the fat pair carries the data pointer and the type
        // identity (section 5.2)
        if (to_resolved.* == .reference and (try self.resolvedOf(to_resolved.reference.child)).* == .interface) {
            if (from_resolved.* == .reference) {
                const pointee = try self.resolvedOf(from_resolved.reference.child);
                if (pointee.* == .interface) return operand;
                if (pointee.* != .declared) {
                    return self.report(span, "only named types convert to interface objects (section 5.2)", .{});
                }
                const descriptor = try self.typeDescriptor(pointee.declared.definition);
                const pair = try self.aggregateSlot(.{ .size = 16, .alignment = 8 });
                try self.instruction("store ptr {s}, ptr {s}", .{ operand.scalar.text, pair });
                const identity_pointer = try self.byteOffset(pair, 8);
                try self.instruction("store ptr @\"{s}\", ptr {s}", .{ descriptor, identity_pointer });
                return .{ .memory = .{ .pointer = pair, .layout = .{ .size = 16, .alignment = 8 }, .fresh = true } };
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
        self.body.writer.print("  " ++ format ++ "\n", arguments) catch return error.OutOfMemory;
    }

    fn startBlock(self: *Codegen, label: []const u8) Error!void {
        if (!self.terminated) {
            self.body.writer.print("  br label %{s}\n", .{label}) catch return error.OutOfMemory;
        }
        self.body.writer.print("{s}:\n", .{label}) catch return error.OutOfMemory;
        self.terminated = false;
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
        try self.scopes.append(self.arena, .{ .locals = .empty });
    }

    fn popFrame(self: *Codegen) void {
        _ = self.scopes.pop();
    }

    // scope-end drop (section 4.2): owning locals release their heap when
    // the frame closes normally; a terminated frame already dropped on its
    // break or return path
    fn closeFrame(self: *Codegen) Error!void {
        if (!self.terminated) {
            try self.emitFrameDrops(&self.scopes.items[self.scopes.items.len - 1], null);
        }
        _ = self.scopes.pop();
    }

    fn emitFrameDrops(self: *Codegen, frame: *const Frame, skip_pointer: ?[]const u8) Error!void {
        var index = frame.locals.items.len;
        while (index > 0) {
            index -= 1;
            const local = frame.locals.items[index];
            if (local.borrowed) continue;
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
    }

    // binds a local the scope must not drop (a borrowed capture, section 4.4)
    fn bindBorrowedLocal(self: *Codegen, name: []const u8, pointer: []const u8, declared_type: *const Type) Error!void {
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        try frame.locals.append(self.arena, .{ .name = name, .pointer = pointer, .declared_type = declared_type, .borrowed = true });
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

const void_type: Type = .void_type;
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
