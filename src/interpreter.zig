//! Tree-walking interpreter over the checked merged unit, the execution
//! stage behind 'alloyc run' and, later, compile-time evaluation (section
//! 6). It consumes the checker's side tables (expression types and resolved
//! call targets) so no semantic analysis is repeated here.
//!
//! Subset covered: integer/float/bool arithmetic with overflow faults
//! (section 4.2), strings, structs, enums with payloads, fixed and heap
//! arrays, slices, pointers with move semantics and deep-copy uniqueness,
//! all control flow including value yielding, ranges, extension calls,
//! lambdas with captured environments, interface-object dispatch with
//! downcasting, the cursor protocol for custom iterables, and a printf
//! extern. The checker also drives this interpreter in a sandboxed mode
//! (comptime_mode plus step_budget) as the section 6 compile-time
//! evaluator, where macros, the built-in macros, and '#Type' reflection
//! methods (section 3.4) are available through the reflection hooks.
//! 'as' beyond primitives (section 3.5) runs through checker-recorded byte
//! shapes (types.Shape): the value serializes into its little-endian
//! C-layout image (section 3.9) and reads back as the target. Deliberately
//! deferred: 'as' on references ('&S as &T' aliases memory, which value
//! semantics cannot model) and on pointer-bearing values.

const std = @import("std");
const tokenizer_module = @import("tokenizer.zig");
const Token = tokenizer_module.Token;
const ast = @import("ast.zig");
const types = @import("types.zig");
const Type = types.Type;
const resolution = @import("resolution.zig");

pub const Interpreter = struct {
    arena: std.mem.Allocator,
    views: []const resolution.ModuleView,
    globals: *const std.StringHashMapUnmanaged(resolution.SymbolList),
    injected: []const resolution.InjectedMap,
    expression_types: *const std.AutoHashMapUnmanaged(*const ast.Expression, *const Type),
    call_targets: *const std.AutoHashMapUnmanaged(*const ast.Expression, resolution.Symbol),
    call_type_bindings: *const std.AutoHashMapUnmanaged(*const ast.Expression, []const Type.Binding),
    // values precomputed by compile-time evaluation (section 6.1), keyed by
    // the '#' expression node; runtime substitutes instead of re-evaluating
    comptime_values: *const std.AutoHashMapUnmanaged(*const ast.Expression, Value),
    // byte shapes per non-primitive 'as' cast (section 3.5), recorded by
    // the checker from the section 3.9 layout rules
    cast_shapes: *const std.AutoHashMapUnmanaged(*const ast.Expression, types.CastShapes),
    type_targets: *const std.AutoHashMapUnmanaged(*const ast.Expression, types.TypeIdentity),
    output: *std.Io.Writer,
    scopes: std.ArrayList(Frame),
    current_view: usize,
    // the active call's resolved type arguments (section 3.7)
    current_type_bindings: []const Type.Binding,
    // compile-time sandboxing (section 6.2): externs fault and evaluation
    // is bounded by a step budget
    comptime_mode: bool,
    step_budget: ?u64,
    reflection: ?ReflectionHooks,
    call_depth: usize,
    debug_hook: ?DebugHook = null,
    debug_stack: std.ArrayList(DebugFrame) = .empty,
    // host filesystem behind the std::io externs (fopen and friends);
    // null makes file externs fault, keeping tests hermetic
    host_io: ?std.Io = null,
    // call nodes whose bare reference result pierces to a copy at the use
    // site (section 4.2 reference-binding explicitness), from the checker
    pierced_results: ?*const std.AutoHashMapUnmanaged(*const ast.Expression, void) = null,
    open_files: std.ArrayList(OpenFile) = .empty,
    // the argv tail behind std::process::arguments (section 5.1a)
    process_arguments: []const []const u8 = &.{},
    fault_message: ?[]const u8,
    pending_return: ?Value,
    // carried by error.Break / error.Yield across expression boundaries
    pending_break: ?Value = null,
    pending_yield: Value = .void_value,

    const Frame = struct {
        bindings: std.StringHashMapUnmanaged(*Value),
        // a barrier frame starts a function body; lookups stop there
        barrier: bool,
    };

    // one fopen'd stream: reads run over the whole file buffered at open,
    // writes buffer until fclose flushes them to disk, matching C stdio
    // closely enough for the std::io wrappers
    const OpenFile = struct {
        path: []const u8,
        writing: bool,
        contents: []const u8,
        cursor: usize,
        write_buffer: std.ArrayList(u8),
        closed: bool,
    };

    /// A debugger's view of one active call, maintained only while a
    /// debug hook is installed; the hook updates the top frame's offset.
    pub const DebugFrame = struct {
        name: []const u8,
        view_index: usize,
        offset: usize,
        // the scope stack height at call entry, so a debugger can slice
        // each call's frames out of the shared stack
        scope_floor: usize,
    };

    /// Called before every statement while installed, so a debug adapter
    /// can pause execution, inspect scopes, and resume.
    pub const DebugHook = struct {
        context: *anyopaque,
        on_statement: *const fn (context: *anyopaque, machine: *Interpreter, statement: *const ast.Statement) Error!void,
    };

    // 'Return' unwinds a 'return' written inside a value-yielding construct
    // up to the enclosing function call, carrying 'pending_return'
    pub const Error = error{ OutOfMemory, RuntimeFault, WriteFailed, Return, Break, Yield };

    pub const Value = union(enum) {
        void_value,
        integer: Integer,
        float: Float,
        bool_value: bool,
        struct_value: *StructInstance,
        enum_value: *EnumInstance,
        // a fixed array value: deep copies on assignment
        array: *ArrayInstance,
        // an owned '*[T]' allocation: unique, deep copies, null = moved-from
        heap_array: ?*ArrayInstance,
        // a '&[T]' view: copies alias
        slice: *ArrayInstance,
        // a '*T' allocation: unique, null = moved-from
        pointer: ?*Value,
        // a '&T' borrow
        reference: *Value,
        function: resolution.Symbol,
        closure: *Closure,
        // a '#Type' reflection value (section 3.4), compile time only
        type_value: *TypeDescription,

        pub const Integer = struct {
            // wide enough for u64 and i64 alike; the primitive bounds it
            value: i128,
            primitive: ?types.Primitive,
        };

        pub const Float = struct {
            value: f64,
            primitive: ?types.Primitive,
        };

        pub const StructInstance = struct {
            fields: []Field,
            // the declared type's name; empty for structural values; kept
            // for display in faults and reflection
            type_name: []const u8,
            // the nominal identity behind interface dispatch, downcasts,
            // and match arms (section 5.2); null for structural values
            identity: ?types.TypeIdentity = null,
        };

        pub const Field = struct {
            name: []const u8,
            value: Value,
        };

        pub const EnumInstance = struct {
            variant: []const u8,
            payload: ?Value,
        };

        pub const ArrayInstance = struct {
            elements: []Value,
        };

        // a lambda with its captured environment (section 4.4); each entry
        // keeps its capture mode so copies of the closure know what to clone
        pub const Closure = struct {
            lambda: *const ast.Lambda,
            view_index: usize,
            environment: []EnvironmentEntry,
            // the enclosing call's type bindings at creation (section 3.7)
            type_bindings: []const Type.Binding,
        };

        pub const EnvironmentEntry = struct {
            name: []const u8,
            cell: *Value,
            mode: CaptureMode,
        };

        // a first-class, mutable description of a type (section 3.4); the
        // checker reflects real types in, macros may synthesise fresh ones
        pub const TypeDescription = struct {
            kind: Kind,
            name: []const u8,
            primitive: ?types.Primitive,
            members: std.ArrayList(Member),
            interface_names: []const []const u8,
            // the checker type this reflects; null for synthesised values
            origin: ?*const Type,

            pub const Kind = enum { struct_kind, enum_kind, primitive_kind, interface_kind, other_kind };

            pub const Member = struct {
                name: []const u8,
                description: *TypeDescription,
            };
        };
    };

    // the checker supplies these so compile-time code can reflect on types
    // (section 3.4) without the interpreter re-implementing type resolution
    pub const ReflectionHooks = struct {
        context: *anyopaque,
        reflect_named: *const fn (context: *anyopaque, name: []const u8) error{OutOfMemory}!?*Value.TypeDescription,
        reflect_expression: *const fn (context: *anyopaque, expression: *const ast.Expression) error{OutOfMemory}!?*Value.TypeDescription,
        // every type in the merged unit implementing the described
        // interface (section 6.4); null when the description is not an
        // interface
        reflect_implementers: *const fn (context: *anyopaque, description: *Value.TypeDescription) error{OutOfMemory}!?[]const Value,
        // whether the subject type implements the interface (section 3.4),
        // by definition identity including lang items; null when the
        // interface description carries no resolved interface
        reflect_implements: *const fn (context: *anyopaque, subject: *Value.TypeDescription, interface: *Value.TypeDescription) error{OutOfMemory}!?bool,
    };

    // statements propagate flow directly; crossing an expression boundary
    // switches to the error channel (error.Break / error.Yield /
    // error.Return with the pending value), so 'break' reaches the
    // enclosing loop and 'yield' the enclosing value construct no matter
    // how deeply they nest (section 4.3)
    const Flow = union(enum) {
        normal,
        break_value: ?Value,
        yield_value: Value,
        return_value: Value,
    };

    pub fn init(
        arena: std.mem.Allocator,
        views: []const resolution.ModuleView,
        unit: *const resolution.MergedUnit,
        expression_types: *const std.AutoHashMapUnmanaged(*const ast.Expression, *const Type),
        call_targets: *const std.AutoHashMapUnmanaged(*const ast.Expression, resolution.Symbol),
        call_type_bindings: *const std.AutoHashMapUnmanaged(*const ast.Expression, []const Type.Binding),
        comptime_values: *const std.AutoHashMapUnmanaged(*const ast.Expression, Value),
        cast_shapes: *const std.AutoHashMapUnmanaged(*const ast.Expression, types.CastShapes),
        type_targets: *const std.AutoHashMapUnmanaged(*const ast.Expression, types.TypeIdentity),
        output: *std.Io.Writer,
    ) Interpreter {
        return .{
            .arena = arena,
            .views = views,
            .globals = &unit.globals,
            .injected = unit.injected,
            .expression_types = expression_types,
            .call_targets = call_targets,
            .call_type_bindings = call_type_bindings,
            .comptime_values = comptime_values,
            .cast_shapes = cast_shapes,
            .type_targets = type_targets,
            .output = output,
            .scopes = .empty,
            .current_view = 0,
            .current_type_bindings = &.{},
            .comptime_mode = false,
            .step_budget = null,
            .reflection = null,
            .call_depth = 0,
            .fault_message = null,
            .pending_return = null,
        };
    }

    // unqualified visibility from the executing module (section 5.4): own
    // library plus the 'exp' symbols of libraries it imported without an
    // alias, mirroring the checker's rule
    fn symbolVisible(self: *const Interpreter, symbol: resolution.Symbol) bool {
        if (resolution.sameLibrary(self.views[self.current_view].library, self.views[symbol.view_index].library)) return true;
        if (symbol.visibility != .exported) return false;
        const library = self.views[symbol.view_index].library orelse return false;
        return self.injected[self.current_view].contains(library);
    }

    fn firstVisible(self: *const Interpreter, name: []const u8) ?resolution.Symbol {
        return self.firstVisibleFrom(name, self.current_view);
    }

    fn firstVisibleFrom(self: *const Interpreter, name: []const u8, view_index: usize) ?resolution.Symbol {
        const symbols = self.globals.get(name) orelse return null;
        for (symbols.items) |symbol| {
            if (resolution.sameLibrary(self.views[view_index].library, self.views[symbol.view_index].library)) return symbol;
            if (symbol.visibility != .exported) continue;
            const library = self.views[symbol.view_index].library orelse continue;
            if (self.injected[view_index].contains(library)) return symbol;
        }
        return null;
    }

    /// Runs 'main' and returns its integer result (the process exit code).
    pub fn run(self: *Interpreter) Error!i64 {
        const symbols = self.globals.get("main") orelse return self.fault("no 'main' function", .{});
        const symbol = for (symbols.items) |candidate| {
            if (candidate.definition.kind == .fn_def) break candidate;
        } else return self.fault("no 'main' function", .{});
        const result = try self.callFunction(symbol, &.{}, &.{});
        return switch (result) {
            .integer => |integer| @intCast(integer.value),
            else => 0,
        };
    }

    /// Evaluates one expression, resolving an inner 'return' to its value.
    /// The compile-time evaluator (section 6.1) drives this entry point.
    pub fn evaluate(self: *Interpreter, expression: *const ast.Expression) Error!Value {
        return self.evalExpression(expression) catch |err| switch (err) {
            error.Return => self.pending_return orelse .void_value,
            else => err,
        };
    }

    /// Opens the compile-time environment frame for bindValue/unbindValue.
    pub fn pushEnvironment(self: *Interpreter) Error!void {
        try self.pushFrame(true);
    }

    pub fn bindValue(self: *Interpreter, name: []const u8, value: Value) Error!void {
        try self.bind(name, value);
    }

    /// Poisons a name (a runtime binding shadowing a compile-time constant).
    pub fn unbindValue(self: *Interpreter, name: []const u8) void {
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        _ = frame.bindings.remove(name);
    }

    fn fault(self: *Interpreter, comptime format: []const u8, arguments: anytype) Error {
        self.fault_message = std.fmt.allocPrint(self.arena, format, arguments) catch return error.OutOfMemory;
        return error.RuntimeFault;
    }

    fn source(self: *const Interpreter) []const u8 {
        return self.views[self.current_view].source;
    }

    fn pushFrame(self: *Interpreter, barrier: bool) Error!void {
        try self.scopes.append(self.arena, .{ .bindings = .empty, .barrier = barrier });
    }

    fn popFrame(self: *Interpreter) void {
        _ = self.scopes.pop();
    }

    fn bind(self: *Interpreter, name: []const u8, value: Value) Error!void {
        const cell = try self.arena.create(Value);
        cell.* = value;
        try self.bindCell(name, cell);
    }

    fn bindCell(self: *Interpreter, name: []const u8, cell: *Value) Error!void {
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        try frame.bindings.put(self.arena, name, cell);
    }

    fn lookup(self: *Interpreter, name: []const u8) ?*Value {
        var index = self.scopes.items.len;
        while (index > 0) {
            index -= 1;
            const frame = &self.scopes.items[index];
            if (frame.bindings.get(name)) |cell| return cell;
            if (frame.barrier) break;
        }
        return null;
    }

    // deep copy is the default value semantics (section 4.2): owned
    // allocations are duplicated so pointer uniqueness always holds
    fn deepCopy(self: *Interpreter, value: Value) Error!Value {
        switch (value) {
            .void_value, .integer, .float, .bool_value, .function => return value,
            // borrows copy as aliases
            .reference, .slice => return value,
            .struct_value => |instance| {
                const fields = try self.arena.alloc(Value.Field, instance.fields.len);
                for (instance.fields, fields) |original, *copy| {
                    copy.* = .{ .name = original.name, .value = try self.deepCopy(original.value) };
                }
                const fresh = try self.arena.create(Value.StructInstance);
                fresh.* = .{ .fields = fields, .type_name = instance.type_name, .identity = instance.identity };
                return .{ .struct_value = fresh };
            },
            .enum_value => |instance| {
                const fresh = try self.arena.create(Value.EnumInstance);
                fresh.* = .{
                    .variant = instance.variant,
                    .payload = if (instance.payload) |payload| try self.deepCopy(payload) else null,
                };
                return .{ .enum_value = fresh };
            },
            .array => |instance| return .{ .array = try self.copyArray(instance) },
            .heap_array => |instance| return .{ .heap_array = if (instance) |alive| try self.copyArray(alive) else null },
            .pointer => |target| {
                const alive = target orelse return .{ .pointer = null };
                const cell = try self.arena.create(Value);
                cell.* = try self.deepCopy(alive.*);
                return .{ .pointer = cell };
            },
            .closure => |original| {
                // a closure owns its environment (section 4.2): owned
                // captures are duplicated, reference captures stay aliases
                const environment = try self.arena.alloc(Value.EnvironmentEntry, original.environment.len);
                for (original.environment, environment) |entry, *copy| {
                    copy.* = entry;
                    if (entry.mode != .reference) {
                        const cell = try self.arena.create(Value);
                        cell.* = try self.deepCopy(entry.cell.*);
                        copy.cell = cell;
                    }
                }
                const fresh = try self.arena.create(Value.Closure);
                fresh.* = .{
                    .lambda = original.lambda,
                    .view_index = original.view_index,
                    .environment = environment,
                    .type_bindings = original.type_bindings,
                };
                return .{ .closure = fresh };
            },
            .type_value => |original| {
                // '#Type' is a value (section 3.4): mutation never reaches
                // the description it was copied from
                const fresh = try self.arena.create(Value.TypeDescription);
                fresh.* = .{
                    .kind = original.kind,
                    .name = original.name,
                    .primitive = original.primitive,
                    .members = .empty,
                    .interface_names = original.interface_names,
                    .origin = original.origin,
                };
                try fresh.members.appendSlice(self.arena, original.members.items);
                return .{ .type_value = fresh };
            },
        }
    }

    fn copyArray(self: *Interpreter, instance: *Value.ArrayInstance) Error!*Value.ArrayInstance {
        const elements = try self.arena.alloc(Value, instance.elements.len);
        for (instance.elements, elements) |original, *copy| {
            copy.* = try self.deepCopy(original);
        }
        const fresh = try self.arena.create(Value.ArrayInstance);
        fresh.* = .{ .elements = elements };
        return fresh;
    }

    // pointee transparency (section 4.2): reading through pointers and
    // references reaches the pointee; a null pointer is a use-after-move
    fn pierceCell(self: *Interpreter, cell: *Value) Error!*Value {
        var current = cell;
        var depth: usize = 0;
        while (depth < 16) : (depth += 1) {
            switch (current.*) {
                .pointer => |target| current = target orelse return self.fault("use of a moved-from pointer", .{}),
                .reference => |target| current = target,
                else => return current,
            }
        }
        return current;
    }

    fn execStatement(self: *Interpreter, statement: *const ast.Statement) Error!Flow {
        if (self.debug_hook) |hook| {
            try hook.on_statement(hook.context, self, statement);
        }
        switch (statement.*) {
            .var_def => |var_def| {
                const value = try self.evalExpression(var_def.value);
                try self.bind(var_def.name.slice(self.source()), value);
                return .normal;
            },
            .block => |statements| {
                try self.pushFrame(false);
                defer self.popFrame();
                for (statements) |inner| {
                    const flow = try self.execStatement(inner);
                    if (flow != .normal) return flow;
                }
                return .normal;
            },
            .break_stmt => |break_stmt| {
                const value: ?Value = if (break_stmt.value) |expression| try self.evalExpression(expression) else null;
                return .{ .break_value = value };
            },
            .yield_stmt => |yield_stmt| {
                return .{ .yield_value = try self.evalExpression(yield_stmt.value) };
            },
            .return_stmt => |return_stmt| {
                const value: Value = if (return_stmt.value) |expression| try self.evalExpression(expression) else .void_value;
                return .{ .return_value = value };
            },
            .assign => |assign| return self.execAssign(assign),
            .expression => |expression| {
                // statement-position ifs and matches pass 'break' and
                // 'yield' through to their enclosing constructs
                _ = switch (expression.*) {
                    .if_expr => |if_expr| try self.evalIf(if_expr, false),
                    .match_expr => |match_expr| try self.evalMatch(match_expr, false),
                    else => try self.evalExpression(expression),
                };
                return .normal;
            },
        }
    }

    fn execAssign(self: *Interpreter, assign: anytype) Error!Flow {
        const place = try self.evalPlace(assign.target) orelse return self.fault("assignment to a non-assignable expression", .{});
        if (assign.operator.tag == .equal) {
            place.* = try self.evalExpression(assign.value);
            return .normal;
        }
        // compound assignment computes 'place op value' in the pointee's
        // type (pointee transparency, section 4.2)
        const pierced = try self.pierceCell(place);
        const right = try self.evalExpression(assign.value);
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
            else => return self.fault("unsupported assignment operator", .{}),
        };
        pierced.* = try self.applyBinary(operator, pierced.*, right, null);
        return .normal;
    }

    fn evalExpression(self: *Interpreter, expression: *const ast.Expression) Error!Value {
        if (self.step_budget) |budget| {
            if (budget == 0) return self.fault("compile-time evaluation exceeded its step budget (section 6.1)", .{});
            self.step_budget = budget - 1;
        }
        switch (expression.*) {
            .integer_literal => |token| return self.integerLiteral(expression, token),
            .float_literal => |token| return .{ .float = .{
                .value = std.fmt.parseFloat(f64, token.slice(self.source())) catch 0,
                .primitive = self.primitiveOf(expression),
            } },
            .bool_literal => |literal| return .{ .bool_value = literal.value },
            .string_literal => |token| return self.stringValue(token),
            .character_literal => |token| return self.characterLiteral(expression, token),
            .path => return self.evalPath(expression),
            .implied_variant => |token| return self.makeEnum(token.slice(self.source()), null),
            .grouped => |inner| return self.evalExpression(inner),
            .comptime_expr => |inner| {
                // substituted by compile-time evaluation (section 6.1); the
                // fallback covers '#' nodes inside other comptime evaluations
                if (self.comptime_values.get(expression)) |value| return self.deepCopy(value);
                return self.evalExpression(inner);
            },
            .unary => return self.evalUnary(expression),
            .binary => return self.evalBinary(expression),
            .cast => return self.evalCast(expression),
            .call => {
                const value = try self.evalCall(expression);
                // a bare '&T' result at a use site is a deep copy of the
                // pointee, never an alias (section 4.2)
                if (self.pierced_results) |pierced| {
                    if (value == .reference and pierced.contains(expression)) {
                        // the reference may point at another reference cell
                        // (a reborrowed parameter): chase like any read
                        return self.deepCopy((try self.pierceCell(value.reference)).*);
                    }
                }
                return value;
            },
            .member => {
                const place = try self.evalPlace(expression) orelse return self.fault("cannot read this member", .{});
                // pointee transparency (section 4.2): reading a pointer or
                // reference field yields a copy of the pointee
                return self.deepCopy((try self.pierceCell(place)).*);
            },
            .index => |index| {
                if (try self.evalPlace(expression)) |place| {
                    return self.deepCopy((try self.pierceCell(place)).*);
                }
                // a temporary subject (a call result) indexes by value
                const subject = try self.evalExpression(index.object);
                const instance = switch (subject) {
                    .array => |instance| instance,
                    .slice => |instance| instance,
                    .heap_array => |instance| instance orelse return self.fault("use of a moved-from array", .{}),
                    else => return self.fault("cannot read this element", .{}),
                };
                const subscript = try self.integerOf(try self.evalExpression(index.subscript));
                if (subscript < 0 or subscript >= instance.elements.len) {
                    return self.fault("index {d} out of bounds for length {d}", .{ subscript, instance.elements.len });
                }
                return self.deepCopy(instance.elements[@intCast(subscript)]);
            },
            .subslice => |subslice| {
                // 'arr[start..end]' borrows a slice aliasing the subject's
                // element range in place (section 3.2)
                var subject: Value = undefined;
                if (try self.evalPlace(subslice.object)) |place| {
                    subject = (try self.pierceCell(place)).*;
                } else {
                    subject = try self.evalExpression(subslice.object);
                }
                const instance = switch (subject) {
                    .array => |instance| instance,
                    .slice => |instance| instance,
                    .heap_array => |instance| instance orelse return self.fault("use of a moved-from array", .{}),
                    else => return self.fault("cannot subslice this value", .{}),
                };
                const start: i128 = if (subslice.start) |start_expression|
                    try self.integerOf(try self.evalExpression(start_expression))
                else
                    0;
                const end = try self.integerOf(try self.evalExpression(subslice.end));
                if (start < 0 or end < start or end > instance.elements.len) {
                    return self.fault("subslice {d}..{d} out of bounds for length {d}", .{ start, end, instance.elements.len });
                }
                const view = try self.arena.create(Value.ArrayInstance);
                view.* = .{ .elements = instance.elements[@intCast(start)..@intCast(end)] };
                return .{ .slice = view };
            },
            .struct_init => |struct_init| {
                const fields = try self.arena.alloc(Value.Field, struct_init.members.len);
                for (struct_init.members, fields) |member, *field| {
                    field.* = .{
                        .name = member.name.slice(self.source()),
                        .value = try self.evalExpression(member.value),
                    };
                }
                const recorded = self.expression_types.get(expression);
                var type_name: []const u8 = "";
                var identity: ?types.TypeIdentity = null;
                if (recorded) |result_type| {
                    const resolved = self.resolveTypeParameter(result_type);
                    if (resolved.* == .declared) {
                        type_name = resolved.declared.name;
                        identity = .{
                            .definition = resolved.declared.definition,
                            .view_index = resolved.declared.view_index,
                        };
                    }
                }
                const instance = try self.arena.create(Value.StructInstance);
                instance.* = .{ .fields = fields, .type_name = type_name, .identity = identity };
                return .{ .struct_value = instance };
            },
            .array_literal => |elements| {
                const values = try self.arena.alloc(Value, elements.len);
                for (elements, values) |element, *slot| {
                    slot.* = try self.evalExpression(element);
                }
                const instance = try self.arena.create(Value.ArrayInstance);
                instance.* = .{ .elements = values };
                return .{ .array = instance };
            },
            .array_fill => |array_fill| {
                const value = try self.evalExpression(array_fill.value);
                const count = try self.integerOf(try self.evalExpression(array_fill.count));
                return .{ .array = try self.filledArray(value, count) };
            },
            .array_range => |array_range| {
                const bounds = try self.rangeBounds(array_range);
                const length: usize = @intCast(bounds.end - bounds.start);
                const values = try self.arena.alloc(Value, length);
                for (values, 0..) |*slot, offset| {
                    slot.* = .{ .integer = .{ .value = bounds.start + @as(i128, @intCast(offset)), .primitive = .i32 } };
                }
                const instance = try self.arena.create(Value.ArrayInstance);
                instance.* = .{ .elements = values };
                return .{ .array = instance };
            },
            .if_expr => |if_expr| return self.evalIf(if_expr, true),
            .while_expr => |while_expr| return self.evalWhile(while_expr),
            .for_expr => |for_expr| return self.evalFor(for_expr),
            .match_expr => |match_expr| return self.evalMatch(match_expr, true),
            .lambda => |*lambda| return self.makeClosure(lambda),
        }
    }

    fn integerLiteral(self: *Interpreter, expression: *const ast.Expression, token: Token) Error!Value {
        const text = token.slice(self.source());
        const value = parseIntegerText(text) catch return self.fault("invalid integer literal '{s}'", .{text});
        // an integer literal in float context is a float (section 3.3)
        if (self.primitiveOf(expression)) |primitive| {
            if (primitive.isFloat()) {
                return .{ .float = .{ .value = @floatFromInt(value), .primitive = primitive } };
            }
            return .{ .integer = .{ .value = value, .primitive = primitive } };
        }
        return .{ .integer = .{ .value = value, .primitive = .i32 } };
    }

    fn characterLiteral(self: *Interpreter, expression: *const ast.Expression, token: Token) Error!Value {
        const text = token.slice(self.source());
        const bytes = try self.unescape(text[1 .. text.len - 1]);
        var value: i128 = 0;
        for (bytes) |byte| value = (value << 8) | byte;
        return .{ .integer = .{ .value = value, .primitive = self.primitiveOf(expression) } };
    }

    fn stringValue(self: *Interpreter, token: Token) Error!Value {
        const text = token.slice(self.source());
        return self.bytesValue(try self.unescape(text[1 .. text.len - 1]));
    }

    fn bytesValue(self: *Interpreter, bytes: []const u8) Error!Value {
        const values = try self.arena.alloc(Value, bytes.len);
        for (bytes, values) |byte, *slot| {
            slot.* = .{ .integer = .{ .value = byte, .primitive = .u8 } };
        }
        const instance = try self.arena.create(Value.ArrayInstance);
        instance.* = .{ .elements = values };
        return .{ .slice = instance };
    }

    fn reflectName(self: *Interpreter, name: []const u8) Error!?Value {
        const hooks = self.reflection orelse return null;
        const description = (try hooks.reflect_named(hooks.context, name)) orelse return null;
        return .{ .type_value = description };
    }

    fn unescape(self: *Interpreter, text: []const u8) Error![]const u8 {
        return tokenizer_module.unescape(self.arena, text);
    }

    fn primitiveOf(self: *Interpreter, expression: *const ast.Expression) ?types.Primitive {
        const recorded = self.expression_types.get(expression) orelse return null;
        const resolved = self.resolveTypeParameter(recorded);
        return if (resolved.* == .primitive) resolved.primitive else null;
    }

    // a type recorded inside a generic body may be a type parameter; the
    // active call's bindings (section 3.7) name the concrete type
    fn resolveTypeParameter(self: *Interpreter, recorded: *const Type) *const Type {
        var current = recorded;
        var depth: usize = 0;
        while (current.* == .type_parameter and depth < 8) : (depth += 1) {
            const name = current.type_parameter.name;
            current = for (self.current_type_bindings) |binding| {
                if (std.mem.eql(u8, binding.name, name)) break binding.bound;
            } else return current;
        }
        return current;
    }

    fn evalPath(self: *Interpreter, expression: *const ast.Expression) Error!Value {
        const path = expression.path;
        if (path.len == 1) {
            const name = path[0].slice(self.source());
            if (self.lookup(name)) |cell| {
                const pierced = try self.pierceCell(cell);
                return self.deepCopy(pierced.*);
            }
            // a global function name used as a value
            if (self.firstVisible(name)) |symbol| {
                switch (symbol.definition.kind) {
                    // a type name at compile time reflects ('#Packet')
                    .type_def, .interface_def => if (self.comptime_mode) {
                        if (try self.reflectName(name)) |value| return value;
                    },
                    else => {},
                }
                return .{ .function = symbol };
            }
            if (self.comptime_mode) {
                // primitive names reflect at compile time ('#u32')
                if (try self.reflectName(name)) |value| return value;
            }
            return self.fault("unknown name '{s}'", .{name});
        }
        // a qualified path names a payload-less enum variant
        return self.makeEnum(path[path.len - 1].slice(self.source()), null);
    }

    // builds a closure: copies capture by value now, '&' captures alias the
    // outer cell, '*' captures take ownership and leave the outer moved-from
    fn makeClosure(self: *Interpreter, lambda: *const ast.Lambda) Error!Value {
        const environment = try self.arena.alloc(Value.EnvironmentEntry, lambda.captures.len);
        for (lambda.captures, environment) |capture, *entry| {
            const name = capture.name.slice(self.source());
            const outer = self.lookup(name) orelse return self.fault("unknown capture '{s}'", .{name});
            const mode = captureMode(capture);
            entry.* = .{ .name = name, .cell = outer, .mode = mode };
            switch (mode) {
                .reference => {},
                .copy => {
                    const cell = try self.arena.create(Value);
                    cell.* = try self.deepCopy((try self.pierceCell(outer)).*);
                    entry.cell = cell;
                },
                .owning => {
                    const taken = outer.*;
                    switch (taken) {
                        .pointer => outer.* = .{ .pointer = null },
                        .heap_array => outer.* = .{ .heap_array = null },
                        else => return self.fault("an owning capture needs a pointer-typed variable (section 2.1)", .{}),
                    }
                    const cell = try self.arena.create(Value);
                    cell.* = taken;
                    entry.cell = cell;
                },
            }
        }
        const instance = try self.arena.create(Value.Closure);
        instance.* = .{
            .lambda = lambda,
            .view_index = self.current_view,
            .environment = environment,
            .type_bindings = self.current_type_bindings,
        };
        return .{ .closure = instance };
    }

    fn makeEnum(self: *Interpreter, variant: []const u8, payload: ?Value) Error!Value {
        const instance = try self.arena.create(Value.EnumInstance);
        instance.* = .{ .variant = variant, .payload = payload };
        return .{ .enum_value = instance };
    }

    fn evalUnary(self: *Interpreter, expression: *const ast.Expression) Error!Value {
        const unary = expression.unary;
        switch (unary.operator.tag) {
            .minus => {
                const operand = try self.evalExpression(unary.operand);
                switch (operand) {
                    .integer => |integer| {
                        const result: Value = .{ .integer = .{ .value = -integer.value, .primitive = integer.primitive orelse self.primitiveOf(expression) } };
                        try self.checkIntegerRange(result.integer);
                        return result;
                    },
                    .float => |float| return .{ .float = .{ .value = -float.value, .primitive = float.primitive } },
                    else => return self.fault("'-' needs a numeric operand", .{}),
                }
            },
            .tilde => {
                const operand = try self.integerValue(try self.evalExpression(unary.operand));
                const primitive = operand.primitive orelse .i32;
                const masked = truncateToPrimitive(~operand.value, primitive);
                return .{ .integer = .{ .value = masked, .primitive = primitive } };
            },
            .bang => {
                const operand = try self.evalExpression(unary.operand);
                if (operand != .bool_value) return self.fault("'!' needs a bool operand", .{});
                return .{ .bool_value = !operand.bool_value };
            },
            .ampersand => {
                const place = try self.evalPlace(unary.operand) orelse {
                    // '&' on a reference-typed call result keeps the borrow
                    // (section 4.2); the checker gated everything else
                    const value = try self.evalExpression(unary.operand);
                    if (value == .reference or value == .slice) return value;
                    return self.fault("'&' needs an addressable operand", .{});
                };
                // borrowing a heap array yields a slice aliasing its
                // elements in place (section 4.2)
                if (place.* == .heap_array) {
                    const instance = place.heap_array orelse return self.fault("use of a moved-from array", .{});
                    return .{ .slice = instance };
                }
                return .{ .reference = place };
            },
            .keyword_new => return self.evalNew(unary.operand),
            .keyword_move => {
                const place = try self.evalPlace(unary.operand) orelse return self.fault("'move' needs an addressable operand", .{});
                const taken = place.*;
                switch (taken) {
                    .pointer => place.* = .{ .pointer = null },
                    .heap_array => place.* = .{ .heap_array = null },
                    else => return self.fault("'move' needs a pointer-typed operand", .{}),
                }
                return taken;
            },
            else => return self.fault("unsupported unary operator", .{}),
        }
    }

    fn evalNew(self: *Interpreter, operand: *const ast.Expression) Error!Value {
        switch (operand.*) {
            .array_literal, .array_fill, .array_range => {
                const value = try self.evalExpression(operand);
                return .{ .heap_array = value.array };
            },
            else => {
                const value = try self.evalExpression(operand);
                const cell = try self.arena.create(Value);
                cell.* = value;
                return .{ .pointer = cell };
            },
        }
    }

    fn evalBinary(self: *Interpreter, expression: *const ast.Expression) Error!Value {
        const binary = expression.binary;
        // logical operators short-circuit (section 4.1)
        if (binary.operator.tag == .ampersand_ampersand or binary.operator.tag == .pipe_pipe) {
            const left = try self.evalExpression(binary.left);
            if (left != .bool_value) return self.fault("logical operators need bool operands", .{});
            if (binary.operator.tag == .ampersand_ampersand and !left.bool_value) return .{ .bool_value = false };
            if (binary.operator.tag == .pipe_pipe and left.bool_value) return .{ .bool_value = true };
            const right = try self.evalExpression(binary.right);
            if (right != .bool_value) return self.fault("logical operators need bool operands", .{});
            return .{ .bool_value = right.bool_value };
        }
        const left = try self.evalExpression(binary.left);
        const right = try self.evalExpression(binary.right);
        return self.applyBinary(binary.operator.tag, left, right, self.primitiveOf(expression));
    }

    fn applyBinary(self: *Interpreter, operator: Token.Tag, left: Value, right: Value, result_primitive: ?types.Primitive) Error!Value {
        if (left == .float or right == .float) {
            const left_float = try self.floatOf(left);
            const right_float = try self.floatOf(right);
            return self.applyFloat(operator, left_float, right_float, result_primitive);
        }
        if (left == .integer and right == .integer) {
            return self.applyInteger(operator, left.integer, right.integer, result_primitive);
        }
        if (left == .bool_value and right == .bool_value) {
            return switch (operator) {
                .equal_equal => .{ .bool_value = left.bool_value == right.bool_value },
                .bang_equal => .{ .bool_value = left.bool_value != right.bool_value },
                else => self.fault("unsupported bool operation", .{}),
            };
        }
        return self.fault("unsupported operand types", .{});
    }

    fn applyInteger(self: *Interpreter, operator: Token.Tag, left: Value.Integer, right: Value.Integer, result_primitive: ?types.Primitive) Error!Value {
        const primitive = result_primitive orelse left.primitive orelse right.primitive;
        const a = left.value;
        const b = right.value;
        const arithmetic: ?i128 = switch (operator) {
            .plus => a + b,
            .minus => a - b,
            .asterisk => a * b,
            .slash => if (b == 0) return self.fault("division by zero", .{}) else @divTrunc(a, b),
            .percent => if (b == 0) return self.fault("division by zero", .{}) else @rem(a, b),
            .shift_left => if (b < 0 or b > 63) return self.fault("shift amount out of range", .{}) else a << @intCast(b),
            .shift_right => if (b < 0 or b > 63) return self.fault("shift amount out of range", .{}) else a >> @intCast(b),
            .ampersand => a & b,
            .pipe => a | b,
            .caret => a ^ b,
            else => null,
        };
        if (arithmetic) |value| {
            const result: Value.Integer = .{ .value = value, .primitive = primitive };
            // shifts and bitwise operations stay within the width by masking
            switch (operator) {
                .shift_left, .ampersand, .pipe, .caret => if (primitive) |bound| {
                    return .{ .integer = .{ .value = truncateToPrimitive(value, bound), .primitive = bound } };
                },
                else => {},
            }
            try self.checkIntegerRange(result);
            return .{ .integer = result };
        }
        return switch (operator) {
            .equal_equal => .{ .bool_value = a == b },
            .bang_equal => .{ .bool_value = a != b },
            .angle_left => .{ .bool_value = a < b },
            .angle_left_equal => .{ .bool_value = a <= b },
            .angle_right => .{ .bool_value = a > b },
            .angle_right_equal => .{ .bool_value = a >= b },
            else => self.fault("unsupported integer operation", .{}),
        };
    }

    fn applyFloat(self: *Interpreter, operator: Token.Tag, left: f64, right: f64, result_primitive: ?types.Primitive) Error!Value {
        const arithmetic: ?f64 = switch (operator) {
            .plus => left + right,
            .minus => left - right,
            .asterisk => left * right,
            .slash => left / right,
            .percent => @rem(left, right),
            else => null,
        };
        if (arithmetic) |value| return .{ .float = .{ .value = value, .primitive = result_primitive } };
        return switch (operator) {
            .equal_equal => .{ .bool_value = left == right },
            .bang_equal => .{ .bool_value = left != right },
            .angle_left => .{ .bool_value = left < right },
            .angle_left_equal => .{ .bool_value = left <= right },
            .angle_right => .{ .bool_value = left > right },
            .angle_right_equal => .{ .bool_value = left >= right },
            else => self.fault("unsupported float operation", .{}),
        };
    }

    // overflow is a fault in the interpreter (section 4.2)
    fn checkIntegerRange(self: *Interpreter, integer: Value.Integer) Error!void {
        const primitive = integer.primitive orelse return;
        if (integer.value < minimumOf(primitive) or integer.value > maximumOf(primitive)) {
            return self.fault("integer overflow: {d} does not fit {s}", .{ integer.value, @tagName(primitive) });
        }
    }

    fn evalCast(self: *Interpreter, expression: *const ast.Expression) Error!Value {
        const cast = expression.cast;
        switch (cast.operator.tag) {
            .keyword_to => {
                const operand = try self.evalExpression(cast.operand);
                const target = self.primitiveOf(expression) orelse return self.fault("'to' target is not a primitive", .{});
                if (target.isFloat()) {
                    const value = switch (operand) {
                        .integer => |integer| @as(f64, @floatFromInt(integer.value)),
                        .float => |float| float.value,
                        else => return self.fault("'to' needs a numeric operand", .{}),
                    };
                    return .{ .float = .{ .value = value, .primitive = target } };
                }
                const wide: i128 = switch (operand) {
                    .integer => |integer| integer.value,
                    .float => |float| @intFromFloat(float.value),
                    else => return self.fault("'to' needs a numeric operand", .{}),
                };
                // conversion truncates into the target range (section 3.5)
                return .{ .integer = .{ .value = truncateToPrimitive(wide, target), .primitive = target } };
            },
            .keyword_is => {
                const operand = try self.evalExpression(cast.operand);
                const target_token = cast.target.named.path[cast.target.named.path.len - 1];
                const target_name = target_token.slice(self.source());
                return switch (operand) {
                    .enum_value => |instance| .{ .bool_value = std.mem.eql(u8, instance.variant, target_name) },
                    // an interface object's concrete-type test compares the
                    // checker-resolved identity (section 3.2)
                    .struct_value => |instance| verdict: {
                        if (self.type_targets.get(expression)) |target| {
                            const identity = instance.identity orelse break :verdict .{ .bool_value = false };
                            break :verdict .{ .bool_value = identity.definition == target.definition };
                        }
                        break :verdict .{ .bool_value = std.mem.eql(u8, instance.type_name, target_name) };
                    },
                    else => self.fault("'is' needs an enum or interface-object subject", .{}),
                };
            },
            .keyword_as => {
                const operand = try self.evalExpression(cast.operand);
                // a shaped cast reinterprets through a byte image (section
                // 3.5); the checker records shapes for non-primitive sides
                if (self.cast_shapes.get(expression)) |shapes| {
                    return self.reinterpretShaped(operand, shapes);
                }
                const target = self.primitiveOf(expression) orelse return self.fault("'as' through references or pointer-bearing values is not supported by the interpreter", .{});
                return self.reinterpret(operand, target);
            },
            else => return self.fault("unsupported cast", .{}),
        }
    }

    // 'as' reinterprets the bytes of same-width primitives (section 3.5)
    fn reinterpret(self: *Interpreter, operand: Value, target: types.Primitive) Error!Value {
        const bits: u64 = switch (operand) {
            .integer => |integer| @truncate(@as(u128, @bitCast(integer.value))),
            .float => |float| if (float.primitive == .f32)
                @as(u32, @bitCast(@as(f32, @floatCast(float.value))))
            else
                @as(u64, @bitCast(float.value)),
            .bool_value => |value| @intFromBool(value),
            else => return self.fault("'as' on non-primitive values is not yet supported by the interpreter", .{}),
        };
        if (target.isFloat()) {
            const value: f64 = if (target == .f32)
                @as(f32, @bitCast(@as(u32, @truncate(bits))))
            else
                @as(f64, @bitCast(bits));
            return .{ .float = .{ .value = value, .primitive = target } };
        }
        if (target == .bool) return .{ .bool_value = bits != 0 };
        const masked = truncateToPrimitive(@as(i128, bits), target);
        return .{ .integer = .{ .value = masked, .primitive = target } };
    }

    // 'as' beyond primitives (section 3.5): the value is serialized into its
    // little-endian C-layout byte image and read back as the target shape
    fn reinterpretShaped(self: *Interpreter, operand: Value, shapes: types.CastShapes) Error!Value {
        const size: usize = @intCast(shapes.source.byteSize());
        const buffer = try self.arena.alloc(u8, size);
        @memset(buffer, 0);
        try self.serializeValue(operand, shapes.source, buffer);
        return self.deserializeValue(shapes.target, buffer);
    }

    fn serializeValue(self: *Interpreter, value: Value, shape: *const types.Shape, bytes: []u8) Error!void {
        switch (shape.*) {
            .primitive => |primitive| try self.serializePrimitive(value, primitive, bytes),
            .record => |record| {
                if (value != .struct_value) return self.fault("'as' source value does not match its checked layout", .{});
                for (record.fields) |field| {
                    const found = for (value.struct_value.fields) |candidate| {
                        if (std.mem.eql(u8, candidate.name, field.name)) break candidate.value;
                    } else return self.fault("'as' source value does not match its checked layout", .{});
                    try self.serializeValue(found, field.shape, bytes[@intCast(field.offset)..][0..@intCast(field.shape.byteSize())]);
                }
            },
            .array => |array| {
                const instance = switch (value) {
                    .array => |instance| instance,
                    .slice => |instance| instance,
                    else => return self.fault("'as' source value does not match its checked layout", .{}),
                };
                if (instance.elements.len != array.count) {
                    return self.fault("'as' source value does not match its checked layout", .{});
                }
                const stride: usize = @intCast(array.stride);
                const element_size: usize = @intCast(array.element.byteSize());
                for (instance.elements, 0..) |element, index| {
                    try self.serializeValue(element, array.element, bytes[index * stride ..][0..element_size]);
                }
            },
            .tagged => |tagged| {
                if (value != .enum_value) return self.fault("'as' source value does not match its checked layout", .{});
                const instance = value.enum_value;
                const index = for (tagged.variants, 0..) |variant, position| {
                    if (std.mem.eql(u8, variant.name, instance.variant)) break position;
                } else return self.fault("'as' source value does not match its checked layout", .{});
                writeUnsigned(bytes[0..@intCast(tagged.tag_size)], index);
                const payload_shape = tagged.variants[index].payload orelse return;
                const payload = instance.payload orelse return self.fault("'as' source value does not match its checked layout", .{});
                try self.serializeValue(payload, payload_shape, bytes[@intCast(tagged.payload_offset)..][0..@intCast(payload_shape.byteSize())]);
            },
        }
    }

    fn serializePrimitive(self: *Interpreter, value: Value, primitive: types.Primitive, bytes: []u8) Error!void {
        switch (primitive) {
            .bool => {
                if (value != .bool_value) return self.fault("'as' source value does not match its checked layout", .{});
                bytes[0] = @intFromBool(value.bool_value);
            },
            .f32, .f64 => {
                if (value != .float) return self.fault("'as' source value does not match its checked layout", .{});
                if (primitive == .f32) {
                    std.mem.writeInt(u32, bytes[0..4], @bitCast(@as(f32, @floatCast(value.float.value))), .little);
                } else {
                    std.mem.writeInt(u64, bytes[0..8], @bitCast(value.float.value), .little);
                }
            },
            else => {
                if (value != .integer) return self.fault("'as' source value does not match its checked layout", .{});
                writeUnsigned(bytes, @truncate(@as(u128, @bitCast(value.integer.value))));
            },
        }
    }

    fn writeUnsigned(bytes: []u8, value: u64) void {
        for (bytes, 0..) |*slot, index| {
            slot.* = @truncate(value >> @intCast(index * 8));
        }
    }

    fn readUnsigned(bytes: []const u8) u64 {
        var value: u64 = 0;
        for (bytes, 0..) |byte, index| {
            value |= @as(u64, byte) << @intCast(index * 8);
        }
        return value;
    }

    fn deserializeValue(self: *Interpreter, shape: *const types.Shape, bytes: []const u8) Error!Value {
        switch (shape.*) {
            .primitive => |primitive| return self.deserializePrimitive(primitive, bytes),
            .record => |record| {
                const fields = try self.arena.alloc(Value.Field, record.fields.len);
                for (record.fields, fields) |field, *slot| {
                    slot.* = .{
                        .name = field.name,
                        .value = try self.deserializeValue(field.shape, bytes[@intCast(field.offset)..][0..@intCast(field.shape.byteSize())]),
                    };
                }
                const instance = try self.arena.create(Value.StructInstance);
                instance.* = .{ .fields = fields, .type_name = record.name, .identity = record.identity };
                return .{ .struct_value = instance };
            },
            .array => |array| {
                const elements = try self.arena.alloc(Value, @intCast(array.count));
                const stride: usize = @intCast(array.stride);
                const element_size: usize = @intCast(array.element.byteSize());
                for (elements, 0..) |*slot, index| {
                    slot.* = try self.deserializeValue(array.element, bytes[index * stride ..][0..element_size]);
                }
                const instance = try self.arena.create(Value.ArrayInstance);
                instance.* = .{ .elements = elements };
                return .{ .array = instance };
            },
            .tagged => |tagged| {
                const index = readUnsigned(bytes[0..@intCast(tagged.tag_size)]);
                if (index >= tagged.variants.len) {
                    return self.fault("'as' produced a tag of {d}, which names no variant (section 3.5)", .{index});
                }
                const variant = tagged.variants[@intCast(index)];
                var payload: ?Value = null;
                if (variant.payload) |payload_shape| {
                    payload = try self.deserializeValue(payload_shape, bytes[@intCast(tagged.payload_offset)..][0..@intCast(payload_shape.byteSize())]);
                }
                const instance = try self.arena.create(Value.EnumInstance);
                instance.* = .{ .variant = variant.name, .payload = payload };
                return .{ .enum_value = instance };
            },
        }
    }

    fn deserializePrimitive(self: *Interpreter, primitive: types.Primitive, bytes: []const u8) Error!Value {
        _ = self;
        switch (primitive) {
            .bool => return .{ .bool_value = bytes[0] != 0 },
            .f32 => return .{ .float = .{ .value = @as(f32, @bitCast(std.mem.readInt(u32, bytes[0..4], .little))), .primitive = primitive } },
            .f64 => return .{ .float = .{ .value = @bitCast(std.mem.readInt(u64, bytes[0..8], .little)), .primitive = primitive } },
            else => {
                const raw = readUnsigned(bytes);
                // signed primitives sign-extend from their width
                const value: i128 = if (primitive.isSigned()) switch (primitive.width()) {
                    1 => @as(i8, @bitCast(@as(u8, @truncate(raw)))),
                    2 => @as(i16, @bitCast(@as(u16, @truncate(raw)))),
                    4 => @as(i32, @bitCast(@as(u32, @truncate(raw)))),
                    else => @as(i64, @bitCast(raw)),
                } else @as(i128, raw);
                return .{ .integer = .{ .value = value, .primitive = primitive } };
            },
        }
    }

    fn evalPlace(self: *Interpreter, expression: *const ast.Expression) Error!?*Value {
        switch (expression.*) {
            .grouped => |inner| return self.evalPlace(inner),
            .path => |path| {
                if (path.len != 1) return null;
                return self.lookup(path[0].slice(self.source()));
            },
            .member => |member| {
                const object = try self.evalPlace(member.object) orelse return null;
                const pierced = try self.pierceCell(object);
                if (pierced.* != .struct_value) return null;
                const name = member.name.slice(self.source());
                for (pierced.struct_value.fields) |*field| {
                    if (std.mem.eql(u8, field.name, name)) return &field.value;
                }
                return null;
            },
            .index => |index| {
                const object = try self.evalPlace(index.object) orelse return null;
                const pierced = try self.pierceCell(object);
                const instance = switch (pierced.*) {
                    .array => |instance| instance,
                    .slice => |instance| instance,
                    .heap_array => |instance| instance orelse return self.fault("use of a moved-from array", .{}),
                    else => return null,
                };
                const subscript = try self.integerOf(try self.evalExpression(index.subscript));
                if (subscript < 0 or subscript >= instance.elements.len) {
                    return self.fault("index {d} out of bounds for length {d}", .{ subscript, instance.elements.len });
                }
                return &instance.elements[@intCast(subscript)];
            },
            else => return null,
        }
    }

    fn evalCall(self: *Interpreter, expression: *const ast.Expression) Error!Value {
        const call = expression.call;
        const callee = unwrapGrouped(call.callee);

        // enum variant construction
        if (callee.* == .implied_variant) {
            const payload = try self.singlePayload(call.arguments);
            return self.makeEnum(callee.implied_variant.slice(self.source()), payload);
        }
        if (self.call_targets.get(expression)) |symbol| {
            const type_bindings = self.call_type_bindings.get(expression) orelse &.{};
            return self.callResolved(symbol, call, callee, type_bindings);
        }
        if (callee.* == .path) {
            const path = callee.path;
            if (path.len >= 2) {
                const payload = try self.singlePayload(call.arguments);
                return self.makeEnum(path[path.len - 1].slice(self.source()), payload);
            }
            // a local binding holding a function or closure value
            if (self.lookup(path[0].slice(self.source()))) |cell| {
                const pierced = try self.pierceCell(cell);
                if (pierced.* == .function) {
                    const arguments = try self.evalArguments(call.arguments);
                    return self.callFunction(pierced.function, arguments, &.{});
                }
                if (pierced.* == .closure) {
                    const arguments = try self.evalArguments(call.arguments);
                    return self.callClosure(pierced.closure, arguments);
                }
            }
            if (self.comptime_mode) {
                const name = path[0].slice(self.source());
                if (try self.comptimeNameCall(name, call)) |value| return value;
            }
            return self.fault("cannot call '{s}' here (not yet supported by the interpreter)", .{path[path.len - 1].slice(self.source())});
        }
        if (callee.* == .member) {
            return self.callMethod(callee.member, call);
        }
        // any other callee expression (an immediately invoked lambda, a
        // grouped expression, ...) evaluates to a callable value
        const value = try self.evalExpression(callee);
        switch (value) {
            .closure => |instance| return self.callClosure(instance, try self.evalArguments(call.arguments)),
            .function => |symbol| return self.callFunction(symbol, try self.evalArguments(call.arguments), &.{}),
            else => return self.fault("this call form is not yet supported by the interpreter", .{}),
        }
    }

    fn singlePayload(self: *Interpreter, arguments: []const *const ast.Expression) Error!?Value {
        if (arguments.len == 0) return null;
        return try self.evalExpression(arguments[0]);
    }

    fn evalArguments(self: *Interpreter, expressions: []const *const ast.Expression) Error![]Value {
        const values = try self.arena.alloc(Value, expressions.len);
        for (expressions, values) |expression, *slot| {
            slot.* = try self.evalExpression(expression);
        }
        return values;
    }

    fn callResolved(self: *Interpreter, symbol: resolution.Symbol, call: anytype, callee: *const ast.Expression, type_bindings: []const Type.Binding) Error!Value {
        switch (symbol.definition.kind) {
            .extern_def => |extern_def| {
                const view_source = self.views[symbol.view_index].source;
                const name = extern_def.name.slice(view_source);
                const arguments = try self.evalArguments(call.arguments);
                return self.callExtern(name, arguments);
            },
            .fn_def => |fn_def| {
                // std::process::arguments is compiler-provided (section
                // 5.1a): the host argv materializes as a slice of slices
                if (self.processArgumentsLangItem(symbol)) {
                    return self.processArgumentsValue();
                }
                const has_self = fn_def.function.parameters.len != 0 and fn_def.function.parameters[0].is_self;
                if (has_self and callee.* == .member) {
                    return self.callExtension(symbol, fn_def, callee.member, call, type_bindings);
                }
                const arguments = try self.evalArguments(call.arguments);
                return self.callFunction(symbol, arguments, type_bindings);
            },
            else => return self.fault("this callee is not callable", .{}),
        }
    }

    fn callMethod(self: *Interpreter, member: anytype, call: anytype) Error!Value {
        // built-in '.length()' on every array form (section 5.1); the
        // receiver may be a temporary, not only a place
        const name = member.name.slice(self.source());
        if (std.mem.eql(u8, name, "length") and call.arguments.len == 0) {
            const receiver: Value = receiver: {
                if (try self.evalPlace(member.object)) |place| {
                    break :receiver (try self.pierceCell(place)).*;
                }
                break :receiver try self.evalExpression(member.object);
            };
            const length: ?usize = switch (receiver) {
                .array => |instance| instance.elements.len,
                .slice => |instance| instance.elements.len,
                .heap_array => |instance| if (instance) |alive| alive.elements.len else return self.fault("use of a moved-from array", .{}),
                else => null,
            };
            if (length) |value| return .{ .integer = .{ .value = @intCast(value), .primitive = .u64 } };
        }
        // '#Type' reflection methods (section 3.4), compile time only
        if (self.comptime_mode) {
            if (try self.typeValueReceiver(member.object)) |description| {
                return self.typeMethod(description, name, call);
            }
        }
        // a call without a static target dispatches at runtime through the
        // receiver's concrete type (section 5.2); extensions win over
        // function-typed fields, matching the native dispatch order
        if (try self.evalPlace(member.object)) |place| {
            const pierced = try self.pierceCell(place);
            if (pierced.* == .struct_value) {
                if (pierced.struct_value.identity) |identity| {
                    if (self.findDispatchTarget(identity, name)) |symbol| {
                        return self.callExtension(symbol, symbol.definition.kind.fn_def, member, call, &.{});
                    }
                }
                // a function-typed field calls through its value (section
                // 4.4) when no extension implements the name
                for (pierced.struct_value.fields) |*field| {
                    if (!std.mem.eql(u8, field.name, name)) continue;
                    switch (field.value) {
                        .closure => |instance| return self.callClosure(instance, try self.evalArguments(call.arguments)),
                        .function => |symbol| return self.callFunction(symbol, try self.evalArguments(call.arguments), &.{}),
                        else => {},
                    }
                }
                if (pierced.struct_value.type_name.len != 0) {
                    return self.fault("no extension '{s}' found for '{s}' at runtime", .{ name, pierced.struct_value.type_name });
                }
            }
        } else {
            // a temporary receiver (a call result) also calls through its
            // function-typed field (section 4.4)
            const receiver = try self.evalExpression(member.object);
            if (receiver == .struct_value) {
                for (receiver.struct_value.fields) |*field| {
                    if (!std.mem.eql(u8, field.name, name)) continue;
                    switch (field.value) {
                        .closure => |instance| return self.callClosure(instance, try self.evalArguments(call.arguments)),
                        .function => |symbol| return self.callFunction(symbol, try self.evalArguments(call.arguments), &.{}),
                        else => {},
                    }
                }
            }
        }
        // compile-time code may run before the extension's module is
        // checked (eager macros evaluate mid-check), so no call target or
        // struct identity exists yet: extensions resolve by name and
        // arity, like every other comptime call (section 6.3)
        if (self.comptime_mode) {
            if (self.globals.get(name)) |symbols| {
                for (symbols.items) |symbol| {
                    if (!self.symbolVisible(symbol)) continue;
                    if (symbol.definition.kind != .fn_def) continue;
                    const fn_def = symbol.definition.kind.fn_def;
                    const parameters = fn_def.function.parameters;
                    if (parameters.len == 0 or !parameters[0].is_self) continue;
                    if (parameters.len != call.arguments.len + 1) continue;
                    return self.callExtension(symbol, fn_def, member, call, &.{});
                }
            }
        }
        return self.fault("no resolved target for the call to '{s}' (not yet supported by the interpreter)", .{name});
    }

    fn typeValueReceiver(self: *Interpreter, object: *const ast.Expression) Error!?*Value.TypeDescription {
        if (try self.evalPlace(object)) |place| {
            const pierced = try self.pierceCell(place);
            return if (pierced.* == .type_value) pierced.type_value else null;
        }
        // a reflection expression ('#Packet', '#type_of(x)') as receiver
        const value = try self.evalExpression(object);
        return if (value == .type_value) value.type_value else null;
    }

    // the '#Type' method set (section 3.4)
    fn typeMethod(self: *Interpreter, description: *Value.TypeDescription, name: []const u8, call: anytype) Error!Value {
        const arguments = try self.evalArguments(call.arguments);
        if (std.mem.eql(u8, name, "is_struct")) return .{ .bool_value = description.kind == .struct_kind };
        if (std.mem.eql(u8, name, "is_enum")) return .{ .bool_value = description.kind == .enum_kind };
        if (std.mem.eql(u8, name, "is_primitive")) return .{ .bool_value = description.kind == .primitive_kind };
        if (std.mem.eql(u8, name, "is_interface")) return .{ .bool_value = description.kind == .interface_kind };
        if (std.mem.eql(u8, name, "name")) return self.bytesValue(description.name);
        if (std.mem.eql(u8, name, "equals")) {
            const other = try self.typeArgument(arguments, 0, "equals");
            return .{ .bool_value = descriptionsEqual(description, other) };
        }
        if (std.mem.eql(u8, name, "implements_interface")) {
            const other = try self.typeArgument(arguments, 0, "implements_interface");
            if (other.kind != .interface_kind) return self.fault("'implements_interface' needs an interface '#Type' (section 3.4)", .{});
            // the checker answers by definition identity, covering lang
            // items; marker names remain only for synthesised descriptions
            if (self.reflection) |hooks| {
                if (try hooks.reflect_implements(hooks.context, description, other)) |conforms| {
                    return .{ .bool_value = conforms };
                }
            }
            for (description.interface_names) |marker| {
                if (std.mem.eql(u8, marker, other.name)) return .{ .bool_value = true };
            }
            return .{ .bool_value = false };
        }
        if (std.mem.eql(u8, name, "add_member")) {
            if (arguments.len != 2) return self.fault("'add_member' expects a name and a '#Type' (section 3.4)", .{});
            const member_name = try self.byteSlice(arguments[0]);
            const member_type = try self.typeArgument(arguments, 1, "add_member");
            try description.members.append(self.arena, .{ .name = member_name, .description = member_type });
            return .void_value;
        }
        if (std.mem.eql(u8, name, "remove_member")) {
            if (arguments.len != 1) return self.fault("'remove_member' expects a member name (section 3.4)", .{});
            const member_name = try self.byteSlice(arguments[0]);
            for (description.members.items, 0..) |member, index| {
                if (std.mem.eql(u8, member.name, member_name)) {
                    _ = description.members.orderedRemove(index);
                    return .void_value;
                }
            }
            return self.fault("no member '{s}' to remove (section 3.4)", .{member_name});
        }
        if (std.mem.eql(u8, name, "member_names")) {
            const values = try self.arena.alloc(Value, description.members.items.len);
            for (description.members.items, values) |member, *slot| {
                slot.* = try self.bytesValue(member.name);
            }
            const instance = try self.arena.create(Value.ArrayInstance);
            instance.* = .{ .elements = values };
            return .{ .slice = instance };
        }
        if (std.mem.eql(u8, name, "member_types")) {
            const values = try self.arena.alloc(Value, description.members.items.len);
            for (description.members.items, values) |member, *slot| {
                slot.* = .{ .type_value = member.description };
            }
            const instance = try self.arena.create(Value.ArrayInstance);
            instance.* = .{ .elements = values };
            return .{ .slice = instance };
        }
        return self.fault("'#Type' has no method '{s}' (section 3.4)", .{name});
    }

    fn typeArgument(self: *Interpreter, arguments: []const Value, index: usize, method: []const u8) Error!*Value.TypeDescription {
        if (index >= arguments.len or arguments[index] != .type_value) {
            return self.fault("'{s}' expects a '#Type' argument (section 3.4)", .{method});
        }
        return arguments[index].type_value;
    }

    fn descriptionsEqual(left: *Value.TypeDescription, right: *Value.TypeDescription) bool {
        if (left == right) return true;
        const left_origin = left.origin orelse return false;
        const right_origin = right.origin orelse return false;
        return left_origin.eql(right_origin);
    }

    // resolves an interface-object call by the receiver's nominal identity:
    // a type-specific extension wins over an interface default (section 5.2)
    fn findDispatchTarget(self: *Interpreter, receiver: types.TypeIdentity, method_name: []const u8) ?resolution.Symbol {
        const symbols = self.globals.get(method_name) orelse return null;
        var default_implementation: ?resolution.Symbol = null;
        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            const parameters = fn_def.function.parameters;
            if (parameters.len == 0 or !parameters[0].is_self) continue;
            const view_source = self.views[symbol.view_index].source;
            const self_name = selfTypeName(parameters[0].parameter_type, view_source) orelse continue;
            // the self type resolves in the extension's own view, so two
            // libraries' same-named types never cross wires
            const self_symbol = self.firstVisibleFrom(self_name, symbol.view_index) orelse continue;
            switch (self_symbol.definition.kind) {
                .type_def => if (self_symbol.definition == receiver.definition) return symbol,
                .interface_def => if (self.typeImplements(receiver, self_symbol.definition)) {
                    default_implementation = symbol;
                },
                else => {},
            }
        }
        return default_implementation;
    }

    // whether the receiver's type marks itself with the interface; markers
    // resolve in the type's declaring view
    fn typeImplements(self: *Interpreter, receiver: types.TypeIdentity, interface_definition: *const ast.Definition) bool {
        if (receiver.definition.kind != .type_def) return false;
        const view_source = self.views[receiver.view_index].source;
        for (receiver.definition.kind.type_def.interfaces) |marker| {
            const marker_symbol = self.firstVisibleFrom(marker.slice(view_source), receiver.view_index) orelse continue;
            if (marker_symbol.definition == interface_definition) return true;
        }
        return false;
    }

    fn selfTypeName(parameter_type: *const ast.TypeExpression, view_source: []const u8) ?[]const u8 {
        var current = parameter_type;
        while (current.* == .modified) current = current.modified.child;
        if (current.* != .named) return null;
        const path = current.named.path;
        return path[path.len - 1].slice(view_source);
    }

    // an extension call: the receiver becomes the self argument per the
    // self parameter's indirection (section 4.5)
    fn callExtension(self: *Interpreter, symbol: resolution.Symbol, fn_def: ast.FnDef, member: anytype, call: anytype, type_bindings: []const Type.Binding) Error!Value {
        const receiver_place = try self.evalPlace(member.object);
        const self_parameter = fn_def.function.parameters[0];
        const self_value: Value = switch (selfIndirection(self_parameter.parameter_type)) {
            .reference => reference: {
                if (receiver_place) |place| break :reference .{ .reference = try self.pierceCell(place) };
                // a temporary receiver materializes for the call's duration
                // (section 4.5); the checker only lets immutable '&' through
                const cell = try self.arena.create(Value);
                cell.* = try self.evalExpression(member.object);
                break :reference .{ .reference = try self.pierceCell(cell) };
            },
            .pointer => pointer: {
                // an inline allocation ('new T {}') is its own pointer
                const place = receiver_place orelse break :pointer try self.evalExpression(member.object);
                break :pointer place.*;
            },
            .value => value: {
                if (receiver_place) |place| {
                    break :value try self.deepCopy((try self.pierceCell(place)).*);
                }
                break :value try self.evalExpression(member.object);
            },
        };
        var arguments: std.ArrayList(Value) = .empty;
        try arguments.append(self.arena, self_value);
        for (call.arguments) |argument| {
            try arguments.append(self.arena, try self.evalExpression(argument));
        }
        return self.callFunction(symbol, arguments.items, type_bindings);
    }

    const SelfIndirection = enum { reference, pointer, value };

    fn selfIndirection(parameter_type: *const ast.TypeExpression) SelfIndirection {
        if (parameter_type.* != .modified) return .value;
        return switch (parameter_type.modified.modifier) {
            .reference, .reference_var => .reference,
            .pointer, .pointer_var => .pointer,
        };
    }

    fn callFunction(self: *Interpreter, symbol: resolution.Symbol, arguments: []const Value, type_bindings: []const Type.Binding) Error!Value {
        if (symbol.definition.kind != .fn_def) return self.fault("this callee is not callable", .{});
        const fn_def = symbol.definition.kind.fn_def;
        if (self.call_depth >= 1024) return self.fault("call stack exhausted (1024 frames)", .{});
        self.call_depth += 1;
        defer self.call_depth -= 1;
        if (self.debug_hook != null) {
            try self.debug_stack.append(self.arena, .{
                .name = fn_def.name.slice(self.views[symbol.view_index].source),
                .view_index = symbol.view_index,
                .offset = fn_def.name.location.start,
                .scope_floor = self.scopes.items.len,
            });
        }
        defer if (self.debug_hook != null) {
            _ = self.debug_stack.pop();
        };

        const saved_view = self.current_view;
        self.current_view = symbol.view_index;
        defer self.current_view = saved_view;

        const saved_bindings = self.current_type_bindings;
        self.current_type_bindings = type_bindings;
        defer self.current_type_bindings = saved_bindings;

        try self.pushFrame(true);
        defer self.popFrame();
        const view_source = self.views[symbol.view_index].source;
        for (fn_def.function.parameters, 0..) |parameter, index| {
            const value: Value = if (index < arguments.len) arguments[index] else .void_value;
            try self.bind(parameter.name.slice(view_source), value);
        }
        const flow = self.execStatement(fn_def.function.body) catch |err| switch (err) {
            // a 'return' raised inside a value-yielding construct lands here
            error.Return => return self.pending_return orelse .void_value,
            else => return err,
        };
        return switch (flow) {
            .return_value => |value| value,
            else => .void_value,
        };
    }

    // the built-in macros (section 6.4)
    fn builtinMacroCall(self: *Interpreter, name: []const u8, call: anytype) Error!?Value {
        if (std.mem.eql(u8, name, "struct_type") or std.mem.eql(u8, name, "enum_type")) {
            const description = try self.arena.create(Value.TypeDescription);
            description.* = .{
                .kind = if (name[0] == 's') .struct_kind else .enum_kind,
                .name = "",
                .primitive = null,
                .members = .empty,
                .interface_names = &.{},
                .origin = null,
            };
            return .{ .type_value = description };
        }
        if (std.mem.eql(u8, name, "type_of")) {
            if (call.arguments.len != 1) return self.fault("'type_of' expects one argument (section 6.4)", .{});
            const hooks = self.reflection orelse return self.fault("reflection is unavailable here", .{});
            const description = (try hooks.reflect_expression(hooks.context, call.arguments[0])) orelse
                return self.fault("the type of this expression is not known at compile time (section 6.4)", .{});
            return .{ .type_value = description };
        }
        if (std.mem.eql(u8, name, "implementers_of")) {
            if (call.arguments.len != 1) return self.fault("'implementers_of' expects one argument (section 6.4)", .{});
            const hooks = self.reflection orelse return self.fault("reflection is unavailable here", .{});
            const argument = try self.evalExpression(call.arguments[0]);
            if (argument != .type_value) {
                return self.fault("'implementers_of' expects an interface (section 6.4)", .{});
            }
            const implementers = (try hooks.reflect_implementers(hooks.context, argument.type_value)) orelse
                return self.fault("'implementers_of' expects an interface (section 6.4)", .{});
            const instance = try self.arena.create(Value.ArrayInstance);
            instance.* = .{ .elements = try self.arena.dupe(Value, implementers) };
            return .{ .array = instance };
        }
        if (std.mem.eql(u8, name, "name_of")) {
            if (call.arguments.len != 1) return self.fault("'name_of' expects one argument (section 6.4)", .{});
            var argument = try self.evalExpression(call.arguments[0]);
            // a borrowed or owned enum reads as its pointee
            while (true) {
                switch (argument) {
                    .reference => |target| argument = target.*,
                    .pointer => |target| argument = (target orelse return self.fault("'name_of' read a moved-from pointer", .{})).*,
                    else => break,
                }
            }
            if (argument != .enum_value) {
                return self.fault("'name_of' expects an enum value (section 6.4)", .{});
            }
            return try self.bytesValue(argument.enum_value.variant);
        }
        return null;
    }

    // compile-time name resolution for macro bodies, which carry no checked
    // call targets: macros match by name, functions by name and arity
    fn comptimeNameCall(self: *Interpreter, name: []const u8, call: anytype) Error!?Value {
        const symbols = self.globals.get(name) orelse return null;
        for (symbols.items) |symbol| {
            if (!self.symbolVisible(symbol)) continue;
            switch (symbol.definition.kind) {
                .macro_def => |macro_def| return try self.callMacro(symbol, macro_def, call),
                .fn_def => |fn_def| {
                    if (fn_def.function.parameters.len == call.arguments.len) {
                        const arguments = try self.evalArguments(call.arguments);
                        return try self.callFunction(symbol, arguments, &.{});
                    }
                },
                else => {},
            }
        }
        return null;
    }

    fn callMacro(self: *Interpreter, symbol: resolution.Symbol, macro_def: ast.MacroDef, call: anytype) Error!Value {
        if (!self.comptime_mode) {
            return self.fault("macros run at compile time; invoke with '#' (section 6.3)", .{});
        }
        if (call.arguments.len != macro_def.parameters.len) {
            return self.fault("this macro expects {d} argument(s), found {d}", .{ macro_def.parameters.len, call.arguments.len });
        }
        // a declaration-only macro is implemented by the compiler
        if (macro_def.body == null) {
            const name = macro_def.name.slice(self.views[symbol.view_index].source);
            return (try self.builtinMacroCall(name, call)) orelse
                self.fault("macro '{s}' is declared but not implemented (section 6.4)", .{name});
        }
        if (self.call_depth >= 1024) return self.fault("call stack exhausted (1024 frames)", .{});
        self.call_depth += 1;
        defer self.call_depth -= 1;

        // arguments evaluate in the caller's view, the body in its own
        const arguments = try self.evalArguments(call.arguments);
        const saved_view = self.current_view;
        self.current_view = symbol.view_index;
        defer self.current_view = saved_view;

        try self.pushFrame(true);
        defer self.popFrame();
        const view_source = self.views[symbol.view_index].source;
        for (macro_def.parameters, 0..) |parameter, index| {
            const value: Value = if (index < arguments.len) arguments[index] else .void_value;
            try self.bind(parameter.name.slice(view_source), value);
        }
        const flow = self.execStatement(macro_def.body.?) catch |err| switch (err) {
            error.Return => return self.pending_return orelse .void_value,
            else => return err,
        };
        return switch (flow) {
            .return_value => |value| value,
            else => .void_value,
        };
    }

    // calls an extension on a value cell, shaping the self argument to the
    // extension's declared indirection (section 4.5)
    fn callWithSelf(self: *Interpreter, symbol: resolution.Symbol, cell: *Value) Error!Value {
        const fn_def = symbol.definition.kind.fn_def;
        const self_value: Value = switch (selfIndirection(fn_def.function.parameters[0].parameter_type)) {
            .reference => .{ .reference = cell },
            .pointer => cell.*,
            .value => try self.deepCopy(cell.*),
        };
        return self.callFunction(symbol, &.{self_value}, &.{});
    }

    fn callClosure(self: *Interpreter, instance: *Value.Closure, arguments: []const Value) Error!Value {
        if (self.call_depth >= 1024) return self.fault("call stack exhausted (1024 frames)", .{});
        self.call_depth += 1;
        defer self.call_depth -= 1;

        const saved_view = self.current_view;
        self.current_view = instance.view_index;
        defer self.current_view = saved_view;

        const saved_bindings = self.current_type_bindings;
        self.current_type_bindings = instance.type_bindings;
        defer self.current_type_bindings = saved_bindings;

        try self.pushFrame(true);
        defer self.popFrame();
        // captured cells first, then parameters (which may shadow them)
        for (instance.environment) |entry| {
            try self.bindCell(entry.name, entry.cell);
        }
        const view_source = self.views[instance.view_index].source;
        for (instance.lambda.function.parameters, 0..) |parameter, index| {
            const value: Value = if (index < arguments.len) arguments[index] else .void_value;
            try self.bind(parameter.name.slice(view_source), value);
        }
        const flow = self.execStatement(instance.lambda.function.body) catch |err| switch (err) {
            error.Return => return self.pending_return orelse .void_value,
            else => return err,
        };
        return switch (flow) {
            .return_value => |value| value,
            else => .void_value,
        };
    }

    // the std::process::arguments lang item (section 5.1a), recognized by
    // its canonical module key and name
    fn processArgumentsLangItem(self: *const Interpreter, symbol: resolution.Symbol) bool {
        if (symbol.definition.kind != .fn_def) return false;
        const key = self.views[symbol.view_index].key orelse return false;
        if (!std.mem.eql(u8, key, "std::process")) return false;
        const name = symbol.definition.kind.fn_def.name.slice(self.views[symbol.view_index].source);
        return std.mem.eql(u8, name, "arguments");
    }

    fn processArgumentsValue(self: *Interpreter) Error!Value {
        if (self.comptime_mode) {
            return self.fault("process arguments are unavailable at compile time (section 6.2)", .{});
        }
        const outer = try self.arena.alloc(Value, self.process_arguments.len);
        for (self.process_arguments, outer) |argument, *slot| {
            const bytes = try self.arena.alloc(Value, argument.len);
            for (argument, bytes) |byte, *element| {
                element.* = .{ .integer = .{ .value = byte, .primitive = .u8 } };
            }
            const inner = try self.arena.create(Value.ArrayInstance);
            inner.* = .{ .elements = bytes };
            slot.* = .{ .slice = inner };
        }
        const instance = try self.arena.create(Value.ArrayInstance);
        instance.* = .{ .elements = outer };
        return .{ .slice = instance };
    }

    fn callExtern(self: *Interpreter, name: []const u8, arguments: []const Value) Error!Value {
        if (self.comptime_mode) {
            return self.fault("extern functions cannot be called at compile time (section 6.2)", .{});
        }
        if (std.mem.eql(u8, name, "printf")) {
            return self.printfExtern(arguments);
        }
        // the std::io externs (section 5.4): C stdio names implemented over
        // the host filesystem, so 'alloyc run' and the debugger execute the
        // standard library unchanged
        if (std.mem.eql(u8, name, "__acrt_iob_func")) {
            if (arguments.len != 1) return self.fault("__acrt_iob_func expects one argument", .{});
            // stream ids pass through: 1 is standard output, 2 standard error
            return .{ .integer = .{ .value = try self.integerOf(arguments[0]), .primitive = .i64 } };
        }
        if (std.mem.eql(u8, name, "fopen")) return self.fopenExtern(arguments);
        if (std.mem.eql(u8, name, "fclose")) return self.fcloseExtern(arguments);
        if (std.mem.eql(u8, name, "fread")) return self.freadExtern(arguments);
        if (std.mem.eql(u8, name, "fwrite")) return self.fwriteExtern(arguments);
        if (std.mem.eql(u8, name, "fseek")) return self.fseekExtern(arguments);
        if (std.mem.eql(u8, name, "ftell")) return self.ftellExtern(arguments);
        return self.fault("extern '{s}' is not available in the interpreter", .{name});
    }

    // handles: 1 and 2 are the standard streams, files count from 3
    const first_file_handle: i128 = 3;

    fn openFileOf(self: *Interpreter, handle: i128) Error!*OpenFile {
        const index = handle - first_file_handle;
        if (index < 0 or index >= self.open_files.items.len) {
            return self.fault("use of an invalid file handle", .{});
        }
        const file = &self.open_files.items[@intCast(index)];
        if (file.closed) return self.fault("use of a closed file handle", .{});
        return file;
    }

    // C strings arriving from Alloy carry an explicit NUL terminator the
    // wrappers appended; the interpreter reads up to it
    fn cString(self: *Interpreter, value: Value) Error![]const u8 {
        const bytes = try self.byteSlice(value);
        const terminator = std.mem.indexOfScalar(u8, bytes, 0) orelse bytes.len;
        return bytes[0..terminator];
    }

    fn fopenExtern(self: *Interpreter, arguments: []const Value) Error!Value {
        if (arguments.len != 2) return self.fault("fopen expects a path and a mode", .{});
        const io = self.host_io orelse return self.fault("file io is unavailable here", .{});
        const path = try self.arena.dupe(u8, try self.cString(arguments[0]));
        const mode = try self.cString(arguments[1]);
        const writing = std.mem.indexOfScalar(u8, mode, 'w') != null;
        const failure: Value = .{ .integer = .{ .value = 0, .primitive = .i64 } };
        var contents: []const u8 = &.{};
        if (!writing) {
            const max_size = 64 * 1024 * 1024;
            contents = std.Io.Dir.cwd().readFileAlloc(io, path, self.arena, .limited(max_size)) catch return failure;
        }
        try self.open_files.append(self.arena, .{
            .path = path,
            .writing = writing,
            .contents = contents,
            .cursor = 0,
            .write_buffer = .empty,
            .closed = false,
        });
        const handle = first_file_handle + @as(i128, @intCast(self.open_files.items.len - 1));
        return .{ .integer = .{ .value = handle, .primitive = .i64 } };
    }

    fn fcloseExtern(self: *Interpreter, arguments: []const Value) Error!Value {
        if (arguments.len != 1) return self.fault("fclose expects a stream", .{});
        const file = try self.openFileOf(try self.integerOf(arguments[0]));
        const success: Value = .{ .integer = .{ .value = 0, .primitive = .i32 } };
        const failure: Value = .{ .integer = .{ .value = -1, .primitive = .i32 } };
        file.closed = true;
        if (file.writing) {
            const io = self.host_io orelse return failure;
            std.Io.Dir.cwd().writeFile(io, .{ .sub_path = file.path, .data = file.write_buffer.items }) catch return failure;
        }
        return success;
    }

    fn freadExtern(self: *Interpreter, arguments: []const Value) Error!Value {
        if (arguments.len != 4) return self.fault("fread expects a buffer, sizes, and a stream", .{});
        const zero: Value = .{ .integer = .{ .value = 0, .primitive = .u64 } };
        const instance = switch (arguments[0]) {
            .slice => |instance| instance,
            .array => |instance| instance,
            .heap_array => |instance| instance orelse return self.fault("use of a moved-from array", .{}),
            else => return self.fault("fread needs a byte buffer", .{}),
        };
        const size = try self.integerOf(arguments[1]);
        const count = try self.integerOf(arguments[2]);
        const file = try self.openFileOf(try self.integerOf(arguments[3]));
        if (size <= 0 or count <= 0 or file.writing) return zero;
        const requested: usize = @intCast(size * count);
        const remaining = file.contents.len - @min(file.cursor, file.contents.len);
        const total = @min(requested, @min(remaining, instance.elements.len));
        for (file.contents[file.cursor..][0..total], instance.elements[0..total]) |byte, *element| {
            element.* = .{ .integer = .{ .value = byte, .primitive = .u8 } };
        }
        file.cursor += total;
        return .{ .integer = .{ .value = @intCast(total / @as(usize, @intCast(size))), .primitive = .u64 } };
    }

    fn fwriteExtern(self: *Interpreter, arguments: []const Value) Error!Value {
        if (arguments.len != 4) return self.fault("fwrite expects a buffer, sizes, and a stream", .{});
        const zero: Value = .{ .integer = .{ .value = 0, .primitive = .u64 } };
        const bytes = try self.byteSlice(arguments[0]);
        const size = try self.integerOf(arguments[1]);
        const count = try self.integerOf(arguments[2]);
        const stream = try self.integerOf(arguments[3]);
        if (size <= 0 or count <= 0) return zero;
        const total = @min(@as(usize, @intCast(size * count)), bytes.len);
        // both standard streams share the interpreter's output writer
        if (stream == 1 or stream == 2) {
            try self.output.writeAll(bytes[0..total]);
        } else {
            const file = try self.openFileOf(stream);
            if (!file.writing) return zero;
            try file.write_buffer.appendSlice(self.arena, bytes[0..total]);
        }
        return .{ .integer = .{ .value = @intCast(total / @as(usize, @intCast(size))), .primitive = .u64 } };
    }

    fn fseekExtern(self: *Interpreter, arguments: []const Value) Error!Value {
        if (arguments.len != 3) return self.fault("fseek expects a stream, an offset, and an origin", .{});
        const file = try self.openFileOf(try self.integerOf(arguments[0]));
        const offset = try self.integerOf(arguments[1]);
        const origin = try self.integerOf(arguments[2]);
        const failure: Value = .{ .integer = .{ .value = -1, .primitive = .i32 } };
        const base: i128 = switch (origin) {
            0 => 0,
            1 => @intCast(file.cursor),
            2 => @intCast(file.contents.len),
            else => return failure,
        };
        const target = base + offset;
        if (target < 0 or target > file.contents.len) return failure;
        file.cursor = @intCast(target);
        return .{ .integer = .{ .value = 0, .primitive = .i32 } };
    }

    fn ftellExtern(self: *Interpreter, arguments: []const Value) Error!Value {
        if (arguments.len != 1) return self.fault("ftell expects a stream", .{});
        const file = try self.openFileOf(try self.integerOf(arguments[0]));
        return .{ .integer = .{ .value = @intCast(file.cursor), .primitive = .i32 } };
    }

    fn printfExtern(self: *Interpreter, arguments: []const Value) Error!Value {
        if (arguments.len == 0) return .{ .integer = .{ .value = 0, .primitive = .i32 } };
        const format = try self.byteSlice(arguments[0]);
        var argument_index: usize = 1;
        var index: usize = 0;
        while (index < format.len) : (index += 1) {
            const byte = format[index];
            if (byte != '%' or index + 1 >= format.len) {
                try self.output.writeByte(byte);
                continue;
            }
            index += 1;
            const specifier = format[index];
            if (specifier == '%') {
                try self.output.writeByte('%');
                continue;
            }
            const argument: Value = if (argument_index < arguments.len) arguments[argument_index] else .void_value;
            argument_index += 1;
            switch (specifier) {
                'd', 'i', 'u' => {
                    const value = try self.integerOf(argument);
                    try self.output.print("{d}", .{value});
                },
                'f' => {
                    const value = try self.floatOf(argument);
                    try self.output.print("{d:.6}", .{value});
                },
                'c' => {
                    const value = try self.integerOf(argument);
                    try self.output.writeByte(@intCast(@as(u8, @truncate(@as(u128, @bitCast(value))))));
                },
                's' => {
                    const bytes = try self.byteSlice(argument);
                    try self.output.writeAll(bytes);
                },
                else => return self.fault("printf specifier '%{c}' is not supported", .{specifier}),
            }
        }
        return .{ .integer = .{ .value = @intCast(format.len), .primitive = .i32 } };
    }

    fn byteSlice(self: *Interpreter, value: Value) Error![]const u8 {
        const instance = switch (value) {
            .slice => |instance| instance,
            .array => |instance| instance,
            .heap_array => |instance| instance orelse return self.fault("use of a moved-from array", .{}),
            else => return self.fault("expected a byte array", .{}),
        };
        const bytes = try self.arena.alloc(u8, instance.elements.len);
        for (instance.elements, bytes) |element, *byte| {
            byte.* = @intCast(@as(u8, @truncate(@as(u128, @bitCast(try self.integerOf(element))))));
        }
        return bytes;
    }

    // a statement-position if passes 'yield' through to the enclosing
    // value construct; only a value-position if consumes it (section 4.3)
    fn evalIf(self: *Interpreter, if_expr: ast.IfExpression, as_value: bool) Error!Value {
        var taken = false;

        try self.pushFrame(false);
        if (if_expr.capture) |capture| {
            // 'if (x is Variant) |payload|' binds when the variant matches
            taken = try self.bindIsCapture(if_expr.condition, capture);
        } else {
            const condition = try self.evalExpression(if_expr.condition);
            taken = condition == .bool_value and condition.bool_value;
        }
        if (taken) {
            const flow = if (as_value) try self.execYieldingBody(if_expr.then_branch) else try self.execStatement(if_expr.then_branch);
            self.popFrame();
            switch (flow) {
                .yield_value => |value| return if (as_value) value else self.flowYield(flow),
                .break_value => return self.flowBreak(flow),
                .return_value => return self.flowReturn(flow),
                .normal => {},
            }
        } else {
            self.popFrame();
            if (if_expr.else_branch) |else_branch| {
                const flow = if (as_value) try self.execYieldingBody(else_branch) else try self.execStatement(else_branch);
                switch (flow) {
                    .yield_value => |value| return if (as_value) value else self.flowYield(flow),
                    .break_value => return self.flowBreak(flow),
                    .return_value => return self.flowReturn(flow),
                    .normal => {},
                }
            }
        }
        return .void_value;
    }

    // a 'return' inside a value-yielding construct unwinds to the enclosing
    // function call via error.Return
    fn flowReturn(self: *Interpreter, flow: Flow) Error!Value {
        self.pending_return = flow.return_value;
        return error.Return;
    }

    // a 'break' inside an if or match unwinds to the enclosing loop
    fn flowBreak(self: *Interpreter, flow: Flow) Error!Value {
        self.pending_break = flow.break_value;
        return error.Break;
    }

    // a 'yield' inside a loop unwinds to the enclosing if or match
    fn flowYield(self: *Interpreter, flow: Flow) Error!Value {
        self.pending_yield = flow.yield_value;
        return error.Yield;
    }

    // a loop body: a nested if or match rethrew 'break' on the error
    // channel; the loop is where it lands (section 4.3)
    fn execLoopBody(self: *Interpreter, statement: *const ast.Statement) Error!Flow {
        return self.execStatement(statement) catch |err| switch (err) {
            error.Break => .{ .break_value = self.pending_break },
            else => err,
        };
    }

    // an if or match body: a nested loop rethrew 'yield' on the error
    // channel; the value construct is where it lands (section 4.3)
    fn execYieldingBody(self: *Interpreter, statement: *const ast.Statement) Error!Flow {
        return self.execStatement(statement) catch |err| switch (err) {
            error.Yield => .{ .yield_value = self.pending_yield },
            else => err,
        };
    }

    fn bindIsCapture(self: *Interpreter, condition: *const ast.Expression, capture: ast.Capture) Error!bool {
        const unwrapped = unwrapGrouped(condition);
        if (unwrapped.* != .cast or unwrapped.cast.operator.tag != .keyword_is) return false;
        const cast = unwrapped.cast;
        const place = try self.evalPlace(cast.operand) orelse return false;
        const pierced = try self.pierceCell(place);
        const target_token = cast.target.named.path[cast.target.named.path.len - 1];
        const target_name = target_token.slice(self.source());
        // a downcast: the capture borrows the concrete value in place,
        // mirroring the subject's indirection (section 3.2)
        if (pierced.* == .struct_value) {
            if (self.type_targets.get(unwrapped)) |target| {
                const identity = pierced.struct_value.identity orelse return false;
                if (identity.definition != target.definition) return false;
            } else if (!std.mem.eql(u8, pierced.struct_value.type_name, target_name)) {
                return false;
            }
            try self.bind(capture.name.slice(self.source()), .{ .reference = pierced });
            return true;
        }
        if (pierced.* != .enum_value) return self.fault("'is' needs an enum or interface-object subject", .{});
        if (!std.mem.eql(u8, pierced.enum_value.variant, target_name)) return false;
        const payload = pierced.enum_value.payload orelse return true;
        try self.bindCaptured(capture, payload, pierced);
        return true;
    }

    // capture typing (section 2.1): deep copy by default, '&' borrows in
    // place, '*' takes a pointer payload out of the subject
    fn bindCaptured(self: *Interpreter, capture: ast.Capture, payload: Value, subject: *Value) Error!void {
        const name = capture.name.slice(self.source());
        const mode = captureMode(capture);
        switch (mode) {
            .copy => try self.bind(name, try self.deepCopy(payload)),
            .reference => {
                if (subject.* == .enum_value) {
                    const cell: *Value = &subject.enum_value.payload.?;
                    try self.bind(name, .{ .reference = cell });
                } else {
                    try self.bind(name, try self.deepCopy(payload));
                }
            },
            .owning => {
                try self.bind(name, payload);
                // the payload moves out; the subject is moved-from
                if (subject.* == .enum_value) {
                    subject.enum_value.payload = switch (payload) {
                        .pointer => .{ .pointer = null },
                        .heap_array => .{ .heap_array = null },
                        else => payload,
                    };
                }
            },
        }
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

    fn evalWhile(self: *Interpreter, while_expr: ast.WhileExpression) Error!Value {
        while (true) {
            const condition = try self.evalExpression(while_expr.condition);
            if (condition != .bool_value or !condition.bool_value) break;
            const flow = try self.execLoopBody(while_expr.body);
            switch (flow) {
                .break_value => |value| return value orelse .void_value,
                .yield_value => return self.flowYield(flow),
                .return_value => return self.flowReturn(flow),
                .normal => {},
            }
        }
        if (while_expr.else_branch) |else_branch| {
            const flow = try self.execLoopBody(else_branch);
            switch (flow) {
                .break_value => |value| return value orelse .void_value,
                .yield_value => return self.flowYield(flow),
                .return_value => return self.flowReturn(flow),
                .normal => {},
            }
        }
        return .void_value;
    }

    fn evalFor(self: *Interpreter, for_expr: ast.ForExpression) Error!Value {
        const Subject = union(enum) {
            counter: struct { next: i128, end: i128, primitive: types.Primitive },
            elements: *Value.ArrayInstance,
            // the cursor protocol (section 4.3): 'next()' advances this cell
            cursor: struct { next_symbol: resolution.Symbol, cell: *Value },
        };
        var subjects: std.ArrayList(Subject) = .empty;
        var length: ?usize = null;
        for (for_expr.subjects) |subject_expression| {
            if (subject_expression.* == .array_range) {
                const bounds = try self.rangeBounds(subject_expression.array_range);
                try subjects.append(self.arena, .{ .counter = .{ .next = bounds.start, .end = bounds.end, .primitive = .i32 } });
                try self.checkLockstep(&length, @intCast(bounds.end - bounds.start));
                continue;
            }
            const pierced: *Value = pierced: {
                if (try self.evalPlace(subject_expression)) |place| {
                    break :pierced try self.pierceCell(place);
                }
                const cell = try self.arena.create(Value);
                cell.* = try self.evalExpression(subject_expression);
                break :pierced cell;
            };
            if (pierced.* == .struct_value) {
                const type_name = pierced.struct_value.type_name;
                const identity = pierced.struct_value.identity orelse
                    return self.fault("'{s}' is not iterable: no 'iterator()' extension found (section 4.3)", .{type_name});
                const iterator_symbol = self.findDispatchTarget(identity, "iterator") orelse
                    return self.fault("'{s}' is not iterable: no 'iterator()' extension found (section 4.3)", .{type_name});
                const cursor_value = try self.callWithSelf(iterator_symbol, pierced);
                const cursor_cell = try self.arena.create(Value);
                cursor_cell.* = cursor_value;
                const cursor_name = if (cursor_value == .struct_value) cursor_value.struct_value.type_name else "";
                const cursor_identity = if (cursor_value == .struct_value) cursor_value.struct_value.identity else null;
                const next_symbol = (if (cursor_identity) |alive| self.findDispatchTarget(alive, "next") else null) orelse
                    return self.fault("the cursor '{s}' has no 'next()' extension (section 4.3)", .{cursor_name});
                try subjects.append(self.arena, .{ .cursor = .{ .next_symbol = next_symbol, .cell = cursor_cell } });
                continue;
            }
            const instance = try self.subjectInstance(pierced);
            try subjects.append(self.arena, .{ .elements = instance });
            try self.checkLockstep(&length, instance.elements.len);
        }

        var iteration: usize = 0;
        while (true) : (iteration += 1) {
            // every subject must produce an element this pass or the loop
            // ends; a partial pass is a length mismatch (section 4.3)
            var any_produced = false;
            var any_exhausted = false;
            if (length) |total| {
                if (iteration < total) any_produced = true else any_exhausted = true;
            }
            const payloads = try self.arena.alloc(?Value, subjects.items.len);
            for (subjects.items, payloads) |subject, *payload| {
                payload.* = null;
                if (subject != .cursor) continue;
                const result = try self.callWithSelf(subject.cursor.next_symbol, subject.cursor.cell);
                if (result == .enum_value and std.mem.eql(u8, result.enum_value.variant, "Some")) {
                    payload.* = result.enum_value.payload orelse .void_value;
                    any_produced = true;
                } else {
                    any_exhausted = true;
                }
            }
            if (any_exhausted) {
                if (any_produced) return self.fault("for subjects disagree on length (section 4.3)", .{});
                break;
            }
            if (!any_produced) break;

            try self.pushFrame(false);
            for (for_expr.captures, 0..) |capture, capture_index| {
                if (capture_index >= subjects.items.len) break;
                switch (subjects.items[capture_index]) {
                    .counter => |counter| {
                        const value: Value = .{ .integer = .{ .value = counter.next + @as(i128, @intCast(iteration)), .primitive = counter.primitive } };
                        try self.bind(capture.name.slice(self.source()), value);
                    },
                    .elements => |instance| {
                        const cell = &instance.elements[iteration];
                        switch (captureMode(capture)) {
                            .copy => try self.bind(capture.name.slice(self.source()), try self.deepCopy(cell.*)),
                            .reference => try self.bind(capture.name.slice(self.source()), .{ .reference = cell }),
                            .owning => try self.bind(capture.name.slice(self.source()), cell.*),
                        }
                    },
                    .cursor => {
                        const value = payloads[capture_index] orelse .void_value;
                        try self.bind(capture.name.slice(self.source()), value);
                    },
                }
            }
            const flow = try self.execLoopBody(for_expr.body);
            self.popFrame();
            switch (flow) {
                .break_value => |value| return value orelse .void_value,
                .yield_value => return self.flowYield(flow),
                .return_value => return self.flowReturn(flow),
                .normal => {},
            }
        }
        if (for_expr.else_branch) |else_branch| {
            const flow = try self.execLoopBody(else_branch);
            switch (flow) {
                .break_value => |value| return value orelse .void_value,
                .yield_value => return self.flowYield(flow),
                .return_value => return self.flowReturn(flow),
                .normal => {},
            }
        }
        return .void_value;
    }

    // multi-subject loops iterate in lockstep over equal lengths (section
    // 4.3); a mismatch is a runtime fault
    fn checkLockstep(self: *Interpreter, length: *?usize, candidate: usize) Error!void {
        if (length.*) |existing| {
            if (existing != candidate) {
                return self.fault("for subjects disagree on length: {d} versus {d} (section 4.3)", .{ existing, candidate });
            }
        } else {
            length.* = candidate;
        }
    }

    fn subjectInstance(self: *Interpreter, value: *Value) Error!*Value.ArrayInstance {
        return switch (value.*) {
            .array => |instance| instance,
            .slice => |instance| instance,
            .heap_array => |instance| instance orelse self.fault("use of a moved-from array", .{}),
            else => self.fault("this subject is not iterable", .{}),
        };
    }

    const RangeBounds = struct { start: i128, end: i128 };

    fn rangeBounds(self: *Interpreter, array_range: anytype) Error!RangeBounds {
        const start: i128 = if (array_range.start) |expression| try self.integerOf(try self.evalExpression(expression)) else 0;
        const end: i128 = try self.integerOf(try self.evalExpression(array_range.end));
        if (end < start) return self.fault("range end {d} is below start {d}", .{ end, start });
        return .{ .start = start, .end = end };
    }

    // a statement-position match passes 'yield' through like an if
    fn evalMatch(self: *Interpreter, match_expr: ast.MatchExpression, as_value: bool) Error!Value {
        const subject_place = try self.evalPlace(match_expr.subject);
        var subject_storage: Value = undefined;
        const subject: *Value = if (subject_place) |place| try self.pierceCell(place) else subject: {
            subject_storage = try self.evalExpression(match_expr.subject);
            break :subject &subject_storage;
        };

        for (match_expr.arms) |arm| {
            const matched = if (arm.pattern) |pattern| try self.matchesPattern(subject, pattern) else true;
            if (!matched) continue;
            try self.pushFrame(false);
            if (arm.capture) |capture| {
                if (subject.* == .enum_value) {
                    if (subject.enum_value.payload) |payload| {
                        try self.bindCaptured(capture, payload, subject);
                    }
                } else if (subject.* == .struct_value) {
                    // an interface-object arm borrows the concrete value
                    try self.bind(capture.name.slice(self.source()), .{ .reference = subject });
                }
            }
            const flow = if (as_value) try self.execYieldingBody(arm.body) else try self.execStatement(arm.body);
            self.popFrame();
            switch (flow) {
                .yield_value => |value| return if (as_value) value else self.flowYield(flow),
                .break_value => return self.flowBreak(flow),
                .return_value => return self.flowReturn(flow),
                // the arm completed without 'yield': the external else runs
                .normal => {
                    if (match_expr.else_branch) |else_branch| {
                        const else_flow = if (as_value) try self.execYieldingBody(else_branch) else try self.execStatement(else_branch);
                        switch (else_flow) {
                            .yield_value => |value| return if (as_value) value else self.flowYield(else_flow),
                            .break_value => return self.flowBreak(else_flow),
                            .return_value => return self.flowReturn(else_flow),
                            .normal => {},
                        }
                    }
                    return .void_value;
                },
            }
        }
        return .void_value;
    }

    fn matchesPattern(self: *Interpreter, subject: *Value, pattern: *const ast.Expression) Error!bool {
        const unwrapped = unwrapGrouped(pattern);
        if (subject.* == .enum_value) {
            const variant_name = switch (unwrapped.*) {
                .path => |path| path[path.len - 1].slice(self.source()),
                .implied_variant => |token| token.slice(self.source()),
                else => return false,
            };
            return std.mem.eql(u8, subject.enum_value.variant, variant_name);
        }
        // an interface-object subject matches arms naming its concrete type
        if (subject.* == .struct_value and subject.struct_value.type_name.len != 0) {
            if (self.type_targets.get(pattern)) |target| {
                const identity = subject.struct_value.identity orelse return false;
                return identity.definition == target.definition;
            }
            const type_name = switch (unwrapped.*) {
                .path => |path| path[path.len - 1].slice(self.source()),
                else => return false,
            };
            return std.mem.eql(u8, subject.struct_value.type_name, type_name);
        }
        const pattern_value = try self.evalExpression(pattern);
        if (subject.* == .integer and pattern_value == .integer) {
            return subject.integer.value == pattern_value.integer.value;
        }
        if (subject.* == .slice and pattern_value == .slice) {
            const left = try self.byteSlice(subject.*);
            const right = try self.byteSlice(pattern_value);
            return std.mem.eql(u8, left, right);
        }
        return false;
    }

    fn integerOf(self: *Interpreter, value: Value) Error!i128 {
        return switch (value) {
            .integer => |integer| integer.value,
            .bool_value => |flag| @intFromBool(flag),
            else => self.fault("expected an integer value", .{}),
        };
    }

    fn integerValue(self: *Interpreter, value: Value) Error!Value.Integer {
        return switch (value) {
            .integer => |integer| integer,
            else => self.fault("expected an integer value", .{}),
        };
    }

    fn floatOf(self: *Interpreter, value: Value) Error!f64 {
        return switch (value) {
            .float => |float| float.value,
            .integer => |integer| @floatFromInt(integer.value),
            else => self.fault("expected a numeric value", .{}),
        };
    }

    fn filledArray(self: *Interpreter, value: Value, count: i128) Error!*Value.ArrayInstance {
        if (count < 0) return self.fault("array fill count is negative", .{});
        const elements = try self.arena.alloc(Value, @intCast(count));
        for (elements) |*slot| {
            slot.* = try self.deepCopy(value);
        }
        const instance = try self.arena.create(Value.ArrayInstance);
        instance.* = .{ .elements = elements };
        return instance;
    }
};

fn unwrapGrouped(expression: *const ast.Expression) *const ast.Expression {
    var current = expression;
    while (current.* == .grouped) current = current.grouped;
    return current;
}

pub fn parseIntegerText(text: []const u8) !i128 {
    if (text.len > 2 and text[0] == '0') {
        switch (text[1]) {
            'x', 'X' => return std.fmt.parseInt(i128, text[2..], 16),
            'b', 'B' => return std.fmt.parseInt(i128, text[2..], 2),
            'o', 'O' => return std.fmt.parseInt(i128, text[2..], 8),
            else => {},
        }
    }
    return std.fmt.parseInt(i128, text, 10);
}

fn minimumOf(primitive: types.Primitive) i128 {
    return switch (primitive) {
        .u8, .u16, .u32, .u64, .bool => 0,
        .i8 => std.math.minInt(i8),
        .i16 => std.math.minInt(i16),
        .i32 => std.math.minInt(i32),
        .i64 => std.math.minInt(i64),
        .f32, .f64 => 0,
    };
}

fn maximumOf(primitive: types.Primitive) i128 {
    return switch (primitive) {
        .u8 => std.math.maxInt(u8),
        .u16 => std.math.maxInt(u16),
        .u32 => std.math.maxInt(u32),
        .u64 => std.math.maxInt(u64),
        .i8 => std.math.maxInt(i8),
        .i16 => std.math.maxInt(i16),
        .i32 => std.math.maxInt(i32),
        .i64 => std.math.maxInt(i64),
        .bool => 1,
        .f32, .f64 => 0,
    };
}

// wraps a wide value into the primitive's range (two's complement)
fn truncateToPrimitive(value: i128, primitive: types.Primitive) i128 {
    return switch (primitive) {
        .u8 => @as(u8, @truncate(@as(u128, @bitCast(value)))),
        .u16 => @as(u16, @truncate(@as(u128, @bitCast(value)))),
        .u32 => @as(u32, @truncate(@as(u128, @bitCast(value)))),
        .u64 => @as(u64, @truncate(@as(u128, @bitCast(value)))),
        .i8 => @as(i8, @truncate(value)),
        .i16 => @as(i16, @truncate(value)),
        .i32 => @as(i32, @truncate(value)),
        .i64 => @as(i64, @truncate(value)),
        .bool => @intFromBool(value != 0),
        .f32, .f64 => value,
    };
}
