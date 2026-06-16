//! Type checking, the whole-program stage after name resolution. Implements
//! section 3 of LANGUAGE_SPEC.md (type compatibility, casting, overloading,
//! inference, mutability) together with the typing consequences of pointee
//! transparency (section 4.2): reading a pointer or reference always yields
//! its pointee, so only 'move', 'new', and unary '&' see the address itself.
//!
//! Also covered: extension function calls and receiver resolution (section
//! 4.5), interface objects, verification, default implementations, and
//! downcasting via 'is' and 'match' (sections 3.2 and 5.2), generic
//! constraint enforcement (section 3.7), the cursor protocol for custom
//! iterables (section 4.3), and compile-time evaluation (section 6):
//! '#' expressions are collected during checking with a snapshot of the
//! visible constants and evaluated through the interpreter in sandboxed
//! mode once every side table is complete. Macro calls and '#Type'
//! reflection (sections 6.3 and 3.4) instead evaluate eagerly at the
//! invocation site, so they may only use earlier definitions.
//!
//! Data layout (section 3.9) is computed by layoutOf: C-compatible struct
//! layout and tag-plus-payload enums, which gives 'as' reinterpretation
//! (section 3.5) its width check on every type.
//!
//! Not yet covered, deliberately: flow analysis (a value-yielding 'if'
//! simply requires an 'else' branch).

const std = @import("std");
const Token = @import("tokenizer.zig").Token;
const ast = @import("ast.zig");
const types = @import("types.zig");
const Type = types.Type;
const resolution = @import("resolution.zig");
const Diagnostic = @import("diagnostic.zig").Diagnostic;
const Interpreter = @import("interpreter.zig").Interpreter;

pub const Checker = struct {
    arena: std.mem.Allocator,
    views: []const resolution.ModuleView,
    globals: *const std.StringHashMapUnmanaged(resolution.SymbolList),
    diagnostics: *std.ArrayList(Diagnostic),
    diagnostics_allocator: std.mem.Allocator,
    current_view: usize,
    scopes: std.ArrayList(Frame),
    // the enclosing function's return type; null infers (lambda inference)
    return_type: ?*const Type,
    inferred_return: ?*const Type,
    // the enclosing function's type parameters, visible to every type
    // expression written inside the body (section 3.7)
    scope_types: *const TypeEnvironment,
    // innermost-first stack of value-yielding constructs 'break' can target
    yield_frames: std.ArrayList(YieldFrame),
    // outputs for the later stages: the type of every expression, and the
    // overload each call site resolved to (keyed by the call expression)
    expression_types: std.AutoHashMapUnmanaged(*const ast.Expression, *const Type) = .empty,
    call_targets: std.AutoHashMapUnmanaged(*const ast.Expression, resolution.Symbol) = .empty,
    call_type_bindings: std.AutoHashMapUnmanaged(*const ast.Expression, []const Type.Binding) = .empty,
    // values computed by compile-time evaluation (section 6.1), keyed by the
    // '#' expression node; evaluation runs after checking so every side
    // table is complete (see runPendingComptime)
    comptime_values: std.AutoHashMapUnmanaged(*const ast.Expression, Interpreter.Value) = .empty,
    // serialization shapes per non-primitive 'as' cast (section 3.5), so the
    // interpreter reinterprets values without re-deriving type structure
    cast_shapes: std.AutoHashMapUnmanaged(*const ast.Expression, types.CastShapes) = .empty,
    pending_comptime: std.ArrayList(PendingComptime) = .empty,
    // synthesised types per 'type T = #...' expression (section 3.4)
    comptime_type_cache: std.AutoHashMapUnmanaged(*const ast.Expression, *const Type) = .empty,
    // nonzero while checking inside a '#' expression: macro calls are only
    // legal there (section 6.3)
    comptime_depth: usize = 0,

    const Frame = struct {
        bindings: std.ArrayList(Binding),
        barrier: bool,
    };

    const Binding = struct {
        name: []const u8,
        binding_type: *const Type,
        mutable: bool,
        // a const binding's initializer, the source of compile-time
        // constants visible to '#' expressions (section 6.1)
        initializer: ?*const ast.Expression = null,
    };

    const PendingComptime = struct {
        outer: *const ast.Expression,
        inner: *const ast.Expression,
        view_index: usize,
        environment: []const ComptimeEnvironmentEntry,
    };

    const ComptimeEnvironmentEntry = struct {
        name: []const u8,
        // null poisons the name: a runtime binding shadows any constant
        initializer: ?*const ast.Expression,
    };

    const YieldFrame = struct {
        yielded: ?*const Type,
    };

    pub const Error = error{OutOfMemory};

    const unknown_type: Type = .unknown;
    const void_type: Type = .void_type;
    const bool_type: Type = .{ .primitive = .bool

    };

    pub fn init(
        arena: std.mem.Allocator,
        views: []const resolution.ModuleView,
        globals: *const std.StringHashMapUnmanaged(resolution.SymbolList),
        diagnostics: *std.ArrayList(Diagnostic),
        diagnostics_allocator: std.mem.Allocator,
    ) Checker {
        return .{
            .arena = arena,
            .views = views,
            .globals = globals,
            .diagnostics = diagnostics,
            .diagnostics_allocator = diagnostics_allocator,
            .current_view = 0,
            .scopes = .empty,
            .return_type = null,
            .inferred_return = null,
            .scope_types = &empty_type_environment,
            .yield_frames = .empty,
        };
    }

    pub fn run(self: *Checker) Error!void {
        try self.checkRedeclarations();
        for (self.views, 0..) |view, view_index| {
            self.current_view = view_index;
            for (view.module.definitions) |*definition| {
                try self.checkDefinition(definition);
            }
        }
        try self.runPendingComptime();
    }

    // section 3.6: functions overload only when their parameter type lists
    // differ; two definitions that share a name with an identical parameter
    // type list are a redeclaration. The resolver permits any fn/fn name
    // sharing and defers identical-signature detection to here, where the
    // parameter types are resolved. (Non-function clashes are caught earlier.)
    fn checkRedeclarations(self: *Checker) Error!void {
        var entries = self.globals.iterator();
        while (entries.next()) |entry| {
            const symbols = entry.value_ptr.items;
            if (symbols.len < 2) continue;
            for (symbols, 0..) |symbol, index| {
                if (symbol.definition.kind != .fn_def) continue;
                for (symbols[0..index]) |previous| {
                    if (previous.definition.kind != .fn_def) continue;
                    if (!try self.sameSignature(previous, symbol)) continue;
                    const previous_path = self.views[previous.view_index].path;
                    self.current_view = symbol.view_index;
                    try self.report(symbol.definition.kind.fn_def.name.location, "redeclaration of '{s}'; an overload with an identical parameter type list already exists in {s} (section 3.6)", .{ entry.key_ptr.*, previous_path });
                    break;
                }
            }
        }
    }

    // whether two functions share a parameter type list (section 3.6). Any
    // type-resolution diagnostics it triggers are rolled back, since the main
    // check pass reports those in context.
    fn sameSignature(self: *Checker, a: resolution.Symbol, b: resolution.Symbol) Error!bool {
        const a_fn = a.definition.kind.fn_def;
        const b_fn = b.definition.kind.fn_def;
        if (a_fn.function.parameters.len != b_fn.function.parameters.len) return false;
        const snapshot = self.diagnostics.items.len;
        defer self.diagnostics.shrinkRetainingCapacity(snapshot);
        const a_env = try self.typeParameterEnvironment(a_fn.type_parameters, a.view_index);
        const b_env = try self.typeParameterEnvironment(b_fn.type_parameters, b.view_index);
        for (a_fn.function.parameters, b_fn.function.parameters) |a_param, b_param| {
            const a_type = try self.typeFromExpressionIn(a_param.parameter_type, a_env, a.view_index);
            const b_type = try self.typeFromExpressionIn(b_param.parameter_type, b_env, b.view_index);
            if (!a_type.eql(b_type)) return false;
        }
        return true;
    }

    fn checkDefinition(self: *Checker, definition: *const ast.Definition) Error!void {
        switch (definition.kind) {
            .fn_def => |fn_def| {
                try self.pushFrame(false);
                defer self.popFrame();
                const environment = try self.typeParameterEnvironment(fn_def.type_parameters, self.current_view);
                for (fn_def.function.parameters, 0..) |parameter, index| {
                    // 'self' marks an extension receiver and is only valid on
                    // the first parameter (section 4.5)
                    if (index != 0 and parameter.is_self) {
                        try self.report(parameter.name.location, "'self' may only appear on the first parameter (section 4.5)", .{});
                    }
                    const parameter_type = try self.typeFromExpression(parameter.parameter_type, environment);
                    try self.bind(parameter.name, parameter_type, false);
                }
                const declared_return: *const Type = if (fn_def.function.return_type) |return_expression|
                    try self.typeFromExpression(return_expression, environment)
                else
                    &void_type;
                const saved_return = self.return_type;
                self.return_type = declared_return;
                defer self.return_type = saved_return;
                const saved_scope_types = self.scope_types;
                self.scope_types = environment;
                defer self.scope_types = saved_scope_types;
                try self.checkStatement(fn_def.function.body);
            },
            .macro_def => |macro_def| {
                // macro bodies run in the compile-time interpreter (section
                // 6.3); they are name-resolved but not yet type checked
                _ = macro_def;
            },
            .type_def => |type_def| try self.verifyInterfaces(definition, type_def),
            // extern and interface declarations carry no bodies; their type
            // expressions were validated during name resolution and are
            // materialized on use
            .extern_def, .interface_def => {},
        }
    }

    // static verification that a marked type satisfies its interfaces
    // (section 5.2, compilation and verification mechanics)
    fn verifyInterfaces(self: *Checker, definition: *const ast.Definition, type_def: ast.TypeDef) Error!void {
        if (type_def.interfaces.len == 0) return;
        const type_name = type_def.name.slice(self.source());
        const environment = try self.typeParameterEnvironment(type_def.type_parameters, self.current_view);
        var arguments: std.ArrayList(*const Type) = .empty;
        for (type_def.type_parameters) |type_parameter| {
            try arguments.append(self.arena, environment.get(type_parameter.name.slice(self.source())).?);
        }
        const concrete = try self.makeType(.{ .declared = .{
            .definition = definition,
            .view_index = self.current_view,
            .name = type_name,
            .arguments = try arguments.toOwnedSlice(self.arena),
        } });
        for (type_def.interfaces) |marker| {
            const marker_name = marker.slice(self.source());
            const symbols = self.globals.get(marker_name) orelse continue;
            const symbol = symbols.items[0];
            // a non-interface marker was already reported during resolution
            if (symbol.definition.kind != .interface_def) continue;
            try self.verifyInterface(concrete, .{
                .definition = symbol.definition,
                .view_index = symbol.view_index,
                .name = marker_name,
            }, marker.location);
        }
    }

    fn verifyInterface(self: *Checker, concrete: *const Type, interface: Type.Interface, span: Token.Location) Error!void {
        const interface_def = interface.definition.kind.interface_def;
        const interface_source = self.views[interface.view_index].source;
        for (interface_def.functions) |function| {
            switch (try self.findSatisfyingExtension(concrete, interface, function)) {
                .satisfied => {},
                .missing => try self.report(span, "'{s}' does not implement '{s}': no extension function '{s}' is visible (section 5.2)", .{
                    concrete.declared.name,
                    interface.name,
                    function.name.slice(interface_source),
                }),
                .mismatched => try self.report(span, "the extension '{s}' for '{s}' does not match the signature declared by '{s}' (section 5.2)", .{
                    function.name.slice(interface_source),
                    concrete.declared.name,
                    interface.name,
                }),
            }
        }
    }

    const SatisfactionVerdict = enum { satisfied, missing, mismatched };

    // searches the merged unit for an extension satisfying one interface
    // function: a type-specific extension or an interface default, with the
    // parameter sequence and return type matching precisely (section 5.2)
    fn findSatisfyingExtension(self: *Checker, concrete: *const Type, interface: Type.Interface, function: ast.InterfaceFn) Error!SatisfactionVerdict {
        const interface_source = self.views[interface.view_index].source;
        const function_name = function.name.slice(interface_source);
        const symbols = self.globals.get(function_name) orelse return .missing;
        var found_receiver = false;
        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            const parameters = fn_def.function.parameters;
            if (parameters.len == 0 or !parameters[0].is_self) continue;
            const candidate_environment = try self.typeParameterEnvironment(fn_def.type_parameters, symbol.view_index);
            const self_type = try self.typeFromExpressionIn(parameters[0].parameter_type, candidate_environment, symbol.view_index);
            const self_child = try self.pierce(self_type);
            const receiver_matches = switch (self_child.*) {
                .interface => |child| child.definition == interface.definition,
                .declared => |child| child.definition == concrete.declared.definition,
                else => false,
            };
            if (!receiver_matches) continue;
            found_receiver = true;
            if (parameters.len - 1 != function.parameters.len) continue;
            var matches = true;
            for (parameters[1..], function.parameters) |candidate_parameter, declared_parameter| {
                const candidate_type = try self.typeFromExpressionIn(candidate_parameter.parameter_type, candidate_environment, symbol.view_index);
                const declared_type = try self.typeFromExpressionIn(declared_parameter.parameter_type, &empty_type_environment, interface.view_index);
                if (!candidate_type.eql(declared_type)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                const candidate_return: *const Type = if (fn_def.function.return_type) |expression|
                    try self.typeFromExpressionIn(expression, candidate_environment, symbol.view_index)
                else
                    &void_type;
                const declared_return: *const Type = if (function.return_type) |expression|
                    try self.typeFromExpressionIn(expression, &empty_type_environment, interface.view_index)
                else
                    &void_type;
                if (!candidate_return.eql(declared_return)) matches = false;
            }
            if (matches) return .satisfied;
        }
        return if (found_receiver) .mismatched else .missing;
    }

    pub const TypeEnvironment = std.StringHashMapUnmanaged(*const Type);

    fn typeParameterEnvironment(self: *Checker, type_parameters: []const ast.TypeParameter, view_index: usize) Error!*TypeEnvironment {
        const view_source = self.views[view_index].source;
        const environment = try self.arena.create(TypeEnvironment);
        environment.* = .empty;
        for (type_parameters) |type_parameter| {
            const name = type_parameter.name.slice(view_source);
            const constraint = self.interfaceOfConstraint(type_parameter.constraint, view_source);
            const parameter = try self.makeType(.{ .type_parameter = .{ .name = name, .constraint = constraint } });
            try environment.put(self.arena, name, parameter);
        }
        return environment;
    }

    // resolves a constraint token ('T: Number') to its interface
    fn interfaceOfConstraint(self: *const Checker, constraint_token: ?Token, view_source: []const u8) ?Type.Interface {
        const token = constraint_token orelse return null;
        const name = token.slice(view_source);
        const symbols = self.globals.get(name) orelse return null;
        const symbol = symbols.items[0];
        if (symbol.definition.kind != .interface_def) return null;
        return .{
            .definition = symbol.definition,
            .view_index = symbol.view_index,
            .name = name,
        };
    }

    // builds a Type from a syntactic type expression (section 3.2)
    fn typeFromExpression(self: *Checker, expression: *const ast.TypeExpression, environment: *const TypeEnvironment) Error!*const Type {
        return self.typeFromExpressionIn(expression, environment, self.current_view);
    }

    pub fn typeFromExpressionIn(self: *Checker, expression: *const ast.TypeExpression, environment: *const TypeEnvironment, view_index: usize) Error!*const Type {
        const view_source = self.views[view_index].source;
        switch (expression.*) {
            .modified => |modified| {
                // '&[T]' is a slice and '*[T]' a heap array, not an
                // indirection onto an unsized array (section 3.2)
                if (modified.child.* == .array and modified.child.array.length == null) {
                    const element = try self.typeFromExpressionIn(modified.child.array.element, environment, view_index);
                    return switch (modified.modifier) {
                        .reference => self.makeType(.{ .slice = .{ .mutable = false, .child = element } }),
                        .reference_var => self.makeType(.{ .slice = .{ .mutable = true, .child = element } }),
                        .pointer => self.makeType(.{ .heap_array = .{ .mutable = false, .child = element } }),
                        .pointer_var => self.makeType(.{ .heap_array = .{ .mutable = true, .child = element } }),
                    };
                }
                // an interface behind an indirection is an interface object
                // (section 5.2); bare interface types are rejected below
                const child = if (try self.interfaceOfNamed(modified.child, environment, view_index)) |interface_type|
                    interface_type
                else
                    try self.typeFromExpressionIn(modified.child, environment, view_index);
                return switch (modified.modifier) {
                    .reference => self.makeType(.{ .reference = .{ .mutable = false, .child = child } }),
                    .reference_var => self.makeType(.{ .reference = .{ .mutable = true, .child = child } }),
                    .pointer => self.makeType(.{ .pointer = .{ .mutable = false, .child = child } }),
                    .pointer_var => self.makeType(.{ .pointer = .{ .mutable = true, .child = child } }),
                };
            },
            .named => |named| {
                if (named.implied) {
                    // '::Variant' is a value whose enum is implied, never a type
                    try self.report(named.path[0].location, "'::{s}' implies an enum variant and is not a type", .{named.path[0].slice(view_source)});
                    return &unknown_type;
                }
                if (named.path.len == 1) {
                    const name = named.path[0].slice(view_source);
                    if (environment.get(name)) |bound| return bound;
                    if (primitiveByName(name)) |primitive| {
                        return self.makeType(.{ .primitive = primitive });
                    }
                }
                const last = named.path[named.path.len - 1].slice(view_source);
                const symbols = self.globals.get(last) orelse return &unknown_type;
                const symbol = symbols.items[0];
                switch (symbol.definition.kind) {
                    .type_def => |type_def| {
                        var arguments: std.ArrayList(*const Type) = .empty;
                        for (named.type_arguments) |argument| {
                            try arguments.append(self.arena, try self.typeFromExpressionIn(argument, environment, view_index));
                        }
                        if (arguments.items.len != type_def.type_parameters.len) {
                            try self.report(named.path[0].location, "'{s}' expects {d} type argument{s}, found {d}", .{
                                last,
                                type_def.type_parameters.len,
                                if (type_def.type_parameters.len == 1) "" else "s",
                                arguments.items.len,
                            });
                            return &unknown_type;
                        }
                        return self.makeType(.{ .declared = .{
                            .definition = symbol.definition,
                            .view_index = symbol.view_index,
                            .name = last,
                            .arguments = try arguments.toOwnedSlice(self.arena),
                        } });
                    },
                    .interface_def => {
                        try self.report(named.path[0].location, "interface '{s}' can only be used behind an indirection ('&{s}', '*{s}') or as a constraint (section 5.2)", .{ last, last, last });
                        return &unknown_type;
                    },
                    else => {
                        try self.report(named.path[0].location, "'{s}' is not a type", .{last});
                        return &unknown_type;
                    },
                }
            },
            .struct_type => |members| {
                var fields: std.ArrayList(Type.Field) = .empty;
                for (members) |member| {
                    try fields.append(self.arena, .{
                        .name = member.name.slice(view_source),
                        .field_type = try self.typeFromExpressionIn(member.member_type, environment, view_index),
                    });
                }
                return self.makeType(.{ .structural = try fields.toOwnedSlice(self.arena) });
            },
            // an inline enum in a type position, compared structurally
            // against other enum types (section 3.3)
            .enum_type => |members| return self.makeType(.{ .inline_enum = .{ .members = members, .view_index = view_index } }),
            .array => |array| {
                const element = try self.typeFromExpressionIn(array.element, environment, view_index);
                const length_token = array.length orelse {
                    // a bare unsized array only exists behind & or * above
                    return &unknown_type;
                };
                const length = parseIntegerLiteral(length_token.slice(view_source)) catch 0;
                // a fixed array's length is positive (section 3.2)
                if (length == 0) {
                    try self.report(length_token.location, "a fixed array needs at least one element (section 3.2)", .{});
                }
                return self.makeType(.{ .fixed_array = .{ .element = element, .length = length } });
            },
            .function => |function| {
                var parameter_types: std.ArrayList(*const Type) = .empty;
                for (function.parameter_types) |parameter_type| {
                    try parameter_types.append(self.arena, try self.typeFromExpressionIn(parameter_type, environment, view_index));
                }
                const return_type: *const Type = if (function.return_type) |return_expression|
                    try self.typeFromExpressionIn(return_expression, environment, view_index)
                else
                    &void_type;
                return self.makeType(.{ .function = .{
                    .parameter_types = try parameter_types.toOwnedSlice(self.arena),
                    .return_type = return_type,
                } });
            },
            // a type synthesised by compile-time evaluation (section 3.4)
            .comptime_type => |inner| return self.comptimeType(inner, view_index),
        }
    }

    // evaluates 'type T = #...' (section 3.4): the expression must yield a
    // '#Type'; the result is cached per node so repeated uses agree
    fn comptimeType(self: *Checker, expression: *const ast.Expression, view_index: usize) Error!*const Type {
        if (self.comptime_type_cache.get(expression)) |cached| return cached;
        const machine = try self.comptimeMachine(view_index, &.{});
        const span = self.expressionSpan(expression);
        const result: *const Type = result: {
            const value = machine.evaluate(expression) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => {
                    try self.report(span, "comptime evaluation failed: {s} (section 6.1)", .{machine.fault_message orelse "unspecified fault"});
                    break :result &unknown_type;
                },
            };
            if (value != .type_value) {
                try self.report(span, "a '#' expression in type position must yield a '#Type' (section 3.4)", .{});
                break :result &unknown_type;
            }
            break :result try self.descriptionToType(value.type_value, span);
        };
        try self.comptime_type_cache.put(self.arena, expression, result);
        return result;
    }

    // recognizes a named type expression denoting an interface, yielding the
    // interface type used inside interface objects (section 5.2)
    fn interfaceOfNamed(self: *Checker, expression: *const ast.TypeExpression, environment: *const TypeEnvironment, view_index: usize) Error!?*const Type {
        if (expression.* != .named) return null;
        const named = expression.named;
        const view_source = self.views[view_index].source;
        if (named.path.len == 1) {
            const name = named.path[0].slice(view_source);
            if (environment.get(name) != null) return null;
            if (primitiveByName(name) != null) return null;
        }
        const last = named.path[named.path.len - 1].slice(view_source);
        const symbols = self.globals.get(last) orelse return null;
        const symbol = symbols.items[0];
        if (symbol.definition.kind != .interface_def) return null;
        return try self.makeType(.{ .interface = .{
            .definition = symbol.definition,
            .view_index = symbol.view_index,
            .name = last,
        } });
    }

    // resolves a named alias one level: 'type Meters = u32' yields u32; a
    // struct or enum bodied definition is nominal and resolves to itself
    pub fn resolveAlias(self: *Checker, candidate: *const Type) Error!*const Type {
        var current = candidate;
        var depth: usize = 0;
        while (current.* == .declared and depth < 64) : (depth += 1) {
            const type_def = current.declared.definition.kind.type_def;
            switch (type_def.base.*) {
                .struct_type, .enum_type => return current,
                else => {},
            }
            const environment = try self.declaredEnvironment(current.declared);
            current = try self.typeFromExpressionIn(type_def.base, environment, current.declared.view_index);
        }
        return current;
    }

    // the substitution environment of an instantiated declared type:
    // its definition's type parameters bound to the instance arguments
    fn declaredEnvironment(self: *Checker, declared: Type.Declared) Error!*TypeEnvironment {
        const environment = try self.arena.create(TypeEnvironment);
        environment.* = .empty;
        const type_def = declared.definition.kind.type_def;
        const definition_source = self.views[declared.view_index].source;
        for (type_def.type_parameters, 0..) |type_parameter, index| {
            if (index >= declared.arguments.len) break;
            try environment.put(self.arena, type_parameter.name.slice(definition_source), declared.arguments[index]);
        }
        return environment;
    }

    const StructBody = struct {
        members: []const ast.StructMember,
        environment: *TypeEnvironment,
        view_index: usize,
    };

    fn structBody(self: *Checker, candidate: *const Type) Error!?StructBody {
        const resolved = try self.resolveAlias(candidate);
        if (resolved.* != .declared) return null;
        const type_def = resolved.declared.definition.kind.type_def;
        if (type_def.base.* != .struct_type) return null;
        return .{
            .members = type_def.base.struct_type,
            .environment = try self.declaredEnvironment(resolved.declared),
            .view_index = resolved.declared.view_index,
        };
    }

    pub const EnumBody = struct {
        variants: []const Type.EnumVariant,
        name: []const u8,
    };

    // a resolved view of any enum type: declared, inline, or synthesised;
    // variant payloads arrive as checker types, not syntax
    pub fn enumBody(self: *Checker, candidate: *const Type) Error!?EnumBody {
        const resolved = try self.resolveAlias(candidate);
        switch (resolved.*) {
            .structural_enum => |variants| return .{
                .variants = variants,
                .name = if (candidate.* == .declared) candidate.declared.name else "enum { ... }",
            },
            .inline_enum => |inline_enum| {
                const environment = try self.arena.create(TypeEnvironment);
                environment.* = .empty;
                return .{
                    .variants = try self.resolveEnumMembers(inline_enum.members, environment, inline_enum.view_index),
                    .name = "enum { ... }",
                };
            },
            .declared => |declared| {
                const type_def = declared.definition.kind.type_def;
                if (type_def.base.* != .enum_type) return null;
                return .{
                    .variants = try self.resolveEnumMembers(type_def.base.enum_type, try self.declaredEnvironment(declared), declared.view_index),
                    .name = declared.name,
                };
            },
            else => return null,
        }
    }

    fn resolveEnumMembers(self: *Checker, members: []const ast.EnumMember, environment: *TypeEnvironment, view_index: usize) Error![]const Type.EnumVariant {
        const view_source = self.views[view_index].source;
        var variants: std.ArrayList(Type.EnumVariant) = .empty;
        for (members) |member| {
            try variants.append(self.arena, .{
                .name = member.name.slice(view_source),
                .payload = if (member.payload) |payload_expression|
                    try self.typeFromExpressionIn(payload_expression, environment, view_index)
                else
                    null,
            });
        }
        return variants.toOwnedSlice(self.arena);
    }

    // structural enum compatibility (section 3.3 rule 7): identical ordered
    // variant lists with identical payload types
    fn enumLayoutsMatch(self: *Checker, left: *const Type, right: *const Type) Error!bool {
        const left_body = try self.enumBody(left) orelse return false;
        const right_body = try self.enumBody(right) orelse return false;
        if (left_body.variants.len != right_body.variants.len) return false;
        for (left_body.variants, right_body.variants) |left_variant, right_variant| {
            if (!std.mem.eql(u8, left_variant.name, right_variant.name)) return false;
            if ((left_variant.payload == null) != (right_variant.payload == null)) return false;
            if (left_variant.payload) |left_payload| {
                if (!left_payload.eql(right_variant.payload.?)) return false;
            }
        }
        return true;
    }

    // type compatibility (section 3.3)
    fn coerce(self: *Checker, from: *const Type, to: *const Type) Error!bool {
        if (from.* == .unknown or to.* == .unknown) return true;
        if (from.eql(to)) return true;

        // rules 2 and 3: untyped literals against primitives and aliases
        if (from.* == .untyped_integer) {
            if (to.* == .untyped_float) return true;
            const resolved = try self.resolveAlias(to);
            return resolved.* == .primitive and resolved.primitive.isNumeric();
        }
        if (from.* == .untyped_float) {
            const resolved = try self.resolveAlias(to);
            return resolved.* == .primitive and resolved.primitive.isFloat();
        }

        // rule 4: named alias transparency, both directions
        if (from.* == .declared) {
            const resolved = try self.resolveAlias(from);
            if (resolved.* != .declared) return self.coerce(resolved, to);
        }
        if (to.* == .declared) {
            const resolved = try self.resolveAlias(to);
            if (resolved.* != .declared) return self.coerce(from, resolved);
        }

        // rule 5: numeric widening within one sign class
        if (from.* == .primitive and to.* == .primitive) {
            const source_primitive = from.primitive;
            const target_primitive = to.primitive;
            if (source_primitive.width() <= target_primitive.width()) {
                if (source_primitive.isUnsigned() and target_primitive.isUnsigned()) return true;
                if (source_primitive.isSigned() and target_primitive.isSigned()) return true;
                if (source_primitive.isFloat() and target_primitive.isFloat()) return true;
            }
            return false;
        }

        // a type converts to an interface it implements; behind an
        // indirection this forms an interface object (section 5.2)
        if (to.* == .interface) {
            return self.implements(from, to.interface);
        }

        // mutable indirections convert to their immutable counterparts
        if (from.* == .pointer and to.* == .pointer) {
            if (!(from.pointer.mutable or !to.pointer.mutable)) return false;
            if (to.pointer.child.* == .interface) return self.implements(from.pointer.child, to.pointer.child.interface);
            return from.pointer.child.eql(to.pointer.child);
        }
        if (from.* == .reference and to.* == .reference) {
            if (!(from.reference.mutable or !to.reference.mutable)) return false;
            if (to.reference.child.* == .interface) return self.implements(from.reference.child, to.reference.child.interface);
            return from.reference.child.eql(to.reference.child);
        }
        if (from.* == .heap_array and to.* == .heap_array) {
            if (!(from.heap_array.mutable or !to.heap_array.mutable)) return false;
            if (from.heap_array.child.eql(to.heap_array.child)) return true;
            // a fresh allocation's untyped element adopts the target element
            return from.heap_array.child.* == .untyped_integer or from.heap_array.child.* == .untyped_float;
        }
        if (from.* == .slice and to.* == .slice) {
            return (from.slice.mutable or !to.slice.mutable) and from.slice.child.eql(to.slice.child);
        }
        if (from.* == .fixed_array and to.* == .fixed_array) {
            if (from.fixed_array.length != to.fixed_array.length) return false;
            if (from.fixed_array.element.eql(to.fixed_array.element)) return true;
            // a literal array's untyped element adopts the target element
            if (from.fixed_array.element.* == .untyped_integer or from.fixed_array.element.* == .untyped_float) {
                return self.coerce(from.fixed_array.element, to.fixed_array.element);
            }
            return false;
        }

        // rule 7: an inline or synthesised enum compares structurally with
        // any enum type whose ordered variant list matches exactly (section
        // 3.3); synthesised enums have no syntax, so structural is all they
        // have (section 3.4)
        if (from.* == .inline_enum or to.* == .inline_enum or from.* == .structural_enum or to.* == .structural_enum) {
            return self.enumLayoutsMatch(from, to);
        }

        // rule 6: a structural target accepts any value providing its fields
        if (to.* == .structural) {
            const body = try self.structuralFieldsOf(from);
            if (body) |from_fields| {
                for (to.structural) |required| {
                    const found = for (from_fields) |provided| {
                        if (std.mem.eql(u8, provided.name, required.name)) break provided;
                    } else return false;
                    if (!try self.coerce(found.field_type, required.field_type)) return false;
                }
                return true;
            }
        }

        return false;
    }

    // the interface behind an interface-object value (section 5.2)
    pub fn interfaceObject(self: *Checker, candidate: *const Type) Error!?Type.Interface {
        const resolved = try self.resolveAlias(try self.pierce(candidate));
        if (resolved.* == .interface) return resolved.interface;
        return null;
    }

    // whether a type satisfies an interface (section 5.2): the declared
    // type's marker list, the lang items' implicit implementers (section
    // 5.1), and generic parameters constrained by the same interface
    fn implements(self: *Checker, candidate: *const Type, interface: Type.Interface) Error!bool {
        const resolved = try self.resolveAlias(candidate);
        switch (resolved.*) {
            .unknown => return true,
            .interface => |other| return other.definition == interface.definition,
            .type_parameter => |parameter| {
                const constraint = parameter.constraint orelse return false;
                return constraint.definition == interface.definition;
            },
            else => {},
        }
        if (self.langItem(interface)) |item| {
            switch (item) {
                .number => if (resolved.isNumeric() and resolved.* != .declared) return true,
                .iterable => switch (resolved.*) {
                    .slice, .heap_array, .fixed_array => return true,
                    else => {},
                },
            }
        }
        if (resolved.* != .declared) return false;
        const type_def = resolved.declared.definition.kind.type_def;
        const definition_source = self.views[resolved.declared.view_index].source;
        for (type_def.interfaces) |marker| {
            const symbols = self.globals.get(marker.slice(definition_source)) orelse continue;
            if (symbols.items[0].definition == interface.definition) return true;
        }
        return false;
    }

    // a type parameter constrained by the Number lang item behaves
    // numerically inside the generic body (sections 3.1, 5.2)
    fn isNumericOperand(self: *const Checker, candidate: *const Type) bool {
        if (candidate.isNumeric()) return true;
        if (candidate.* != .type_parameter) return false;
        const constraint = candidate.type_parameter.constraint orelse return false;
        return self.langItem(constraint) == .number;
    }

    const LangItem = enum { number, iterable };

    // the two standard interfaces are recognized by canonical path
    // (section 5.1a), not by name alone
    fn langItem(self: *const Checker, interface: Type.Interface) ?LangItem {
        const key = self.views[interface.view_index].key orelse return null;
        if (std.mem.eql(u8, key, "std::number") and std.mem.eql(u8, interface.name, "Number")) return .number;
        if (std.mem.eql(u8, key, "std::iterable") and std.mem.eql(u8, interface.name, "Iterable")) return .iterable;
        return null;
    }

    pub fn structuralFieldsOf(self: *Checker, candidate: *const Type) Error!?[]const Type.Field {
        const resolved = try self.resolveAlias(candidate);
        // synthesised types ('type T = #...', section 3.4) resolve straight
        // to a structural field list
        if (resolved.* == .structural) return resolved.structural;
        const body = try self.structBody(candidate) orelse return null;
        var fields: std.ArrayList(Type.Field) = .empty;
        const body_source = self.views[body.view_index].source;
        for (body.members) |member| {
            try fields.append(self.arena, .{
                .name = member.name.slice(body_source),
                .field_type = try self.typeFromExpressionIn(member.member_type, body.environment, body.view_index),
            });
        }
        return try fields.toOwnedSlice(self.arena);
    }

    pub const Layout = struct {
        size: u64,
        alignment: u64,
    };

    pub fn alignForward(value: u64, alignment: u64) u64 {
        return (value + alignment - 1) / alignment * alignment;
    }

    // C-compatible data layout (section 3.9): declaration order, natural
    // alignment, size rounded up to the largest member alignment; null when
    // the type has no defined runtime layout (unknown, type parameters, ...)
    pub fn layoutOf(self: *Checker, candidate: *const Type, depth: usize) Error!?Layout {
        if (depth > 16) return null;
        const resolved = try self.resolveAlias(try self.defaulted(candidate));
        switch (resolved.*) {
            .primitive => |primitive| {
                const width: u64 = primitive.width();
                return .{ .size = width, .alignment = width };
            },
            // an interface object is a fat pointer (section 3.2)
            .pointer => |indirection| return if (indirection.child.* == .interface)
                .{ .size = 16, .alignment = 8 }
            else
                .{ .size = 8, .alignment = 8 },
            .reference => |indirection| return if (indirection.child.* == .interface)
                .{ .size = 16, .alignment = 8 }
            else
                .{ .size = 8, .alignment = 8 },
            .heap_array, .function => return .{ .size = 8, .alignment = 8 },
            .slice => return .{ .size = 16, .alignment = 8 },
            .fixed_array => |array| {
                const element = try self.layoutOf(array.element, depth + 1) orelse return null;
                return .{ .size = element.size * array.length, .alignment = element.alignment };
            },
            .structural, .declared, .inline_enum, .structural_enum => {
                if (try self.enumBody(resolved)) |body| return self.enumLayout(body, depth);
                const fields = try self.structuralFieldsOf(resolved) orelse return null;
                var offset: u64 = 0;
                var max_alignment: u64 = 1;
                for (fields) |field| {
                    const field_layout = try self.layoutOf(field.field_type, depth + 1) orelse return null;
                    max_alignment = @max(max_alignment, field_layout.alignment);
                    offset = alignForward(offset, field_layout.alignment) + field_layout.size;
                }
                return .{ .size = alignForward(offset, max_alignment), .alignment = max_alignment };
            },
            else => return null,
        }
    }

    pub const FieldSlot = struct {
        name: []const u8,
        offset: u64,
        field_type: *const Type,
    };

    /// The byte offset of every struct field under the section 3.9 layout
    /// rules; null when the candidate is not a struct or a field has no
    /// defined layout. Code generation reads aggregates through these slots.
    pub fn fieldSlots(self: *Checker, candidate: *const Type) Error!?[]const FieldSlot {
        const resolved = try self.resolveAlias(try self.defaulted(candidate));
        if (try self.enumBody(resolved) != null) return null;
        const fields = try self.structuralFieldsOf(resolved) orelse return null;
        var slots: std.ArrayList(FieldSlot) = .empty;
        var offset: u64 = 0;
        for (fields) |field| {
            const field_layout = try self.layoutOf(field.field_type, 1) orelse return null;
            offset = alignForward(offset, field_layout.alignment);
            try slots.append(self.arena, .{ .name = field.name, .offset = offset, .field_type = field.field_type });
            offset += field_layout.size;
        }
        return try slots.toOwnedSlice(self.arena);
    }

    pub const EnumFrame = struct {
        tag_size: u64,
        payload_offset: u64,
        layout: Layout,
        variants: []const Type.EnumVariant,
    };

    /// The runtime frame of any enum type (section 3.9): tag width, payload
    /// offset, full layout, and the resolved variant list whose order is the
    /// tag numbering. Code generation builds and matches enums through this.
    pub fn enumFrame(self: *Checker, candidate: *const Type) Error!?EnumFrame {
        const resolved = try self.resolveAlias(try self.defaulted(candidate));
        const body = try self.enumBody(resolved) orelse return null;
        const layout = try self.enumLayout(body, 0) orelse return null;
        const tag_size: u64 = if (body.variants.len <= 1 << 8) 1 else if (body.variants.len <= 1 << 16) 2 else 4;
        var payload_alignment: u64 = 1;
        for (body.variants) |variant| {
            const payload_type = variant.payload orelse continue;
            const payload_layout = try self.layoutOf(payload_type, 1) orelse return null;
            payload_alignment = @max(payload_alignment, payload_layout.alignment);
        }
        return .{
            .tag_size = tag_size,
            .payload_offset = alignForward(tag_size, payload_alignment),
            .layout = layout,
            .variants = body.variants,
        };
    }

    // a tag sized to the variant count, then the payload area sized and
    // aligned for the largest payload (section 3.9)
    fn enumLayout(self: *Checker, body: EnumBody, depth: usize) Error!?Layout {
        const tag_size: u64 = if (body.variants.len <= 1 << 8) 1 else if (body.variants.len <= 1 << 16) 2 else 4;
        var payload_size: u64 = 0;
        var payload_alignment: u64 = 1;
        for (body.variants) |variant| {
            const payload_type = variant.payload orelse continue;
            const payload_layout = try self.layoutOf(payload_type, depth + 1) orelse return null;
            payload_size = @max(payload_size, payload_layout.size);
            payload_alignment = @max(payload_alignment, payload_layout.alignment);
        }
        const alignment = @max(tag_size, payload_alignment);
        const payload_offset = alignForward(tag_size, payload_alignment);
        return .{ .size = alignForward(payload_offset + alignForward(payload_size, payload_alignment), alignment), .alignment = alignment };
    }

    fn makeShape(self: *Checker, shape: types.Shape) Error!*const types.Shape {
        const stored = try self.arena.create(types.Shape);
        stored.* = shape;
        return stored;
    }

    // flattens a type's layout (section 3.9) into byte offsets for 'as'
    // reinterpretation (section 3.5); null when the type carries pointers,
    // references, or closures, which have no portable byte image
    fn shapeOf(self: *Checker, candidate: *const Type, depth: usize) Error!?*const types.Shape {
        if (depth > 16) return null;
        const resolved = try self.resolveAlias(try self.defaulted(candidate));
        switch (resolved.*) {
            .primitive => |primitive| return try self.makeShape(.{ .primitive = primitive }),
            .fixed_array => |array| {
                const element_layout = try self.layoutOf(array.element, depth + 1) orelse return null;
                const element = try self.shapeOf(array.element, depth + 1) orelse return null;
                return try self.makeShape(.{ .array = .{
                    .size = element_layout.size * array.length,
                    .stride = element_layout.size,
                    .count = array.length,
                    .element = element,
                } });
            },
            .structural, .declared, .inline_enum, .structural_enum => {
                if (try self.enumBody(resolved)) |body| {
                    const tag_size: u64 = if (body.variants.len <= 1 << 8) 1 else if (body.variants.len <= 1 << 16) 2 else 4;
                    var payload_size: u64 = 0;
                    var payload_alignment: u64 = 1;
                    var variants: std.ArrayList(types.Shape.Variant) = .empty;
                    for (body.variants) |body_variant| {
                        var payload_shape: ?*const types.Shape = null;
                        if (body_variant.payload) |payload_type| {
                            const payload_layout = try self.layoutOf(payload_type, depth + 1) orelse return null;
                            payload_shape = (try self.shapeOf(payload_type, depth + 1)) orelse return null;
                            payload_size = @max(payload_size, payload_layout.size);
                            payload_alignment = @max(payload_alignment, payload_layout.alignment);
                        }
                        try variants.append(self.arena, .{ .name = body_variant.name, .payload = payload_shape });
                    }
                    const alignment = @max(tag_size, payload_alignment);
                    const payload_offset = alignForward(tag_size, payload_alignment);
                    return try self.makeShape(.{ .tagged = .{
                        .size = alignForward(payload_offset + alignForward(payload_size, payload_alignment), alignment),
                        .tag_size = tag_size,
                        .payload_offset = payload_offset,
                        .variants = try variants.toOwnedSlice(self.arena),
                    } });
                }
                const fields = try self.structuralFieldsOf(resolved) orelse return null;
                var offset: u64 = 0;
                var max_alignment: u64 = 1;
                var shaped: std.ArrayList(types.Shape.Field) = .empty;
                for (fields) |field| {
                    const field_layout = try self.layoutOf(field.field_type, depth + 1) orelse return null;
                    const field_shape = try self.shapeOf(field.field_type, depth + 1) orelse return null;
                    max_alignment = @max(max_alignment, field_layout.alignment);
                    offset = alignForward(offset, field_layout.alignment);
                    try shaped.append(self.arena, .{ .name = field.name, .offset = offset, .shape = field_shape });
                    offset += field_layout.size;
                }
                return try self.makeShape(.{ .record = .{
                    .size = alignForward(offset, max_alignment),
                    .name = if (resolved.* == .declared) resolved.declared.name else "",
                    .fields = try shaped.toOwnedSlice(self.arena),
                } });
            },
            else => return null,
        }
    }

    // an untyped literal's default when no contextual type exists (rules 2, 3)
    pub fn defaulted(self: *Checker, candidate: *const Type) Error!*const Type {
        return switch (candidate.*) {
            .untyped_integer => self.makeType(.{ .primitive = .i32 }),
            .untyped_float => self.makeType(.{ .primitive = .f32 }),
            .fixed_array => |array| self.makeType(.{ .fixed_array = .{
                .element = try self.defaulted(array.element),
                .length = array.length,
            } }),
            .heap_array => |indirection| self.makeType(.{ .heap_array = .{
                .mutable = indirection.mutable,
                .child = try self.defaulted(indirection.child),
            } }),
            .pointer => |indirection| self.makeType(.{ .pointer = .{
                .mutable = indirection.mutable,
                .child = try self.defaulted(indirection.child),
            } }),
            else => candidate,
        };
    }

    fn checkStatement(self: *Checker, statement: *const ast.Statement) Error!void {
        switch (statement.*) {
            .block => |statements| {
                try self.pushFrame(false);
                defer self.popFrame();
                for (statements) |child| {
                    try self.checkStatement(child);
                }
            },
            .var_def => |var_def| {
                var binding_type: *const Type = undefined;
                if (var_def.declared_type) |declared_expression| {
                    const declared_type = try self.typeFromExpression(declared_expression, self.scope_types);
                    const value_type = try self.checkExpression(var_def.value, declared_type);
                    try self.expectAssignable(value_type, declared_type, var_def.value, var_def.name.location);
                    binding_type = declared_type;
                } else {
                    const value_type = try self.checkExpression(var_def.value, null);
                    binding_type = try self.defaulted(value_type);
                }
                try self.bind(var_def.name, binding_type, var_def.mutable);
                // a const binding's initializer feeds compile-time
                // evaluation (section 6.1)
                if (!var_def.mutable) {
                    const frame = &self.scopes.items[self.scopes.items.len - 1];
                    frame.bindings.items[frame.bindings.items.len - 1].initializer = var_def.value;
                }
            },
            .assign => |assign| {
                const place = try self.lvalueOf(assign.target);
                const target_type: *const Type = if (place) |info| target: {
                    // plain '=' on a pointer or reference place rebinds the
                    // place itself, so 'move'/'new'/'&' is required (section
                    // 4.2); compound assignment writes through to the
                    // pointee (pointee transparency)
                    if (assign.operator.tag == .equal) {
                        const raw_resolved = try self.resolveAlias(info.raw);
                        switch (raw_resolved.*) {
                            .pointer, .heap_array, .reference => break :target info.raw,
                            else => {},
                        }
                    }
                    break :target info.pierced;
                } else try self.checkExpression(assign.target, null);
                if (place == null) {
                    try self.report(self.statementSpan(statement), "the assignment target is not an assignable location", .{});
                } else if (!place.?.mutable) {
                    try self.report(self.statementSpan(statement), "cannot assign through an immutable binding; declare it 'var' or reach it through '&var'/'*var' (section 3.8)", .{});
                }
                const value_type = try self.checkExpression(assign.value, target_type);
                if (assign.operator.tag == .equal) {
                    try self.expectAssignable(value_type, target_type, assign.value, assign.operator.location);
                } else {
                    // compound assignment: target op= value needs numeric or
                    // integer operands depending on the operator
                    const integer_only = switch (assign.operator.tag) {
                        .shift_left_equal, .shift_right_equal, .ampersand_equal, .pipe_equal, .caret_equal, .percent_equal => true,
                        else => false,
                    };
                    // aliases resolve so 'type Header = u32' stays numeric
                    const target_resolved = try self.resolveAlias(target_type);
                    const value_resolved = try self.resolveAlias(value_type);
                    if (integer_only and !(target_resolved.isInteger() and value_resolved.isInteger())) {
                        try self.report(assign.operator.location, "'{s}' requires integer operands", .{assign.operator.tag.lexeme().?});
                    } else if (!target_resolved.isNumeric() or !value_resolved.isNumeric()) {
                        try self.report(assign.operator.location, "'{s}' requires numeric operands", .{assign.operator.tag.lexeme().?});
                    } else if (!try self.coerce(value_type, target_type)) {
                        try self.typeMismatch(assign.operator.location, value_type, target_type);
                    }
                }
            },
            .return_stmt => |return_stmt| {
                const expected = self.return_type;
                if (return_stmt.value) |value| {
                    const value_type = try self.checkExpression(value, expected);
                    if (expected) |return_type| {
                        if (return_type.* == .void_type) {
                            try self.report(return_stmt.keyword.location, "this function does not return a value", .{});
                        } else if (!try self.coerce(value_type, return_type)) {
                            try self.typeMismatch(return_stmt.keyword.location, value_type, return_type);
                        }
                    } else {
                        // lambda return inference: unify across return sites
                        try self.inferReturn(value_type, return_stmt.keyword.location);
                    }
                } else {
                    if (expected) |return_type| {
                        if (return_type.* != .void_type) {
                            const rendered = try return_type.render(self.arena);
                            try self.report(return_stmt.keyword.location, "expected a return value of type {s}", .{rendered});
                        }
                    } else {
                        try self.inferReturn(&void_type, return_stmt.keyword.location);
                    }
                }
            },
            .break_stmt => |break_stmt| {
                if (self.yield_frames.items.len == 0) {
                    try self.report(break_stmt.keyword.location, "'break' must be inside an if, loop, or match", .{});
                    if (break_stmt.value) |value| _ = try self.checkExpression(value, null);
                    return;
                }
                const frame = &self.yield_frames.items[self.yield_frames.items.len - 1];
                if (break_stmt.value) |value| {
                    const value_type = try self.checkExpression(value, frame.yielded);
                    if (frame.yielded) |previous| {
                        const unified = try self.unify(previous, value_type);
                        if (unified == null) {
                            const left = try previous.render(self.arena);
                            const right = try value_type.render(self.arena);
                            try self.report(break_stmt.keyword.location, "'break' values disagree: {s} versus {s}", .{ left, right });
                        } else {
                            frame.yielded = unified;
                        }
                    } else {
                        frame.yielded = value_type;
                    }
                }
            },
            .expression => |expression| {
                _ = try self.checkExpressionAsStatement(expression);
            },
        }
    }

    // section 4.3: loop 'else' and the external match 'else' are only
    // permitted when the construct is evaluated as an expression
    fn checkExpressionAsStatement(self: *Checker, expression: *const ast.Expression) Error!*const Type {
        switch (expression.*) {
            .if_expr => |if_expr| return self.checkIf(if_expr, expression, false),
            .while_expr => |while_expr| {
                if (while_expr.else_branch != null) {
                    try self.report(self.expressionSpan(expression), "a loop 'else' is only permitted when the loop is used as an expression (section 4.3)", .{});
                }
                return self.checkWhile(while_expr, expression, false);
            },
            .for_expr => |for_expr| {
                if (for_expr.else_branch != null) {
                    try self.report(self.expressionSpan(expression), "a loop 'else' is only permitted when the loop is used as an expression (section 4.3)", .{});
                }
                return self.checkFor(for_expr, expression, false);
            },
            .match_expr => |match_expr| {
                if (match_expr.else_branch != null) {
                    try self.report(self.expressionSpan(expression), "an external match 'else' is only permitted when the match is used as an expression (section 4.3)", .{});
                }
                return self.checkMatch(match_expr, expression, false);
            },
            else => return self.checkExpression(expression, null),
        }
    }

    // the result of every expression is recorded for the later stages
    // (interpreter and code generation); re-checks overwrite with the final
    // verdict, so the table always holds the winning typing
    fn checkExpression(self: *Checker, expression: *const ast.Expression, expected: ?*const Type) Error!*const Type {
        const result = try self.checkExpressionInner(expression, expected);
        // an untyped literal records its contextual type so the later stages
        // know its width; everything else records its own type
        var recorded = result;
        if (expected) |context| {
            if ((result.* == .untyped_integer or result.* == .untyped_float) and try self.coerce(result, context)) {
                recorded = try self.resolveAlias(context);
            }
        }
        try self.expression_types.put(self.arena, expression, recorded);
        return result;
    }

    fn checkExpressionInner(self: *Checker, expression: *const ast.Expression, expected: ?*const Type) Error!*const Type {
        switch (expression.*) {
            .integer_literal => return &untyped_integer_type,
            .float_literal => return &untyped_float_type,
            .bool_literal => return &bool_type,
            // a string literal is a static '&[u8]' slice (section 1.6)
            .string_literal => |token| {
                try self.validateEscapes(token);
                return self.makeType(.{ .slice = .{ .mutable = false, .child = try self.makeType(.{ .primitive = .u8 }) } });
            },
            .character_literal => |token| return self.characterLiteralType(token),
            .path => return self.checkPath(expression, expected),
            .implied_variant => return self.checkImpliedVariant(expression, expected),
            .grouped => |inner| return self.checkExpression(inner, expected),
            // compile-time evaluation substitutes the computed value (section
            // 6.1); macro and reflection results type from their value, so
            // they evaluate now -- everything else defers until checking
            // completes and may call later definitions
            .comptime_expr => |inner| {
                self.comptime_depth += 1;
                const result = try self.checkExpression(inner, expected);
                self.comptime_depth -= 1;
                // a '#' nested inside another '#' evaluates as part of the
                // outer expression, never on its own
                if (self.comptime_depth > 0) return result;
                if (self.needsEagerComptime(inner)) {
                    const environment = try self.snapshotComptimeEnvironment();
                    if (try self.runValueComptime(expression, inner, self.current_view, environment)) |value| {
                        return self.typeOfComptimeValue(value, self.expressionSpan(expression));
                    }
                    return &unknown_type;
                }
                try self.deferComptime(expression, inner);
                return result;
            },
            .unary => return self.checkUnary(expression, expected),
            .binary => return self.checkBinary(expression),
            .cast => return self.checkCast(expression),
            .call => return self.checkCall(expression, expected),
            .member => return self.checkMember(expression),
            .index => return self.checkIndex(expression),
            .struct_init => return self.checkStructInit(expression, expected),
            .array_literal => |elements| {
                var element_type: *const Type = &unknown_type;
                var first = true;
                const expected_element: ?*const Type = if (expected != null and expected.?.* == .fixed_array) expected.?.fixed_array.element else null;
                for (elements) |element| {
                    const candidate = try self.checkExpression(element, expected_element);
                    if (first) {
                        element_type = candidate;
                        first = false;
                    } else if (try self.unify(element_type, candidate)) |unified| {
                        element_type = unified;
                    } else {
                        const left = try element_type.render(self.arena);
                        const right = try candidate.render(self.arena);
                        try self.report(self.expressionSpan(element), "array elements disagree: {s} versus {s}", .{ left, right });
                    }
                }
                return self.makeType(.{ .fixed_array = .{ .element = element_type, .length = elements.len } });
            },
            .array_fill => return self.checkArrayFill(expression, expected, false),
            .array_range => return self.checkArrayRange(expression, expected, false, false),
            .if_expr => |if_expr| return self.checkIf(if_expr, expression, true),
            .while_expr => |while_expr| return self.checkWhile(while_expr, expression, true),
            .for_expr => |for_expr| return self.checkFor(for_expr, expression, true),
            .match_expr => |match_expr| return self.checkMatch(match_expr, expression, true),
            .lambda => return self.checkLambda(expression),
        }
    }

    fn checkPath(self: *Checker, expression: *const ast.Expression, expected: ?*const Type) Error!*const Type {
        const path = expression.path;
        if (path.len == 1) {
            const name = path[0].slice(self.source());
            if (self.lookup(name)) |binding| {
                // pointee transparency: reading a pointer or reference yields
                // a copy of the pointee (section 4.2)
                return self.pierce(binding.binding_type);
            }
            if (self.globals.get(name)) |symbols| {
                return self.symbolValueType(symbols, path[0].location);
            }
            return &unknown_type;
        }
        // 'Enum::Variant' without a payload is a value of the enum type
        if (try self.variantOfPath(path)) |variant| {
            if (variant.payload != null) {
                try self.report(self.expressionSpan(expression), "variant '{s}' carries a payload; construct it with '(...)'", .{variant.name});
            }
            // a generic enum takes its type arguments from context
            if (expected) |context| {
                if (context.* == .declared and context.declared.definition == variant.enum_type.declared.definition) {
                    return context;
                }
            }
            try self.reportUnboundVariantParameters(variant, self.scope_types, self.expressionSpan(expression));
            return variant.enum_type;
        }
        if (self.globals.get(path[path.len - 1].slice(self.source()))) |symbols| {
            return self.symbolValueType(symbols, path[0].location);
        }
        return &unknown_type;
    }

    fn symbolValueType(self: *Checker, symbols: resolution.SymbolList, span: Token.Location) Error!*const Type {
        const symbol = symbols.items[0];
        switch (symbol.definition.kind) {
            .fn_def => |fn_def| {
                if (symbols.items.len > 1) {
                    try self.report(span, "'{s}' is overloaded; a function value needs a unique function", .{fn_def.name.slice(self.views[symbol.view_index].source)});
                    return &unknown_type;
                }
                if (fn_def.type_parameters.len != 0) return &unknown_type;
                return self.functionType(symbol);
            },
            .extern_def => return self.functionType(symbol),
            // a bare type or interface name is not a value; '#T' reflection
            // is compile-time evaluation
            else => return &unknown_type,
        }
    }

    fn functionType(self: *Checker, symbol: resolution.Symbol) Error!*const Type {
        var parameter_types: std.ArrayList(*const Type) = .empty;
        var return_type: *const Type = &void_type;
        switch (symbol.definition.kind) {
            .fn_def => |fn_def| {
                for (fn_def.function.parameters) |parameter| {
                    try parameter_types.append(self.arena, try self.typeFromExpressionIn(parameter.parameter_type, &empty_type_environment, symbol.view_index));
                }
                if (fn_def.function.return_type) |return_expression| {
                    return_type = try self.typeFromExpressionIn(return_expression, &empty_type_environment, symbol.view_index);
                }
            },
            .extern_def => |extern_def| {
                for (extern_def.parameters) |parameter| {
                    try parameter_types.append(self.arena, try self.typeFromExpressionIn(parameter.parameter_type, &empty_type_environment, symbol.view_index));
                }
                if (extern_def.return_type) |return_expression| {
                    return_type = try self.typeFromExpressionIn(return_expression, &empty_type_environment, symbol.view_index);
                }
            },
            else => unreachable,
        }
        return self.makeType(.{ .function = .{
            .parameter_types = try parameter_types.toOwnedSlice(self.arena),
            .return_type = return_type,
        } });
    }

    fn checkUnary(self: *Checker, expression: *const ast.Expression, expected: ?*const Type) Error!*const Type {
        const unary = expression.unary;
        switch (unary.operator.tag) {
            .minus => {
                const operand = try self.checkExpression(unary.operand, expected);
                if (!self.isNumericOperand(operand)) {
                    try self.report(unary.operator.location, "'-' requires a numeric operand", .{});
                    return &unknown_type;
                }
                if (operand.* == .primitive and operand.primitive.isUnsigned()) {
                    try self.report(unary.operator.location, "cannot negate the unsigned type {s}", .{@tagName(operand.primitive)});
                }
                return operand;
            },
            .tilde => {
                const operand = try self.checkExpression(unary.operand, expected);
                if (!operand.isInteger()) {
                    try self.report(unary.operator.location, "'~' requires an integer operand", .{});
                    return &unknown_type;
                }
                return operand;
            },
            .bang => {
                const operand = try self.checkExpression(unary.operand, null);
                if (!operand.isBool()) {
                    try self.report(unary.operator.location, "'!' requires a bool operand", .{});
                }
                return &bool_type;
            },
            .ampersand => {
                // '&' references any value in place (section 4.2); the
                // reference is mutable when the referenced location is
                const place = try self.lvalueOf(unary.operand) orelse {
                    _ = try self.checkExpression(unary.operand, null);
                    try self.report(unary.operator.location, "'&' requires an addressable value (a variable, field, or element)", .{});
                    return &unknown_type;
                };
                return self.makeType(.{ .reference = .{ .mutable = place.mutable, .child = place.pierced } });
            },
            .keyword_new => {
                // 'new' deep-copies the operand value into a fresh heap
                // allocation (section 4.2); an array operand allocates *[T]
                if (unary.operand.* == .array_fill) {
                    return self.checkArrayFill(unary.operand, expected, true);
                }
                if (unary.operand.* == .array_range) {
                    return self.checkArrayRange(unary.operand, expected, true, false);
                }
                // a string literal's bytes copy into an owned '*[u8]' heap
                // array (section 1.6); the literal itself is a static '&[u8]'
                if (unary.operand.* == .string_literal) {
                    _ = try self.checkExpression(unary.operand, null);
                    return self.makeType(.{ .heap_array = .{ .mutable = true, .child = try self.makeType(.{ .primitive = .u8 }) } });
                }
                const expected_child: ?*const Type = if (expected) |context| switch (context.*) {
                    .pointer => |indirection| indirection.child,
                    .heap_array => context,
                    else => null,
                } else null;
                const operand = try self.checkExpression(unary.operand, expected_child);
                if (operand.* == .fixed_array) {
                    return self.makeType(.{ .heap_array = .{ .mutable = true, .child = operand.fixed_array.element } });
                }
                // the contextual pointee type fixes an untyped operand
                if (expected_child) |context| {
                    if (context.* != .heap_array and try self.coerce(operand, context)) {
                        return self.makeType(.{ .pointer = .{ .mutable = true, .child = context } });
                    }
                }
                return self.makeType(.{ .pointer = .{ .mutable = true, .child = try self.defaulted(operand) } });
            },
            .keyword_move => {
                // 'move' is the only address-reading operator: it transfers
                // an owning pointer and clears the source (section 4.2)
                const place = try self.lvalueOf(unary.operand) orelse {
                    _ = try self.checkExpression(unary.operand, null);
                    try self.report(unary.operator.location, "'move' requires an owning pointer variable or field", .{});
                    return &unknown_type;
                };
                switch (place.raw.*) {
                    .pointer, .heap_array => {},
                    .unknown => return &unknown_type,
                    else => {
                        const rendered = try place.raw.render(self.arena);
                        try self.report(unary.operator.location, "'move' requires a pointer (*T or *[T]), found {s}", .{rendered});
                        return &unknown_type;
                    },
                }
                if (!place.mutable) {
                    try self.report(unary.operator.location, "'move' clears its source, which must be mutable (section 3.8)", .{});
                }
                return place.raw;
            },
            else => {
                _ = try self.checkExpression(unary.operand, null);
                return &unknown_type;
            },
        }
    }

    fn checkBinary(self: *Checker, expression: *const ast.Expression) Error!*const Type {
        const binary = expression.binary;
        const operator = binary.operator;
        switch (operator.tag) {
            .plus, .minus, .asterisk, .slash, .percent => {
                const left = try self.checkExpression(binary.left, null);
                const right = try self.checkExpression(binary.right, null);
                if (!self.isNumericOperand(left) or !self.isNumericOperand(right)) {
                    try self.report(operator.location, "'{s}' requires numeric operands", .{operator.tag.lexeme().?});
                    return &unknown_type;
                }
                return try self.unify(left, right) orelse {
                    try self.operandMismatch(operator, left, right);
                    return &unknown_type;
                };
            },
            .shift_left, .shift_right => {
                const left = try self.checkExpression(binary.left, null);
                const right = try self.checkExpression(binary.right, null);
                if (!left.isInteger() or !right.isInteger()) {
                    try self.report(operator.location, "'{s}' requires integer operands", .{operator.tag.lexeme().?});
                    return &unknown_type;
                }
                return left;
            },
            .ampersand, .pipe, .caret => {
                const left = try self.checkExpression(binary.left, null);
                const right = try self.checkExpression(binary.right, null);
                if (!left.isInteger() or !right.isInteger()) {
                    try self.report(operator.location, "'{s}' requires integer operands", .{operator.tag.lexeme().?});
                    return &unknown_type;
                }
                return try self.unify(left, right) orelse {
                    try self.operandMismatch(operator, left, right);
                    return &unknown_type;
                };
            },
            .angle_left, .angle_left_equal, .angle_right, .angle_right_equal => {
                const left = try self.checkExpression(binary.left, null);
                const right = try self.checkExpression(binary.right, null);
                if (!self.isNumericOperand(left) or !self.isNumericOperand(right) or try self.unify(left, right) == null) {
                    try self.operandMismatch(operator, left, right);
                }
                return &bool_type;
            },
            .equal_equal, .bang_equal => {
                const left = try self.checkExpression(binary.left, null);
                const right = try self.checkExpression(binary.right, null);
                if (try self.unify(left, right) == null) {
                    try self.operandMismatch(operator, left, right);
                }
                return &bool_type;
            },
            .ampersand_ampersand, .pipe_pipe => {
                const left = try self.checkExpression(binary.left, null);
                const right = try self.checkExpression(binary.right, null);
                if (!left.isBool() or !right.isBool()) {
                    try self.report(operator.location, "'{s}' requires bool operands", .{operator.tag.lexeme().?});
                }
                return &bool_type;
            },
            else => {
                _ = try self.checkExpression(binary.left, null);
                _ = try self.checkExpression(binary.right, null);
                return &unknown_type;
            },
        }
    }

    // the common type of two operands: either coerces into the other
    fn unify(self: *Checker, left: *const Type, right: *const Type) Error!?*const Type {
        if (try self.coerce(left, right)) return right;
        if (try self.coerce(right, left)) return left;
        return null;
    }

    fn checkCast(self: *Checker, expression: *const ast.Expression) Error!*const Type {
        const cast = expression.cast;
        const operand = try self.checkExpression(cast.operand, null);
        switch (cast.operator.tag) {
            // 'x as T' reinterprets bytes; widths must match (section 3.5)
            .keyword_as => {
                const target = try self.typeFromExpression(cast.target, self.scope_types);
                const source_resolved = try self.resolveAlias(try self.defaulted(operand));
                const target_resolved = try self.resolveAlias(target);
                // '&S as &T' views the pointee in place, so the pointee
                // layouts are what must agree (section 3.5)
                var source_measured = source_resolved;
                var target_measured = target_resolved;
                if (source_resolved.* == .reference and target_resolved.* == .reference) {
                    source_measured = try self.resolveAlias(source_resolved.reference.child);
                    target_measured = try self.resolveAlias(target_resolved.reference.child);
                }
                const source_layout = try self.layoutOf(source_measured, 0);
                const target_layout = try self.layoutOf(target_measured, 0);
                if (source_layout != null and target_layout != null and source_layout.?.size != target_layout.?.size) {
                    try self.report(cast.operator.location, "'as' reinterprets bytes in place: {s} is {d} byte(s) but {s} is {d} byte(s) (section 3.5)", .{
                        try source_measured.render(self.arena),
                        source_layout.?.size,
                        try target_measured.render(self.arena),
                        target_layout.?.size,
                    });
                } else if ((source_resolved.* == .reference and target_resolved.* == .reference) or
                    (source_resolved.* != .reference and target_resolved.* != .reference and
                        !(source_resolved.* == .primitive and target_resolved.* == .primitive)))
                {
                    // record the byte shapes so the interpreter can reinterpret
                    // (section 3.5): the pointee shapes for '&S as &T', the
                    // value shapes for a non-primitive value cast
                    const source_shape = try self.shapeOf(source_measured, 0);
                    const target_shape = try self.shapeOf(target_measured, 0);
                    if (source_shape != null and target_shape != null) {
                        try self.cast_shapes.put(self.arena, expression, .{ .source = source_shape.?, .target = target_shape.? });
                    }
                }
                return target;
            },
            // 'x to T' converts between Number types (section 3.5)
            .keyword_to => {
                const target = try self.typeFromExpression(cast.target, self.scope_types);
                const target_resolved = try self.resolveAlias(target);
                const target_numeric = target_resolved.isNumeric() and target_resolved.* != .unknown or target_resolved.* == .type_parameter;
                const operand_resolved = try self.resolveAlias(operand);
                if (!operand_resolved.isNumeric() and operand_resolved.* != .type_parameter) {
                    try self.report(cast.operator.location, "'to' converts Number values; the operand is not numeric (section 3.5)", .{});
                } else if (!target_numeric) {
                    const rendered = try target.render(self.arena);
                    try self.report(cast.operator.location, "'to' converts to Number types, found {s} (section 3.5)", .{rendered});
                }
                return target;
            },
            // 'x is T' tests an enum variant or an interface object's
            // concrete type, yielding bool (section 3.2)
            .keyword_is => {
                if (try self.enumBody(operand)) |body| {
                    const variant_token = cast.target.named.path[cast.target.named.path.len - 1];
                    const variant_name = variant_token.slice(self.source());
                    if (cast.target.named.path.len == 1 and !cast.target.named.implied) {
                        try self.report(variant_token.location, "an enum variant test needs '::{s}' or '{s}::{s}' (section 3.2)", .{ variant_name, body.name, variant_name });
                    }
                    var found = false;
                    for (body.variants) |variant| {
                        if (std.mem.eql(u8, variant.name, variant_name)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        try self.report(variant_token.location, "enum '{s}' has no variant '{s}'", .{ body.name, variant_name });
                    }
                } else if (try self.interfaceObject(operand)) |interface| {
                    const target = try self.typeFromExpression(cast.target, self.scope_types);
                    if (target.* != .unknown and !try self.implements(target, interface)) {
                        const rendered = try target.render(self.arena);
                        try self.report(cast.operator.location, "{s} does not implement '{s}', so this 'is' test can never succeed (section 3.2)", .{ rendered, interface.name });
                    }
                } else if (operand.* != .unknown) {
                    const rendered = try operand.render(self.arena);
                    try self.report(cast.operator.location, "'is' tests enum variants and interface objects; the subject is {s} (section 3.2)", .{rendered});
                }
                return &bool_type;
            },
            else => return &unknown_type,
        }
    }

    fn checkIndex(self: *Checker, expression: *const ast.Expression) Error!*const Type {
        const index = expression.index;
        const base = try self.checkExpression(index.object, null);
        const subscript = try self.checkExpression(index.subscript, null);
        if (!subscript.isInteger()) {
            try self.report(self.expressionSpan(index.subscript), "an array index must be an integer", .{});
        }
        const resolved = try self.resolveAlias(base);
        // reading an element pierces like any read (section 4.2)
        return switch (resolved.*) {
            .slice => |slice| self.pierce(slice.child),
            .heap_array => |heap| self.pierce(heap.child),
            .fixed_array => |array| self.pierce(array.element),
            .unknown => &unknown_type,
            else => {
                const rendered = try base.render(self.arena);
                try self.report(self.expressionSpan(index.object), "{s} cannot be indexed", .{rendered});
                return &unknown_type;
            },
        };
    }

    fn checkMember(self: *Checker, expression: *const ast.Expression) Error!*const Type {
        const member = expression.member;
        const base = try self.checkExpression(member.object, null);
        // pointee transparency (section 4.2): reading a pointer or
        // reference field yields a copy of the pointee
        return self.pierce(try self.memberType(base, member.name));
    }

    fn memberType(self: *Checker, base: *const Type, name_token: Token) Error!*const Type {
        if (base.* == .unknown) return &unknown_type;
        const name = name_token.slice(self.source());
        // covers declared structs, synthesised aliases, and structural
        // values alike
        if (try self.structuralFieldsOf(base)) |fields| {
            for (fields) |field| {
                if (std.mem.eql(u8, field.name, name)) return field.field_type;
            }
        }
        const rendered = try base.render(self.arena);
        try self.report(name_token.location, "{s} has no member '{s}'", .{ rendered, name });
        return &unknown_type;
    }

    fn checkStructInit(self: *Checker, expression: *const ast.Expression, expected: ?*const Type) Error!*const Type {
        const struct_init = expression.struct_init;
        if (struct_init.name) |name_token| {
            const name = name_token.slice(self.source());
            const symbols = self.globals.get(name) orelse return &unknown_type;
            const symbol = symbols.items[0];
            const type_def = switch (symbol.definition.kind) {
                .type_def => |*type_def| type_def,
                else => {
                    try self.report(name_token.location, "'{s}' is not a struct type", .{name});
                    return &unknown_type;
                },
            };
            // generic instantiation comes from the contextual type
            // ('var v: Vec<u32> = Vec { ... }'); bare generic inits without
            // context cannot bind their type parameters
            var arguments: []const *const Type = &.{};
            if (type_def.type_parameters.len != 0) {
                if (expected != null and expected.?.* == .declared and expected.?.declared.definition == symbol.definition) {
                    arguments = expected.?.declared.arguments;
                } else {
                    try self.report(name_token.location, "cannot infer the type arguments of '{s}' here; annotate the target type", .{name});
                    return &unknown_type;
                }
            }
            const initialized = try self.makeType(.{ .declared = .{
                .definition = symbol.definition,
                .view_index = symbol.view_index,
                .name = name,
                .arguments = arguments,
            } });
            try self.checkStructInitMembers(initialized, struct_init.members, name_token.location);
            return initialized;
        }
        // anonymous initializer: a structural value (section 3.3 rule 6)
        var fields: std.ArrayList(Type.Field) = .empty;
        for (struct_init.members) |member| {
            const value_type = try self.checkExpression(member.value, null);
            try fields.append(self.arena, .{
                .name = member.name.slice(self.source()),
                .field_type = try self.defaulted(value_type),
            });
        }
        return self.makeType(.{ .structural = try fields.toOwnedSlice(self.arena) });
    }

    fn checkStructInitMembers(self: *Checker, initialized: *const Type, members: []const ast.MemberInit, span: Token.Location) Error!void {
        const fields = try self.structuralFieldsOf(initialized) orelse {
            const rendered = try initialized.render(self.arena);
            try self.report(span, "{s} is not a struct type", .{rendered});
            return;
        };
        for (members) |member| {
            const member_name = member.name.slice(self.source());
            const field = for (fields) |candidate| {
                if (std.mem.eql(u8, candidate.name, member_name)) break candidate;
            } else {
                const rendered = try initialized.render(self.arena);
                try self.report(member.name.location, "{s} has no member '{s}'", .{ rendered, member_name });
                _ = try self.checkExpression(member.value, null);
                continue;
            };
            const value_type = try self.checkExpression(member.value, field.field_type);
            try self.expectAssignable(value_type, field.field_type, member.value, member.name.location);
        }
        // every member must be initialized
        for (fields) |candidate| {
            const initialized_here = for (members) |member| {
                if (std.mem.eql(u8, member.name.slice(self.source()), candidate.name)) break true;
            } else false;
            if (!initialized_here) {
                const rendered = try initialized.render(self.arena);
                try self.report(span, "member '{s}' of {s} is not initialized", .{ candidate.name, rendered });
            }
        }
    }

    // a stack array fill's count must be compile-time evaluatable (section
    // 2.1), verified here rather than by the grammar; folds consts and '#'
    // expressions, yielding null for a genuine runtime value (the comptime
    // snapshot poisons mutable vars, so those never fold)
    fn comptimeArrayCount(self: *Checker, count: *const ast.Expression) Error!?u64 {
        const environment = try self.snapshotComptimeEnvironment();
        const machine = try self.comptimeMachine(self.current_view, environment);
        const value = machine.evaluate(count) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return null,
        };
        return switch (value) {
            .integer => |integer| if (integer.value >= 0 and integer.value <= std.math.maxInt(u64))
                @intCast(integer.value)
            else
                null,
            else => null,
        };
    }

    fn checkArrayFill(self: *Checker, expression: *const ast.Expression, expected: ?*const Type, under_new: bool) Error!*const Type {
        const array_fill = expression.array_fill;
        const expected_element: ?*const Type = if (expected) |context| switch (context.*) {
            .fixed_array => |array| array.element,
            .heap_array => |heap| heap.child,
            .slice => |slice| slice.child,
            else => null,
        } else null;
        var value_type = try self.checkExpression(array_fill.value, expected_element);
        if (expected_element) |element| {
            if (try self.coerce(value_type, element)) value_type = element;
        }
        const count_type = try self.checkExpression(array_fill.count, null);
        if (!count_type.isInteger()) {
            try self.report(self.expressionSpan(array_fill.count), "the fill count must be an integer", .{});
        }
        // a compile-time count makes a fixed array; a runtime count is only
        // valid under 'new', producing a heap array (sections 2.1 and 4.2).
        // the count is any compile-time-evaluatable expression, not just an
        // integer-literal token (section 2.1)
        const fixed_length: ?u64 = if (array_fill.count.* == .integer_literal)
            (parseIntegerLiteral(array_fill.count.integer_literal.slice(self.source())) catch 0)
        else if (count_type.isInteger())
            try self.comptimeArrayCount(array_fill.count)
        else
            null;
        if (fixed_length) |length| {
            if (under_new) {
                return self.makeType(.{ .heap_array = .{ .mutable = true, .child = try self.defaulted(value_type) } });
            }
            // a fixed array's length is positive (section 3.2)
            if (length == 0) {
                try self.report(self.expressionSpan(array_fill.count), "a fixed array needs at least one element (section 3.2)", .{});
            }
            return self.makeType(.{ .fixed_array = .{ .element = value_type, .length = length } });
        }
        if (!under_new) {
            try self.report(self.expressionSpan(expression), "a runtime-sized array fill requires 'new' (section 4.2); a stack array's count must be compile-time evaluatable", .{});
            return &unknown_type;
        }
        return self.makeType(.{ .heap_array = .{ .mutable = true, .child = try self.defaulted(value_type) } });
    }

    // '[start..end]' generates the integers start..end-1 (section 2.1):
    // literal bounds make a fixed array, runtime bounds need 'new' (heap
    // array) or a 'for' subject position, where no array materializes
    fn checkArrayRange(self: *Checker, expression: *const ast.Expression, expected: ?*const Type, under_new: bool, as_loop_subject: bool) Error!*const Type {
        const array_range = expression.array_range;
        const expected_element: ?*const Type = if (expected) |context| switch (context.*) {
            .fixed_array => |array| array.element,
            .heap_array => |heap| heap.child,
            .slice => |slice| slice.child,
            else => null,
        } else null;
        const start_type: ?*const Type = if (array_range.start) |start| try self.checkExpression(start, expected_element) else null;
        const end_type = try self.checkExpression(array_range.end, expected_element);
        var bounds_ok = true;
        if (start_type != null and !start_type.?.isInteger()) {
            try self.report(self.expressionSpan(array_range.start.?), "a range bound must be an integer", .{});
            bounds_ok = false;
        }
        if (!end_type.isInteger()) {
            try self.report(self.expressionSpan(array_range.end), "a range bound must be an integer", .{});
            bounds_ok = false;
        }
        if (!bounds_ok) return &unknown_type;
        var element_type = end_type;
        if (start_type) |first| {
            element_type = try self.unify(first, end_type) orelse mismatch: {
                try self.operandMismatchAt(array_range.operator.location, first, end_type);
                break :mismatch &unknown_type;
            };
        }
        if (expected_element) |element| {
            if (try self.coerce(element_type, element)) element_type = element;
        }
        const start_literal: ?u64 = if (array_range.start) |start| literal: {
            if (start.* != .integer_literal) break :literal null;
            break :literal parseIntegerLiteral(start.integer_literal.slice(self.source())) catch 0;
        } else 0;
        const end_literal: ?u64 = if (array_range.end.* == .integer_literal)
            parseIntegerLiteral(array_range.end.integer_literal.slice(self.source())) catch 0
        else
            null;
        if (start_literal != null and end_literal != null) {
            if (end_literal.? < start_literal.?) {
                try self.report(array_range.operator.location, "the range end {d} is less than its start {d}", .{ end_literal.?, start_literal.? });
            }
            const length = end_literal.? -| start_literal.?;
            if (under_new) {
                return self.makeType(.{ .heap_array = .{ .mutable = true, .child = try self.defaulted(element_type) } });
            }
            return self.makeType(.{ .fixed_array = .{ .element = element_type, .length = length } });
        }
        if (under_new) {
            return self.makeType(.{ .heap_array = .{ .mutable = true, .child = try self.defaulted(element_type) } });
        }
        // a 'for' subject never materializes; it lowers to a counting loop
        if (as_loop_subject) {
            return self.makeType(.{ .slice = .{ .mutable = false, .child = try self.defaulted(element_type) } });
        }
        try self.report(self.expressionSpan(expression), "a runtime-bounded range requires 'new' outside a 'for' subject (section 2.1); a stack array's size must be compile-time evaluatable", .{});
        return &unknown_type;
    }

    // the tokenizer validates escape introducers; the payloads (\xHH digit
    // count, \u{...} contents) are checked here (section 1.6)
    fn validateEscapes(self: *Checker, token: Token) Error!void {
        const text = token.slice(self.source());
        if (text.len < 2) return;
        const end = text.len - 1;
        var index: usize = 1;
        while (index < end) : (index += 1) {
            if (text[index] != '\\' or index + 1 >= end) continue;
            const introducer = index + 1;
            switch (text[introducer]) {
                'x', 'X' => {
                    const stop = @min(introducer + 3, end);
                    const digits = text[@min(introducer + 1, stop)..stop];
                    if (digits.len != 2 or !isHexDigit(digits[0]) or !isHexDigit(digits[1])) {
                        try self.report(escapeLocation(token, index, stop), "'\\x' needs exactly two hex digits (section 1.6)", .{});
                        return;
                    }
                    index = introducer + 2;
                },
                'u', 'U' => {
                    if (introducer + 1 >= end or text[introducer + 1] != '{') {
                        try self.report(escapeLocation(token, index, @min(introducer + 2, end)), "'\\u' needs '{{' hex digits '}}' (section 1.6)", .{});
                        return;
                    }
                    var close = introducer + 2;
                    while (close < end and text[close] != '}') close += 1;
                    const digits = text[introducer + 2 .. close];
                    if (close >= end or digits.len == 0) {
                        try self.report(escapeLocation(token, index, close + 1), "'\\u' needs '{{' hex digits '}}' (section 1.6)", .{});
                        return;
                    }
                    const scalar = std.fmt.parseInt(u21, digits, 16) catch {
                        try self.report(escapeLocation(token, index, close + 1), "'\\u{{...}}' is not a valid Unicode scalar value (section 1.6)", .{});
                        return;
                    };
                    if (scalar > 0x10FFFF or (scalar >= 0xD800 and scalar <= 0xDFFF)) {
                        try self.report(escapeLocation(token, index, close + 1), "'\\u{{...}}' is not a valid Unicode scalar value (section 1.6)", .{});
                        return;
                    }
                    index = close;
                },
                else => index = introducer,
            }
        }
    }

    fn isHexDigit(byte: u8) bool {
        return switch (byte) {
            '0'...'9', 'a'...'f', 'A'...'F' => true,
            else => false,
        };
    }

    fn escapeLocation(token: Token, start: usize, stop: usize) Token.Location {
        return .{ .start = token.location.start + start, .end = token.location.start + stop };
    }

    fn characterLiteralType(self: *Checker, token: Token) Error!*const Type {
        try self.validateEscapes(token);
        const text = token.slice(self.source());
        const content = text[1 .. text.len - 1];
        var byte_count: usize = 0;
        var position: usize = 0;
        while (position < content.len) {
            if (content[position] == '\\' and position + 1 < content.len) {
                switch (content[position + 1]) {
                    'x', 'X' => {
                        byte_count += 1;
                        position += 4;
                    },
                    'u', 'U' => {
                        // \u{...}: the scalar's exact UTF-8 width
                        const brace_start = position + 2;
                        var brace_end = brace_start;
                        while (brace_end < content.len and content[brace_end] != '}') brace_end += 1;
                        const digits = if (brace_end > brace_start + 1) content[brace_start + 1 .. brace_end] else "";
                        const scalar = std.fmt.parseInt(u21, digits, 16) catch 0;
                        byte_count += std.unicode.utf8CodepointSequenceLength(scalar) catch 1;
                        position = brace_end + 1;
                    },
                    else => {
                        byte_count += 1;
                        position += 2;
                    },
                }
            } else {
                byte_count += 1;
                position += 1;
            }
        }
        // the smallest unsigned width holding the bytes (section 1.6)
        const primitive: types.Primitive = switch (byte_count) {
            0, 1 => .u8,
            2 => .u16,
            3, 4 => .u32,
            5, 6, 7, 8 => .u64,
            else => {
                try self.report(token.location, "character literal exceeds 8 bytes (section 1.6)", .{});
                return &unknown_type;
            },
        };
        return self.makeType(.{ .primitive = primitive });
    }

    fn checkLambda(self: *Checker, expression: *const ast.Expression) Error!*const Type {
        const lambda = expression.lambda;
        // capture values are computed in the enclosing scope (section 4.4)
        var capture_bindings: std.ArrayList(Binding) = .empty;
        for (lambda.captures) |capture| {
            const name = capture.name.slice(self.source());
            const outer = self.lookup(name) orelse {
                try capture_bindings.append(self.arena, .{ .name = name, .binding_type = &unknown_type, .mutable = false });
                continue;
            };
            const bound = try self.captureBinding(capture, outer.binding_type, outer.mutable, try self.pierce(outer.binding_type));
            try capture_bindings.append(self.arena, bound);
        }

        try self.pushFrame(true);
        defer self.popFrame();
        for (capture_bindings.items) |binding| {
            const frame = &self.scopes.items[self.scopes.items.len - 1];
            try frame.bindings.append(self.arena, binding);
        }

        var parameter_types: std.ArrayList(*const Type) = .empty;
        for (lambda.function.parameters) |parameter| {
            const parameter_type = try self.typeFromExpression(parameter.parameter_type, self.scope_types);
            try parameter_types.append(self.arena, parameter_type);
            try self.bind(parameter.name, parameter_type, false);
        }

        const declared_return: ?*const Type = if (lambda.function.return_type) |return_expression|
            try self.typeFromExpression(return_expression, self.scope_types)
        else
            null;

        const saved_return = self.return_type;
        const saved_inferred = self.inferred_return;
        const saved_yields = self.yield_frames;
        self.return_type = declared_return;
        self.inferred_return = null;
        self.yield_frames = .empty;
        try self.checkStatement(lambda.function.body);
        const inferred = self.inferred_return;
        self.return_type = saved_return;
        self.inferred_return = saved_inferred;
        self.yield_frames = saved_yields;

        const return_type = declared_return orelse (inferred orelse &void_type);
        return self.makeType(.{ .function = .{
            .parameter_types = try parameter_types.toOwnedSlice(self.arena),
            .return_type = return_type,
        } });
    }

    fn inferReturn(self: *Checker, value_type: *const Type, span: Token.Location) Error!void {
        const concrete = try self.defaulted(value_type);
        if (self.inferred_return) |previous| {
            if (try self.unify(previous, concrete)) |unified| {
                self.inferred_return = unified;
            } else {
                const left = try previous.render(self.arena);
                const right = try concrete.render(self.arena);
                try self.report(span, "return types disagree: {s} versus {s}", .{ left, right });
            }
        } else {
            self.inferred_return = concrete;
        }
    }

    // capture typing (section 2.1): deep copy by default, '&'/' &var' borrow
    // in place, '*'/'*var' take ownership of a pointer-typed value
    // a downcast capture mirrors the subject's indirection: a '&Shape'
    // subject yields '&Circle', '*var Shape' yields '*var Circle' (section 3.2)
    fn downcastBinding(self: *Checker, capture: ast.Capture, subject_raw: *const Type, target: *const Type) Error!Binding {
        if (capture.modifier != null or capture.annotation != null) {
            try self.report(capture.name.location, "a downcast capture mirrors the subject's indirection and takes no modifier or annotation (section 3.2)", .{});
        }
        const name = capture.name.slice(self.source());
        return switch (subject_raw.*) {
            .reference => |indirection| .{
                .name = name,
                .binding_type = try self.makeType(.{ .reference = .{ .mutable = indirection.mutable, .child = target } }),
                .mutable = indirection.mutable,
            },
            .pointer => |indirection| .{
                .name = name,
                .binding_type = try self.makeType(.{ .pointer = .{ .mutable = indirection.mutable, .child = target } }),
                .mutable = indirection.mutable,
            },
            // a pierced subject reads as a reference to the concrete value
            else => .{
                .name = name,
                .binding_type = try self.makeType(.{ .reference = .{ .mutable = false, .child = target } }),
                .mutable = false,
            },
        };
    }

    fn captureBinding(self: *Checker, capture: ast.Capture, subject_raw: *const Type, subject_mutable: bool, subject_value: *const Type) Error!Binding {
        const name = capture.name.slice(self.source());
        if (capture.annotation) |annotation| {
            const annotated = try self.typeFromExpression(annotation, self.scope_types);
            if (annotated.* == .reference) {
                if (annotated.reference.mutable and !subject_mutable) {
                    try self.report(capture.name.location, "a '&var' capture requires a mutable subject (section 2.1)", .{});
                }
                if (!try self.coerce(subject_value, annotated.reference.child)) {
                    try self.typeMismatch(capture.name.location, subject_value, annotated.reference.child);
                }
            } else if (!try self.coerce(subject_value, annotated)) {
                try self.typeMismatch(capture.name.location, subject_value, annotated);
            }
            return .{ .name = name, .binding_type = annotated, .mutable = annotated.* == .reference and annotated.reference.mutable };
        }
        if (capture.modifier) |modifier| {
            switch (modifier) {
                .reference => {
                    const bound = try self.makeType(.{ .reference = .{ .mutable = false, .child = try self.defaulted(subject_value) } });
                    return .{ .name = name, .binding_type = bound, .mutable = false };
                },
                .reference_var => {
                    if (!subject_mutable) {
                        try self.report(capture.name.location, "a '&var' capture requires a mutable subject (section 2.1)", .{});
                    }
                    const bound = try self.makeType(.{ .reference = .{ .mutable = true, .child = try self.defaulted(subject_value) } });
                    return .{ .name = name, .binding_type = bound, .mutable = true };
                },
                .pointer, .pointer_var => {
                    // owning capture: only pointers are movable (section 2.1)
                    const is_pointer = subject_raw.* == .pointer or subject_raw.* == .heap_array or subject_raw.* == .unknown;
                    if (!is_pointer) {
                        const rendered = try subject_raw.render(self.arena);
                        try self.report(capture.name.location, "an owning capture requires a pointer-typed value, found {s} (section 2.1)", .{rendered});
                        return .{ .name = name, .binding_type = &unknown_type, .mutable = modifier == .pointer_var };
                    }
                    if (!subject_mutable) {
                        try self.report(capture.name.location, "an owning capture moves out of its subject, which must be mutable (section 2.1)", .{});
                    }
                    return .{ .name = name, .binding_type = subject_raw, .mutable = modifier == .pointer_var };
                },
            }
        }
        // deep copy default
        return .{ .name = name, .binding_type = try self.defaulted(subject_value), .mutable = false };
    }

    fn checkIf(self: *Checker, if_expr: ast.IfExpression, expression: *const ast.Expression, as_value: bool) Error!*const Type {
        const condition_type = try self.checkExpression(if_expr.condition, null);
        if (!condition_type.isBool()) {
            try self.report(self.expressionSpan(if_expr.condition), "an if condition must be bool", .{});
        }

        try self.yield_frames.append(self.arena, .{ .yielded = null });

        try self.pushFrame(false);
        if (if_expr.capture) |capture| {
            try self.bindIsCapture(if_expr.condition, capture);
        }
        try self.checkStatement(if_expr.then_branch);
        self.popFrame();

        if (if_expr.else_branch) |else_branch| {
            try self.checkStatement(else_branch);
        }

        const frame = self.yield_frames.pop().?;
        if (!as_value) return &void_type;
        const yielded = frame.yielded orelse {
            try self.report(self.expressionSpan(expression), "this if is used as a value but no branch does 'break value' (section 4.3)", .{});
            return &unknown_type;
        };
        if (if_expr.else_branch == null) {
            try self.report(self.expressionSpan(expression), "an if used as a value needs an else branch", .{});
        }
        return yielded;
    }

    // 'if (subject is Enum::Variant) |capture|' binds the variant payload
    fn bindIsCapture(self: *Checker, condition: *const ast.Expression, capture: ast.Capture) Error!void {
        const unwrapped = unwrapGrouped(condition);
        if (unwrapped.* != .cast or unwrapped.cast.operator.tag != .keyword_is) {
            try self.report(capture.name.location, "a capture on 'if' requires an 'is' condition (section 3.2)", .{});
            try self.bind(capture.name, &unknown_type, false);
            return;
        }
        const cast = unwrapped.cast;
        const subject_type = try self.checkExpression(cast.operand, null);
        if (try self.interfaceObject(subject_type)) |_| {
            const target = try self.typeFromExpression(cast.target, self.scope_types);
            const place = try self.lvalueOf(cast.operand);
            const raw = if (place) |info| info.raw else subject_type;
            const binding = try self.downcastBinding(capture, raw, target);
            const frame = &self.scopes.items[self.scopes.items.len - 1];
            try frame.bindings.append(self.arena, binding);
            return;
        }
        const body = try self.enumBody(subject_type) orelse {
            try self.bind(capture.name, &unknown_type, false);
            return;
        };
        const variant_token = cast.target.named.path[cast.target.named.path.len - 1];
        const variant_name = variant_token.slice(self.source());
        const payload: ?*const Type = for (body.variants) |variant| {
            if (std.mem.eql(u8, variant.name, variant_name)) break variant.payload;
        } else null;
        const payload_type = payload orelse {
            try self.report(capture.name.location, "variant '{s}' carries no payload to capture (section 3.2)", .{variant_name});
            try self.bind(capture.name, &unknown_type, false);
            return;
        };
        const place = try self.lvalueOf(cast.operand);
        const subject_mutable = if (place) |info| info.mutable else false;
        const binding = try self.captureBinding(capture, payload_type, subject_mutable, try self.pierce(payload_type));
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        try frame.bindings.append(self.arena, binding);
    }

    fn checkWhile(self: *Checker, while_expr: ast.WhileExpression, expression: *const ast.Expression, as_value: bool) Error!*const Type {
        const condition_type = try self.checkExpression(while_expr.condition, null);
        if (!condition_type.isBool()) {
            try self.report(self.expressionSpan(while_expr.condition), "a while condition must be bool", .{});
        }
        try self.yield_frames.append(self.arena, .{ .yielded = null });
        try self.checkStatement(while_expr.body);
        if (while_expr.else_branch) |else_branch| {
            try self.checkStatement(else_branch);
        }
        const frame = self.yield_frames.pop().?;
        if (!as_value) return &void_type;
        if (while_expr.else_branch == null) {
            try self.report(self.expressionSpan(expression), "a loop used as a value requires an 'else' branch (section 4.3)", .{});
        }
        return frame.yielded orelse {
            try self.report(self.expressionSpan(expression), "this loop is used as a value but never does 'break value' (section 4.3)", .{});
            return &unknown_type;
        };
    }

    fn checkFor(self: *Checker, for_expr: ast.ForExpression, expression: *const ast.Expression, as_value: bool) Error!*const Type {
        var element_types: std.ArrayList(*const Type) = .empty;
        var element_mutability: std.ArrayList(bool) = .empty;
        for (for_expr.subjects) |subject| {
            // a range subject never materializes: runtime bounds are fine
            // and the loop lowers to a counting loop (section 4.3)
            const subject_type = if (subject.* == .array_range)
                try self.checkArrayRange(subject, null, false, true)
            else
                try self.checkExpression(subject, null);
            const resolved = try self.resolveAlias(subject_type);
            var element: ?*const Type = switch (resolved.*) {
                .slice => |slice| slice.child,
                .heap_array => |heap| heap.child,
                .fixed_array => |array| array.element,
                .unknown => &unknown_type,
                else => null,
            };
            var mutable = switch (resolved.*) {
                .slice => |slice| slice.mutable,
                .heap_array => |heap| heap.mutable,
                .fixed_array => if (try self.lvalueOf(subject)) |place| place.mutable else false,
                else => false,
            };
            if (element == null) {
                // the cursor protocol (section 4.3): 'iterator()' yields a
                // cursor whose 'next()' returns Option<T>
                if (try self.cursorElement(subject, subject_type)) |cursor_element| {
                    element = cursor_element;
                    mutable = false;
                }
            }
            if (element == null) {
                const rendered = try subject_type.render(self.arena);
                try self.report(self.expressionSpan(subject), "{s} is not iterable: provide 'iterator()' and 'next()' extension functions (section 4.3)", .{rendered});
            }
            try element_types.append(self.arena, element orelse &unknown_type);
            try element_mutability.append(self.arena, mutable);
        }

        try self.yield_frames.append(self.arena, .{ .yielded = null });
        try self.pushFrame(false);
        for (for_expr.captures, 0..) |capture, capture_index| {
            const element = if (capture_index < element_types.items.len) element_types.items[capture_index] else &unknown_type;
            const mutable = if (capture_index < element_mutability.items.len) element_mutability.items[capture_index] else false;
            const binding = try self.captureBinding(capture, element, mutable, try self.pierce(element));
            const frame = &self.scopes.items[self.scopes.items.len - 1];
            try frame.bindings.append(self.arena, binding);
        }
        if (for_expr.captures.len != for_expr.subjects.len and for_expr.captures.len != 0) {
            try self.report(self.expressionSpan(expression), "{d} subject(s) but {d} capture(s)", .{ for_expr.subjects.len, for_expr.captures.len });
        }
        try self.checkStatement(for_expr.body);
        self.popFrame();
        if (for_expr.else_branch) |else_branch| {
            try self.checkStatement(else_branch);
        }
        const frame = self.yield_frames.pop().?;
        if (!as_value) return &void_type;
        if (for_expr.else_branch == null) {
            try self.report(self.expressionSpan(expression), "a loop used as a value requires an 'else' branch (section 4.3)", .{});
        }
        return frame.yielded orelse {
            try self.report(self.expressionSpan(expression), "this loop is used as a value but never does 'break value' (section 4.3)", .{});
            return &unknown_type;
        };
    }

    // resolves the cursor protocol for one 'for' subject (section 4.3):
    // 'subject.iterator()' yields a cursor, 'cursor.next()' yields
    // 'Option<T>', and the loop variable binds to T
    fn cursorElement(self: *Checker, subject: *const ast.Expression, subject_type: *const Type) Error!?*const Type {
        const place = try self.lvalueOf(subject);
        const raw: *const Type = if (place) |found| found.raw else subject_type;
        const receiver: MethodReceiver = .{
            .raw = raw,
            .pierced = try self.pierce(raw),
            .mutable = if (place) |found| piercedMutability(found) else false,
        };
        const cursor_type = (try self.quietMethodReturn(receiver, "iterator")) orelse return null;
        // the cursor is a fresh mutable local advanced by each 'next' call
        const cursor_receiver: MethodReceiver = .{
            .raw = cursor_type,
            .pierced = try self.pierce(cursor_type),
            .mutable = true,
        };
        const next_type = (try self.quietMethodReturn(cursor_receiver, "next")) orelse return null;
        return self.optionPayload(next_type);
    }

    const QuietCandidate = struct {
        symbol: resolution.Symbol,
        return_type: *const Type,
        type_bindings: []const Type.Binding,
    };

    // resolves a zero-argument extension call without emitting diagnostics
    fn quietMethodCandidate(self: *Checker, receiver: MethodReceiver, name: []const u8) Error!?QuietCandidate {
        const symbols = self.globals.get(name) orelse return null;
        const empty_call = .{ .type_arguments = @as([]const *const ast.TypeExpression, &.{}) };
        var result: ?QuietCandidate = null;
        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            if (fn_def.function.parameters.len == 0 or !fn_def.function.parameters[0].is_self) continue;
            if (try self.tryCandidate(symbol, empty_call, &.{}, receiver)) |candidate| {
                result = .{
                    .symbol = symbol,
                    .return_type = candidate.return_type,
                    .type_bindings = candidate.type_bindings,
                };
            }
        }
        return result;
    }

    fn quietMethodReturn(self: *Checker, receiver: MethodReceiver, name: []const u8) Error!?*const Type {
        const candidate = (try self.quietMethodCandidate(receiver, name)) orelse return null;
        return candidate.return_type;
    }

    pub const CursorProtocol = struct {
        iterator: resolution.Symbol,
        iterator_bindings: []const Type.Binding,
        cursor_type: *const Type,
        next: resolution.Symbol,
        next_bindings: []const Type.Binding,
        option_type: *const Type,
        element: *const Type,
    };

    /// Resolves the cursor protocol (section 4.3) for code generation,
    /// keeping the chosen symbols and their inferred type bindings:
    /// 'subject.iterator()' yields the cursor, 'cursor.next()' yields
    /// 'Option<element>'.
    pub fn cursorProtocolOf(self: *Checker, subject_type: *const Type) Error!?CursorProtocol {
        const receiver: MethodReceiver = .{
            .raw = subject_type,
            .pierced = try self.pierce(subject_type),
            .mutable = true,
        };
        const iterator = (try self.quietMethodCandidate(receiver, "iterator")) orelse return null;
        const cursor_receiver: MethodReceiver = .{
            .raw = iterator.return_type,
            .pierced = try self.pierce(iterator.return_type),
            .mutable = true,
        };
        const next = (try self.quietMethodCandidate(cursor_receiver, "next")) orelse return null;
        const element = (try self.optionPayload(next.return_type)) orelse return null;
        return .{
            .iterator = iterator.symbol,
            .iterator_bindings = iterator.type_bindings,
            .cursor_type = iterator.return_type,
            .next = next.symbol,
            .next_bindings = next.type_bindings,
            .option_type = next.return_type,
            .element = element,
        };
    }

    // recognizes an instance of the 'Option<T>' lang item (section 5.1a)
    fn optionPayload(self: *Checker, candidate: *const Type) Error!?*const Type {
        const resolved = try self.resolveAlias(candidate);
        if (resolved.* != .declared) return null;
        const declared = resolved.declared;
        const key = self.views[declared.view_index].key orelse return null;
        if (!std.mem.eql(u8, key, "std::option") or !std.mem.eql(u8, declared.name, "Option")) return null;
        if (declared.arguments.len != 1) return null;
        return declared.arguments[0];
    }

    // every match must be exhaustive: an enum subject by covering all
    // variants or an 'else' arm, every other subject by an 'else' arm
    // (section 4.3)
    fn checkExhaustiveness(self: *Checker, match_expr: ast.MatchExpression, subject_enum: ?EnumBody, expression: *const ast.Expression) Error!void {
        const body = subject_enum orelse {
            try self.report(self.expressionSpan(expression), "a match over this subject can never cover every value: add an 'else' arm (section 4.3)", .{});
            return;
        };
        var missing: std.ArrayList([]const u8) = .empty;
        for (body.variants) |variant| {
            const covered = for (match_expr.arms) |arm| {
                const pattern = arm.pattern orelse continue;
                const unwrapped = unwrapGrouped(pattern);
                const arm_token = switch (unwrapped.*) {
                    .path => |path| path[path.len - 1],
                    .implied_variant => |token| token,
                    else => continue,
                };
                if (std.mem.eql(u8, arm_token.slice(self.source()), variant.name)) break true;
            } else false;
            if (!covered) try missing.append(self.arena, variant.name);
        }
        if (missing.items.len == 0) return;
        const joined = try std.mem.join(self.arena, "', '", missing.items);
        try self.report(self.expressionSpan(expression), "this match does not cover variant{s} '{s}' of '{s}': add arms or an 'else' arm (section 4.3)", .{
            if (missing.items.len == 1) "" else "s",
            joined,
            body.name,
        });
    }

    fn checkMatch(self: *Checker, match_expr: ast.MatchExpression, expression: *const ast.Expression, as_value: bool) Error!*const Type {
        const subject_type = try self.checkExpression(match_expr.subject, null);
        const subject_enum = try self.enumBody(subject_type);
        const subject_interface = try self.interfaceObject(subject_type);
        const subject_place = try self.lvalueOf(match_expr.subject);
        const subject_mutable = if (subject_place) |place| place.mutable else false;
        const subject_raw = if (subject_place) |place| place.raw else subject_type;

        try self.yield_frames.append(self.arena, .{ .yielded = null });
        var has_else_arm = false;
        for (match_expr.arms) |arm| {
            if (arm.pattern == null) has_else_arm = true;
            try self.pushFrame(false);
            if (arm.pattern) |pattern| {
                if (subject_enum) |body| {
                    try self.checkEnumArm(body, pattern, arm.capture, subject_mutable);
                } else if (subject_interface) |interface| {
                    try self.checkInterfaceArm(interface, pattern, arm.capture, subject_raw);
                } else {
                    // numeric, character, and string subjects match literal
                    // patterns; captures are enum-only (section 4.3)
                    const pattern_type = try self.checkExpression(pattern, subject_type);
                    if (try self.unify(pattern_type, subject_type) == null) {
                        try self.operandMismatchAt(self.expressionSpan(pattern), pattern_type, subject_type);
                    }
                    if (arm.capture) |capture| {
                        try self.report(capture.name.location, "pattern captures are only valid on enum payload variants (section 4.3)", .{});
                        try self.bind(capture.name, &unknown_type, false);
                    }
                }
            } else if (arm.capture) |capture| {
                try self.report(capture.name.location, "an 'else' arm has no payload to capture", .{});
                try self.bind(capture.name, &unknown_type, false);
            }
            try self.checkStatement(arm.body);
            self.popFrame();
        }
        if (!has_else_arm and subject_type.* != .unknown) {
            try self.checkExhaustiveness(match_expr, subject_enum, expression);
        }
        if (match_expr.else_branch) |else_branch| {
            try self.checkStatement(else_branch);
        }
        const frame = self.yield_frames.pop().?;
        if (!as_value) return &void_type;
        return frame.yielded orelse {
            try self.report(self.expressionSpan(expression), "this match is used as a value but no arm does 'break value' (section 4.3)", .{});
            return &unknown_type;
        };
    }

    fn checkEnumArm(self: *Checker, body: EnumBody, pattern: *const ast.Expression, capture: ?ast.Capture, subject_mutable: bool) Error!void {
        const unwrapped = unwrapGrouped(pattern);
        const variant_token = switch (unwrapped.*) {
            .path => |path| path[path.len - 1],
            .implied_variant => |token| token,
            else => {
                try self.report(self.expressionSpan(pattern), "an enum match arm pattern must name a variant of '{s}'", .{body.name});
                return;
            },
        };
        const variant_name = variant_token.slice(self.source());
        const member = for (body.variants) |candidate| {
            if (std.mem.eql(u8, candidate.name, variant_name)) break candidate;
        } else {
            try self.report(variant_token.location, "enum '{s}' has no variant '{s}'", .{ body.name, variant_name });
            return;
        };
        if (capture) |arm_capture| {
            const payload_type = member.payload orelse {
                try self.report(arm_capture.name.location, "variant '{s}' carries no payload to capture (section 4.3)", .{variant_name});
                try self.bind(arm_capture.name, &unknown_type, false);
                return;
            };
            const binding = try self.captureBinding(arm_capture, payload_type, subject_mutable, try self.pierce(payload_type));
            const frame = &self.scopes.items[self.scopes.items.len - 1];
            try frame.bindings.append(self.arena, binding);
        }
    }

    // a match arm on an interface object names a concrete type and captures
    // the downcasted value (section 3.2)
    fn checkInterfaceArm(self: *Checker, interface: Type.Interface, pattern: *const ast.Expression, capture: ?ast.Capture, subject_raw: *const Type) Error!void {
        const unwrapped = unwrapGrouped(pattern);
        if (unwrapped.* != .path) {
            try self.report(self.expressionSpan(pattern), "a match arm on an interface object must name a concrete type implementing '{s}' (section 3.2)", .{interface.name});
            return;
        }
        const path = unwrapped.path;
        const name_token = path[path.len - 1];
        const name = name_token.slice(self.source());
        const target: *const Type = resolve: {
            const symbols = self.globals.get(name) orelse break :resolve null;
            const symbol = symbols.items[0];
            if (symbol.definition.kind != .type_def) break :resolve null;
            break :resolve try self.makeType(.{ .declared = .{
                .definition = symbol.definition,
                .view_index = symbol.view_index,
                .name = name,
                .arguments = &.{},
            } });
        } orelse {
            try self.report(name_token.location, "'{s}' is not a type, so it cannot match an interface object's concrete type (section 3.2)", .{name});
            return;
        };
        if (!try self.implements(target, interface)) {
            try self.report(name_token.location, "'{s}' does not implement '{s}', so this arm can never match (section 3.2)", .{ name, interface.name });
        }
        if (capture) |arm_capture| {
            const binding = try self.downcastBinding(arm_capture, subject_raw, target);
            const frame = &self.scopes.items[self.scopes.items.len - 1];
            try frame.bindings.append(self.arena, binding);
        }
    }

    fn checkCall(self: *Checker, expression: *const ast.Expression, expected: ?*const Type) Error!*const Type {
        const call = expression.call;
        const callee = unwrapGrouped(call.callee);

        // '::Variant(payload)' with the enum implied from context
        if (callee.* == .implied_variant) {
            if (try self.impliedVariant(callee.implied_variant, expected)) |variant| {
                return self.callVariantConstructor(variant, call, expected, callee.implied_variant.location);
            }
            for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
            return &unknown_type;
        }

        if (callee.* == .path) {
            const path = callee.path;
            const name = path[path.len - 1].slice(self.source());
            // a local binding of function type
            if (path.len == 1) {
                if (self.lookup(path[0].slice(self.source()))) |binding| {
                    return self.callFunctionValue(binding.binding_type, call, path[0].location);
                }
                if (builtinMacro(name)) {
                    if (self.comptime_depth == 0) {
                        try self.report(path[0].location, "a macro call must be invoked with '#' (section 6.3)", .{});
                    }
                    for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
                    return &unknown_type;
                }
            }
            // enum variant construction: 'Holder::Boxed(value)'
            if (try self.variantOfPath(path)) |variant| {
                return self.callVariantConstructor(variant, call, expected, path[path.len - 1].location);
            }
            if (self.globals.get(name)) |symbols| {
                return self.callOverloads(name, symbols, call, expression, path[0].location);
            }
            return &unknown_type;
        }

        if (callee.* == .member) {
            const member = callee.member;
            const name = member.name.slice(self.source());
            const place = try self.lvalueOf(member.object);
            const raw: *const Type = if (place) |found| found.raw else try self.checkExpression(member.object, null);
            const receiver: MethodReceiver = .{
                .raw = raw,
                .pierced = try self.pierce(raw),
                .mutable = if (place) |found| piercedMutability(found) else false,
            };
            // built-in '.length()' on every array form (section 5.1)
            const resolved = try self.resolveAlias(receiver.pierced);
            const is_array = resolved.* == .slice or resolved.* == .heap_array or resolved.* == .fixed_array;
            if (is_array and std.mem.eql(u8, name, "length") and call.arguments.len == 0) {
                return self.makeType(.{ .primitive = .u64 });
            }
            if (receiver.pierced.* == .unknown) {
                for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
                return &unknown_type;
            }
            // calls through an interface object dispatch at runtime via the
            // vtable; the interface declaration is the signature (section 5.2)
            if (receiver.pierced.* == .interface) {
                return self.callInterfaceFunction(receiver.pierced.interface, member.name, call);
            }
            // a constrained generic value exposes its constraint's interface
            // functions, resolved statically per instantiation (section 5.2)
            if (receiver.pierced.* == .type_parameter) {
                if (receiver.pierced.type_parameter.constraint) |constraint| {
                    return self.callInterfaceFunction(constraint, member.name, call);
                }
            }
            // a struct member of function type is an ordinary indirect call
            if (try self.structBody(receiver.pierced)) |body| {
                const body_source = self.views[body.view_index].source;
                for (body.members) |struct_member| {
                    if (!std.mem.eql(u8, struct_member.name.slice(body_source), name)) continue;
                    const member_type = try self.typeFromExpressionIn(struct_member.member_type, body.environment, body.view_index);
                    if (member_type.* == .function) {
                        return self.callFunctionValue(member_type, call, member.name.location);
                    }
                }
            }
            // extension function call (section 4.5)
            const symbols = self.globals.get(name) orelse {
                const rendered = try receiver.pierced.render(self.arena);
                try self.report(member.name.location, "no extension function '{s}' for {s} (section 4.5)", .{ name, rendered });
                for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
                return &unknown_type;
            };
            const result = try self.callMethodOverloads(name, symbols, call, receiver, expression, member.name.location);
            // a '*T' self parameter takes ownership of the receiver (section
            // 4.2): a place must transfer it explicitly with 'move'; only a
            // fresh value ('new T {}', a call result) passes bare
            if (place != null) {
                if (self.call_targets.get(expression)) |target| {
                    if (target.definition.kind == .fn_def) {
                        const parameters = target.definition.kind.fn_def.function.parameters;
                        if (parameters.len != 0 and parameters[0].is_self and selfTakesOwnership(parameters[0].parameter_type)) {
                            try self.report(member.name.location, "'{s}' takes ownership of its receiver: transfer it with '(move ...).{s}(...)' or allocate inline (section 4.2)", .{ name, name });
                        }
                    }
                }
            }
            return result;
        }

        const callee_type = try self.checkExpression(call.callee, null);
        return self.callFunctionValue(callee_type, call, self.expressionSpan(call.callee));
    }

    fn selfTakesOwnership(parameter_type: *const ast.TypeExpression) bool {
        if (parameter_type.* != .modified) return false;
        return switch (parameter_type.modified.modifier) {
            .pointer, .pointer_var => true,
            .reference, .reference_var => false,
        };
    }

    fn callFunctionValue(self: *Checker, callee_type: *const Type, call: anytype, span: Token.Location) Error!*const Type {
        if (callee_type.* == .unknown) {
            for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
            return &unknown_type;
        }
        if (callee_type.* != .function) {
            const rendered = try callee_type.render(self.arena);
            try self.report(span, "{s} is not callable", .{rendered});
            for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
            return &unknown_type;
        }
        const function = callee_type.function;
        if (call.arguments.len != function.parameter_types.len) {
            try self.report(span, "expected {d} argument(s), found {d}", .{ function.parameter_types.len, call.arguments.len });
        }
        const checked = @min(call.arguments.len, function.parameter_types.len);
        for (call.arguments[0..checked], function.parameter_types[0..checked]) |argument, parameter_type| {
            const argument_type = try self.checkExpression(argument, parameter_type);
            try self.expectAssignable(argument_type, parameter_type, argument, self.expressionSpan(argument));
        }
        for (call.arguments[checked..]) |argument| _ = try self.checkExpression(argument, null);
        return function.return_type;
    }

    const Variant = struct {
        enum_type: *const Type,
        name: []const u8,
        payload: ?*const Type,
    };

    // resolves 'Enum::Variant' / 'module::Enum::Variant' paths
    fn variantOfPath(self: *Checker, path: []const Token) Error!?Variant {
        if (path.len < 2) return null;
        const type_name = path[path.len - 2].slice(self.source());
        const symbols = self.globals.get(type_name) orelse return null;
        return self.variantOfSymbol(symbols.items[0], type_name, path[path.len - 1].slice(self.source()));
    }

    fn variantOfSymbol(self: *Checker, symbol: resolution.Symbol, type_name: []const u8, variant_name: []const u8) Error!?Variant {
        if (symbol.definition.kind != .type_def) return null;
        const type_def = symbol.definition.kind.type_def;
        // enum syntax or a comptime base that may synthesise an enum
        // (section 3.4); anything else cannot carry variants
        if (type_def.base.* != .enum_type and type_def.base.* != .comptime_type) return null;
        // generic enums get their arguments inferred at the construction site
        var arguments: std.ArrayList(*const Type) = .empty;
        for (type_def.type_parameters) |type_parameter| {
            const parameter_name = type_parameter.name.slice(self.views[symbol.view_index].source);
            try arguments.append(self.arena, try self.makeType(.{ .type_parameter = .{ .name = parameter_name, .constraint = null } }));
        }
        const enum_type = try self.makeType(.{ .declared = .{
            .definition = symbol.definition,
            .view_index = symbol.view_index,
            .name = type_name,
            .arguments = try arguments.toOwnedSlice(self.arena),
        } });
        const body = (try self.enumBody(enum_type)) orelse return null;
        for (body.variants) |variant| {
            if (std.mem.eql(u8, variant.name, variant_name)) {
                return .{
                    .enum_type = enum_type,
                    .name = variant_name,
                    .payload = variant.payload,
                };
            }
        }
        return null;
    }

    // '::Variant' resolution (section 3.2): the contextual type when it is
    // an enum carrying the variant, otherwise the single visible enum
    // definition carrying it
    fn impliedVariant(self: *Checker, name_token: Token, expected: ?*const Type) Error!?Variant {
        const variant_name = name_token.slice(self.source());
        if (expected) |context| {
            const resolved = try self.resolveAlias(context);
            if (try self.enumBody(resolved)) |body| {
                for (body.variants) |variant| {
                    if (std.mem.eql(u8, variant.name, variant_name)) {
                        return .{
                            .enum_type = resolved,
                            .name = variant_name,
                            .payload = variant.payload,
                        };
                    }
                }
                try self.report(name_token.location, "'{s}' has no variant '{s}'", .{ body.name, variant_name });
                return null;
            }
        }
        var found: ?Variant = null;
        var count: usize = 0;
        var iterator = self.globals.iterator();
        while (iterator.next()) |entry| {
            if (try self.variantOfSymbol(entry.value_ptr.items[0], entry.key_ptr.*, variant_name)) |variant| {
                found = variant;
                count += 1;
            }
        }
        if (count == 1) return found;
        if (count == 0) {
            try self.report(name_token.location, "no enum in scope has a variant '{s}' (section 3.2)", .{variant_name});
        } else {
            try self.report(name_token.location, "'::{s}' is ambiguous: {d} enums in scope have this variant; name the enum (section 3.2)", .{ variant_name, count });
        }
        return null;
    }

    // a payload-less '::Variant' expression; '::Variant(x)' arrives as a
    // call and is handled in checkCall
    fn checkImpliedVariant(self: *Checker, expression: *const ast.Expression, expected: ?*const Type) Error!*const Type {
        const token = expression.implied_variant;
        const variant = (try self.impliedVariant(token, expected)) orelse return &unknown_type;
        const empty_call: EmptyCall = .{};
        return self.callVariantConstructor(variant, empty_call, expected, token.location);
    }

    const EmptyCall = struct {
        arguments: []const *const ast.Expression = &.{},
    };

    // every type parameter must be bound by the payload or the context
    // when a variant is constructed (section 3.7)
    fn reportUnboundVariantParameters(self: *Checker, variant: Variant, bindings: *const TypeEnvironment, span: Token.Location) Error!void {
        if (variant.enum_type.* != .declared) return;
        for (variant.enum_type.declared.arguments) |argument| {
            if (argument.* != .type_parameter) continue;
            if (bindings.get(argument.type_parameter.name) != null) continue;
            try self.report(span, "cannot infer type parameter '{s}' of '{s}': annotate the expected type (section 3.7)", .{ argument.type_parameter.name, variant.enum_type.declared.name });
        }
    }

    fn callVariantConstructor(self: *Checker, variant: Variant, call: anytype, expected: ?*const Type, span: Token.Location) Error!*const Type {
        // a contextual type of the same enum fixes the type parameters first
        // ('var maybe: Option<u32> = Option::Some(7)', section 3.7)
        var bindings: TypeEnvironment = .empty;
        if (expected) |context| {
            if (context.* == .declared and variant.enum_type.* == .declared and context.declared.definition == variant.enum_type.declared.definition) {
                for (variant.enum_type.declared.arguments, context.declared.arguments) |placeholder, bound| {
                    if (placeholder.* == .type_parameter) {
                        try bindings.put(self.arena, placeholder.type_parameter.name, bound);
                    }
                }
            }
        }
        const declared_payload = variant.payload orelse {
            if (call.arguments.len != 0) {
                try self.report(span, "variant '{s}' carries no payload", .{variant.name});
            }
            for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
            try self.reportUnboundVariantParameters(variant, &bindings, span);
            return self.substitute(variant.enum_type, &bindings);
        };
        if (call.arguments.len != 1) {
            try self.report(span, "variant '{s}' takes exactly one payload argument", .{variant.name});
            for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
            return self.substitute(variant.enum_type, &bindings);
        }
        const payload_type = try self.substitute(declared_payload, &bindings);
        const argument_type = try self.checkExpression(call.arguments[0], payload_type);
        // remaining parameters infer from the payload value (section 3.7)
        if (!try self.unifyParameter(payload_type, argument_type, &bindings)) {
            try self.expectAssignable(argument_type, payload_type, call.arguments[0], span);
        } else if (!try self.coerce(argument_type, try self.substitute(payload_type, &bindings))) {
            try self.expectAssignable(argument_type, try self.substitute(payload_type, &bindings), call.arguments[0], span);
        }
        try self.reportUnboundVariantParameters(variant, &bindings, span);
        return self.substitute(variant.enum_type, &bindings);
    }

    // an argument whose type depends on the parameter it lands in: an
    // implied variant or a generic enum variant construction (sections 3.2,
    // 3.7); these are typed per overload candidate instead of up front
    fn contextualArgument(self: *Checker, expression: *const ast.Expression) Error!bool {
        const unwrapped = unwrapGrouped(expression);
        switch (unwrapped.*) {
            .implied_variant => return true,
            .path => |path| {
                const variant = (try self.variantOfPath(path)) orelse return false;
                return variant.enum_type.declared.arguments.len != 0;
            },
            .call => |call| {
                const callee = unwrapGrouped(call.callee);
                if (callee.* == .implied_variant) return true;
                if (callee.* != .path) return false;
                const variant = (try self.variantOfPath(callee.path)) orelse return false;
                return variant.enum_type.declared.arguments.len != 0;
            },
            else => return false,
        }
    }

    // types an expression against an expected type, keeping no diagnostics;
    // any diagnostic means the candidate under trial does not fit
    fn quietExpressionType(self: *Checker, expression: *const ast.Expression, expected: *const Type) Error!?*const Type {
        const saved = self.diagnostics.items.len;
        const result = try self.checkExpression(expression, expected);
        const failed = self.diagnostics.items.len != saved;
        self.diagnostics.shrinkRetainingCapacity(saved);
        return if (failed) null else result;
    }

    // snapshots the names visible to a '#' expression: const bindings carry
    // their initializer, runtime bindings poison the name (section 6.1)
    fn snapshotComptimeEnvironment(self: *Checker) Error![]const ComptimeEnvironmentEntry {
        var environment: std.ArrayList(ComptimeEnvironmentEntry) = .empty;
        for (self.scopes.items) |frame| {
            for (frame.bindings.items) |binding| {
                try environment.append(self.arena, .{ .name = binding.name, .initializer = binding.initializer });
            }
        }
        return environment.toOwnedSlice(self.arena);
    }

    fn deferComptime(self: *Checker, outer: *const ast.Expression, inner: *const ast.Expression) Error!void {
        try self.pending_comptime.append(self.arena, .{
            .outer = outer,
            .inner = inner,
            .view_index = self.current_view,
            .environment = try self.snapshotComptimeEnvironment(),
        });
    }

    // compile-time evaluation (section 6.1) runs once checking has filled
    // every side table, so '#' expressions may call functions defined later
    fn runPendingComptime(self: *Checker) Error!void {
        for (self.pending_comptime.items) |pending| {
            self.current_view = pending.view_index;
            try self.evaluatePendingComptime(pending);
        }
    }

    fn evaluatePendingComptime(self: *Checker, pending: PendingComptime) Error!void {
        _ = try self.runValueComptime(pending.outer, pending.inner, pending.view_index, pending.environment);
    }

    // builds a sandboxed interpreter with the constants of one snapshot
    // bound in declaration order; a failing initializer simply is not
    // compile-time-known
    fn comptimeMachine(self: *Checker, view_index: usize, environment: []const ComptimeEnvironmentEntry) Error!*Interpreter {
        const sink = try self.arena.create(std.Io.Writer.Allocating);
        sink.* = .init(self.arena);
        const machine = try self.arena.create(Interpreter);
        machine.* = Interpreter.init(
            self.arena,
            self.views,
            self.globals,
            &self.expression_types,
            &self.call_targets,
            &self.call_type_bindings,
            &self.comptime_values,
            &self.cast_shapes,
            &sink.writer,
        );
        machine.comptime_mode = true;
        machine.step_budget = 1_000_000;
        machine.current_view = view_index;
        machine.reflection = .{
            .context = @ptrCast(self),
            .reflect_named = reflectNamedHook,
            .reflect_expression = reflectExpressionHook,
        };
        machine.pushEnvironment() catch return error.OutOfMemory;
        for (environment) |entry| {
            const initializer = entry.initializer orelse {
                machine.unbindValue(entry.name);
                continue;
            };
            const value = machine.evaluate(initializer) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => {
                    machine.unbindValue(entry.name);
                    continue;
                },
            };
            machine.bindValue(entry.name, value) catch return error.OutOfMemory;
        }
        return machine;
    }

    // evaluates a '#' expression in value position: faults become compile
    // errors, the pointer barrier applies, and the value is recorded for
    // runtime substitution
    fn runValueComptime(self: *Checker, outer: *const ast.Expression, inner: *const ast.Expression, view_index: usize, environment: []const ComptimeEnvironmentEntry) Error!?Interpreter.Value {
        const machine = try self.comptimeMachine(view_index, environment);
        const value = machine.evaluate(inner) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                try self.report(self.expressionSpan(inner), "comptime evaluation failed: {s} (section 6.1)", .{machine.fault_message orelse "unspecified fault"});
                return null;
            },
        };
        if (violatesPointerBarrier(value)) {
            try self.report(self.expressionSpan(outer), "a compile-time result cannot carry a '&T' or '*T' indirection into runtime (section 6.2)", .{});
            return null;
        }
        try self.comptime_values.put(self.arena, outer, value);
        return value;
    }

    // whether a '#' expression must evaluate during checking: macro calls,
    // built-in macros, and type reflection produce values whose TYPE the
    // checker needs. Nested '#' constructs handle their own evaluation, so
    // control-flow bodies are not scanned.
    fn needsEagerComptime(self: *Checker, expression: *const ast.Expression) bool {
        switch (expression.*) {
            .path => |path| {
                if (path.len != 1) return false;
                const name = path[0].slice(self.source());
                if (self.lookup(name) != null) return false;
                if (primitiveByName(name) != null) return true;
                if (std.mem.eql(u8, name, "void")) return true;
                if (self.globals.get(name)) |symbols| {
                    return switch (symbols.items[0].definition.kind) {
                        .type_def, .interface_def => true,
                        else => false,
                    };
                }
                return false;
            },
            .call => |call| {
                const callee = call.callee;
                if (callee.* == .path and callee.path.len == 1) {
                    const name = callee.path[0].slice(self.source());
                    if (self.lookup(name) == null) {
                        if (builtinMacro(name)) return true;
                        if (self.globals.get(name)) |symbols| {
                            if (symbols.items[0].definition.kind == .macro_def) return true;
                        }
                    }
                }
                if (self.needsEagerComptime(call.callee)) return true;
                for (call.arguments) |argument| {
                    if (self.needsEagerComptime(argument)) return true;
                }
                return false;
            },
            .member => |member| return self.needsEagerComptime(member.object),
            .index => |index| return self.needsEagerComptime(index.object) or self.needsEagerComptime(index.subscript),
            .grouped => |inner| return self.needsEagerComptime(inner),
            .unary => |unary| return self.needsEagerComptime(unary.operand),
            .binary => |binary| return self.needsEagerComptime(binary.left) or self.needsEagerComptime(binary.right),
            .cast => |cast| return self.needsEagerComptime(cast.operand),
            // a nested '#' marking reflection makes the whole expression
            // statically untypable, so the outer '#' must run eagerly too
            .comptime_expr => |inner| return self.needsEagerComptime(inner),
            else => return false,
        }
    }

    // the runtime type of a compile-time value, for '#' results in value
    // position (section 6.1, plain values only)
    fn typeOfComptimeValue(self: *Checker, value: Interpreter.Value, span: Token.Location) Error!*const Type {
        switch (value) {
            .void_value => return &void_type,
            .integer => |integer| return self.makeType(.{ .primitive = integer.primitive orelse .i32 }),
            .float => |float| return self.makeType(.{ .primitive = float.primitive orelse .f64 }),
            .bool_value => return &bool_type,
            .slice => |instance| return self.makeType(.{ .slice = .{ .mutable = false, .child = try self.comptimeElementType(instance, span) } }),
            .heap_array => |instance| {
                const alive = instance orelse return &unknown_type;
                return self.makeType(.{ .heap_array = .{ .mutable = false, .child = try self.comptimeElementType(alive, span) } });
            },
            .array => |instance| return self.makeType(.{ .fixed_array = .{
                .length = instance.elements.len,
                .element = try self.comptimeElementType(instance, span),
            } }),
            .struct_value => |instance| {
                if (instance.type_name.len != 0) {
                    if (self.globals.get(instance.type_name)) |symbols| {
                        const symbol = symbols.items[0];
                        if (symbol.definition.kind == .type_def) {
                            return self.makeType(.{ .declared = .{
                                .definition = symbol.definition,
                                .view_index = symbol.view_index,
                                .name = instance.type_name,
                                .arguments = &.{},
                            } });
                        }
                    }
                }
                try self.report(span, "this compile-time struct value has no nameable runtime type (section 6.1)", .{});
                return &unknown_type;
            },
            .enum_value => |instance| return self.comptimeEnumType(instance.variant, span),
            .function => |symbol| return self.functionType(symbol),
            .type_value => {
                try self.report(span, "a '#Type' cannot be retained in a runtime declaration (section 3.4)", .{});
                return &unknown_type;
            },
            .pointer, .reference, .closure => return &unknown_type,
        }
    }

    fn comptimeElementType(self: *Checker, instance: *Interpreter.Value.ArrayInstance, span: Token.Location) Error!*const Type {
        if (instance.elements.len == 0) return self.makeType(.{ .primitive = .u8 });
        return self.typeOfComptimeValue(instance.elements[0], span);
    }

    // names the unique enum declaring this variant; mirrors implied-variant
    // inference (section 3.2)
    fn comptimeEnumType(self: *Checker, variant_name: []const u8, span: Token.Location) Error!*const Type {
        var found: ?*const Type = null;
        var count: usize = 0;
        var iterator = self.globals.iterator();
        while (iterator.next()) |entry| {
            for (entry.value_ptr.items) |symbol| {
                if (symbol.definition.kind != .type_def) continue;
                const type_def = symbol.definition.kind.type_def;
                if (type_def.base.* != .enum_type or type_def.type_parameters.len != 0) continue;
                const view_source = self.views[symbol.view_index].source;
                for (type_def.base.enum_type) |member| {
                    if (std.mem.eql(u8, member.name.slice(view_source), variant_name)) {
                        count += 1;
                        found = try self.makeType(.{ .declared = .{
                            .definition = symbol.definition,
                            .view_index = symbol.view_index,
                            .name = type_def.name.slice(view_source),
                            .arguments = &.{},
                        } });
                    }
                }
            }
        }
        if (count == 1) return found.?;
        try self.report(span, "the runtime enum type of this compile-time value is ambiguous or unknown (section 6.1)", .{});
        return &unknown_type;
    }

    // the pointer barrier (section 6.2): references, pointers, and closures
    // cannot escape compile time; slices and heap arrays materialize
    fn violatesPointerBarrier(value: Interpreter.Value) bool {
        switch (value) {
            .pointer, .reference, .closure => return true,
            .struct_value => |instance| {
                for (instance.fields) |field| {
                    if (violatesPointerBarrier(field.value)) return true;
                }
                return false;
            },
            .enum_value => |instance| {
                const payload = instance.payload orelse return false;
                return violatesPointerBarrier(payload);
            },
            .array => |instance| return arrayViolatesPointerBarrier(instance),
            .slice => |instance| return arrayViolatesPointerBarrier(instance),
            .heap_array => |instance| {
                const alive = instance orelse return true;
                return arrayViolatesPointerBarrier(alive);
            },
            else => return false,
        }
    }

    fn arrayViolatesPointerBarrier(instance: *Interpreter.Value.ArrayInstance) bool {
        for (instance.elements) |element| {
            if (violatesPointerBarrier(element)) return true;
        }
        return false;
    }

    fn reflectNamedHook(context: *anyopaque, name: []const u8) error{OutOfMemory}!?*Interpreter.Value.TypeDescription {
        const self: *Checker = @ptrCast(@alignCast(context));
        return self.reflectName(name);
    }

    fn reflectExpressionHook(context: *anyopaque, expression: *const ast.Expression) error{OutOfMemory}!?*Interpreter.Value.TypeDescription {
        const self: *Checker = @ptrCast(@alignCast(context));
        const recorded = self.expression_types.get(expression) orelse return null;
        return try self.describeType(recorded, 0);
    }

    // '#T' reflection (section 3.4): a primitive, type, or interface name
    // becomes a '#Type' description
    fn reflectName(self: *Checker, name: []const u8) Error!?*Interpreter.Value.TypeDescription {
        if (primitiveByName(name)) |primitive| {
            return try self.describeType(try self.makeType(.{ .primitive = primitive }), 0);
        }
        // '#void' marks a payload-less enum variant in 'add_member'
        // (section 3.4)
        if (std.mem.eql(u8, name, "void")) {
            return try self.voidDescription();
        }
        const symbols = self.globals.get(name) orelse return null;
        const symbol = symbols.items[0];
        switch (symbol.definition.kind) {
            .type_def => |type_def| {
                // generic types reflect only as instances, not as templates
                if (type_def.type_parameters.len != 0) return null;
                const declared = try self.makeType(.{ .declared = .{
                    .definition = symbol.definition,
                    .view_index = symbol.view_index,
                    .name = name,
                    .arguments = &.{},
                } });
                return try self.describeType(declared, 0);
            },
            .interface_def => {
                const description = try self.arena.create(Interpreter.Value.TypeDescription);
                description.* = .{
                    .kind = .interface_kind,
                    .name = name,
                    .primitive = null,
                    .members = .empty,
                    .interface_names = &.{},
                    .origin = null,
                };
                return description;
            },
            else => return null,
        }
    }

    // converts a checker type into a '#Type' description (section 3.4)
    fn describeType(self: *Checker, candidate: *const Type, depth: u32) Error!*Interpreter.Value.TypeDescription {
        const description = try self.arena.create(Interpreter.Value.TypeDescription);
        // untyped literals default first (section 3.3), so '#type_of(7)'
        // reflects i32
        const resolved = try self.resolveAlias(try self.defaulted(candidate));
        description.* = .{
            .kind = .other_kind,
            .name = "",
            .primitive = null,
            .members = .empty,
            .interface_names = &.{},
            .origin = resolved,
        };
        if (depth > 16) {
            description.name = "...";
            return description;
        }
        switch (resolved.*) {
            .primitive => |primitive| {
                description.kind = .primitive_kind;
                description.name = @tagName(primitive);
                description.primitive = primitive;
            },
            .interface => |interface| {
                description.kind = .interface_kind;
                description.name = interface.name;
            },
            // a synthesised alias resolves straight to a structural list
            .structural => |fields| {
                description.kind = .struct_kind;
                if (candidate.* == .declared) description.name = candidate.declared.name;
                for (fields) |field| {
                    try description.members.append(self.arena, .{
                        .name = field.name,
                        .description = try self.describeType(field.field_type, depth + 1),
                    });
                }
            },
            .declared, .inline_enum, .structural_enum => {
                if (try self.structBody(resolved)) |body| {
                    description.kind = .struct_kind;
                    description.name = resolved.declared.name;
                    const member_source = self.views[body.view_index].source;
                    for (body.members) |member| {
                        const member_type = try self.typeFromExpressionIn(member.member_type, body.environment, body.view_index);
                        try description.members.append(self.arena, .{
                            .name = member.name.slice(member_source),
                            .description = try self.describeType(member_type, depth + 1),
                        });
                    }
                } else if (try self.enumBody(resolved)) |body| {
                    description.kind = .enum_kind;
                    // a synthesised enum reflects under its alias name
                    description.name = if (candidate.* == .declared) candidate.declared.name else body.name;
                    for (body.variants) |variant| {
                        const payload: *Interpreter.Value.TypeDescription = if (variant.payload) |payload_type|
                            try self.describeType(payload_type, depth + 1)
                        else
                            try self.voidDescription();
                        try description.members.append(self.arena, .{
                            .name = variant.name,
                            .description = payload,
                        });
                    }
                } else {
                    description.name = try resolved.render(self.arena);
                }
                if (resolved.* == .declared) {
                    const type_def = resolved.declared.definition.kind.type_def;
                    const marker_source = self.views[resolved.declared.view_index].source;
                    var markers: std.ArrayList([]const u8) = .empty;
                    for (type_def.interfaces) |marker| {
                        try markers.append(self.arena, marker.slice(marker_source));
                    }
                    description.interface_names = try markers.toOwnedSlice(self.arena);
                }
            },
            else => {
                description.name = try resolved.render(self.arena);
            },
        }
        return description;
    }

    fn isVoidDescription(description: *Interpreter.Value.TypeDescription) bool {
        return description.kind == .other_kind and description.origin == null and std.mem.eql(u8, description.name, "void");
    }

    fn voidDescription(self: *Checker) Error!*Interpreter.Value.TypeDescription {
        const description = try self.arena.create(Interpreter.Value.TypeDescription);
        description.* = .{
            .kind = .other_kind,
            .name = "void",
            .primitive = null,
            .members = .empty,
            .interface_names = &.{},
            .origin = null,
        };
        return description;
    }

    // converts a '#Type' result back into a checker type for
    // 'type T = #...' synthesis (section 3.4)
    fn descriptionToType(self: *Checker, description: *Interpreter.Value.TypeDescription, span: Token.Location) Error!*const Type {
        if (description.origin) |origin| return origin;
        switch (description.kind) {
            .primitive_kind => {
                const primitive = description.primitive orelse return &unknown_type;
                return self.makeType(.{ .primitive = primitive });
            },
            .struct_kind => {
                var fields: std.ArrayList(Type.Field) = .empty;
                for (description.members.items) |member| {
                    try fields.append(self.arena, .{
                        .name = member.name,
                        .field_type = try self.descriptionToType(member.description, span),
                    });
                }
                return self.makeType(.{ .structural = try fields.toOwnedSlice(self.arena) });
            },
            .enum_kind => {
                var variants: std.ArrayList(Type.EnumVariant) = .empty;
                for (description.members.items) |member| {
                    // a void member description marks a payload-less variant,
                    // mirroring how reflection encodes them (section 3.4)
                    const payload: ?*const Type = if (isVoidDescription(member.description))
                        null
                    else
                        try self.descriptionToType(member.description, span);
                    try variants.append(self.arena, .{ .name = member.name, .payload = payload });
                }
                return self.makeType(.{ .structural_enum = try variants.toOwnedSlice(self.arena) });
            },
            .interface_kind, .other_kind => {
                try self.report(span, "this '#Type' does not describe a usable runtime type (section 3.4)", .{});
                return &unknown_type;
            },
        }
    }

    fn recordCallTarget(self: *Checker, target_key: ?*const ast.Expression, symbol: resolution.Symbol, type_bindings: []const Type.Binding) Error!void {
        const key = target_key orelse return;
        try self.call_targets.put(self.arena, key, symbol);
        if (type_bindings.len != 0) {
            try self.call_type_bindings.put(self.arena, key, type_bindings);
        }
    }

    // the reporting pass for contextual arguments, run once the overload
    // outcome is known; without a winner they are checked context-free
    fn recheckContextualArguments(self: *Checker, call: anytype, expectations: []const *const Type) Error!void {
        for (call.arguments, 0..) |argument, index| {
            if (!try self.contextualArgument(argument)) continue;
            const expected: ?*const Type = if (index < expectations.len) expectations[index] else null;
            _ = try self.checkExpression(argument, expected);
        }
    }

    fn callOverloads(self: *Checker, name: []const u8, symbols: resolution.SymbolList, call: anytype, target_key: ?*const ast.Expression, span: Token.Location) Error!*const Type {
        const Candidate = struct {
            symbol: resolution.Symbol,
            return_type: *const Type,
            exact: bool,
            argument_expectations: []const *const Type,
            type_bindings: []const Type.Binding,
        };
        var viable: std.ArrayList(Candidate) = .empty;
        var function_count: usize = 0;

        // argument types are computed once, without parameter context;
        // untyped literals participate in unification via their defaults;
        // contextual arguments are typed per candidate instead
        var argument_types: std.ArrayList(*const Type) = .empty;
        for (call.arguments) |argument| {
            const argument_type: *const Type = if (try self.contextualArgument(argument))
                &unknown_type
            else
                try self.checkExpression(argument, null);
            try argument_types.append(self.arena, argument_type);
        }

        // surplus explicit type arguments are an error (section 3.7): no
        // candidate can bind more type arguments than it declares parameters
        if (call.type_arguments.len != 0) {
            var max_type_parameters: usize = 0;
            var saw_function = false;
            for (symbols.items) |symbol| {
                if (symbol.definition.kind != .fn_def) continue;
                saw_function = true;
                max_type_parameters = @max(max_type_parameters, symbol.definition.kind.fn_def.type_parameters.len);
            }
            if (saw_function and call.type_arguments.len > max_type_parameters) {
                try self.report(span, "too many type arguments for '{s}': {d} supplied but at most {d} type parameter(s) declared (section 3.7)", .{ name, call.type_arguments.len, max_type_parameters });
                return &unknown_type;
            }
        }

        for (symbols.items) |symbol| {
            switch (symbol.definition.kind) {
                .macro_def => |macro_def| {
                    // macro calls evaluate at compile time (section 6.3)
                    if (self.comptime_depth == 0) {
                        try self.report(span, "a macro call must be invoked with '#' (section 6.3)", .{});
                    }
                    if (call.arguments.len != macro_def.parameters.len) {
                        try self.report(span, "macro '{s}' expects {d} argument(s), found {d}", .{ name, macro_def.parameters.len, call.arguments.len });
                    } else {
                        // macro parameters are strictly typed (section 6.3)
                        for (macro_def.parameters, argument_types.items, 0..) |parameter, argument_type, index| {
                            const parameter_type = try self.typeFromExpressionIn(parameter.parameter_type, &empty_type_environment, symbol.view_index);
                            if (!try self.coerce(argument_type, parameter_type)) {
                                try self.typeMismatch(self.expressionSpan(call.arguments[index]), argument_type, parameter_type);
                            }
                        }
                    }
                    return &unknown_type;
                },
                .fn_def, .extern_def => function_count += 1,
                else => {
                    try self.report(span, "'{s}' is not callable", .{name});
                    return &unknown_type;
                },
            }
            if (try self.tryCandidate(symbol, call, argument_types.items, null)) |result| {
                try viable.append(self.arena, .{
                    .symbol = symbol,
                    .return_type = result.return_type,
                    .exact = result.exact,
                    .argument_expectations = result.argument_expectations,
                    .type_bindings = result.type_bindings,
                });
            }
        }

        // overload resolution (section 3.6)
        if (viable.items.len == 1) {
            try self.recordCallTarget(target_key, viable.items[0].symbol, viable.items[0].type_bindings);
            try self.recheckContextualArguments(call, viable.items[0].argument_expectations);
            return viable.items[0].return_type;
        }
        if (viable.items.len > 1) {
            var exact: ?Candidate = null;
            var exact_count: usize = 0;
            for (viable.items) |candidate| {
                if (candidate.exact) {
                    exact = candidate;
                    exact_count += 1;
                }
            }
            if (exact_count == 1) {
                try self.recordCallTarget(target_key, exact.?.symbol, exact.?.type_bindings);
                try self.recheckContextualArguments(call, exact.?.argument_expectations);
                return exact.?.return_type;
            }
            try self.report(span, "the call to '{s}' is ambiguous: {d} overloads are viable (section 3.6)", .{ name, viable.items.len });
            return &unknown_type;
        }
        try self.report(span, "no overload of '{s}' matches these argument types ({d} candidate{s})", .{ name, function_count, if (function_count == 1) "" else "s" });
        try self.recheckContextualArguments(call, &.{});
        return &unknown_type;
    }

    const MethodReceiver = struct {
        // the receiver location's declared type, indirections intact
        raw: *const Type,
        // the receiver value after pointee transparency
        pierced: *const Type,
        // mutability of the pierced location (section 3.8)
        mutable: bool,
    };

    // overload resolution for dot-notation calls: the receiver acts as the
    // implicit first argument, and a type-specific extension is preferred
    // over an interface default implementation (sections 4.5, 5.2)
    fn callMethodOverloads(self: *Checker, name: []const u8, symbols: resolution.SymbolList, call: anytype, receiver: MethodReceiver, target_key: ?*const ast.Expression, span: Token.Location) Error!*const Type {
        const Candidate = struct {
            symbol: resolution.Symbol,
            return_type: *const Type,
            exact: bool,
            default_implementation: bool,
            argument_expectations: []const *const Type,
            type_bindings: []const Type.Binding,
        };
        var viable: std.ArrayList(Candidate) = .empty;
        var extension_count: usize = 0;

        var argument_types: std.ArrayList(*const Type) = .empty;
        for (call.arguments) |argument| {
            const argument_type: *const Type = if (try self.contextualArgument(argument))
                &unknown_type
            else
                try self.checkExpression(argument, null);
            try argument_types.append(self.arena, argument_type);
        }

        for (symbols.items) |symbol| {
            if (symbol.definition.kind != .fn_def) continue;
            const fn_def = symbol.definition.kind.fn_def;
            if (fn_def.function.parameters.len == 0 or !fn_def.function.parameters[0].is_self) continue;
            extension_count += 1;
            if (try self.tryCandidate(symbol, call, argument_types.items, receiver)) |result| {
                try viable.append(self.arena, .{
                    .symbol = symbol,
                    .return_type = result.return_type,
                    .exact = result.exact,
                    .default_implementation = result.default_implementation,
                    .argument_expectations = result.argument_expectations,
                    .type_bindings = result.type_bindings,
                });
            }
        }

        if (extension_count == 0) {
            const rendered = try receiver.pierced.render(self.arena);
            try self.report(span, "no extension function '{s}' for {s} (section 4.5)", .{ name, rendered });
            return &unknown_type;
        }

        var has_specific = false;
        for (viable.items) |candidate| {
            if (!candidate.default_implementation) has_specific = true;
        }
        var remaining: std.ArrayList(Candidate) = .empty;
        for (viable.items) |candidate| {
            if (has_specific and candidate.default_implementation) continue;
            try remaining.append(self.arena, candidate);
        }

        if (remaining.items.len == 1) {
            try self.recordCallTarget(target_key, remaining.items[0].symbol, remaining.items[0].type_bindings);
            try self.recheckContextualArguments(call, remaining.items[0].argument_expectations);
            return remaining.items[0].return_type;
        }
        if (remaining.items.len > 1) {
            var exact: ?Candidate = null;
            var exact_count: usize = 0;
            for (remaining.items) |candidate| {
                if (candidate.exact) {
                    exact = candidate;
                    exact_count += 1;
                }
            }
            if (exact_count == 1) {
                try self.recordCallTarget(target_key, exact.?.symbol, exact.?.type_bindings);
                try self.recheckContextualArguments(call, exact.?.argument_expectations);
                return exact.?.return_type;
            }
            try self.report(span, "the call to '{s}' is ambiguous: {d} overloads are viable (section 3.6)", .{ name, remaining.items.len });
            return &unknown_type;
        }
        const rendered = try receiver.pierced.render(self.arena);
        try self.report(span, "no overload of '{s}' matches {s} and these argument types ({d} candidate{s})", .{ name, rendered, extension_count, if (extension_count == 1) "" else "s" });
        try self.recheckContextualArguments(call, &.{});
        return &unknown_type;
    }

    // a call through an interface object resolves at runtime through the
    // vtable; statically it checks against the declared signature (5.2)
    fn callInterfaceFunction(self: *Checker, interface: Type.Interface, name_token: Token, call: anytype) Error!*const Type {
        const interface_def = interface.definition.kind.interface_def;
        const interface_source = self.views[interface.view_index].source;
        const name = name_token.slice(self.source());
        const function = for (interface_def.functions) |candidate| {
            if (std.mem.eql(u8, candidate.name.slice(interface_source), name)) break candidate;
        } else {
            try self.report(name_token.location, "interface '{s}' declares no function '{s}' (section 5.2)", .{ interface.name, name });
            for (call.arguments) |argument| _ = try self.checkExpression(argument, null);
            return &unknown_type;
        };
        if (call.arguments.len != function.parameters.len) {
            try self.report(name_token.location, "'{s}' expects {d} argument(s), found {d}", .{ name, function.parameters.len, call.arguments.len });
        }
        const checked = @min(call.arguments.len, function.parameters.len);
        for (call.arguments[0..checked], function.parameters[0..checked]) |argument, parameter| {
            const parameter_type = try self.typeFromExpressionIn(parameter.parameter_type, &empty_type_environment, interface.view_index);
            const argument_type = try self.checkExpression(argument, parameter_type);
            try self.expectAssignable(argument_type, parameter_type, argument, self.expressionSpan(argument));
        }
        for (call.arguments[checked..]) |argument| _ = try self.checkExpression(argument, null);
        return if (function.return_type) |return_expression|
            try self.typeFromExpressionIn(return_expression, &empty_type_environment, interface.view_index)
        else
            &void_type;
    }

    // how the receiver expression enters overload resolution as the implicit
    // first argument (section 4.5): a reference receiver auto-references the
    // place, a pointer receiver passes the binding's own pointer, and a
    // value receiver passes a copy
    fn methodReceiverArgument(self: *Checker, self_type: *const Type, receiver: MethodReceiver) Error!*const Type {
        return switch (self_type.*) {
            .reference => self.makeType(.{ .reference = .{ .mutable = receiver.mutable, .child = receiver.pierced } }),
            .pointer => if (receiver.raw.* == .pointer) receiver.raw else receiver.pierced,
            else => receiver.pierced,
        };
    }

    const CandidateResult = struct {
        return_type: *const Type,
        exact: bool,
        default_implementation: bool = false,
        // substituted parameter types per argument index, for the final
        // reporting pass over contextual arguments; empty when none occur
        argument_expectations: []const *const Type = &.{},
        // resolved type arguments, recorded at the winning call site so
        // later stages can resolve type parameters (section 3.7)
        type_bindings: []const Type.Binding = &.{},
    };

    // checks one overload candidate: arity, generic inference by unification
    // (section 3.7), then per-argument compatibility (section 3.3); a
    // non-null receiver consumes the leading 'self' parameter (section 4.5)
    fn tryCandidate(self: *Checker, symbol: resolution.Symbol, call: anytype, argument_types: []const *const Type, receiver: ?MethodReceiver) Error!?CandidateResult {
        var parameters: []const ast.Parameter = undefined;
        var variadic = false;
        var type_parameters: []const ast.TypeParameter = &.{};
        var return_expression: ?*const ast.TypeExpression = null;
        switch (symbol.definition.kind) {
            .fn_def => |fn_def| {
                parameters = fn_def.function.parameters;
                type_parameters = fn_def.type_parameters;
                return_expression = fn_def.function.return_type;
            },
            .extern_def => |extern_def| {
                parameters = extern_def.parameters;
                variadic = extern_def.variadic;
                return_expression = extern_def.return_type;
            },
            else => return null,
        }

        const has_self = parameters.len != 0 and parameters[0].is_self;
        if (receiver != null) {
            if (!has_self) return null;
            if (argument_types.len != parameters.len - 1) return null;
        } else {
            // extension functions are called via dot notation only (4.5)
            if (has_self) return null;
            if (variadic) {
                if (argument_types.len < parameters.len) return null;
            } else if (argument_types.len != parameters.len) {
                return null;
            }
        }

        const definition_source = self.views[symbol.view_index].source;
        const environment = try self.arena.create(TypeEnvironment);
        environment.* = .empty;
        // explicit type arguments bind left-to-right (section 3.7); more
        // arguments than the function has type parameters makes it non-viable
        if (call.type_arguments.len > type_parameters.len) return null;
        for (call.type_arguments, 0..) |explicit, index| {
            const bound = try self.typeFromExpression(explicit, self.scope_types);
            try environment.put(self.arena, type_parameters[index].name.slice(definition_source), bound);
        }
        // unbound parameters appear as type_parameter placeholders
        var placeholders: TypeEnvironment = .empty;
        for (type_parameters) |type_parameter| {
            const parameter_name = type_parameter.name.slice(definition_source);
            if (environment.get(parameter_name) != null) {
                try placeholders.put(self.arena, parameter_name, environment.get(parameter_name).?);
                continue;
            }
            const constraint = self.interfaceOfConstraint(type_parameter.constraint, definition_source);
            try placeholders.put(self.arena, parameter_name, try self.makeType(.{ .type_parameter = .{ .name = parameter_name, .constraint = constraint } }));
        }

        var exact = true;
        var default_implementation = false;
        const offset: usize = if (receiver != null) 1 else 0;
        if (receiver) |method_receiver| {
            const self_type = try self.typeFromExpressionIn(parameters[0].parameter_type, &placeholders, symbol.view_index);
            const self_child = try self.pierce(self_type);
            if (self_child.* == .interface) default_implementation = true;
            const receiver_argument = try self.methodReceiverArgument(self_type, method_receiver);
            if (!try self.unifyParameter(self_type, receiver_argument, environment)) return null;
            const substituted = try self.substitute(self_type, environment);
            if (!receiver_argument.eql(substituted)) {
                exact = false;
                if (!try self.coerce(receiver_argument, substituted)) return null;
            }
        }
        // contextual arguments resolve against this candidate's parameter
        // types instead of their precomputed placeholder (sections 3.2, 3.7)
        const arguments: []const *const ast.Expression = if (@hasField(@TypeOf(call), "arguments")) call.arguments else &.{};
        var has_contextual = false;
        for (parameters[offset..], 0..) |parameter, index| {
            if (index >= argument_types.len) break;
            const parameter_type = try self.typeFromExpressionIn(parameter.parameter_type, &placeholders, symbol.view_index);
            var argument_type = argument_types[index];
            if (index < arguments.len and try self.contextualArgument(arguments[index])) {
                has_contextual = true;
                const expected_here = try self.substitute(parameter_type, environment);
                argument_type = (try self.quietExpressionType(arguments[index], expected_here)) orelse return null;
            }
            if (!try self.unifyParameter(parameter_type, argument_type, environment)) return null;
            const substituted = try self.substitute(parameter_type, environment);
            if (!argument_type.eql(substituted)) {
                exact = false;
                if (!try self.coerce(argument_type, substituted)) return null;
            }
        }

        // every type parameter must be bound after unification, and the
        // bound type must satisfy the declared constraint (section 3.7)
        var type_bindings: []Type.Binding = &.{};
        if (type_parameters.len != 0) {
            type_bindings = try self.arena.alloc(Type.Binding, type_parameters.len);
        }
        for (type_parameters, 0..) |type_parameter, index| {
            const parameter_name = type_parameter.name.slice(definition_source);
            const bound = environment.get(parameter_name) orelse return null;
            if (self.interfaceOfConstraint(type_parameter.constraint, definition_source)) |constraint| {
                if (!try self.implements(bound, constraint)) return null;
            }
            type_bindings[index] = .{ .name = parameter_name, .bound = bound };
        }

        const return_type: *const Type = if (return_expression) |expression|
            try self.substitute(try self.typeFromExpressionIn(expression, &placeholders, symbol.view_index), environment)
        else
            &void_type;

        var argument_expectations: []const *const Type = &.{};
        if (has_contextual) {
            var expectations: std.ArrayList(*const Type) = .empty;
            for (parameters[offset..]) |parameter| {
                const parameter_type = try self.typeFromExpressionIn(parameter.parameter_type, &placeholders, symbol.view_index);
                try expectations.append(self.arena, try self.substitute(parameter_type, environment));
            }
            argument_expectations = try expectations.toOwnedSlice(self.arena);
        }
        return .{
            .return_type = return_type,
            .exact = exact,
            .default_implementation = default_implementation,
            .argument_expectations = argument_expectations,
            .type_bindings = type_bindings,
        };
    }

    // structural unification of a parameter type against an argument type,
    // binding type parameters as they are encountered (section 3.7)
    fn unifyParameter(self: *Checker, parameter: *const Type, argument: *const Type, bindings: *TypeEnvironment) Error!bool {
        switch (parameter.*) {
            .type_parameter => |type_parameter| {
                if (bindings.get(type_parameter.name)) |bound| {
                    return try self.coerce(argument, bound) or bound.eql(argument);
                }
                // an untyped literal binds as its default (section 3.7)
                try bindings.put(self.arena, type_parameter.name, try self.defaulted(argument));
                return true;
            },
            .pointer => |indirection| {
                if (argument.* != .pointer) return argument.* == .unknown;
                return self.unifyParameter(indirection.child, argument.pointer.child, bindings);
            },
            .reference => |indirection| {
                if (argument.* != .reference) return argument.* == .unknown;
                return self.unifyParameter(indirection.child, argument.reference.child, bindings);
            },
            .heap_array => |indirection| {
                if (argument.* != .heap_array) return argument.* == .unknown;
                return self.unifyParameter(indirection.child, argument.heap_array.child, bindings);
            },
            .slice => |slice| {
                if (argument.* != .slice) return argument.* == .unknown;
                return self.unifyParameter(slice.child, argument.slice.child, bindings);
            },
            .fixed_array => |array| {
                if (argument.* != .fixed_array) return argument.* == .unknown;
                return self.unifyParameter(array.element, argument.fixed_array.element, bindings);
            },
            .function => |function| {
                if (argument.* != .function) return argument.* == .unknown;
                if (function.parameter_types.len != argument.function.parameter_types.len) return false;
                for (function.parameter_types, argument.function.parameter_types) |left, right| {
                    if (!try self.unifyParameter(left, right, bindings)) return false;
                }
                return self.unifyParameter(function.return_type, argument.function.return_type, bindings);
            },
            .declared => |declared| {
                if (argument.* != .declared) {
                    if (argument.* == .unknown) return true;
                    return self.coerce(argument, parameter);
                }
                if (declared.definition != argument.declared.definition) return false;
                for (declared.arguments, argument.declared.arguments) |left, right| {
                    if (!try self.unifyParameter(left, right, bindings)) return false;
                }
                return true;
            },
            else => return try self.coerce(argument, parameter),
        }
    }

    pub fn substitute(self: *Checker, candidate: *const Type, bindings: *const TypeEnvironment) Error!*const Type {
        switch (candidate.*) {
            .type_parameter => |type_parameter| {
                return bindings.get(type_parameter.name) orelse candidate;
            },
            .pointer => |indirection| return self.makeType(.{ .pointer = .{ .mutable = indirection.mutable, .child = try self.substitute(indirection.child, bindings) } }),
            .reference => |indirection| return self.makeType(.{ .reference = .{ .mutable = indirection.mutable, .child = try self.substitute(indirection.child, bindings) } }),
            .heap_array => |indirection| return self.makeType(.{ .heap_array = .{ .mutable = indirection.mutable, .child = try self.substitute(indirection.child, bindings) } }),
            .slice => |slice| return self.makeType(.{ .slice = .{ .mutable = slice.mutable, .child = try self.substitute(slice.child, bindings) } }),
            .fixed_array => |array| return self.makeType(.{ .fixed_array = .{ .element = try self.substitute(array.element, bindings), .length = array.length } }),
            .function => |function| {
                var parameter_types: std.ArrayList(*const Type) = .empty;
                for (function.parameter_types) |parameter_type| {
                    try parameter_types.append(self.arena, try self.substitute(parameter_type, bindings));
                }
                return self.makeType(.{ .function = .{
                    .parameter_types = try parameter_types.toOwnedSlice(self.arena),
                    .return_type = try self.substitute(function.return_type, bindings),
                } });
            },
            .declared => |declared| {
                var arguments: std.ArrayList(*const Type) = .empty;
                for (declared.arguments) |argument| {
                    try arguments.append(self.arena, try self.substitute(argument, bindings));
                }
                return self.makeType(.{ .declared = .{
                    .definition = declared.definition,
                    .view_index = declared.view_index,
                    .name = declared.name,
                    .arguments = try arguments.toOwnedSlice(self.arena),
                } });
            },
            else => return candidate,
        }
    }

    const Lvalue = struct {
        // the location's declared type, indirections intact
        raw: *const Type,
        // what a read of the location yields (pointee transparency)
        pierced: *const Type,
        mutable: bool,
    };

    // analyzes an expression as an assignable location (sections 3.8, 4.2)
    fn lvalueOf(self: *Checker, expression: *const ast.Expression) Error!?Lvalue {
        switch (expression.*) {
            .grouped => |inner| return self.lvalueOf(inner),
            .path => |path| {
                if (path.len != 1) return null;
                const binding = self.lookup(path[0].slice(self.source())) orelse return null;
                // the binding's own slot: its mutability is the binding's
                return .{
                    .raw = binding.binding_type,
                    .pierced = try self.pierce(binding.binding_type),
                    .mutable = binding.mutable,
                };
            },
            .member => |member| {
                const base = try self.lvalueOf(member.object) orelse return null;
                const field_type = try self.memberType(base.pierced, member.name);
                return .{
                    .raw = field_type,
                    .pierced = try self.pierce(field_type),
                    .mutable = piercedMutability(base),
                };
            },
            .index => |index| {
                _ = try self.checkExpression(index.subscript, null);
                const base = try self.lvalueOf(index.object) orelse return null;
                const container = try self.resolveAlias(base.pierced);
                const element: *const Type = switch (container.*) {
                    .slice => |slice| slice.child,
                    .heap_array => |heap| heap.child,
                    .fixed_array => |array| array.element,
                    else => return null,
                };
                const mutable = switch (container.*) {
                    .slice => |slice| slice.mutable,
                    .heap_array => |heap| heap.mutable,
                    .fixed_array => piercedMutability(base),
                    else => false,
                };
                return .{ .raw = element, .pierced = try self.pierce(element), .mutable = mutable };
            },
            else => return null,
        }
    }

    // mutability when reaching through a location: behind '&var'/'*var' the
    // pointee is mutable regardless of the binding; behind '&'/'*' it is
    // immutable; direct access inherits the binding (section 3.8)
    fn piercedMutability(place: Lvalue) bool {
        return switch (place.raw.*) {
            .pointer => |indirection| indirection.mutable,
            .reference => |indirection| indirection.mutable,
            else => place.mutable,
        };
    }

    // pointee transparency (section 4.2): a pointer or reference reads as
    // its pointee; slices, arrays, and heap arrays read as themselves
    pub fn pierce(self: *Checker, candidate: *const Type) Error!*const Type {
        _ = self;
        var current = candidate;
        var depth: usize = 0;
        while (depth < 16) : (depth += 1) {
            switch (current.*) {
                .pointer => |indirection| current = indirection.child,
                .reference => |indirection| current = indirection.child,
                else => return current,
            }
        }
        return current;
    }

    fn expectAssignable(self: *Checker, from: *const Type, to: *const Type, value: *const ast.Expression, span: Token.Location) Error!void {
        if (try self.coerce(from, to)) return;
        const from_rendered = try from.render(self.arena);
        const to_rendered = try to.render(self.arena);
        // a pointer-typed target with a pierced read on the right means the
        // user wrote 'p' where pointer transfer needs 'move p' (section 4.2)
        const target_owns = to.* == .pointer or to.* == .heap_array;
        const value_place = try self.lvalueOf(value);
        if (target_owns and value_place != null and try self.coerce(value_place.?.raw, to)) {
            try self.report(span, "assigning to {s} transfers ownership: use 'move' to hand over the allocation or 'new' to copy it (section 4.2)", .{to_rendered});
            return;
        }
        if (to.* == .reference and value_place != null) {
            try self.report(span, "expected {s}; take a reference explicitly with '&' (section 4.2)", .{to_rendered});
            return;
        }
        try self.report(span, "expected {s}, found {s}", .{ to_rendered, from_rendered });
    }

    fn typeMismatch(self: *Checker, span: Token.Location, from: *const Type, to: *const Type) Error!void {
        const from_rendered = try from.render(self.arena);
        const to_rendered = try to.render(self.arena);
        try self.report(span, "expected {s}, found {s}", .{ to_rendered, from_rendered });
    }

    fn operandMismatch(self: *Checker, operator: Token, left: *const Type, right: *const Type) Error!void {
        try self.operandMismatchAt(operator.location, left, right);
    }

    fn operandMismatchAt(self: *Checker, span: Token.Location, left: *const Type, right: *const Type) Error!void {
        const left_rendered = try left.render(self.arena);
        const right_rendered = try right.render(self.arena);
        try self.report(span, "operand types {s} and {s} are incompatible", .{ left_rendered, right_rendered });
    }

    fn pushFrame(self: *Checker, barrier: bool) Error!void {
        try self.scopes.append(self.arena, .{ .bindings = .empty, .barrier = barrier });
    }

    fn popFrame(self: *Checker) void {
        _ = self.scopes.pop();
    }

    fn bind(self: *Checker, name_token: Token, binding_type: *const Type, mutable: bool) Error!void {
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        try frame.bindings.append(self.arena, .{
            .name = name_token.slice(self.source()),
            .binding_type = binding_type,
            .mutable = mutable,
        });
    }

    fn lookup(self: *const Checker, name: []const u8) ?Binding {
        var frame_index = self.scopes.items.len;
        while (frame_index > 0) {
            frame_index -= 1;
            const frame = self.scopes.items[frame_index];
            var binding_index = frame.bindings.items.len;
            while (binding_index > 0) {
                binding_index -= 1;
                if (std.mem.eql(u8, frame.bindings.items[binding_index].name, name)) {
                    return frame.bindings.items[binding_index];
                }
            }
            if (frame.barrier) return null;
        }
        return null;
    }

    fn makeType(self: *Checker, value: Type) Error!*const Type {
        const node = try self.arena.create(Type);
        node.* = value;
        return node;
    }

    fn source(self: *const Checker) []const u8 {
        return self.views[self.current_view].source;
    }

    fn report(self: *Checker, span: Token.Location, comptime format: []const u8, arguments: anytype) Error!void {
        const view = self.views[self.current_view];
        try self.diagnostics.append(self.diagnostics_allocator, .{
            .path = view.path,
            .source = view.source,
            .span = span,
            .message = try std.fmt.allocPrint(self.arena, format, arguments),
        });
    }

    // best-effort source span of an expression for diagnostics
    pub fn expressionSpan(self: *const Checker, expression: *const ast.Expression) Token.Location {
        return switch (expression.*) {
            .integer_literal, .float_literal, .string_literal, .character_literal => |token| token.location,
            .bool_literal => |literal| literal.token.location,
            .path => |path| .{ .start = path[0].location.start, .end = path[path.len - 1].location.end },
            .implied_variant => |token| token.location,
            .binary => |binary| binary.operator.location,
            .unary => |unary| unary.operator.location,
            .cast => |cast| cast.operator.location,
            .call => |call| self.expressionSpan(call.callee),
            .member => |member| member.name.location,
            .index => |index| self.expressionSpan(index.object),
            .struct_init => |struct_init| if (struct_init.name) |name| name.location else if (struct_init.members.len != 0) struct_init.members[0].name.location else .{ .start = 0, .end = 0 },
            .array_literal => |elements| if (elements.len != 0) self.expressionSpan(elements[0]) else .{ .start = 0, .end = 0 },
            .array_fill => |array_fill| self.expressionSpan(array_fill.value),
            .array_range => |array_range| array_range.operator.location,
            .if_expr => |if_expr| self.expressionSpan(if_expr.condition),
            .while_expr => |while_expr| self.expressionSpan(while_expr.condition),
            .for_expr => |for_expr| if (for_expr.subjects.len != 0) self.expressionSpan(for_expr.subjects[0]) else .{ .start = 0, .end = 0 },
            .match_expr => |match_expr| self.expressionSpan(match_expr.subject),
            .lambda => |lambda| if (lambda.captures.len != 0) lambda.captures[0].name.location else .{ .start = 0, .end = 0 },
            .grouped => |inner| self.expressionSpan(inner),
            .comptime_expr => |inner| self.expressionSpan(inner),
        };
    }

    fn statementSpan(self: *const Checker, statement: *const ast.Statement) Token.Location {
        return switch (statement.*) {
            .var_def => |var_def| var_def.name.location,
            .assign => |assign| assign.operator.location,
            .break_stmt => |break_stmt| break_stmt.keyword.location,
            .return_stmt => |return_stmt| return_stmt.keyword.location,
            .expression => |expression| self.expressionSpan(expression),
            .block => .{ .start = 0, .end = 0 },
        };
    }
};

const untyped_integer_type: Type = .untyped_integer;
const untyped_float_type: Type = .untyped_float;
const empty_type_environment: Checker.TypeEnvironment = .empty;

fn unwrapGrouped(expression: *const ast.Expression) *const ast.Expression {
    var current = expression;
    while (current.* == .grouped) current = current.grouped;
    return current;
}

fn primitiveByName(name: []const u8) ?types.Primitive {
    return std.meta.stringToEnum(types.Primitive, name);
}

fn builtinMacro(name: []const u8) bool {
    return std.mem.eql(u8, name, "type_of") or
        std.mem.eql(u8, name, "struct_type") or
        std.mem.eql(u8, name, "enum_type");
}

fn parseIntegerLiteral(text: []const u8) !u64 {
    if (text.len > 2 and text[0] == '0') {
        switch (text[1]) {
            'x' => return std.fmt.parseInt(u64, text[2..], 16),
            'b' => return std.fmt.parseInt(u64, text[2..], 2),
            'o' => return std.fmt.parseInt(u64, text[2..], 8),
            else => {},
        }
    }
    return std.fmt.parseInt(u64, text, 10);
}
