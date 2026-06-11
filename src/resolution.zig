//! Name resolution, the first whole-program stage after the merge point.
//! All module definitions merge into one global symbol table (section 5.4):
//! unqualified names see the entire merged unit, while module-qualified paths
//! (`std::vec::Vec`) additionally pass a pub/exp visibility check. The
//! resolver then walks every definition body and verifies that each name
//! refers to a declaration, reporting every failure as a diagnostic.

const std = @import("std");
const Token = @import("tokenizer.zig").Token;
const ast = @import("ast.zig");
const Diagnostic = @import("diagnostic.zig").Diagnostic;

/// One parsed module as seen by the whole-program stages.
pub const ModuleView = struct {
    // canonical import key ('std::option'), null for the entry module
    key: ?[]const u8,
    path: []const u8,
    source: []const u8,
    module: *const ast.Module,
};

pub const Symbol = struct {
    visibility: ast.Visibility,
    view_index: usize,
    definition: *const ast.Definition,
};

/// All symbols sharing one name. More than one entry only ever holds
/// functions, which may overload (section 3.6).
pub const SymbolList = std.ArrayList(Symbol);

/// The merged compilation unit: every definition of every module by name.
pub const MergedUnit = struct {
    globals: std.StringHashMapUnmanaged(SymbolList),
};

// 'void' is not a value type, but '#void' marks a payload-less enum
// variant in 'add_member' (section 3.4)
const primitive_types = std.StaticStringMap(void).initComptime(.{
    .{"u8"},  .{"u16"}, .{"u32"}, .{"u64"},
    .{"i8"},  .{"i16"}, .{"i32"}, .{"i64"},
    .{"f32"}, .{"f64"}, .{"bool"}, .{"void"},
});

// built-in macros (section 6.4), callable without any declaration
const builtin_macros = std.StaticStringMap(void).initComptime(.{
    .{"type_of"}, .{"struct_type"}, .{"enum_type"},
});

pub const Resolver = struct {
    // long-lived allocations (table, messages) come from the compilation arena
    arena: std.mem.Allocator,
    views: []const ModuleView,
    diagnostics: *std.ArrayList(Diagnostic),
    diagnostics_allocator: std.mem.Allocator,
    globals: std.StringHashMapUnmanaged(SymbolList),
    // canonical import key to view index, for module-qualified paths
    module_keys: std.StringHashMapUnmanaged(usize),
    // import aliases of the module currently being resolved
    aliases: std.StringHashMapUnmanaged([]const u8),
    scopes: std.ArrayList(Frame),
    current_view: usize,

    const Frame = struct {
        bindings: std.ArrayList(Binding),
        // a lambda boundary: lookups for locals stop here (section 4.4)
        barrier: bool,
    };

    const Binding = struct {
        name: []const u8,
    };

    pub const Error = error{OutOfMemory};

    pub fn init(
        arena: std.mem.Allocator,
        views: []const ModuleView,
        diagnostics: *std.ArrayList(Diagnostic),
        diagnostics_allocator: std.mem.Allocator,
    ) Resolver {
        return .{
            .arena = arena,
            .views = views,
            .diagnostics = diagnostics,
            .diagnostics_allocator = diagnostics_allocator,
            .globals = .empty,
            .module_keys = .empty,
            .aliases = .empty,
            .scopes = .empty,
            .current_view = 0,
        };
    }

    /// Runs the merge and the resolve passes. Failures land in the
    /// diagnostics list; the returned unit is valid either way.
    pub fn run(self: *Resolver) Error!MergedUnit {
        try self.collectGlobals();
        for (self.views, 0..) |view, view_index| {
            self.current_view = view_index;
            try self.resolveModule(view);
        }
        return .{ .globals = self.globals };
    }

    // merge pass: every definition of every module lands in one table
    fn collectGlobals(self: *Resolver) Error!void {
        for (self.views, 0..) |view, view_index| {
            if (view.key) |key| {
                try self.module_keys.put(self.arena, key, view_index);
            }
            for (view.module.definitions) |*definition| {
                const name_token = definitionName(definition);
                const name = name_token.slice(view.source);
                const symbol: Symbol = .{
                    .visibility = definition.visibility,
                    .view_index = view_index,
                    .definition = definition,
                };
                const existing = try self.globals.getOrPut(self.arena, name);
                if (!existing.found_existing) {
                    existing.value_ptr.* = .empty;
                    try existing.value_ptr.append(self.arena, symbol);
                    continue;
                }
                // only fn definitions may share a name (overloading, section
                // 3.6); identical-signature detection is the type checker's
                const overloads = definition.kind == .fn_def and
                    existing.value_ptr.items[0].definition.kind == .fn_def;
                if (!overloads) {
                    const previous_path = self.views[existing.value_ptr.items[0].view_index].path;
                    self.current_view = view_index;
                    try self.report(name_token.location, "redeclaration of '{s}'; previously declared in {s}", .{ name, previous_path });
                    continue;
                }
                try existing.value_ptr.append(self.arena, symbol);
            }
        }
    }

    fn resolveModule(self: *Resolver, view: ModuleView) Error!void {
        self.aliases.clearRetainingCapacity();
        for (view.module.imports) |import| {
            if (import.alias) |alias| {
                const key = try self.joinPath(import.path, view.source);
                try self.aliases.put(self.arena, alias.slice(view.source), key);
            }
        }
        for (view.module.definitions) |*definition| {
            try self.resolveDefinition(definition);
        }
    }

    fn resolveDefinition(self: *Resolver, definition: *const ast.Definition) Error!void {
        switch (definition.kind) {
            .type_def => |type_def| {
                try self.pushFrame(false);
                try self.bindTypeParameters(type_def.type_parameters);
                for (type_def.interfaces) |interface_name| {
                    try self.resolveInterfaceName(interface_name);
                }
                try self.resolveTypeExpression(type_def.base, false);
                self.popFrame();
            },
            .fn_def => |fn_def| {
                try self.pushFrame(false);
                try self.bindTypeParameters(fn_def.type_parameters);
                try self.resolveFunction(fn_def.function);
                self.popFrame();
            },
            .extern_def => |extern_def| {
                for (extern_def.parameters) |parameter| {
                    try self.resolveTypeExpression(parameter.parameter_type, false);
                }
                if (extern_def.return_type) |return_type| {
                    try self.resolveTypeExpression(return_type, false);
                }
            },
            .interface_def => |interface_def| {
                for (interface_def.functions) |interface_fn| {
                    for (interface_fn.parameters) |parameter| {
                        try self.resolveTypeExpression(parameter.parameter_type, false);
                    }
                    if (interface_fn.return_type) |return_type| {
                        try self.resolveTypeExpression(return_type, false);
                    }
                }
            },
            .macro_def => |macro_def| {
                try self.pushFrame(false);
                for (macro_def.parameters) |parameter| {
                    try self.resolveTypeExpression(parameter.parameter_type, false);
                    try self.bind(parameter.name);
                }
                try self.resolveStatement(macro_def.body);
                self.popFrame();
            },
        }
    }

    // shared by fn definitions and lambdas; the caller owns the outer frame
    fn resolveFunction(self: *Resolver, function: ast.Function) Error!void {
        for (function.parameters) |parameter| {
            try self.resolveTypeExpression(parameter.parameter_type, false);
            try self.bind(parameter.name);
        }
        if (function.return_type) |return_type| {
            try self.resolveTypeExpression(return_type, false);
        }
        try self.resolveStatement(function.body);
    }

    fn bindTypeParameters(self: *Resolver, type_parameters: []const ast.TypeParameter) Error!void {
        for (type_parameters) |type_parameter| {
            if (type_parameter.constraint) |constraint| {
                try self.resolveInterfaceName(constraint);
            }
            try self.bind(type_parameter.name);
        }
    }

    fn resolveInterfaceName(self: *Resolver, name_token: Token) Error!void {
        const name = name_token.slice(self.source());
        const symbols = self.globals.get(name) orelse {
            return self.report(name_token.location, "use of undeclared interface '{s}'", .{name});
        };
        if (symbols.items[0].definition.kind != .interface_def) {
            return self.report(name_token.location, "'{s}' is not an interface ({s})", .{ name, definitionKindName(symbols.items[0].definition) });
        }
    }

    fn resolveStatement(self: *Resolver, statement: *const ast.Statement) Error!void {
        switch (statement.*) {
            .block => |statements| {
                try self.pushFrame(false);
                for (statements) |child| {
                    try self.resolveStatement(child);
                }
                self.popFrame();
            },
            .var_def => |var_def| {
                // the value resolves first, so 'var x = x' refers outward
                try self.resolveExpression(var_def.value);
                if (var_def.declared_type) |declared_type| {
                    try self.resolveTypeExpression(declared_type, false);
                }
                try self.bind(var_def.name);
            },
            .assign => |assign| {
                try self.resolveExpression(assign.target);
                try self.resolveExpression(assign.value);
            },
            .expression => |expression| try self.resolveExpression(expression),
            .break_stmt => |break_stmt| {
                if (break_stmt.value) |value| try self.resolveExpression(value);
            },
            .return_stmt => |return_stmt| {
                if (return_stmt.value) |value| try self.resolveExpression(value);
            },
        }
    }

    fn resolveExpression(self: *Resolver, expression: *const ast.Expression) Error!void {
        switch (expression.*) {
            .integer_literal, .float_literal, .string_literal, .character_literal, .bool_literal => {},
            .path => |path| try self.resolvePath(path, .value),
            // the checker resolves the implied enum from context
            .implied_variant => {},
            .grouped => |inner| try self.resolveExpression(inner),
            .comptime_expr => |inner| try self.resolveExpression(inner),
            .unary => |unary| try self.resolveExpression(unary.operand),
            .binary => |binary| {
                try self.resolveExpression(binary.left);
                try self.resolveExpression(binary.right);
            },
            .cast => |cast| {
                try self.resolveExpression(cast.operand);
                // an 'is' target may name an enum variant (section 3.2)
                const allow_variant = cast.operator.tag == .keyword_is;
                try self.resolveTypeExpression(cast.target, allow_variant);
            },
            .call => |call| {
                try self.resolveExpression(call.callee);
                for (call.type_arguments) |type_argument| {
                    try self.resolveTypeExpression(type_argument, false);
                }
                for (call.arguments) |argument| {
                    try self.resolveExpression(argument);
                }
            },
            // member names need the object's type; the type checker owns them
            .member => |member| try self.resolveExpression(member.object),
            .index => |index| {
                try self.resolveExpression(index.object);
                try self.resolveExpression(index.subscript);
            },
            .struct_init => |struct_init| {
                if (struct_init.name) |name| {
                    try self.resolvePath(&.{name}, .type);
                }
                for (struct_init.members) |member| {
                    try self.resolveExpression(member.value);
                }
            },
            .array_literal => |elements| {
                for (elements) |element| try self.resolveExpression(element);
            },
            .array_fill => |array_fill| {
                try self.resolveExpression(array_fill.value);
                try self.resolveExpression(array_fill.count);
            },
            .array_range => |array_range| {
                if (array_range.start) |start| try self.resolveExpression(start);
                try self.resolveExpression(array_range.end);
            },
            .if_expr => |if_expr| {
                try self.resolveExpression(if_expr.condition);
                try self.pushFrame(false);
                if (if_expr.capture) |capture| try self.bindCapture(capture);
                try self.resolveStatement(if_expr.then_branch);
                self.popFrame();
                if (if_expr.else_branch) |else_branch| {
                    try self.resolveStatement(else_branch);
                }
            },
            .while_expr => |while_expr| {
                try self.resolveExpression(while_expr.condition);
                try self.resolveStatement(while_expr.body);
                if (while_expr.else_branch) |else_branch| {
                    try self.resolveStatement(else_branch);
                }
            },
            .for_expr => |for_expr| {
                for (for_expr.subjects) |subject| {
                    try self.resolveExpression(subject);
                }
                try self.pushFrame(false);
                for (for_expr.captures) |capture| try self.bindCapture(capture);
                try self.resolveStatement(for_expr.body);
                self.popFrame();
                if (for_expr.else_branch) |else_branch| {
                    try self.resolveStatement(else_branch);
                }
            },
            .match_expr => |match_expr| {
                try self.resolveExpression(match_expr.subject);
                for (match_expr.arms) |arm| {
                    if (arm.pattern) |pattern| {
                        try self.resolveExpression(pattern);
                    }
                    try self.pushFrame(false);
                    if (arm.capture) |capture| try self.bindCapture(capture);
                    try self.resolveStatement(arm.body);
                    self.popFrame();
                }
                if (match_expr.else_branch) |else_branch| {
                    try self.resolveStatement(else_branch);
                }
            },
            .lambda => |lambda| {
                // captures name variables of the enclosing scope (section 4.4)
                for (lambda.captures) |capture| {
                    try self.resolveCapturedVariable(capture.name);
                    if (capture.annotation) |annotation| {
                        try self.resolveTypeExpression(annotation, false);
                    }
                }
                try self.pushFrame(true);
                for (lambda.captures) |capture| {
                    try self.bind(capture.name);
                }
                try self.resolveFunction(lambda.function);
                self.popFrame();
            },
        }
    }

    fn bindCapture(self: *Resolver, capture: ast.Capture) Error!void {
        if (capture.annotation) |annotation| {
            try self.resolveTypeExpression(annotation, false);
        }
        try self.bind(capture.name);
    }

    // a lambda capture must name something visible where the lambda is written
    fn resolveCapturedVariable(self: *Resolver, name_token: Token) Error!void {
        const name = name_token.slice(self.source());
        if (self.lookupLocal(name, false) or self.globals.contains(name)) return;
        try self.report(name_token.location, "use of undeclared identifier '{s}'", .{name});
    }

    fn resolveTypeExpression(self: *Resolver, type_expression: *const ast.TypeExpression, allow_variant: bool) Error!void {
        switch (type_expression.*) {
            .modified => |modified| try self.resolveTypeExpression(modified.child, allow_variant),
            .named => |named| {
                // an implied '::Variant' target is resolved by the checker
                if (!named.implied) {
                    try self.resolvePath(named.path, if (allow_variant) .value else .type);
                }
                for (named.type_arguments) |type_argument| {
                    try self.resolveTypeExpression(type_argument, false);
                }
            },
            .struct_type => |members| {
                for (members) |member| {
                    try self.resolveTypeExpression(member.member_type, false);
                }
            },
            .enum_type => |members| {
                for (members) |member| {
                    if (member.payload) |payload| {
                        try self.resolveTypeExpression(payload, false);
                    }
                }
            },
            // the fixed length is an integer literal, nothing to resolve
            .array => |array| try self.resolveTypeExpression(array.element, false),
            .function => |function| {
                for (function.parameter_types) |parameter_type| {
                    try self.resolveTypeExpression(parameter_type, false);
                }
                if (function.return_type) |return_type| {
                    try self.resolveTypeExpression(return_type, false);
                }
            },
            .comptime_type => |expression| try self.resolveExpression(expression),
        }
    }

    const PathContext = enum { value, type };

    // resolves an identifier path: a local or global name, a module-qualified
    // name ('std::vec::Vec', alias-qualified too), or 'Type::Variant'
    fn resolvePath(self: *Resolver, path: []const Token, context: PathContext) Error!void {
        const span: Token.Location = .{
            .start = path[0].location.start,
            .end = path[path.len - 1].location.end,
        };
        const first = path[0].slice(self.source());

        if (path.len == 1) {
            if (self.lookupLocal(first, false)) return;
            if (self.globals.contains(first)) return;
            // primitives appear in type positions and as '#u32' reflection
            if (primitive_types.has(first)) return;
            if (context == .value and builtin_macros.has(first)) return;
            if (self.lookupLocal(first, true)) {
                // capture lists are value-only: a type name (a scope type
                // parameter) stays visible inside a lambda without capture
                if (context == .type) return;
                return self.report(span, "'{s}' is declared outside this lambda; add it to the capture list to use it", .{first});
            }
            return self.report(span, "use of undeclared identifier '{s}'", .{first});
        }

        // module-qualified: the longest leading segment run that names a
        // loaded module wins, then one definition name, then maybe a variant
        var prefix_length = path.len - 1;
        while (prefix_length >= 1) : (prefix_length -= 1) {
            const key = try self.joinPath(path[0..prefix_length], self.source());
            if (self.module_keys.get(key)) |view_index| {
                return self.resolveQualified(span, key, view_index, path[prefix_length..]);
            }
            if (prefix_length == 1) break;
        }

        // alias-qualified: 'vectors::Vec' after 'import std::vec as vectors'
        if (self.aliases.get(first)) |key| {
            if (self.module_keys.get(key)) |view_index| {
                return self.resolveQualified(span, key, view_index, path[1..]);
            }
        }

        // 'Type::Variant' on a local or global enum type
        if (path.len == 2) {
            if (self.lookupLocal(first, false)) return;
            if (self.globals.get(first)) |symbols| {
                return self.checkVariant(span, symbols.items[0], path[1]);
            }
        }

        const full = try self.joinPath(path, self.source());
        try self.report(span, "use of undeclared name '{s}'", .{full});
    }

    // resolves the remainder of a module-qualified path (section 5.4):
    // qualified access reaches only pub/exp definitions of that module
    fn resolveQualified(self: *Resolver, span: Token.Location, key: []const u8, view_index: usize, remainder: []const Token) Error!void {
        const definition_name = remainder[0].slice(self.source());
        const symbols = self.globals.get(definition_name) orelse SymbolList.empty;
        var found: ?Symbol = null;
        var found_visible = false;
        for (symbols.items) |symbol| {
            if (symbol.view_index != view_index) continue;
            found = symbol;
            if (symbol.visibility != .private) {
                found_visible = true;
                break;
            }
        }
        if (found == null) {
            return self.report(span, "module '{s}' has no definition '{s}'", .{ key, definition_name });
        }
        if (!found_visible) {
            return self.report(span, "'{s}' in module '{s}' is private; mark it 'pub' to allow qualified access", .{ definition_name, key });
        }
        if (remainder.len == 2) {
            return self.checkVariant(span, found.?, remainder[1]);
        }
        if (remainder.len > 2) {
            const full = try self.joinPath(remainder, self.source());
            return self.report(span, "'{s}' does not name a definition in module '{s}'", .{ full, key });
        }
    }

    // verifies 'Type::Variant' when the base is a directly declared enum;
    // alias chains and non-enum bases are left to the type checker
    fn checkVariant(self: *Resolver, span: Token.Location, symbol: Symbol, variant_token: Token) Error!void {
        const type_def = switch (symbol.definition.kind) {
            .type_def => |*type_def| type_def,
            else => return,
        };
        const members = switch (type_def.base.*) {
            .enum_type => |members| members,
            else => return,
        };
        const variant_name = variant_token.slice(self.source());
        const definition_source = self.views[symbol.view_index].source;
        for (members) |member| {
            if (std.mem.eql(u8, member.name.slice(definition_source), variant_name)) return;
        }
        const type_name = definitionName(symbol.definition).slice(definition_source);
        try self.report(span, "enum '{s}' has no variant '{s}'", .{ type_name, variant_name });
    }

    fn pushFrame(self: *Resolver, barrier: bool) Error!void {
        try self.scopes.append(self.arena, .{ .bindings = .empty, .barrier = barrier });
    }

    fn popFrame(self: *Resolver) void {
        _ = self.scopes.pop();
    }

    // binds a name in the innermost frame; same-frame duplicates are errors,
    // shadowing an outer frame is allowed
    fn bind(self: *Resolver, name_token: Token) Error!void {
        const name = name_token.slice(self.source());
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        for (frame.bindings.items) |binding| {
            if (std.mem.eql(u8, binding.name, name)) {
                return self.report(name_token.location, "'{s}' is already declared in this scope", .{name});
            }
        }
        try frame.bindings.append(self.arena, .{ .name = name });
    }

    // searches scope frames innermost-first; without 'past_barriers' the
    // search stops at a lambda boundary so uncaptured locals stay invisible
    fn lookupLocal(self: *const Resolver, name: []const u8, past_barriers: bool) bool {
        var frame_index = self.scopes.items.len;
        while (frame_index > 0) {
            frame_index -= 1;
            const frame = self.scopes.items[frame_index];
            for (frame.bindings.items) |binding| {
                if (std.mem.eql(u8, binding.name, name)) return true;
            }
            if (frame.barrier and !past_barriers) return false;
        }
        return false;
    }

    fn joinPath(self: *Resolver, path: []const Token, path_source: []const u8) Error![]const u8 {
        var joined: std.ArrayList(u8) = .empty;
        for (path, 0..) |segment, segment_index| {
            if (segment_index != 0) try joined.appendSlice(self.arena, "::");
            try joined.appendSlice(self.arena, segment.slice(path_source));
        }
        return joined.toOwnedSlice(self.arena);
    }

    fn source(self: *const Resolver) []const u8 {
        return self.views[self.current_view].source;
    }

    fn report(self: *Resolver, span: Token.Location, comptime format: []const u8, arguments: anytype) Error!void {
        const view = self.views[self.current_view];
        try self.diagnostics.append(self.diagnostics_allocator, .{
            .path = view.path,
            .source = view.source,
            .span = span,
            .message = try std.fmt.allocPrint(self.arena, format, arguments),
        });
    }
};

fn definitionName(definition: *const ast.Definition) Token {
    return switch (definition.kind) {
        .type_def => |def| def.name,
        .fn_def => |def| def.name,
        .extern_def => |def| def.name,
        .interface_def => |def| def.name,
        .macro_def => |def| def.name,
    };
}

fn definitionKindName(definition: *const ast.Definition) []const u8 {
    return switch (definition.kind) {
        .type_def => "a type",
        .fn_def => "a function",
        .extern_def => "an extern function",
        .interface_def => "an interface",
        .macro_def => "a macro",
    };
}
