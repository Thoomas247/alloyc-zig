//! Name resolution, the first whole-program stage after the merge point.
//! All module definitions merge into one global symbol table (section 6.4):
//! unqualified names see their own library plus every 'exp' symbol of other
//! libraries in the unit, while module-qualified paths (`std::vec::Vec`)
//! additionally pass a pub/exp visibility check. The
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
    // the package the module arrived from, null in the executable's unit
    library: ?[]const u8 = null,
    module: *const ast.Module,
};

pub const Symbol = struct {
    visibility: ast.Visibility,
    view_index: usize,
    definition: *const ast.Definition,
};

/// All symbols sharing one name. More than one entry only ever holds
/// functions, which may overload (section 4.6).
pub const SymbolList = std.ArrayList(Symbol);

/// Import alias to canonical module key, one map per module view.
pub const AliasMap = std.StringHashMapUnmanaged([]const u8);

/// Library names whose 'exp' symbols join a module's unqualified namespace,
/// one map per module view; the value is the import token for diagnostics.
pub const InjectedMap = std.StringHashMapUnmanaged(Token);

/// One resolved use of a global definition, recorded for tooling: the
/// language server answers find-references and rename from this table.
pub const NameReference = struct {
    view_index: usize,
    span: Token.Location,
    definition: *const ast.Definition,
};

/// What a local binding declares; captures count as variables.
pub const LocalKind = enum { variable, parameter, type_parameter };

/// One declaration or use of a local binding (parameter, variable,
/// capture, or type parameter), identified by a per-run binding id.
pub const LocalReference = struct {
    view_index: usize,
    span: Token.Location,
    binding_id: usize,
    kind: LocalKind,
};

/// The merged compilation unit: every definition of every module by name,
/// plus the maps later stages need to re-derive what a qualified path pins.
pub const MergedUnit = struct {
    globals: std.StringHashMapUnmanaged(SymbolList),
    // canonical import key ('std::vec', 'pkg::mathx') to view index
    module_keys: std.StringHashMapUnmanaged(usize),
    aliases: []const AliasMap,
    injected: []const InjectedMap,
    references: []const NameReference,
    locals: []const LocalReference,
    // qualified functions ('fn Vector::empty'), living in their type's
    // namespace keyed by the type's definition identity (section 6.4)
    associated: []const AssociatedFunction,
};

pub const AssociatedFunction = struct {
    type_definition: *const ast.Definition,
    name: []const u8,
    symbol: Symbol,
};

/// Whether two modules belong to the same library; null marks the
/// executable's own compilation unit (section 6.4).
pub fn sameLibrary(left: ?[]const u8, right: ?[]const u8) bool {
    if (left == null and right == null) return true;
    if (left == null or right == null) return false;
    return std.mem.eql(u8, left.?, right.?);
}

// 'void' is not a value type, but '#void' marks a payload-less enum
// variant in 'add_member' (section 4.4)
const primitive_types = std.StaticStringMap(void).initComptime(.{
    .{"u8"},  .{"u16"}, .{"u32"}, .{"u64"},
    .{"i8"},  .{"i16"}, .{"i32"}, .{"i64"},
    .{"f32"}, .{"f64"}, .{"bool"}, .{"void"},
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
    // import aliases per module view, kept for the later stages
    alias_maps: []AliasMap,
    // libraries injected into each view's unqualified namespace: imports
    // without an explicit 'as' inject their library's 'exp' names; an
    // aliased import is reachable through the alias only (section 6.4)
    injected_maps: []InjectedMap,
    // every resolved use of a global definition, for tooling
    references: std.ArrayList(NameReference),
    associated: std.ArrayList(AssociatedFunction),
    // every declaration and use of a local binding, for tooling
    local_references: std.ArrayList(LocalReference),
    next_binding_id: usize,
    scopes: std.ArrayList(Frame),
    current_view: usize,

    const Frame = struct {
        bindings: std.ArrayList(Binding),
        // a lambda boundary: lookups for locals stop here (section 5.4)
        barrier: bool,
    };

    const Binding = struct {
        name: []const u8,
        id: usize,
        kind: LocalKind,
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
            .alias_maps = &.{},
            .injected_maps = &.{},
            .references = .empty,
            .associated = .empty,
            .local_references = .empty,
            .next_binding_id = 0,
            .scopes = .empty,
            .current_view = 0,
        };
    }

    /// Runs the merge and the resolve passes. Failures land in the
    /// diagnostics list; the returned unit is valid either way.
    pub fn run(self: *Resolver) Error!MergedUnit {
        try self.collectGlobals();
        self.alias_maps = try self.arena.alloc(AliasMap, self.views.len);
        for (self.alias_maps) |*map| map.* = .empty;
        self.injected_maps = try self.arena.alloc(InjectedMap, self.views.len);
        for (self.injected_maps) |*map| map.* = .empty;
        for (self.views, 0..) |view, view_index| {
            self.current_view = view_index;
            try self.registerImports(view);
        }
        try self.checkInjectedCollisions();
        try self.collectAssociated();
        for (self.views, 0..) |view, view_index| {
            self.current_view = view_index;
            try self.resolveModule(view);
        }
        return .{
            .globals = self.globals,
            .module_keys = self.module_keys,
            .aliases = self.alias_maps,
            .injected = self.injected_maps,
            .references = self.references.items,
            .locals = self.local_references.items,
            .associated = self.associated.items,
        };
    }

    // qualified functions ('fn Vector::empty') attach to any type visible
    // to the defining module, like extension functions; the qualifier
    // resolves through normal unqualified visibility (section 6.4)
    fn collectAssociated(self: *Resolver) Error!void {
        for (self.views, 0..) |view, view_index| {
            self.current_view = view_index;
            for (view.module.definitions) |*definition| {
                if (definition.kind != .fn_def) continue;
                const fn_def = &definition.kind.fn_def;
                const qualifier = fn_def.qualifier orelse continue;
                const type_name = qualifier.slice(view.source);
                const symbols = self.globals.get(type_name) orelse SymbolList.empty;
                const type_symbol = for (symbols.items) |candidate| {
                    if (candidate.definition.kind != .type_def) continue;
                    if (!self.visibleUnqualified(candidate)) continue;
                    break candidate;
                } else {
                    try self.report(qualifier.location, "'{s}' does not name a visible type here (section 6.4)", .{type_name});
                    continue;
                };
                // a qualified function is a plain free function: 'self'
                // receivers belong to extension functions, which already
                // reach the type through dot dispatch (section 5.5)
                if (fn_def.function.parameters.len != 0 and fn_def.function.parameters[0].is_self) {
                    try self.report(fn_def.name.location, "'{s}::{s}' cannot take 'self': a qualified function is not an extension (section 6.4)", .{ type_name, fn_def.name.slice(view.source) });
                    continue;
                }
                const fn_name = fn_def.name.slice(view.source);
                // an enum's variant names stay unambiguous constructors
                const base = type_symbol.definition.kind.type_def.base;
                if (base.* == .enum_type) {
                    const enum_source = self.views[type_symbol.view_index].source;
                    for (base.enum_type) |member| {
                        if (std.mem.eql(u8, member.name.slice(enum_source), fn_name)) {
                            try self.report(fn_def.name.location, "'{s}::{s}' collides with the enum's variant '{s}' (section 6.4)", .{ type_name, fn_name, fn_name });
                        }
                    }
                }
                try self.recordReference(qualifier.location, type_symbol.definition);
                try self.associated.append(self.arena, .{
                    .type_definition = type_symbol.definition,
                    .name = fn_name,
                    .symbol = .{
                        .visibility = definition.visibility,
                        .view_index = view_index,
                        .definition = definition,
                    },
                });
            }
        }
    }

    fn recordLocal(self: *Resolver, span: Token.Location, binding: Binding) Error!void {
        try self.local_references.append(self.arena, .{
            .view_index = self.current_view,
            .span = span,
            .binding_id = binding.id,
            .kind = binding.kind,
        });
    }

    fn recordReference(self: *Resolver, span: Token.Location, definition: *const ast.Definition) Error!void {
        try self.references.append(self.arena, .{
            .view_index = self.current_view,
            .span = span,
            .definition = definition,
        });
    }

    // registers each import's alias and, when the import has no explicit
    // 'as', injects its library's 'exp' names into this module's
    // unqualified namespace (section 6.4)
    fn registerImports(self: *Resolver, view: ModuleView) Error!void {
        for (view.module.imports) |import| {
            // every import gets an alias: the explicit 'as' name, or the
            // last path segment ('import pkg::mathx' aliases 'mathx')
            const key = try self.importKey(import, view);
            const alias_token = import.alias orelse import.path[import.path.len - 1];
            const alias_name = alias_token.slice(view.source);
            const existing = try self.alias_maps[self.current_view].getOrPut(self.arena, alias_name);
            if (existing.found_existing) {
                try self.report(alias_token.location, "the import alias '{s}' is already used in this module; rename one with 'as'", .{alias_name});
                continue;
            }
            existing.value_ptr.* = key;
            if (import.alias != null) continue;
            const target_view = self.module_keys.get(key) orelse continue;
            const target_library = self.views[target_view].library orelse continue;
            if (sameLibrary(view.library, target_library)) continue;
            const injected = try self.injected_maps[self.current_view].getOrPut(self.arena, target_library);
            if (!injected.found_existing) injected.value_ptr.* = alias_token;
        }
    }

    // a name visible unqualified from two different libraries in one module
    // is an error resolved by aliasing an import (section 6.4)
    fn checkInjectedCollisions(self: *Resolver) Error!void {
        var iterator = self.globals.iterator();
        while (iterator.next()) |entry| {
            for (self.views, 0..) |view, view_index| {
                var first_library: ?[]const u8 = null;
                var reported = false;
                for (entry.value_ptr.items) |symbol| {
                    if (reported) break;
                    const symbol_library = self.views[symbol.view_index].library;
                    const own = sameLibrary(view.library, symbol_library);
                    if (!own) {
                        if (symbol.visibility != .exported) continue;
                        const library = symbol_library orelse continue;
                        if (!self.injected_maps[view_index].contains(library)) continue;
                    }
                    const display = if (own) view.library orelse "this compilation unit" else symbol_library.?;
                    if (first_library) |previous| {
                        if (std.mem.eql(u8, previous, display)) continue;
                        // point at an offending import of this module
                        const token = self.injected_maps[view_index].get(if (own) previous else display).?;
                        self.current_view = view_index;
                        try self.report(token.location, "'{s}' is visible from both '{s}' and '{s}'; alias the import ('import ... as name') to disambiguate (section 6.4)", .{ entry.key_ptr.*, previous, display });
                        reported = true;
                    } else {
                        first_library = display;
                    }
                }
            }
        }
    }

    // merge pass: every definition of every module lands in one table
    fn collectGlobals(self: *Resolver) Error!void {
        for (self.views, 0..) |view, view_index| {
            if (view.key) |key| {
                try self.module_keys.put(self.arena, key, view_index);
            }
            for (view.module.definitions) |*definition| {
                // qualified functions live in their type's namespace, not
                // the flat map; collectAssociated registers them
                if (definition.kind == .fn_def and definition.kind.fn_def.qualifier != null) continue;
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
                // only fn definitions may share a name within one library
                // (overloading, section 4.6), and a macro may share a name
                // with functions - '#name' always selects the macro, a bare
                // 'name' never does (section 7.3); different libraries
                // reuse names freely, clashes surface at unqualified use
                // sites instead
                var conflict: ?Symbol = null;
                for (existing.value_ptr.items) |other| {
                    if (!sameLibrary(self.views[other.view_index].library, view.library)) continue;
                    if (callableKind(definition) and callableKind(other.definition) and
                        (definition.kind == .macro_def) != (other.definition.kind == .macro_def)) continue;
                    if (definition.kind == .fn_def and other.definition.kind == .fn_def) continue;
                    conflict = other;
                    break;
                }
                if (conflict) |other| {
                    const previous_path = self.views[other.view_index].path;
                    self.current_view = view_index;
                    try self.report(name_token.location, "redeclaration of '{s}'; previously declared in {s}", .{ name, previous_path });
                    continue;
                }
                try existing.value_ptr.append(self.arena, symbol);
            }
        }
    }

    fn resolveModule(self: *Resolver, view: ModuleView) Error!void {
        for (view.module.definitions) |*definition| {
            try self.resolveDefinition(definition);
        }
    }

    fn resolveDefinition(self: *Resolver, definition: *const ast.Definition) Error!void {
        switch (definition.kind) {
            .type_def => |type_def| {
                try self.pushFrame(false);
                try self.bindTypeParameters(type_def.type_parameters);
                for (type_def.interfaces) |marker| {
                    try self.resolveInterfaceMarker(marker);
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
                // the interface's own type parameters scope over every
                // declared signature ('fn next() -> Option<&T>')
                try self.pushFrame(false);
                try self.bindTypeParameters(interface_def.type_parameters);
                for (interface_def.functions) |interface_fn| {
                    for (interface_fn.parameters) |parameter| {
                        try self.resolveTypeExpression(parameter.parameter_type, false);
                    }
                    if (interface_fn.return_type) |return_type| {
                        try self.resolveTypeExpression(return_type, false);
                    }
                }
                self.popFrame();
            },
            .macro_def => |macro_def| {
                try self.pushFrame(false);
                for (macro_def.parameters) |parameter| {
                    if (parameter.parameter_type) |parameter_type| {
                        try self.resolveTypeExpression(parameter_type, false);
                    }
                    try self.bind(parameter.name, .parameter);
                }
                try self.resolveTypeExpression(macro_def.return_type, false);
                if (macro_def.body) |body| try self.resolveStatement(body);
                self.popFrame();
            },
        }
    }

    // shared by fn definitions and lambdas; the caller owns the outer frame
    fn resolveFunction(self: *Resolver, function: ast.Function) Error!void {
        for (function.parameters) |parameter| {
            try self.resolveTypeExpression(parameter.parameter_type, false);
            try self.bind(parameter.name, .parameter);
        }
        if (function.return_type) |return_type| {
            try self.resolveTypeExpression(return_type, false);
        }
        try self.resolveStatement(function.body);
    }

    fn bindTypeParameters(self: *Resolver, type_parameters: []const ast.TypeParameter) Error!void {
        for (type_parameters) |type_parameter| {
            // a constraint's type arguments may reference the parameters
            // declared to its left ('<T, It: Iterator<T>>', section 4.7)
            if (type_parameter.constraint) |constraint| {
                try self.resolveInterfaceMarker(constraint);
            }
            try self.bind(type_parameter.name, .type_parameter);
        }
    }

    fn callableKind(definition: *const ast.Definition) bool {
        return switch (definition.kind) {
            .fn_def, .extern_def, .macro_def => true,
            else => false,
        };
    }

    fn resolveInterfaceMarker(self: *Resolver, marker: ast.InterfaceMarker) Error!void {
        try self.resolveInterfaceName(marker.name);
        for (marker.type_arguments) |argument| {
            try self.resolveTypeExpression(argument, false);
        }
    }

    fn resolveInterfaceName(self: *Resolver, name_token: Token) Error!void {
        const name = name_token.slice(self.source());
        const lookup = self.lookupUnqualified(name);
        const symbol = lookup.visible orelse {
            if (lookup.hidden != null) {
                return self.report(name_token.location, "interface '{s}' is not exported; mark it 'exp' to allow use outside its library (section 6.4)", .{name});
            }
            return self.report(name_token.location, "use of undeclared interface '{s}'", .{name});
        };
        if (symbol.definition.kind != .interface_def) {
            return self.report(name_token.location, "'{s}' is not an interface ({s})", .{ name, definitionKindName(symbol.definition) });
        }
        try self.recordReference(name_token.location, symbol.definition);
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
                try self.bind(var_def.name, .variable);
            },
            .assign => |assign| {
                try self.resolveExpression(assign.target);
                try self.resolveExpression(assign.value);
            },
            .expression => |expression| try self.resolveExpression(expression),
            .break_stmt => |break_stmt| {
                if (break_stmt.value) |value| try self.resolveExpression(value);
            },
            .yield_stmt => |yield_stmt| try self.resolveExpression(yield_stmt.value),
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
                // an 'is' target may name an enum variant (section 4.2)
                const allow_variant = cast.operator.tag == .keyword_is;
                try self.resolveTypeExpression(cast.target, allow_variant);
                // an inline 'is' capture binds into the enclosing condition
                // frame, visible to later conjuncts and the branch body
                if (cast.capture) |capture| try self.bindCapture(capture);
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
            .subslice => |subslice| {
                try self.resolveExpression(subslice.object);
                if (subslice.start) |start| try self.resolveExpression(start);
                try self.resolveExpression(subslice.end);
            },
            // an inline layout's members resolve like any type expression
            .type_literal => |layout| try self.resolveTypeExpression(layout, false),
            .struct_init => |struct_init| {
                if (struct_init.path) |path| {
                    try self.resolvePath(path, .type);
                }
                for (struct_init.type_arguments) |argument| {
                    try self.resolveTypeExpression(argument, false);
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
                // the frame spans condition and then-branch: inline 'is'
                // captures bind during the condition (section 4.2)
                try self.pushFrame(false);
                try self.resolveExpression(if_expr.condition);
                try self.resolveStatement(if_expr.then_branch);
                self.popFrame();
                if (if_expr.else_branch) |else_branch| {
                    try self.resolveStatement(else_branch);
                }
            },
            .while_expr => |while_expr| {
                // like 'if': condition captures are visible in the body,
                // re-bound each iteration
                try self.pushFrame(false);
                try self.resolveExpression(while_expr.condition);
                try self.resolveStatement(while_expr.body);
                self.popFrame();
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
                // captures name variables of the enclosing scope (section 5.4)
                for (lambda.captures) |capture| {
                    try self.resolveCapturedVariable(capture.name);
                }
                try self.pushFrame(true);
                for (lambda.captures) |capture| {
                    try self.bind(capture.name, .variable);
                }
                try self.resolveFunction(lambda.function);
                self.popFrame();
            },
        }
    }

    fn bindCapture(self: *Resolver, capture: ast.Capture) Error!void {
        try self.bind(capture.name, .variable);
    }

    // a lambda capture must name something visible where the lambda is written
    fn resolveCapturedVariable(self: *Resolver, name_token: Token) Error!void {
        const name = name_token.slice(self.source());
        if (self.lookupLocal(name, false)) |binding_id| {
            return self.recordLocal(name_token.location, binding_id);
        }
        if (self.lookupUnqualified(name).visible != null) return;
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
            .type_description => {},
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
            if (self.lookupLocal(first, false)) |binding_id| {
                return self.recordLocal(span, binding_id);
            }
            const lookup = self.lookupUnqualified(first);
            if (lookup.visible) |symbol| {
                return self.recordReference(span, symbol.definition);
            }
            // primitives appear in type positions and as '#u32' reflection
            if (primitive_types.has(first)) return;
            if (self.lookupLocal(first, true)) |binding_id| {
                // capture lists are value-only: a type name (a scope type
                // parameter) stays visible inside a lambda without capture
                if (context == .type) return self.recordLocal(span, binding_id);
                return self.report(span, "'{s}' is declared outside this lambda; add it to the capture list to use it", .{first});
            }
            if (lookup.hidden) |hidden| {
                if (hidden.visibility == .exported) {
                    const library = self.views[hidden.view_index].library orelse "the program";
                    return self.report(span, "'{s}' is exported by '{s}' but not imported unqualified here; use qualified access or import it without 'as' (section 6.4)", .{ first, library });
                }
                return self.report(span, "'{s}' is not exported; mark it 'exp' to allow use outside its library (section 6.4)", .{first});
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
            // inside a library, relative module keys live under the
            // package namespace (section 6.4)
            if (self.views[self.current_view].library) |package_name| {
                const prefixed = try std.fmt.allocPrint(self.arena, "pkg::{s}::{s}", .{ package_name, key });
                if (self.module_keys.get(prefixed)) |view_index| {
                    return self.resolveQualified(span, prefixed, view_index, path[prefix_length..]);
                }
            }
            if (prefix_length == 1) break;
        }

        // alias-qualified: 'vectors::Vec' after 'import std::vec as vectors'
        if (self.alias_maps[self.current_view].get(first)) |key| {
            if (self.module_keys.get(key)) |view_index| {
                return self.resolveQualified(span, key, view_index, path[1..]);
            }
        }

        // 'Type::Variant' on a local or global enum type
        if (path.len == 2) {
            if (self.lookupLocal(first, false)) |binding_id| {
                return self.recordLocal(path[0].location, binding_id);
            }
            if (self.lookupUnqualified(first).visible) |symbol| {
                try self.recordReference(path[0].location, symbol.definition);
                return self.checkVariant(span, symbol, path[1]);
            }
        }

        const full = try self.joinPath(path, self.source());
        try self.report(span, "use of undeclared name '{s}'", .{full});
    }

    // resolves the remainder of a module-qualified path (section 6.4):
    // qualified access reaches pub/exp definitions within one compilation
    // unit, but only 'exp' definitions across a library boundary
    fn resolveQualified(self: *Resolver, span: Token.Location, key: []const u8, view_index: usize, remainder: []const Token) Error!void {
        const definition_name = remainder[0].slice(self.source());
        const cross_library = !sameLibrary(self.views[self.current_view].library, self.views[view_index].library);
        const symbols = self.globals.get(definition_name) orelse SymbolList.empty;
        var found: ?Symbol = null;
        var found_visible = false;
        for (symbols.items) |symbol| {
            if (symbol.view_index != view_index) continue;
            found = symbol;
            const visible = if (cross_library) symbol.visibility == .exported else symbol.visibility != .private;
            if (visible) {
                found_visible = true;
                break;
            }
        }
        if (found == null) {
            return self.report(span, "module '{s}' has no definition '{s}'", .{ key, definition_name });
        }
        if (!found_visible) {
            if (cross_library) {
                return self.report(span, "'{s}' in module '{s}' is not exported; mark it 'exp' to allow use outside its library (section 6.4)", .{ definition_name, key });
            }
            return self.report(span, "'{s}' in module '{s}' is private; mark it 'pub' to allow qualified access", .{ definition_name, key });
        }
        try self.recordReference(remainder[0].location, found.?.definition);
        if (remainder.len == 2) {
            return self.checkVariant(span, found.?, remainder[1]);
        }
        if (remainder.len > 2) {
            const full = try self.joinPath(remainder, self.source());
            return self.report(span, "'{s}' does not name a definition in module '{s}'", .{ full, key });
        }
    }

    // the canonical module key of an import, mirroring the loader's
    // namespacing: a library's relative imports resolve under its own
    // package prefix (section 6.4)
    fn importKey(self: *Resolver, import: ast.Import, view: ModuleView) Error![]const u8 {
        const joined = try self.joinPath(import.path, view.source);
        if (view.library) |package_name| {
            const first_segment = import.path[0].slice(view.source);
            if (!std.mem.eql(u8, first_segment, "std") and !std.mem.eql(u8, first_segment, "pkg")) {
                return std.fmt.allocPrint(self.arena, "pkg::{s}::{s}", .{ package_name, joined });
            }
        }
        // an import resolves relative to the importing module first (5.4):
        // when the loader found the file there, the module is registered
        // under the directory-qualified key, which this must mirror
        if (view.library == null and
            !std.mem.startsWith(u8, joined, "std::") and
            !std.mem.startsWith(u8, joined, "pkg::"))
        {
            if (view.key) |module_key| {
                if (std.mem.lastIndexOf(u8, module_key, "::")) |prefix_end| {
                    const relative = try std.fmt.allocPrint(self.arena, "{s}::{s}", .{ module_key[0..prefix_end], joined });
                    if (self.module_keys.contains(relative)) return relative;
                }
            }
        }
        return joined;
    }

    // unqualified lookup (section 6.4): a name sees every symbol of its own
    // library plus the 'exp' symbols of libraries this module imported
    // without an alias
    fn visibleUnqualified(self: *const Resolver, symbol: Symbol) bool {
        if (sameLibrary(self.views[self.current_view].library, self.views[symbol.view_index].library)) return true;
        if (symbol.visibility != .exported) return false;
        const library = self.views[symbol.view_index].library orelse return false;
        return self.injected_maps[self.current_view].contains(library);
    }

    const UnqualifiedLookup = struct {
        visible: ?Symbol = null,
        // an invisible match: not exported, or exported behind an alias
        hidden: ?Symbol = null,
    };

    fn lookupUnqualified(self: *const Resolver, name: []const u8) UnqualifiedLookup {
        var result: UnqualifiedLookup = .{};
        const symbols = self.globals.get(name) orelse return result;
        for (symbols.items) |symbol| {
            if (self.visibleUnqualified(symbol)) {
                if (result.visible == null) result.visible = symbol;
            } else if (result.hidden == null) {
                result.hidden = symbol;
            }
        }
        return result;
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
        // a qualified function in the type's namespace is not a variant
        for (self.associated.items) |entry| {
            if (entry.type_definition == symbol.definition and std.mem.eql(u8, entry.name, variant_name)) return;
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
    fn bind(self: *Resolver, name_token: Token, kind: LocalKind) Error!void {
        const name = name_token.slice(self.source());
        const frame = &self.scopes.items[self.scopes.items.len - 1];
        for (frame.bindings.items) |binding| {
            if (std.mem.eql(u8, binding.name, name)) {
                return self.report(name_token.location, "'{s}' is already declared in this scope", .{name});
            }
        }
        const binding: Binding = .{ .name = name, .id = self.next_binding_id, .kind = kind };
        self.next_binding_id += 1;
        try frame.bindings.append(self.arena, binding);
        try self.recordLocal(name_token.location, binding);
    }

    // searches scope frames innermost-first, yielding the binding's id;
    // without 'past_barriers' the search stops at a lambda boundary so
    // uncaptured locals stay invisible
    fn lookupLocal(self: *const Resolver, name: []const u8, past_barriers: bool) ?Binding {
        var frame_index = self.scopes.items.len;
        while (frame_index > 0) {
            frame_index -= 1;
            const frame = self.scopes.items[frame_index];
            for (frame.bindings.items) |binding| {
                if (std.mem.eql(u8, binding.name, name)) return binding;
            }
            if (frame.barrier and !past_barriers) return null;
        }
        return null;
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
