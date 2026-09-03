//! Parser for the Alloy language, implementing section 3 (Syntactic Grammar)
//! of LANGUAGE_SPEC.md. Consumes the token list produced by the tokenizer and
//! builds the tree defined in ast.zig.
//!
//! Terminators are strict (section 3.1): every statement ends with ';',
//! which expectTerminator consumes, and a stray ';' where a statement would
//! start is an error rather than an empty statement.

const std = @import("std");
const tokenizer_module = @import("tokenizer.zig");
const Token = tokenizer_module.Token;
const ast = @import("ast.zig");

pub const Parser = struct {
    arena: std.mem.Allocator,
    source: []const u8,
    tokens: []const Token,
    token_index: usize,
    // a generic close '>>' splits into two '>'; true means one half is pending
    pending_angle: bool,
    // match arm patterns end at '|', so the capture is not eaten as bitwise or
    stop_at_pipe: bool,
    // match arm patterns disallow 'Name { ... }' so the arm body block is kept
    allow_struct_init: bool,
    // a cast target retried without generic arguments: 'x to i32 < y' is a
    // comparison once '<' fails to open a type-argument list
    plain_cast_target: bool = false,
    failure: ?Failure,

    pub const Failure = struct {
        span: Token.Location,
        message: []const u8,
    };

    pub const Error = error{ ParseError, OutOfMemory };

    pub fn init(arena: std.mem.Allocator, source: []const u8, tokens: []const Token) Parser {
        return .{
            .arena = arena,
            .source = source,
            .tokens = tokens,
            .token_index = 0,
            .pending_angle = false,
            .stop_at_pipe = false,
            .allow_struct_init = true,
            .failure = null,
        };
    }

    pub fn parseModule(self: *Parser) Error!ast.Module {
        var imports: std.ArrayList(ast.Import) = .empty;
        var definitions: std.ArrayList(ast.Definition) = .empty;

        try self.rejectStrayTerminator();
        while (self.current().tag == .keyword_import) {
            try imports.append(self.arena, try self.parseImport());
            try self.rejectStrayTerminator();
        }
        while (self.current().tag != .end_of_file) {
            try definitions.append(self.arena, try self.parseDefinition());
            try self.rejectStrayTerminator();
        }
        return .{
            .imports = try imports.toOwnedSlice(self.arena),
            .definitions = try definitions.toOwnedSlice(self.arena),
        };
    }

    fn parseImport(self: *Parser) Error!ast.Import {
        _ = try self.expect(.keyword_import, "'import'");
        var path: std.ArrayList(Token) = .empty;
        try path.append(self.arena, try self.expect(.identifier, "a module name"));
        while (self.match(.colon_colon) != null) {
            try path.append(self.arena, try self.expect(.identifier, "a module name"));
        }
        var alias: ?Token = null;
        if (self.match(.keyword_as) != null) {
            alias = try self.expect(.identifier, "an alias name");
        }
        try self.expectTerminator();
        return .{ .path = try path.toOwnedSlice(self.arena), .alias = alias };
    }

    fn parseDefinition(self: *Parser) Error!ast.Definition {
        var visibility: ast.Visibility = .private;
        if (self.match(.keyword_pub) != null) {
            visibility = .public;
        } else if (self.match(.keyword_exp) != null) {
            visibility = .exported;
        }
        const token = self.current();
        return switch (token.tag) {
            .keyword_type => .{ .visibility = visibility, .kind = .{ .type_def = try self.parseTypeDef() } },
            .keyword_fn => .{ .visibility = visibility, .kind = .{ .fn_def = try self.parseFnDef() } },
            .keyword_extern => .{ .visibility = visibility, .kind = .{ .extern_def = try self.parseExternDef() } },
            .keyword_interface => .{ .visibility = visibility, .kind = .{ .interface_def = try self.parseInterfaceDef() } },
            .keyword_macro => .{ .visibility = visibility, .kind = .{ .macro_def = try self.parseMacroDef() } },
            .keyword_import => self.fail(token, "imports must appear before all definitions", .{}),
            else => self.fail(token, "expected a definition (type, fn, extern, interface, or macro), found {s}", .{try self.describe(token)}),
        };
    }

    fn parseTypeDef(self: *Parser) Error!ast.TypeDef {
        _ = try self.expect(.keyword_type, "'type'");
        const name = try self.expect(.identifier, "a type name");
        const type_parameters = try self.parseTypeParameters();
        var interfaces: std.ArrayList(ast.InterfaceMarker) = .empty;
        if (self.match(.colon) != null) {
            try interfaces.append(self.arena, try self.parseInterfaceMarker());
            while (self.match(.comma) != null) {
                try interfaces.append(self.arena, try self.parseInterfaceMarker());
            }
        }
        _ = try self.expect(.equal, "'='");
        const base = try self.parseBaseType();
        try self.expectTerminator();
        return .{
            .name = name,
            .type_parameters = type_parameters,
            .interfaces = try interfaces.toOwnedSlice(self.arena),
            .base = base,
        };
    }

    fn parseFnDef(self: *Parser) Error!ast.FnDef {
        _ = try self.expect(.keyword_fn, "'fn'");
        var qualifier: ?Token = null;
        var name = try self.expect(.identifier, "a function name");
        // 'fn Vector::empty(...)' defines into the type's namespace
        if (self.match(.colon_colon) != null) {
            qualifier = name;
            name = try self.expect(.identifier, "a function name");
        }
        const type_parameters = try self.parseTypeParameters();
        const function = try self.parseFunction();
        return .{ .qualifier = qualifier, .name = name, .type_parameters = type_parameters, .function = function };
    }

    fn parseExternDef(self: *Parser) Error!ast.ExternDef {
        _ = try self.expect(.keyword_extern, "'extern'");
        const name = try self.expect(.identifier, "a function name");
        const parameter_list = try self.parseParameterList(true);
        var return_type: ?*const ast.TypeExpression = null;
        if (self.match(.arrow) != null) {
            return_type = try self.parseType();
        }
        try self.expectTerminator();
        return .{
            .name = name,
            .parameters = parameter_list.parameters,
            .variadic = parameter_list.variadic,
            .return_type = return_type,
        };
    }

    fn parseInterfaceDef(self: *Parser) Error!ast.InterfaceDef {
        _ = try self.expect(.keyword_interface, "'interface'");
        const name = try self.expect(.identifier, "an interface name");
        const type_parameters = try self.parseTypeParameters();
        _ = try self.expect(.brace_left, "'{'");
        var functions: std.ArrayList(ast.InterfaceFn) = .empty;
        try self.rejectStrayTerminator();
        while (self.current().tag != .brace_right) {
            _ = try self.expect(.keyword_fn, "'fn' or '}'");
            const fn_name = try self.expect(.identifier, "a function name");
            // the receiver indirection comes first, nameless and typeless:
            // the type is whatever implements the interface (section 6.2)
            _ = try self.expect(.parenthesis_left, "'('");
            const receiver_token = self.current();
            if (self.match(.keyword_self) == null) {
                return self.fail(receiver_token, "an interface function declares its receiver first: 'fn {s}(self: &, ...)' (section 6.2)", .{fn_name.slice(self.source)});
            }
            _ = try self.expect(.colon, "':'");
            const receiver = self.parseTypeModifier() orelse {
                return self.fail(self.current(), "expected the receiver indirection ('&', '&var', '*', or '*var') after 'self:' (section 6.2)", .{});
            };
            var parameters: std.ArrayList(ast.Parameter) = .empty;
            while (self.match(.comma) != null) {
                if (self.current().tag == .parenthesis_right) break;
                const parameter_name = try self.expect(.identifier, "a parameter name");
                _ = try self.expect(.colon, "':'");
                const parameter_type = try self.parseType();
                try parameters.append(self.arena, .{ .is_self = false, .name = parameter_name, .parameter_type = parameter_type });
            }
            _ = try self.expect(.parenthesis_right, "')'");
            var return_type: ?*const ast.TypeExpression = null;
            if (self.match(.arrow) != null) {
                return_type = try self.parseType();
            }
            try self.expectTerminator();
            try functions.append(self.arena, .{
                .name = fn_name,
                .receiver = receiver,
                .parameters = try parameters.toOwnedSlice(self.arena),
                .return_type = return_type,
            });
            try self.rejectStrayTerminator();
        }
        _ = try self.expect(.brace_right, "'}'");
        return .{ .name = name, .type_parameters = type_parameters, .functions = try functions.toOwnedSlice(self.arena) };
    }

    fn parseMacroDef(self: *Parser) Error!ast.MacroDef {
        _ = try self.expect(.keyword_macro, "'macro'");
        const name = try self.expect(.identifier, "a macro name");
        _ = try self.expect(.parenthesis_left, "'('");
        var parameters: std.ArrayList(ast.MacroParameter) = .empty;
        if (self.current().tag != .parenthesis_right) {
            while (true) {
                const parameter_name = try self.expect(.identifier, "a parameter name");
                var parameter_type: ?*const ast.TypeExpression = null;
                if (self.match(.colon) != null) {
                    parameter_type = try self.parseType();
                }
                try parameters.append(self.arena, .{ .name = parameter_name, .parameter_type = parameter_type });
                if (self.match(.comma) == null) break;
            }
        }
        _ = try self.expect(.parenthesis_right, "')'");
        // the result type is declared, never inferred (section 7.3)
        if (self.current().tag != .arrow) {
            return self.fail(self.current(), "a macro declares its result type: 'macro {s}(...) -> type' (section 7.3)", .{name.slice(self.source)});
        }
        _ = self.advance();
        const return_type = try self.parseType();
        // a body makes an ordinary macro; a terminator makes a
        // declaration-only macro, implemented by the compiler
        if (self.current().tag != .brace_left) {
            try self.expectTerminator();
            return .{ .name = name, .parameters = try parameters.toOwnedSlice(self.arena), .return_type = return_type, .body = null };
        }
        for (parameters.items) |parameter| {
            if (parameter.parameter_type == null) {
                return self.fail(parameter.name, "a macro with a body types its parameters ('{s}: ...')", .{parameter.name.slice(self.source)});
            }
        }
        const body = try self.parseBlock();
        return .{ .name = name, .parameters = try parameters.toOwnedSlice(self.arena), .return_type = return_type, .body = body };
    }

    fn parseTypeParameters(self: *Parser) Error![]const ast.TypeParameter {
        if (self.match(.angle_left) == null) return &.{};
        return self.parseAngleList(ast.TypeParameter, parseTypeParameter);
    }

    fn parseTypeParameter(self: *Parser) Error!ast.TypeParameter {
        const parameter_name = try self.expect(.identifier, "a type parameter name");
        var constraint: ?ast.InterfaceMarker = null;
        if (self.match(.colon) != null) {
            constraint = try self.parseInterfaceMarker();
        }
        return .{ .name = parameter_name, .constraint = constraint };
    }

    // the comma-separated elements of a '<...>' list up to and including
    // its '>', the opening '<' already consumed
    fn parseAngleList(self: *Parser, comptime T: type, comptime parseElement: fn (*Parser) Error!T) Error![]const T {
        var elements: std.ArrayList(T) = .empty;
        while (true) {
            try elements.append(self.arena, try parseElement(self));
            // a pending half of '>>' closes this list before any comma
            if (self.pending_angle) break;
            if (self.match(.comma) == null) break;
        }
        try self.expectAngleRight();
        return elements.toOwnedSlice(self.arena);
    }

    // an interface name with optional type arguments, used by conformance
    // markers and generic constraints ('It: Iterator<T>', section 6.2)
    fn parseInterfaceMarker(self: *Parser) Error!ast.InterfaceMarker {
        const name = try self.expect(.identifier, "an interface name");
        var type_arguments: []const *const ast.TypeExpression = &.{};
        if (self.match(.angle_left) != null) {
            type_arguments = try self.parseAngleList(*const ast.TypeExpression, parseType);
        }
        return .{ .name = name, .type_arguments = type_arguments };
    }

    const ParameterList = struct {
        parameters: []const ast.Parameter,
        variadic: bool,
    };

    fn parseParameterList(self: *Parser, allow_variadic: bool) Error!ParameterList {
        _ = try self.expect(.parenthesis_left, "'('");
        var parameters: std.ArrayList(ast.Parameter) = .empty;
        var variadic = false;
        if (self.current().tag != .parenthesis_right) {
            while (true) {
                if (allow_variadic and self.current().tag == .ellipsis) {
                    _ = self.advance();
                    variadic = true;
                    break;
                }
                const is_self = self.match(.keyword_self) != null;
                const parameter_name = try self.expect(.identifier, "a parameter name");
                _ = try self.expect(.colon, "':'");
                const parameter_type = try self.parseType();
                try parameters.append(self.arena, .{
                    .is_self = is_self,
                    .name = parameter_name,
                    .parameter_type = parameter_type,
                });
                if (self.match(.comma) == null) break;
            }
        }
        _ = try self.expect(.parenthesis_right, "')'");
        return .{ .parameters = try parameters.toOwnedSlice(self.arena), .variadic = variadic };
    }

    fn parseFunction(self: *Parser) Error!ast.Function {
        const parameter_list = try self.parseParameterList(false);
        var return_type: ?*const ast.TypeExpression = null;
        if (self.match(.arrow) != null) {
            return_type = try self.parseType();
        }
        const body = try self.parseBlock();
        return .{
            .parameters = parameter_list.parameters,
            .return_type = return_type,
            .body = body,
        };
    }

    fn parseType(self: *Parser) Error!*const ast.TypeExpression {
        if (self.parseTypeModifier()) |modifier| {
            const child = try self.parseType();
            return self.create(ast.TypeExpression, .{ .modified = .{ .modifier = modifier, .child = child } });
        }
        return self.parseBaseType();
    }

    // consumes '*' / '*var' / '&' / '&var' when present
    fn parseTypeModifier(self: *Parser) ?ast.TypeModifier {
        return switch (self.current().tag) {
            .asterisk => modifier: {
                _ = self.advance();
                break :modifier if (self.match(.keyword_var) != null) .pointer_var else .pointer;
            },
            .ampersand => modifier: {
                _ = self.advance();
                break :modifier if (self.match(.keyword_var) != null) .reference_var else .reference;
            },
            else => null,
        };
    }

    fn parseBaseType(self: *Parser) Error!*const ast.TypeExpression {
        const token = self.current();
        switch (token.tag) {
            .keyword_struct => {
                _ = self.advance();
                return self.create(ast.TypeExpression, .{ .struct_type = try self.parseStructMembers() });
            },
            .keyword_enum => {
                _ = self.advance();
                return self.create(ast.TypeExpression, .{ .enum_type = try self.parseEnumMembers() });
            },
            .bracket_left => {
                _ = self.advance();
                const element = try self.parseType();
                var length: ?Token = null;
                if (self.match(.colon) != null) {
                    length = try self.expect(.integer_literal, "an integer length");
                }
                _ = try self.expect(.bracket_right, "']'");
                return self.create(ast.TypeExpression, .{ .array = .{ .element = element, .length = length } });
            },
            .parenthesis_left => {
                _ = self.advance();
                var parameter_types: std.ArrayList(*const ast.TypeExpression) = .empty;
                if (self.current().tag != .parenthesis_right) {
                    while (true) {
                        try parameter_types.append(self.arena, try self.parseType());
                        if (self.match(.comma) == null) break;
                    }
                }
                _ = try self.expect(.parenthesis_right, "')'");
                var return_type: ?*const ast.TypeExpression = null;
                if (self.match(.arrow) != null) {
                    return_type = try self.parseType();
                }
                return self.create(ast.TypeExpression, .{ .function = .{
                    .parameter_types = try parameter_types.toOwnedSlice(self.arena),
                    .return_type = return_type,
                } });
            },
            .hash => {
                _ = self.advance();
                // '#Type' in a type position is the descriptor itself,
                // never the reflection of a type named 'Type' (section 4.4)
                const next = self.current();
                if (next.tag == .identifier and std.mem.eql(u8, next.slice(self.source), "Type")) {
                    _ = self.advance();
                    return self.create(ast.TypeExpression, .{ .type_description = next });
                }
                const expression = try self.parseComptimeOperand();
                return self.create(ast.TypeExpression, .{ .comptime_type = expression });
            },
            .identifier => {
                return self.create(ast.TypeExpression, .{ .named = try self.parseNamedType() });
            },
            else => return self.fail(token, "expected a type, found {s}", .{try self.describe(token)}),
        }
    }

    fn parseNamedType(self: *Parser) Error!ast.NamedType {
        var path: std.ArrayList(Token) = .empty;
        try path.append(self.arena, try self.expect(.identifier, "a type name"));
        while (self.match(.colon_colon) != null) {
            try path.append(self.arena, try self.expect(.identifier, "a type name"));
        }
        var type_arguments: []const *const ast.TypeExpression = &.{};
        if (!self.plain_cast_target and self.match(.angle_left) != null) {
            type_arguments = try self.parseAngleList(*const ast.TypeExpression, parseType);
        }
        return .{
            .path = try path.toOwnedSlice(self.arena),
            .type_arguments = type_arguments,
        };
    }

    fn parseStructMembers(self: *Parser) Error![]const ast.StructMember {
        _ = try self.expect(.brace_left, "'{'");
        var members: std.ArrayList(ast.StructMember) = .empty;
        while (self.current().tag != .brace_right) {
            const member_name = try self.expect(.identifier, "a member name");
            _ = try self.expect(.colon, "':'");
            const member_type = try self.parseType();
            try members.append(self.arena, .{ .name = member_name, .member_type = member_type });
            if (self.match(.comma) == null) break;
        }
        _ = try self.expect(.brace_right, "'}'");
        return members.toOwnedSlice(self.arena);
    }

    fn parseEnumMembers(self: *Parser) Error![]const ast.EnumMember {
        _ = try self.expect(.brace_left, "'{'");
        var members: std.ArrayList(ast.EnumMember) = .empty;
        while (self.current().tag != .brace_right) {
            const member_name = try self.expect(.identifier, "a variant name");
            var payload: ?*const ast.TypeExpression = null;
            if (self.match(.colon) != null) {
                payload = try self.parseType();
            }
            try members.append(self.arena, .{ .name = member_name, .payload = payload });
            if (self.match(.comma) == null) break;
        }
        _ = try self.expect(.brace_right, "'}'");
        return members.toOwnedSlice(self.arena);
    }

    fn parseBlock(self: *Parser) Error!*const ast.Statement {
        _ = try self.expect(.brace_left, "'{'");
        var statements: std.ArrayList(*const ast.Statement) = .empty;
        try self.rejectStrayTerminator();
        while (self.current().tag != .brace_right) {
            if (self.current().tag == .end_of_file) {
                return self.fail(self.current(), "expected '}}', found end of file", .{});
            }
            try statements.append(self.arena, try self.parseStatement());
            try self.rejectStrayTerminator();
        }
        _ = try self.expect(.brace_right, "'}'");
        return self.create(ast.Statement, .{ .block = try statements.toOwnedSlice(self.arena) });
    }

    fn parseStatement(self: *Parser) Error!*const ast.Statement {
        const token = self.current();
        switch (token.tag) {
            .keyword_var, .keyword_const => return self.parseVarDef(),
            // '{' in statement position is a block unless '.' follows
            // (an anonymous struct literal, section 3.1)
            .brace_left => {
                if (self.tokens[self.token_index + 1].tag != .dot) return self.parseBlock();
            },
            // a statement starting with one of these always parses as the
            // statement form, never as an expression (section 3.1)
            .keyword_if, .keyword_while, .keyword_for, .keyword_match => {
                return self.create(ast.Statement, .{ .expression = try self.parseControlFlow(token.tag, false) });
            },
            .keyword_break => {
                const keyword = self.advance();
                var value: ?*const ast.Expression = null;
                if (self.current().tag != .semicolon) {
                    value = try self.parseExpression();
                }
                try self.expectTerminator();
                return self.create(ast.Statement, .{ .break_stmt = .{ .keyword = keyword, .value = value } });
            },
            .keyword_continue => {
                const keyword = self.advance();
                try self.expectTerminator();
                return self.create(ast.Statement, .{ .continue_stmt = .{ .keyword = keyword } });
            },
            .keyword_yield => {
                const keyword = self.advance();
                const value = try self.parseExpression();
                try self.expectTerminator();
                return self.create(ast.Statement, .{ .yield_stmt = .{ .keyword = keyword, .value = value } });
            },
            .keyword_return => {
                const keyword = self.advance();
                var value: ?*const ast.Expression = null;
                if (self.current().tag != .semicolon) {
                    value = try self.parseExpression();
                }
                try self.expectTerminator();
                return self.create(ast.Statement, .{ .return_stmt = .{ .keyword = keyword, .value = value } });
            },
            .keyword_panic => {
                const keyword = self.advance();
                var message: ?*const ast.Expression = null;
                if (self.current().tag != .semicolon) {
                    message = try self.parseExpression();
                }
                try self.expectTerminator();
                return self.create(ast.Statement, .{ .panic_stmt = .{ .keyword = keyword, .message = message } });
            },
            else => {},
        }
        // anything else is an expression statement or an assignment
        const expression = try self.parseExpression();
        if (assignOperator(self.current().tag)) {
            const operator = self.advance();
            const value = try self.parseExpression();
            try self.expectTerminator();
            return self.create(ast.Statement, .{ .assign = .{
                .target = expression,
                .operator = operator,
                .value = value,
            } });
        }
        try self.expectTerminator();
        return self.create(ast.Statement, .{ .expression = expression });
    }

    // the constructs that read as either a statement or a value; 'tag' is
    // one of their keywords and 'as_value' selects the position (section 3.1)
    fn parseControlFlow(self: *Parser, tag: Token.Tag, as_value: bool) Error!*const ast.Expression {
        return switch (tag) {
            .keyword_if => self.create(ast.Expression, .{ .if_expr = try self.parseIf(as_value) }),
            .keyword_while => self.create(ast.Expression, .{ .while_expr = try self.parseWhile(as_value) }),
            .keyword_for => self.create(ast.Expression, .{ .for_expr = try self.parseFor(as_value) }),
            .keyword_match => self.create(ast.Expression, .{ .match_expr = try self.parseMatch(as_value) }),
            else => unreachable,
        };
    }

    fn parseVarDef(self: *Parser) Error!*const ast.Statement {
        const keyword = self.advance();
        const mutable = keyword.tag == .keyword_var;
        const name = try self.expect(.identifier, "a variable name");
        var declared_type: ?*const ast.TypeExpression = null;
        if (self.match(.colon) != null) {
            declared_type = try self.parseType();
        }
        _ = try self.expect(.equal, "'='");
        const value = try self.parseExpression();
        try self.expectTerminator();
        return self.create(ast.Statement, .{ .var_def = .{
            .mutable = mutable,
            .name = name,
            .declared_type = declared_type,
            .value = value,
        } });
    }

    /// Parses one freestanding expression, consuming the whole input; the
    /// debugger uses this for watch expressions and breakpoint conditions.
    pub fn parseFreestandingExpression(self: *Parser) Error!*const ast.Expression {
        const expression = try self.parseExpression();
        if (self.current().tag != .end_of_file) {
            return self.fail(self.current(), "expected the end of the expression", .{});
        }
        return expression;
    }

    fn parseExpression(self: *Parser) Error!*const ast.Expression {
        return self.parseBinary(0);
    }

    // an expression wrapped in its own delimiters ('(...)', '[...]', argument
    // and initializer lists) drops the match-arm pattern limits: a '|' inside
    // parentheses is bitwise or again, and struct initializers are legal
    fn parseInnerExpression(self: *Parser) Error!*const ast.Expression {
        const saved_pipe = self.stop_at_pipe;
        const saved_struct = self.allow_struct_init;
        self.stop_at_pipe = false;
        self.allow_struct_init = true;
        defer {
            self.stop_at_pipe = saved_pipe;
            self.allow_struct_init = saved_struct;
        }
        return self.parseExpression();
    }

    fn parseBinary(self: *Parser, minimum_precedence: u8) Error!*const ast.Expression {
        var left = try self.parseCast();
        while (true) {
            const tag = self.current().tag;
            if (tag == .pipe and self.stop_at_pipe) break;
            const precedence = binaryPrecedence(tag) orelse break;
            if (precedence < minimum_precedence) break;
            const operator = self.advance();
            // +1 keeps equal-precedence operators left-associative
            const right = try self.parseBinary(precedence + 1);
            left = try self.create(ast.Expression, .{ .binary = .{
                .operator = operator,
                .left = left,
                .right = right,
            } });
        }
        return left;
    }

    fn parseCast(self: *Parser) Error!*const ast.Expression {
        var operand = try self.parseUnary();
        while (true) {
            const tag = self.current().tag;
            switch (tag) {
                .keyword_is => {
                    const operator = self.advance();
                    // '::Variant' implies the enum from the subject
                    const named: ast.NamedType = if (self.match(.colon_colon) != null) implied: {
                        const path = try self.arena.alloc(Token, 1);
                        path[0] = try self.expect(.identifier, "a variant name");
                        break :implied .{ .path = path, .type_arguments = &.{}, .implied = true };
                    } else try self.parseNamedType();
                    const target = try self.create(ast.TypeExpression, .{ .named = named });
                    // 'x is ::Some |v|' captures the payload inline: a '|'
                    // right after the test's type ALWAYS opens a capture,
                    // never bitwise or (section 3.1); match-arm captures
                    // stay the arm's own
                    var capture: ?ast.Capture = null;
                    if (!self.stop_at_pipe and self.current().tag == .pipe) {
                        _ = self.advance();
                        capture = try self.parseCapture();
                        _ = try self.expect(.pipe, "'|'");
                    }
                    operand = try self.create(ast.Expression, .{ .cast = .{
                        .operator = operator,
                        .operand = operand,
                        .target = target,
                        .capture = capture,
                    } });
                },
                .keyword_as, .keyword_to => {
                    const operator = self.advance();
                    const target = try self.parseCastTarget();
                    operand = try self.create(ast.Expression, .{ .cast = .{
                        .operator = operator,
                        .operand = operand,
                        .target = target,
                    } });
                },
                else => break,
            }
        }
        return operand;
    }

    // a cast target may be generic ('x to Vector<u8>'), but a '<' after it
    // is usually a comparison ('n to i32 < limit'): the argument list is
    // tried first and the target re-read without one when it fails
    fn parseCastTarget(self: *Parser) Error!*const ast.TypeExpression {
        const saved = self.snapshot();
        return self.parseType() catch |err| switch (err) {
            error.OutOfMemory => return err,
            error.ParseError => {
                self.restore(saved);
                self.plain_cast_target = true;
                defer self.plain_cast_target = false;
                return self.parseType();
            },
        };
    }

    fn parseUnary(self: *Parser) Error!*const ast.Expression {
        switch (self.current().tag) {
            .minus, .tilde, .bang, .ampersand, .keyword_new, .keyword_move => {
                const operator = self.advance();
                // '&var' is the mutable form of the borrow (section 5.2)
                const mutable = operator.tag == .ampersand and self.match(.keyword_var) != null;
                const operand = try self.parseUnary();
                return self.create(ast.Expression, .{ .unary = .{ .operator = operator, .operand = operand, .mutable = mutable } });
            },
            else => return self.parsePostfix(),
        }
    }

    fn parsePostfix(self: *Parser) Error!*const ast.Expression {
        return self.parsePostfixFrom(try self.parsePrimary());
    }

    fn parsePostfixFrom(self: *Parser, primary: *const ast.Expression) Error!*const ast.Expression {
        var expression = primary;
        while (true) {
            switch (self.current().tag) {
                .parenthesis_left => {
                    const arguments = try self.parseCallArguments();
                    expression = try self.create(ast.Expression, .{ .call = .{
                        .callee = expression,
                        .type_arguments = &.{},
                        .arguments = arguments,
                    } });
                },
                .dot => {
                    _ = self.advance();
                    const name = try self.expect(.identifier, "a member name");
                    expression = try self.create(ast.Expression, .{ .member = .{
                        .object = expression,
                        .name = name,
                    } });
                },
                .bracket_left => {
                    _ = self.advance();
                    // 'arr[start..end]' and 'arr[..end]' denote the unsized
                    // range value (section 3.1); the checker demands '&',
                    // '&var', or 'new' on it. A lone subscript is an index
                    var start: ?*const ast.Expression = null;
                    if (self.current().tag != .dot_dot) {
                        start = try self.parseInnerExpression();
                    }
                    if (self.current().tag == .dot_dot) {
                        const operator = self.advance();
                        const end = try self.parseInnerExpression();
                        _ = try self.expect(.bracket_right, "']'");
                        expression = try self.create(ast.Expression, .{ .subslice = .{
                            .object = expression,
                            .operator = operator,
                            .start = start,
                            .end = end,
                        } });
                        continue;
                    }
                    const subscript = start orelse
                        return self.fail(self.current(), "expected an index or a '..' subslice, found {s}", .{try self.describe(self.current())});
                    _ = try self.expect(.bracket_right, "']'");
                    expression = try self.create(ast.Expression, .{ .index = .{
                        .object = expression,
                        .subscript = subscript,
                    } });
                },
                .angle_left => {
                    // could be a generic call 'f<T>(x)' or a comparison 'a < b';
                    // try the generic call and fall back to comparison
                    expression = self.tryParseGenericCall(expression) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        error.ParseError => break,
                    };
                },
                else => break,
            }
        }
        return expression;
    }

    const Snapshot = struct {
        token_index: usize,
        pending_angle: bool,
        failure: ?Failure,
    };

    fn snapshot(self: *const Parser) Snapshot {
        return .{
            .token_index = self.token_index,
            .pending_angle = self.pending_angle,
            .failure = self.failure,
        };
    }

    fn restore(self: *Parser, saved: Snapshot) void {
        self.token_index = saved.token_index;
        self.pending_angle = saved.pending_angle;
        self.failure = saved.failure;
    }

    fn tryParseGenericCall(self: *Parser, callee: *const ast.Expression) Error!*const ast.Expression {
        const saved = self.snapshot();
        errdefer self.restore(saved);
        _ = try self.expect(.angle_left, "'<'");
        const type_arguments = try self.parseAngleList(*const ast.TypeExpression, parseType);
        // 'Vector<T> { ... }': a struct literal binding the type's
        // parameters explicitly (section 4.7)
        if (self.current().tag == .brace_left and self.allow_struct_init and callee.* == .path) {
            const members = try self.parseMemberInits();
            return self.create(ast.Expression, .{ .struct_init = .{
                .path = callee.path,
                .type_arguments = type_arguments,
                .members = members,
            } });
        }
        if (self.current().tag != .parenthesis_left) {
            return self.fail(self.current(), "expected '(' or '{{' after generic arguments", .{});
        }
        const arguments = try self.parseCallArguments();
        return self.create(ast.Expression, .{ .call = .{
            .callee = callee,
            .type_arguments = type_arguments,
            .arguments = arguments,
        } });
    }

    fn parseCallArguments(self: *Parser) Error![]const *const ast.Expression {
        _ = try self.expect(.parenthesis_left, "'('");
        var arguments: std.ArrayList(*const ast.Expression) = .empty;
        if (self.current().tag != .parenthesis_right) {
            while (true) {
                try arguments.append(self.arena, try self.parseInnerExpression());
                if (self.match(.comma) == null) break;
            }
        }
        _ = try self.expect(.parenthesis_right, "')'");
        return arguments.toOwnedSlice(self.arena);
    }

    fn parsePrimary(self: *Parser) Error!*const ast.Expression {
        const token = self.current();
        switch (token.tag) {
            .integer_literal => return self.create(ast.Expression, .{ .integer_literal = self.advance() }),
            .float_literal => return self.create(ast.Expression, .{ .float_literal = self.advance() }),
            .string_literal => return self.create(ast.Expression, .{ .string_literal = self.advance() }),
            .character_literal => return self.create(ast.Expression, .{ .character_literal = self.advance() }),
            .keyword_true => return self.create(ast.Expression, .{ .bool_literal = .{ .token = self.advance(), .value = true } }),
            .keyword_false => return self.create(ast.Expression, .{ .bool_literal = .{ .token = self.advance(), .value = false } }),
            .colon_colon => {
                // '::Variant': the enum type is implied from context
                _ = self.advance();
                const name = try self.expect(.identifier, "a variant name");
                return self.create(ast.Expression, .{ .implied_variant = name });
            },
            .identifier => {
                var path: std.ArrayList(Token) = .empty;
                try path.append(self.arena, self.advance());
                while (self.current().tag == .colon_colon) {
                    _ = self.advance();
                    try path.append(self.arena, try self.expect(.identifier, "a name"));
                }
                if (self.current().tag == .brace_left and self.allow_struct_init) {
                    const members = try self.parseMemberInits();
                    return self.create(ast.Expression, .{ .struct_init = .{
                        .path = try path.toOwnedSlice(self.arena),
                        .members = members,
                    } });
                }
                return self.create(ast.Expression, .{ .path = try path.toOwnedSlice(self.arena) });
            },
            .brace_left => {
                const members = try self.parseMemberInits();
                return self.create(ast.Expression, .{ .struct_init = .{ .path = null, .members = members } });
            },
            .parenthesis_left => {
                if (self.looksLikeLambda()) {
                    const function = try self.parseFunction();
                    return self.create(ast.Expression, .{ .lambda = .{ .captures = &.{}, .function = function } });
                }
                _ = self.advance();
                const inner = try self.parseInnerExpression();
                _ = try self.expect(.parenthesis_right, "')'");
                return self.create(ast.Expression, .{ .grouped = inner });
            },
            .pipe_pipe => return self.fail(token, "a lambda with no captures omits the capture list entirely: drop the '||'", .{}),
            .pipe => {
                const captures = try self.parseCaptureList();
                const function = try self.parseFunction();
                return self.create(ast.Expression, .{ .lambda = .{ .captures = captures, .function = function } });
            },
            .bracket_left => return self.parseArrayLiteralOrFill(),
            .keyword_if, .keyword_while, .keyword_for, .keyword_match => return self.parseControlFlow(token.tag, true),
            .hash => {
                _ = self.advance();
                const operand = try self.parseComptimeOperand();
                return self.create(ast.Expression, .{ .comptime_expr = operand });
            },
            else => return self.fail(token, "expected an expression, found {s}", .{try self.describe(token)}),
        }
    }

    // what follows '#': a postfix expression, or an inline 'struct' /
    // 'enum' layout giving that layout's '#Type' (section 3.1)
    fn parseComptimeOperand(self: *Parser) Error!*const ast.Expression {
        if (self.current().tag == .keyword_struct or self.current().tag == .keyword_enum) {
            const layout = try self.parseBaseType();
            // '#' binds the whole postfix chain: '#struct { ... }.name()'
            return self.parsePostfixFrom(try self.create(ast.Expression, .{ .type_literal = layout }));
        }
        return self.parsePostfix();
    }

    fn parseMemberInits(self: *Parser) Error![]const ast.MemberInit {
        _ = try self.expect(.brace_left, "'{'");
        var members: std.ArrayList(ast.MemberInit) = .empty;
        while (self.current().tag != .brace_right) {
            _ = try self.expect(.dot, "'.'");
            const member_name = try self.expect(.identifier, "a member name");
            _ = try self.expect(.equal, "'='");
            const value = try self.parseInnerExpression();
            try members.append(self.arena, .{ .name = member_name, .value = value });
            if (self.match(.comma) == null) break;
        }
        _ = try self.expect(.brace_right, "'}'");
        return members.toOwnedSlice(self.arena);
    }

    fn parseArrayLiteralOrFill(self: *Parser) Error!*const ast.Expression {
        _ = try self.expect(.bracket_left, "'['");
        // '[]' is the empty array literal; its element type comes from
        // context (section 4.2)
        if (self.match(.bracket_right) != null) {
            return self.create(ast.Expression, .{ .array_literal = &.{} });
        }
        // '[..end]' is a range generator starting at 0 (section 3.1)
        if (self.match(.dot_dot)) |operator| {
            const end = try self.parseInnerExpression();
            _ = try self.expect(.bracket_right, "']'");
            return self.create(ast.Expression, .{ .array_range = .{ .operator = operator, .start = null, .end = end } });
        }
        const first = try self.parseInnerExpression();
        // '[start..end]' generates the integers start..end-1 (section 3.1)
        if (self.match(.dot_dot)) |operator| {
            const end = try self.parseInnerExpression();
            _ = try self.expect(.bracket_right, "']'");
            return self.create(ast.Expression, .{ .array_range = .{ .operator = operator, .start = first, .end = end } });
        }
        if (self.match(.colon) != null) {
            // the count is any expression; a later stage verifies it is
            // compile-time evaluatable for stack arrays (section 3.1)
            const count = try self.parseInnerExpression();
            _ = try self.expect(.bracket_right, "']'");
            return self.create(ast.Expression, .{ .array_fill = .{ .value = first, .count = count } });
        }
        var elements: std.ArrayList(*const ast.Expression) = .empty;
        try elements.append(self.arena, first);
        while (self.match(.comma) != null) {
            try elements.append(self.arena, try self.parseInnerExpression());
        }
        _ = try self.expect(.bracket_right, "']'");
        return self.create(ast.Expression, .{ .array_literal = try elements.toOwnedSlice(self.arena) });
    }

    // 'as_value': the construct is in value position, where an 'if'
    // branch or a 'match' arm may be a bare expression and a loop may carry
    // an 'else' (section 3.1)
    fn parseIf(self: *Parser, as_value: bool) Error!ast.IfExpression {
        _ = try self.expect(.keyword_if, "'if'");
        _ = try self.expect(.parenthesis_left, "'('");
        const condition = try self.parseInnerExpression();
        _ = try self.expect(.parenthesis_right, "')'");
        // the retired postfix capture: captures moved inside the condition
        // onto the 'is' test itself (section 4.2)
        if (self.current().tag == .pipe) {
            return self.fail(self.current(), "the capture follows the 'is' test inside the condition: 'if (x is ::Some |v|)' (section 4.2)", .{});
        }
        const then_branch = try self.parseIfBranch(as_value);
        var else_branch: ?*const ast.Statement = null;
        if (self.match(.keyword_else) != null) {
            else_branch = try self.parseIfBranch(as_value);
        } else if (as_value) {
            return self.fail(self.current(), "an 'if' used as a value needs an 'else' branch (section 3.1)", .{});
        }
        return .{
            .condition = condition,
            .then_branch = then_branch,
            .else_branch = else_branch,
        };
    }

    // a branch is a block, or in value position a bare expression that
    // yields implicitly; '{' starts a block unless '.' follows (section 3.1)
    fn parseIfBranch(self: *Parser, as_value: bool) Error!*const ast.Statement {
        const token = self.current();
        if (token.tag == .brace_left and self.tokens[self.token_index + 1].tag != .dot) {
            return self.parseBlock();
        }
        if (token.tag == .keyword_if and !as_value) {
            // 'else if' chains in statement form
            return self.create(ast.Statement, .{ .expression = try self.create(ast.Expression, .{ .if_expr = try self.parseIf(false) }) });
        }
        if (!as_value) {
            return self.fail(token, "the body of an 'if' is a block: write '{{ ... }}' (section 3.1)", .{});
        }
        const expression = try self.parseExpression();
        return self.create(ast.Statement, .{ .expression = expression });
    }

    fn parseLoopBody(self: *Parser, construct: []const u8) Error!*const ast.Statement {
        if (self.current().tag != .brace_left) {
            return self.fail(self.current(), "the body of a '{s}' is a block: write '{{ ... }}' (section 3.1)", .{construct});
        }
        return self.parseBlock();
    }

    // a trailing 'else' on a loop is only valid in value position (section 5.3)
    fn parseLoopElse(self: *Parser, as_value: bool) Error!?*const ast.Statement {
        const token = self.current();
        if (self.match(.keyword_else) == null) return null;
        if (!as_value) {
            return self.fail(token, "an 'else' on a loop is only valid when the loop is used as a value (section 5.3)", .{});
        }
        return try self.parseLoopBody("loop else");
    }

    fn parseWhile(self: *Parser, as_value: bool) Error!ast.WhileExpression {
        _ = try self.expect(.keyword_while, "'while'");
        _ = try self.expect(.parenthesis_left, "'('");
        const condition = try self.parseInnerExpression();
        _ = try self.expect(.parenthesis_right, "')'");
        const body = try self.parseLoopBody("while");
        const else_branch = try self.parseLoopElse(as_value);
        return .{ .condition = condition, .body = body, .else_branch = else_branch };
    }

    fn parseFor(self: *Parser, as_value: bool) Error!ast.ForExpression {
        _ = try self.expect(.keyword_for, "'for'");
        _ = try self.expect(.parenthesis_left, "'('");
        var subjects: std.ArrayList(*const ast.Expression) = .empty;
        while (true) {
            try subjects.append(self.arena, try self.parseInnerExpression());
            if (self.match(.comma) == null) break;
        }
        _ = try self.expect(.parenthesis_right, "')'");
        var captures: []const ast.Capture = &.{};
        if (self.current().tag == .pipe or self.current().tag == .pipe_pipe) {
            captures = try self.parseCaptureList();
        }
        const body = try self.parseLoopBody("for");
        const else_branch = try self.parseLoopElse(as_value);
        return .{
            .subjects = try subjects.toOwnedSlice(self.arena),
            .captures = captures,
            .body = body,
            .else_branch = else_branch,
        };
    }

    fn parseMatch(self: *Parser, as_value: bool) Error!ast.MatchExpression {
        _ = try self.expect(.keyword_match, "'match'");
        _ = try self.expect(.parenthesis_left, "'('");
        const subject = try self.parseInnerExpression();
        _ = try self.expect(.parenthesis_right, "')'");
        _ = try self.expect(.brace_left, "'{'");
        var arms: std.ArrayList(ast.MatchArm) = .empty;
        try self.rejectStrayTerminator();
        while (self.current().tag != .brace_right) {
            if (self.current().tag == .end_of_file) {
                return self.fail(self.current(), "expected '}}', found end of file", .{});
            }
            var pattern: ?*const ast.Expression = null;
            if (self.match(.keyword_else) == null) {
                const saved_pipe = self.stop_at_pipe;
                const saved_struct = self.allow_struct_init;
                self.stop_at_pipe = true;
                self.allow_struct_init = false;
                pattern = try self.parseExpression();
                self.stop_at_pipe = saved_pipe;
                self.allow_struct_init = saved_struct;
            }
            var capture: ?ast.Capture = null;
            if (self.current().tag == .pipe) {
                _ = self.advance();
                capture = try self.parseCapture();
                _ = try self.expect(.pipe, "'|'");
            }
            // an arm body is a block, or in value position a bare
            // expression ended with ';' (section 5.3)
            const body: *const ast.Statement = body: {
                if (self.current().tag == .brace_left) break :body try self.parseBlock();
                if (!as_value) {
                    return self.fail(self.current(), "a match arm body is a block: write '{{ ... }}' (a bare expression arm is only valid when the match is used as a value, section 5.3)", .{});
                }
                const expression = try self.parseExpression();
                try self.expectTerminator();
                break :body try self.create(ast.Statement, .{ .expression = expression });
            };
            try arms.append(self.arena, .{ .pattern = pattern, .capture = capture, .body = body });
            try self.rejectStrayTerminator();
        }
        _ = try self.expect(.brace_right, "'}'");
        var else_branch: ?*const ast.Statement = null;
        if (self.current().tag == .keyword_else) {
            const token = self.advance();
            if (!as_value) {
                return self.fail(token, "an external 'else' on a match is only valid when the match is used as a value (section 5.3)", .{});
            }
            else_branch = try self.parseLoopBody("match else");
        }
        return .{
            .subject = subject,
            .arms = try arms.toOwnedSlice(self.arena),
            .else_branch = else_branch,
        };
    }

    fn parseCaptureList(self: *Parser) Error![]const ast.Capture {
        _ = try self.expect(.pipe, "'|'");
        var captures: std.ArrayList(ast.Capture) = .empty;
        while (true) {
            try captures.append(self.arena, try self.parseCapture());
            if (self.match(.comma) == null) break;
        }
        _ = try self.expect(.pipe, "'|'");
        return captures.toOwnedSlice(self.arena);
    }

    // a capture writes its modifier BEFORE the name and carries no type
    // annotation: '|x|', '|&x|', '|&var x|', '|move x|' (section 3.1)
    fn parseCapture(self: *Parser) Error!ast.Capture {
        if (self.match(.keyword_move) != null) {
            const name = try self.expect(.identifier, "a capture name");
            return .{ .modifier = .pointer, .name = name };
        }
        var modifier: ?ast.TypeModifier = null;
        if (self.match(.ampersand) != null) {
            modifier = if (self.match(.keyword_var) != null) .reference_var else .reference;
        } else if (self.current().tag == .asterisk) {
            return self.fail(self.current(), "an owning capture is written '|move name|' (section 3.1)", .{});
        }
        const name = try self.expect(.identifier, "a capture name");
        if (self.current().tag == .colon) {
            return self.fail(self.current(), "a capture carries no type annotation; its modifier goes before the name: '|&{s}|' (section 3.1)", .{name.slice(self.source)});
        }
        return .{ .modifier = modifier, .name = name };
    }

    // a '(' starts a lambda when it opens a parameter list: '()', '(self ...)',
    // or '(name: ...)'; anything else is a parenthesized expression
    fn looksLikeLambda(self: *const Parser) bool {
        switch (self.tokens[self.token_index + 1].tag) {
            .parenthesis_right => {
                const after_close = self.tokens[self.token_index + 2].tag;
                return after_close == .arrow or after_close == .brace_left;
            },
            .keyword_self => return true,
            .identifier => return self.tokens[self.token_index + 2].tag == .colon,
            else => return false,
        }
    }


    fn binaryPrecedence(tag: Token.Tag) ?u8 {
        return switch (tag) {
            .asterisk, .slash, .percent => 100,
            .plus, .minus => 90,
            .shift_left, .shift_right => 80,
            .angle_left, .angle_left_equal, .angle_right, .angle_right_equal => 70,
            .equal_equal, .bang_equal => 60,
            .ampersand => 50,
            .caret => 40,
            .pipe => 30,
            .ampersand_ampersand => 20,
            .pipe_pipe => 10,
            else => null,
        };
    }

    fn assignOperator(tag: Token.Tag) bool {
        return switch (tag) {
            .equal,
            .plus_equal,
            .minus_equal,
            .asterisk_equal,
            .slash_equal,
            .percent_equal,
            .shift_left_equal,
            .shift_right_equal,
            .ampersand_equal,
            .pipe_equal,
            .caret_equal,
            => true,
            else => false,
        };
    }

    fn current(self: *const Parser) Token {
        return self.tokens[self.token_index];
    }

    fn advance(self: *Parser) Token {
        const token = self.current();
        if (token.tag != .end_of_file) {
            self.token_index += 1;
        }
        return token;
    }

    fn match(self: *Parser, tag: Token.Tag) ?Token {
        if (self.current().tag != tag) return null;
        return self.advance();
    }

    fn expect(self: *Parser, tag: Token.Tag, what: []const u8) Error!Token {
        const token = self.current();
        if (token.tag != tag) {
            return self.fail(token, "expected {s}, found {s}", .{ what, try self.describe(token) });
        }
        return self.advance();
    }

    // consumes '>' or the first half of '>>' when closing generic arguments
    fn expectAngleRight(self: *Parser) Error!void {
        if (self.pending_angle) {
            self.pending_angle = false;
            return;
        }
        const token = self.current();
        switch (token.tag) {
            .angle_right => _ = self.advance(),
            .shift_right => {
                _ = self.advance();
                self.pending_angle = true;
            },
            else => return self.fail(token, "expected '>', found {s}", .{try self.describe(token)}),
        }
    }

    // there is no empty statement: a ';' that terminates nothing is an
    // error (section 3.1)
    fn rejectStrayTerminator(self: *Parser) Error!void {
        if (self.current().tag == .semicolon) {
            return self.fail(self.current(), "this ';' terminates nothing: a block statement ends at its '}}' and takes no ';' (section 3.1)", .{});
        }
    }

    // every statement and declaration ends with ';' (section 3.1)
    fn expectTerminator(self: *Parser) Error!void {
        if (self.match(.semicolon) != null) return;
        const token = self.current();
        return self.fail(token, "expected ';' before {s}", .{try self.describe(token)});
    }

    fn create(self: *Parser, comptime T: type, value: T) Error!*const T {
        const node = try self.arena.create(T);
        node.* = value;
        return node;
    }

    fn fail(self: *Parser, token: Token, comptime format: []const u8, arguments: anytype) Error {
        self.failure = .{
            .span = token.location,
            .message = try std.fmt.allocPrint(self.arena, format, arguments),
        };
        return error.ParseError;
    }

    fn describe(self: *Parser, token: Token) Error![]const u8 {
        if (token.tag.lexeme()) |text| {
            return std.fmt.allocPrint(self.arena, "'{s}'", .{text});
        }
        return switch (token.tag) {
            .identifier => std.fmt.allocPrint(self.arena, "identifier '{s}'", .{token.slice(self.source)}),
            .end_of_file => "end of file",
            else => std.fmt.allocPrint(self.arena, "'{s}'", .{token.slice(self.source)}),
        };
    }
};

const testing = std.testing;

fn parseForTest(arena: *std.heap.ArenaAllocator, source: []const u8) Parser.Error!ast.Module {
    var tokenizer = tokenizer_module.Tokenizer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    try tokenizer.tokenizeAll(arena.allocator(), &tokens);
    var parser = Parser.init(arena.allocator(), source, tokens.items);
    return parser.parseModule();
}

// parses a module that must fail, returning the failure message
fn parseFailure(arena: *std.heap.ArenaAllocator, source: []const u8) ![]const u8 {
    var tokenizer = tokenizer_module.Tokenizer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    try tokenizer.tokenizeAll(arena.allocator(), &tokens);
    var parser = Parser.init(arena.allocator(), source, tokens.items);
    try testing.expectError(error.ParseError, parser.parseModule());
    return parser.failure.?.message;
}

test "imports and definitions parse" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\import std::option;
        \\import std::vec as vectors;
        \\
        \\pub fn main() -> i32 {
        \\    return 0;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    try testing.expectEqual(@as(usize, 2), module.imports.len);
    try testing.expectEqual(@as(usize, 2), module.imports[0].path.len);
    try testing.expectEqualStrings("vectors", module.imports[1].alias.?.slice(source));
    try testing.expectEqual(@as(usize, 1), module.definitions.len);
    try testing.expectEqual(ast.Visibility.public, module.definitions[0].visibility);
    const fn_def = module.definitions[0].kind.fn_def;
    try testing.expectEqualStrings("main", fn_def.name.slice(source));
    try testing.expectEqual(@as(usize, 1), fn_def.function.body.block.len);
}

test "type definition with interfaces, generics, and nested angle close" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\type Matrix<T: Number> : Serializable, Iterable = struct {
        \\    rows: Vec<Vec<T>>,
        \\    tag: enum { A, B: u32 },
        \\};
    ;
    const module = try parseForTest(&arena, source);
    const type_def = module.definitions[0].kind.type_def;
    try testing.expectEqualStrings("Number", type_def.type_parameters[0].constraint.?.name.slice(source));
    try testing.expectEqual(@as(usize, 2), type_def.interfaces.len);
    const members = type_def.base.struct_type;
    const rows_type = members[0].member_type.named;
    try testing.expectEqualStrings("Vec", rows_type.path[0].slice(source));
    try testing.expectEqual(@as(usize, 1), rows_type.type_arguments.len);
    try testing.expectEqual(@as(usize, 2), members[1].member_type.enum_type.len);
}

test "statements span lines freely and share lines with semicolons" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var wide = 1 +
        \\        2 + (3 *
        \\        4); var narrow = 5;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    try testing.expectEqual(@as(usize, 2), body.len);
    const value = body[0].var_def.value;
    // ((1 + 2) + (3 * 4)): left-associative with the group on the right
    try testing.expectEqual(Token.Tag.plus, value.binary.operator.tag);
    try testing.expectEqual(Token.Tag.plus, value.binary.left.binary.operator.tag);
    try testing.expectEqual(Token.Tag.asterisk, value.binary.right.grouped.binary.operator.tag);
}

test "generic call versus comparison" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var a = make<u64>(7);
        \\    var b = 3 < lengths;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const call = body[0].var_def.value.call;
    try testing.expectEqual(@as(usize, 1), call.type_arguments.len);
    try testing.expectEqual(@as(usize, 1), call.arguments.len);
    try testing.expectEqual(Token.Tag.angle_left, body[1].var_def.value.binary.operator.tag);
}

test "match arms with patterns, captures, and external else" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var label = match (value) {
        \\        Option::Some |&v| { yield v; }
        \\        Option::None { yield fallback; }
        \\        else { yield other; }
        \\    } else {
        \\        yield missing;
        \\    };
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const match_expr = body[0].var_def.value.match_expr;
    try testing.expectEqual(@as(usize, 3), match_expr.arms.len);
    try testing.expectEqual(@as(usize, 2), match_expr.arms[0].pattern.?.path.len);
    const capture = match_expr.arms[0].capture.?;
    try testing.expectEqual(ast.TypeModifier.reference, capture.modifier.?);
    try testing.expect(match_expr.arms[2].pattern == null);
    try testing.expect(match_expr.else_branch != null);
}

test "if with is test, owning capture, and value form with bare branches" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    if (h is Holder::Boxed |move boxed|) {
        \\        use(boxed);
        \\    } else {
        \\        nothing();
        \\    }
        \\    var w = if (cond) a else { yield b; };
        \\    var chain = if (a) x else if (b) y else z;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const if_expr = body[0].expression.if_expr;
    try testing.expectEqual(Token.Tag.keyword_is, if_expr.condition.cast.operator.tag);
    try testing.expectEqual(ast.TypeModifier.pointer, if_expr.condition.cast.capture.?.modifier.?);
    try testing.expect(if_expr.else_branch != null);
    const value_if = body[1].var_def.value.if_expr;
    try testing.expect(value_if.then_branch.* == .expression);
    try testing.expect(value_if.else_branch.?.* == .block);
    const chain = body[2].var_def.value.if_expr;
    try testing.expect(chain.else_branch.?.expression.* == .if_expr);
}

test "statement-position constructs reject bare branches and loop else" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const cases = [_]struct { source: []const u8, needle: []const u8 }{
        .{ .source = "fn f() { if (c) doA(); }", .needle = "the body of an 'if' is a block" },
        .{ .source = "fn f() { while (c) { } else { } }", .needle = "only valid when the loop is used as a value" },
        .{ .source = "fn f() { match (c) { 0 run(); } }", .needle = "a match arm body is a block" },
        .{ .source = "fn f() { match (c) { 0 { } } else { } }", .needle = "external 'else' on a match is only valid" },
        .{ .source = "fn f() { var x = if (c) a; }", .needle = "needs an 'else' branch" },
    };
    for (cases) |case| {
        const message = try parseFailure(&arena, case.source);
        try testing.expect(std.mem.indexOf(u8, message, case.needle) != null);
    }
}

test "lambda forms and capture annotations" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    const doubler = |&var x, y, move z| (a: i64) -> i64 { return a; };
        \\    const plain = (b: i64) { use(b); };
        \\    const empty = () { run(); };
        \\    var grouped = (a + b);
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const captures = body[0].var_def.value.lambda.captures;
    try testing.expectEqual(@as(usize, 3), captures.len);
    try testing.expectEqual(ast.TypeModifier.reference_var, captures[0].modifier.?);
    try testing.expect(captures[1].modifier == null);
    try testing.expectEqual(ast.TypeModifier.pointer, captures[2].modifier.?);
    try testing.expectEqual(@as(usize, 1), body[1].var_def.value.lambda.function.parameters.len);
    try testing.expectEqual(@as(usize, 0), body[2].var_def.value.lambda.captures.len);
    try testing.expectEqual(Token.Tag.plus, body[3].var_def.value.grouped.binary.operator.tag);
}

test "loops carry an else in value position and match arms may be bare" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var found = while (running) {
        \\        break if (cond) a else b;
        \\    } else {
        \\        yield 0;
        \\    };
        \\    var y = match (state) {
        \\        ::Idle 0;
        \\        ::Busy |load| load;
        \\        else -1;
        \\    };
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const loop = body[0].var_def.value.while_expr;
    try testing.expect(loop.else_branch != null);
    try testing.expect(loop.body.block[0].break_stmt.value.?.* == .if_expr);
    const match_expr = body[1].var_def.value.match_expr;
    try testing.expectEqual(@as(usize, 3), match_expr.arms.len);
    try testing.expect(match_expr.arms[0].body.* == .expression);
    try testing.expect(match_expr.arms[1].capture != null);
    try testing.expect(match_expr.arms[2].pattern == null);
}

test "continue parses as a bare statement" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    while (running) {
        \\        if (skip) { continue; }
        \\        work();
        \\    }
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const loop = body[0].expression.while_expr;
    const then_branch = loop.body.block[0].expression.if_expr.then_branch;
    try testing.expect(then_branch.block[0].* == .continue_stmt);
}

test "casts chain and pointer assignments parse" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var p: *var Packet = new Packet { .id = 1, .payload = new [0 : n] };
        \\    var q: *Packet = move p;
        \\    var f2 = raw as f32 to f64;
        \\    items[2] += one().two;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    try testing.expectEqual(ast.TypeModifier.pointer_var, body[0].var_def.declared_type.?.modified.modifier);
    try testing.expectEqual(Token.Tag.keyword_new, body[0].var_def.value.unary.operator.tag);
    try testing.expectEqual(Token.Tag.keyword_move, body[1].var_def.value.unary.operator.tag);
    const outer_cast = body[2].var_def.value.cast;
    try testing.expectEqual(Token.Tag.keyword_to, outer_cast.operator.tag);
    try testing.expectEqual(Token.Tag.keyword_as, outer_cast.operand.cast.operator.tag);
    try testing.expectEqual(Token.Tag.plus_equal, body[3].assign.operator.tag);
    try testing.expect(body[3].assign.target.* == .index);
}

test "a semicolon that terminates nothing is an error" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const sources = [_][]const u8{
        "fn f() { var x = 1;; return x; }",
        "fn f() { if (c) { } ; }",
        "fn f() { return 1; };",
    };
    for (sources) |source| {
        const message = try parseFailure(&arena, source);
        try testing.expect(std.mem.indexOf(u8, message, "terminates nothing") != null);
    }
}

test "missing semicolon fails with a clear message" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "fn f() { var x = 1 var y = 2; }";
    const message = try parseFailure(&arena, source);
    try testing.expect(std.mem.indexOf(u8, message, "expected ';'") != null);
}

test "declaration-only macros parse without a body" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\pub macro type_of(value) -> #Type;
        \\macro pair(left, right) -> i64;
        \\macro doubled(x: i64) -> i64 { return x; }
    ;
    const module = try parseForTest(&arena, source);
    try testing.expectEqual(@as(usize, 3), module.definitions.len);
    try testing.expect(module.definitions[0].kind.macro_def.body == null);
    try testing.expect(module.definitions[0].kind.macro_def.parameters[0].parameter_type == null);
    try testing.expectEqual(@as(usize, 2), module.definitions[1].kind.macro_def.parameters.len);
    try testing.expect(module.definitions[2].kind.macro_def.body != null);
    try testing.expect(module.definitions[2].kind.macro_def.parameters[0].parameter_type != null);
}

test "a macro declares its result type" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "macro answer() { return 42; }";
    const message = try parseFailure(&arena, source);
    try testing.expect(std.mem.indexOf(u8, message, "declares its result type") != null);
}

test "a macro body requires typed parameters" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "macro bad(x) -> i64 { return x; }";
    const message = try parseFailure(&arena, source);
    try testing.expect(std.mem.indexOf(u8, message, "types its parameters") != null);
}

test "an empty capture list fails with the omitted form" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "fn f() { const empty = || () { run(); }; }";
    const message = try parseFailure(&arena, source);
    try testing.expect(std.mem.indexOf(u8, message, "drop the '||'") != null);
}

test "an annotated capture fails with the prefix form" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "fn f() { const doubler = |x: &var| (a: i64) -> i64 { return a; }; }";
    const message = try parseFailure(&arena, source);
    try testing.expect(std.mem.indexOf(u8, message, "'|&x|'") != null);
}

test "extern, interface, and macro definitions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\extern printf(format: &[u8], ...) -> i32;
        \\extern exit(code: i32);
        \\extern getpid() -> i32;
        \\interface Shape {
        \\    fn area(self: &) -> f32;
        \\    fn scale(self: &var, factor: f32);
        \\}
        \\macro readTypeFromJson(path: &[u8]) -> #Type {
        \\    return path;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    try testing.expectEqual(@as(usize, 5), module.definitions.len);
    const printf_def = module.definitions[0].kind.extern_def;
    try testing.expect(printf_def.variadic);
    try testing.expectEqual(@as(usize, 1), printf_def.parameters.len);
    try testing.expect(printf_def.return_type != null);
    try testing.expect(module.definitions[1].kind.extern_def.return_type == null);
    try testing.expect(!module.definitions[2].kind.extern_def.variadic);
    const shape = module.definitions[3].kind.interface_def;
    try testing.expectEqual(@as(usize, 2), shape.functions.len);
    try testing.expect(shape.functions[1].return_type == null);
    try testing.expectEqualStrings("readTypeFromJson", module.definitions[4].kind.macro_def.name.slice(source));
}

test "type expressions cover all base forms" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f(fixed: [u8 : 4], slice: &[u8], heap: *[u8], callback: (u32, u32) -> u32) {
        \\    var made: std::vec::Vec<u32> = make();
        \\    var reflected: #type_of(made) = made;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const parameters = module.definitions[0].kind.fn_def.function.parameters;
    try testing.expect(parameters[0].parameter_type.array.length != null);
    try testing.expectEqual(ast.TypeModifier.reference, parameters[1].parameter_type.modified.modifier);
    try testing.expect(parameters[1].parameter_type.modified.child.array.length == null);
    try testing.expectEqual(ast.TypeModifier.pointer, parameters[2].parameter_type.modified.modifier);
    const callback_type = parameters[3].parameter_type.function;
    try testing.expectEqual(@as(usize, 2), callback_type.parameter_types.len);
    try testing.expect(callback_type.return_type != null);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    try testing.expectEqual(@as(usize, 3), body[0].var_def.declared_type.?.named.path.len);
    try testing.expect(body[1].var_def.declared_type.?.* == .comptime_type);
}

test "for with multiple subjects and captures" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var r = for (first, second) |a, &var b| {
        \\        consume(a, b);
        \\    } else {
        \\        yield 0;
        \\    };
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const for_expr = module.definitions[0].kind.fn_def.function.body.block[0].var_def.value.for_expr;
    try testing.expectEqual(@as(usize, 2), for_expr.subjects.len);
    try testing.expectEqual(@as(usize, 2), for_expr.captures.len);
    try testing.expect(for_expr.captures[0].modifier == null);
    try testing.expectEqual(ast.TypeModifier.reference_var, for_expr.captures[1].modifier.?);
    try testing.expect(for_expr.else_branch != null);
}

test "array ranges parse with and without a start" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var full = [2..10];
        \\    var open = [..count];
        \\    var fill = [0 : 4];
        \\    for ([..5]) |i| { consume(i); }
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const full = body[0].var_def.value.array_range;
    try testing.expect(full.start != null);
    try testing.expect(full.end.* == .integer_literal);
    const open = body[1].var_def.value.array_range;
    try testing.expect(open.start == null);
    try testing.expect(open.end.* == .path);
    // '[v : n]' stays an array fill
    try testing.expect(body[2].var_def.value.* == .array_fill);
    const for_expr = body[3].expression.for_expr;
    try testing.expect(for_expr.subjects[0].* == .array_range);
}

test "comptime prefix binds as a postfix expression" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    var a = #a + b;
        \\    var c = #compute(x).field;
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    // '#a + b' is '(#a) + b', not '#(a + b)'
    const sum = body[0].var_def.value.binary;
    try testing.expect(sum.left.* == .comptime_expr);
    // '#compute(x).field' marks the whole postfix chain
    try testing.expect(body[1].var_def.value.comptime_expr.* == .member);
}

test "match-arm pattern limits do not leak into nested expressions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    match (value) {
        \\        (FLAG_A | FLAG_B) { handle(); }
        \\        masked(Config { .raw = bits | extra }) { other(); }
        \\        else { fallback(); }
        \\    }
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const match_expr = module.definitions[0].kind.fn_def.function.body.block[0].expression.match_expr;
    try testing.expectEqual(@as(usize, 3), match_expr.arms.len);
    try testing.expectEqual(Token.Tag.pipe, match_expr.arms[0].pattern.?.grouped.binary.operator.tag);
    const argument = match_expr.arms[1].pattern.?.call.arguments[0];
    try testing.expect(argument.* == .struct_init);
}

test "import after a definition fails with a clear message" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "fn f() { }\nimport std::vec;";
    const message = try parseFailure(&arena, source);
    try testing.expect(std.mem.indexOf(u8, message, "imports must appear before") != null);
}

test "stray token at top level fails with a definition message" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "const x = 1;\n";
    const message = try parseFailure(&arena, source);
    try testing.expect(std.mem.indexOf(u8, message, "expected a definition") != null);
}
