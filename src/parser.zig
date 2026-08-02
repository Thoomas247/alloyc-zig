//! Parser for the Alloy language, implementing section 2 (Syntactic Grammar)
//! of LANGUAGE_SPEC.md. Consumes the token list produced by the tokenizer and
//! builds the tree defined in ast.zig.
//!
//! Statements are terminated by ';' and implicitly before '}', 'else', and
//! end of file. Terminator checks verify but never consume the semicolon;
//! statement-list loops skip semicolon runs, which also makes redundant
//! semicolons parse as empty statements.

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

        self.skipSemicolons();
        while (self.current().tag == .keyword_import) {
            try imports.append(self.arena, try self.parseImport());
            self.skipSemicolons();
        }
        while (self.current().tag != .end_of_file) {
            try definitions.append(self.arena, try self.parseDefinition());
            self.skipSemicolons();
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
        var interfaces: std.ArrayList(Token) = .empty;
        if (self.match(.colon) != null) {
            try interfaces.append(self.arena, try self.expect(.identifier, "an interface name"));
            while (self.match(.comma) != null) {
                try interfaces.append(self.arena, try self.expect(.identifier, "an interface name"));
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
        _ = try self.expect(.brace_left, "'{'");
        var functions: std.ArrayList(ast.InterfaceFn) = .empty;
        self.skipSemicolons();
        while (self.current().tag != .brace_right) {
            _ = try self.expect(.keyword_fn, "'fn' or '}'");
            const fn_name = try self.expect(.identifier, "a function name");
            const parameter_list = try self.parseParameterList(false);
            var return_type: ?*const ast.TypeExpression = null;
            if (self.match(.arrow) != null) {
                return_type = try self.parseType();
            }
            try self.expectTerminator();
            try functions.append(self.arena, .{
                .name = fn_name,
                .parameters = parameter_list.parameters,
                .return_type = return_type,
            });
            self.skipSemicolons();
        }
        _ = try self.expect(.brace_right, "'}'");
        return .{ .name = name, .functions = try functions.toOwnedSlice(self.arena) };
    }

    fn parseMacroDef(self: *Parser) Error!ast.MacroDef {
        _ = try self.expect(.keyword_macro, "'macro'");
        const name = try self.expect(.identifier, "a macro name");
        const parameter_list = try self.parseParameterList(false);
        const body = try self.parseBlock();
        return .{ .name = name, .parameters = parameter_list.parameters, .body = body };
    }

    fn parseTypeParameters(self: *Parser) Error![]const ast.TypeParameter {
        var type_parameters: std.ArrayList(ast.TypeParameter) = .empty;
        if (self.match(.angle_left) != null) {
            while (true) {
                const parameter_name = try self.expect(.identifier, "a type parameter name");
                var constraint: ?Token = null;
                if (self.match(.colon) != null) {
                    constraint = try self.expect(.identifier, "a constraint name");
                }
                try type_parameters.append(self.arena, .{ .name = parameter_name, .constraint = constraint });
                if (self.match(.comma) == null) break;
            }
            try self.expectAngleRight();
        }
        return type_parameters.toOwnedSlice(self.arena);
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
                const expression = try self.parsePostfix();
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
        var type_arguments: std.ArrayList(*const ast.TypeExpression) = .empty;
        if (self.match(.angle_left) != null) {
            while (true) {
                try type_arguments.append(self.arena, try self.parseType());
                // a pending half of '>>' closes this list before any comma
                if (self.pending_angle) break;
                if (self.match(.comma) == null) break;
            }
            try self.expectAngleRight();
        }
        return .{
            .path = try path.toOwnedSlice(self.arena),
            .type_arguments = try type_arguments.toOwnedSlice(self.arena),
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
        self.skipSemicolons();
        while (self.current().tag != .brace_right) {
            if (self.current().tag == .end_of_file) {
                return self.fail(self.current(), "expected '}}', found end of file", .{});
            }
            try statements.append(self.arena, try self.parseStatement());
            self.skipSemicolons();
        }
        _ = try self.expect(.brace_right, "'}'");
        return self.create(ast.Statement, .{ .block = try statements.toOwnedSlice(self.arena) });
    }

    fn parseStatement(self: *Parser) Error!*const ast.Statement {
        const token = self.current();
        switch (token.tag) {
            .keyword_var, .keyword_const => return self.parseVarDef(),
            .brace_left => return self.parseBlock(),
            .keyword_break => {
                const keyword = self.advance();
                var value: ?*const ast.Expression = null;
                if (!self.atTerminator()) {
                    value = try self.parseExpression();
                }
                // a block-like operand is self-terminating per section 2.1
                if (value == null or !isBlockLike(value.?)) {
                    try self.expectTerminator();
                }
                return self.create(ast.Statement, .{ .break_stmt = .{ .keyword = keyword, .value = value } });
            },
            .keyword_yield => {
                const keyword = self.advance();
                const value = try self.parseExpression();
                if (!isBlockLike(value)) {
                    try self.expectTerminator();
                }
                return self.create(ast.Statement, .{ .yield_stmt = .{ .keyword = keyword, .value = value } });
            },
            .keyword_return => {
                const keyword = self.advance();
                var value: ?*const ast.Expression = null;
                if (!self.atTerminator()) {
                    value = try self.parseExpression();
                }
                try self.expectTerminator();
                return self.create(ast.Statement, .{ .return_stmt = .{ .keyword = keyword, .value = value } });
            },
            else => {
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
                // a block-like expression statement needs no semicolon
                if (!isBlockLike(expression)) {
                    try self.expectTerminator();
                }
                return self.create(ast.Statement, .{ .expression = expression });
            },
        }
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
                    operand = try self.create(ast.Expression, .{ .cast = .{
                        .operator = operator,
                        .operand = operand,
                        .target = target,
                    } });
                },
                .keyword_as, .keyword_to => {
                    const operator = self.advance();
                    const target = try self.parseType();
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

    fn parseUnary(self: *Parser) Error!*const ast.Expression {
        switch (self.current().tag) {
            .minus, .tilde, .bang, .ampersand, .keyword_new, .keyword_move => {
                const operator = self.advance();
                const operand = try self.parseUnary();
                return self.create(ast.Expression, .{ .unary = .{ .operator = operator, .operand = operand } });
            },
            else => return self.parsePostfix(),
        }
    }

    fn parsePostfix(self: *Parser) Error!*const ast.Expression {
        var expression = try self.parsePrimary();
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
                    // 'arr[start..end]' and 'arr[..end]' borrow a subslice;
                    // a lone subscript is an element index (section 2.1)
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
        var type_arguments: std.ArrayList(*const ast.TypeExpression) = .empty;
        while (true) {
            try type_arguments.append(self.arena, try self.parseType());
            if (self.pending_angle) break;
            if (self.match(.comma) == null) break;
        }
        try self.expectAngleRight();
        if (self.current().tag != .parenthesis_left) {
            return self.fail(self.current(), "expected '(' after generic arguments", .{});
        }
        const arguments = try self.parseCallArguments();
        return self.create(ast.Expression, .{ .call = .{
            .callee = callee,
            .type_arguments = try type_arguments.toOwnedSlice(self.arena),
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
            .keyword_if => return self.create(ast.Expression, .{ .if_expr = try self.parseIf() }),
            .keyword_while => return self.create(ast.Expression, .{ .while_expr = try self.parseWhile() }),
            .keyword_for => return self.create(ast.Expression, .{ .for_expr = try self.parseFor() }),
            .keyword_match => return self.create(ast.Expression, .{ .match_expr = try self.parseMatch() }),
            .hash => {
                _ = self.advance();
                const operand = try self.parsePostfix();
                return self.create(ast.Expression, .{ .comptime_expr = operand });
            },
            else => return self.fail(token, "expected an expression, found {s}", .{try self.describe(token)}),
        }
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
        // '[..end]' is a range generator starting at 0 (section 2.1)
        if (self.match(.dot_dot)) |operator| {
            const end = try self.parseInnerExpression();
            _ = try self.expect(.bracket_right, "']'");
            return self.create(ast.Expression, .{ .array_range = .{ .operator = operator, .start = null, .end = end } });
        }
        const first = try self.parseInnerExpression();
        // '[start..end]' generates the integers start..end-1 (section 2.1)
        if (self.match(.dot_dot)) |operator| {
            const end = try self.parseInnerExpression();
            _ = try self.expect(.bracket_right, "']'");
            return self.create(ast.Expression, .{ .array_range = .{ .operator = operator, .start = first, .end = end } });
        }
        if (self.match(.colon) != null) {
            // the count is any expression; a later stage verifies it is
            // compile-time evaluatable for stack arrays (section 2.1)
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

    fn parseIf(self: *Parser) Error!ast.IfExpression {
        _ = try self.expect(.keyword_if, "'if'");
        _ = try self.expect(.parenthesis_left, "'('");
        const condition = try self.parseInnerExpression();
        _ = try self.expect(.parenthesis_right, "')'");
        var capture: ?ast.Capture = null;
        if (self.current().tag == .pipe) {
            _ = self.advance();
            capture = try self.parseCapture();
            _ = try self.expect(.pipe, "'|'");
        }
        const then_branch = try self.parseStatement();
        const else_branch = try self.parseElseBranch();
        return .{
            .condition = condition,
            .capture = capture,
            .then_branch = then_branch,
            .else_branch = else_branch,
        };
    }

    fn parseWhile(self: *Parser) Error!ast.WhileExpression {
        _ = try self.expect(.keyword_while, "'while'");
        _ = try self.expect(.parenthesis_left, "'('");
        const condition = try self.parseInnerExpression();
        _ = try self.expect(.parenthesis_right, "')'");
        const body = try self.parseStatement();
        const else_branch = try self.parseElseBranch();
        return .{ .condition = condition, .body = body, .else_branch = else_branch };
    }

    fn parseFor(self: *Parser) Error!ast.ForExpression {
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
        const body = try self.parseStatement();
        const else_branch = try self.parseElseBranch();
        return .{
            .subjects = try subjects.toOwnedSlice(self.arena),
            .captures = captures,
            .body = body,
            .else_branch = else_branch,
        };
    }

    fn parseMatch(self: *Parser) Error!ast.MatchExpression {
        _ = try self.expect(.keyword_match, "'match'");
        _ = try self.expect(.parenthesis_left, "'('");
        const subject = try self.parseInnerExpression();
        _ = try self.expect(.parenthesis_right, "')'");
        _ = try self.expect(.brace_left, "'{'");
        var arms: std.ArrayList(ast.MatchArm) = .empty;
        self.skipSemicolons();
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
            const body = try self.parseStatement();
            try arms.append(self.arena, .{ .pattern = pattern, .capture = capture, .body = body });
            self.skipSemicolons();
        }
        _ = try self.expect(.brace_right, "'}'");
        const else_branch = try self.parseElseBranch();
        return .{
            .subject = subject,
            .arms = try arms.toOwnedSlice(self.arena),
            .else_branch = else_branch,
        };
    }

    // accepts 'else' directly after the previous branch or past its semicolon
    fn parseElseBranch(self: *Parser) Error!?*const ast.Statement {
        if (self.current().tag == .keyword_else) {
            _ = self.advance();
            return try self.parseStatement();
        }
        var index = self.token_index;
        while (self.tokens[index].tag == .semicolon) index += 1;
        if (index != self.token_index and self.tokens[index].tag == .keyword_else) {
            self.token_index = index + 1;
            return try self.parseStatement();
        }
        return null;
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

    fn parseCapture(self: *Parser) Error!ast.Capture {
        // a modifier ahead of the name is the retired prefix form: captures
        // annotate after the name, like every other binding (section 2.1)
        if (self.current().tag == .asterisk or self.current().tag == .ampersand) {
            const token = self.current();
            const modifier = self.parseTypeModifier().?;
            if (self.current().tag == .identifier) {
                const name = self.current().slice(self.source);
                return self.fail(token, "a capture is annotated after its name: write '|{s}: {s}|'", .{ name, modifier.lexeme() });
            }
            return self.fail(token, "expected a capture name, found {s}", .{try self.describe(token)});
        }
        const name = try self.expect(.identifier, "a capture name");
        if (self.match(.colon) == null) {
            return .{ .modifier = null, .name = name, .annotation = null };
        }
        // a bare modifier annotation ('|a: &|', '|a: *var|') ends at '|' or ','
        if (self.current().tag == .asterisk or self.current().tag == .ampersand) {
            const after_modifier = self.peekPastModifier();
            if (after_modifier == .pipe or after_modifier == .comma) {
                const modifier = self.parseTypeModifier().?;
                return .{ .modifier = modifier, .name = name, .annotation = null };
            }
        }
        const annotation = try self.parseType();
        return .{ .modifier = null, .name = name, .annotation = annotation };
    }

    // looks past '&' / '*' and an optional 'var' without consuming anything
    fn peekPastModifier(self: *const Parser) Token.Tag {
        var index = self.token_index + 1;
        if (self.tokens[index].tag == .keyword_var) {
            index += 1;
        }
        return self.tokens[index].tag;
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

    fn isBlockLike(expression: *const ast.Expression) bool {
        return switch (expression.*) {
            .if_expr, .while_expr, .for_expr, .match_expr => true,
            else => false,
        };
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

    // redundant semicolons are empty statements and are skipped silently
    fn skipSemicolons(self: *Parser) void {
        while (self.tokens[self.token_index].tag == .semicolon) {
            self.token_index += 1;
        }
    }

    // a statement may end at ';', or implicitly before '}', 'else', or end
    // of file; the terminator is verified but never consumed here
    fn atTerminator(self: *const Parser) bool {
        return switch (self.current().tag) {
            .semicolon, .brace_right, .keyword_else, .end_of_file => true,
            else => false,
        };
    }

    fn expectTerminator(self: *Parser) Error!void {
        if (self.atTerminator()) return;
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
    try testing.expectEqualStrings("Number", type_def.type_parameters[0].constraint.?.slice(source));
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
        \\        Option::Some |v: &| { yield v; }
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

test "if with is test, owning capture, and else past a semicolon" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    if (h is Holder::Boxed) |boxed: *| {
        \\        use(boxed);
        \\    } else {
        \\        nothing();
        \\    }
        \\    if (cond) run(); else halt();
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const if_expr = body[0].expression.if_expr;
    try testing.expectEqual(Token.Tag.keyword_is, if_expr.condition.cast.operator.tag);
    try testing.expectEqual(ast.TypeModifier.pointer, if_expr.capture.?.modifier.?);
    try testing.expect(if_expr.else_branch != null);
    try testing.expect(body[1].expression.if_expr.else_branch != null);
}

test "lambda forms and capture annotations" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    const doubler = |x: &var, y: &u32, z: *var| (a: i64) -> i64 { return a; };
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
    try testing.expect(captures[1].annotation != null);
    try testing.expectEqual(ast.TypeModifier.pointer_var, captures[2].modifier.?);
    try testing.expectEqual(@as(usize, 1), body[1].var_def.value.lambda.function.parameters.len);
    try testing.expectEqual(@as(usize, 0), body[2].var_def.value.lambda.captures.len);
    try testing.expectEqual(Token.Tag.plus, body[3].var_def.value.grouped.binary.operator.tag);
}

test "break with block-like operand is self-terminating" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    while (running) {
        \\        break if (cond) yield a else yield b
        \\    }
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    const loop_body = body[0].expression.while_expr.body.block;
    const break_value = loop_body[0].break_stmt.value.?;
    try testing.expect(break_value.* == .if_expr);
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

test "redundant semicolons parse as empty statements" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\fn f() {
        \\    ;;
        \\    var x = 1;;
        \\    return x;
        \\};
    ;
    const module = try parseForTest(&arena, source);
    const body = module.definitions[0].kind.fn_def.function.body.block;
    try testing.expectEqual(@as(usize, 2), body.len);
}

test "missing semicolon fails with a clear message" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "fn f() { var x = 1 var y = 2; }";
    var tokenizer = tokenizer_module.Tokenizer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    try tokenizer.tokenizeAll(arena.allocator(), &tokens);
    var parser = Parser.init(arena.allocator(), source, tokens.items);
    try testing.expectError(error.ParseError, parser.parseModule());
    try testing.expect(std.mem.indexOf(u8, parser.failure.?.message, "expected ';'") != null);
}

test "an empty capture list fails with the omitted form" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "fn f() { const empty = || () { run(); }; }";
    var tokenizer = tokenizer_module.Tokenizer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    try tokenizer.tokenizeAll(arena.allocator(), &tokens);
    var parser = Parser.init(arena.allocator(), source, tokens.items);
    try testing.expectError(error.ParseError, parser.parseModule());
    try testing.expect(std.mem.indexOf(u8, parser.failure.?.message, "drop the '||'") != null);
}

test "a modifier before a capture name fails with the annotated form" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "fn f() { const doubler = |&var x| (a: i64) -> i64 { return a; }; }";
    var tokenizer = tokenizer_module.Tokenizer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    try tokenizer.tokenizeAll(arena.allocator(), &tokens);
    var parser = Parser.init(arena.allocator(), source, tokens.items);
    try testing.expectError(error.ParseError, parser.parseModule());
    try testing.expect(std.mem.indexOf(u8, parser.failure.?.message, "'|x: &var|'") != null);
}

test "extern, interface, and macro definitions" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source =
        \\extern printf(format: &[u8], ...) -> i32;
        \\extern exit(code: i32);
        \\extern getpid() -> i32;
        \\interface Shape {
        \\    fn area() -> f32;
        \\    fn scale(factor: f32)
        \\}
        \\macro readTypeFromJson(path: &[u8]) {
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
        \\    for (first, second) |a, b: &var| {
        \\        consume(a, b);
        \\    } else {
        \\        break 0;
        \\    }
        \\}
    ;
    const module = try parseForTest(&arena, source);
    const for_expr = module.definitions[0].kind.fn_def.function.body.block[0].expression.for_expr;
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
    var tokenizer = tokenizer_module.Tokenizer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    try tokenizer.tokenizeAll(arena.allocator(), &tokens);
    var parser = Parser.init(arena.allocator(), source, tokens.items);
    try testing.expectError(error.ParseError, parser.parseModule());
    try testing.expect(std.mem.indexOf(u8, parser.failure.?.message, "imports must appear before") != null);
}

test "stray token at top level fails with a definition message" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const source = "const x = 1;\n";
    var tokenizer = tokenizer_module.Tokenizer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    try tokenizer.tokenizeAll(arena.allocator(), &tokens);
    var parser = Parser.init(arena.allocator(), source, tokens.items);
    try testing.expectError(error.ParseError, parser.parseModule());
    try testing.expect(std.mem.indexOf(u8, parser.failure.?.message, "expected a definition") != null);
}
