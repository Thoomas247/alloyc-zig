//! Tokenizer for the Alloy language, implementing section 1 (Lexical Grammar) of
//! LANGUAGE_SPEC.md.

const std = @import("std");

/// Decodes the escape sequences (section 2.6) of a string or character
/// literal body (the text between the quotes) into raw bytes. Shared by the
/// interpreter and code generation so both decode literals identically.
pub fn unescape(allocator: std.mem.Allocator, text: []const u8) error{OutOfMemory}![]const u8 {
    var bytes: std.ArrayList(u8) = .empty;
    var index: usize = 0;
    while (index < text.len) : (index += 1) {
        const byte = text[index];
        if (byte != '\\' or index + 1 >= text.len) {
            try bytes.append(allocator, byte);
            continue;
        }
        index += 1;
        switch (text[index]) {
            'n' => try bytes.append(allocator, '\n'),
            't' => try bytes.append(allocator, '\t'),
            'r' => try bytes.append(allocator, '\r'),
            '0' => try bytes.append(allocator, 0),
            '\\' => try bytes.append(allocator, '\\'),
            '\'' => try bytes.append(allocator, '\''),
            '"' => try bytes.append(allocator, '"'),
            'x' => {
                if (index + 2 < text.len) {
                    const parsed = std.fmt.parseInt(u8, text[index + 1 .. index + 3], 16) catch 0;
                    try bytes.append(allocator, parsed);
                    index += 2;
                }
            },
            'u' => {
                // '\u{...}' appends the scalar's UTF-8 bytes
                if (index + 1 < text.len and text[index + 1] == '{') {
                    const close = std.mem.indexOfScalarPos(u8, text, index, '}') orelse text.len - 1;
                    const scalar = std.fmt.parseInt(u21, text[index + 2 .. close], 16) catch 0;
                    var encoded: [4]u8 = undefined;
                    const length = std.unicode.utf8Encode(scalar, &encoded) catch 1;
                    try bytes.appendSlice(allocator, encoded[0..length]);
                    index = close;
                }
            },
            else => try bytes.append(allocator, text[index]),
        }
    }
    return bytes.toOwnedSlice(allocator);
}

pub const Token = struct {
    tag: Tag,
    location: Location,

    pub const Location = struct {
        start: usize,
        end: usize,
    };

    pub const Tag = enum {
        invalid,
        end_of_file,
        unterminated_block_comment,
        unterminated_string,
        unterminated_character,
        malformed_integer_literal,
        empty_character_literal,
        invalid_escape_sequence,

        identifier,
        integer_literal,
        float_literal,
        string_literal,
        character_literal,
        // emitted only when 'emit_comments' is set (the formatter)
        line_comment,
        block_comment,

        keyword_import,
        keyword_as,
        keyword_extern,
        keyword_type,
        keyword_enum,
        keyword_struct,
        keyword_const,
        keyword_var,
        keyword_fn,
        keyword_if,
        keyword_else,
        keyword_while,
        keyword_for,
        keyword_match,
        keyword_break,
        keyword_continue,
        keyword_yield,
        keyword_return,
        keyword_new,
        keyword_move,
        keyword_self,
        keyword_pub,
        keyword_exp,
        keyword_true,
        keyword_false,
        keyword_interface,
        keyword_macro,
        keyword_is,
        keyword_to,

        plus, // +
        plus_equal, // +=
        minus, // -
        minus_equal, // -=
        arrow, // ->
        asterisk, // *
        asterisk_equal, // *=
        slash, // /
        slash_equal, // /=
        percent, // %
        percent_equal, // %=
        shift_left, // <<
        shift_left_equal, // <<=
        shift_right, // >>
        shift_right_equal, // >>=
        ampersand, // &
        ampersand_ampersand, // &&
        ampersand_equal, // &=
        pipe, // |
        pipe_pipe, // ||
        pipe_equal, // |=
        caret, // ^
        caret_equal, // ^=
        tilde, // ~
        bang, // !
        bang_equal, // !=
        equal, // =
        equal_equal, // ==
        angle_left, // <
        angle_left_equal, // <=
        angle_right, // >
        angle_right_equal, // >=
        colon, // :
        colon_colon, // ::
        dot, // .
        dot_dot, // ..
        ellipsis, // ...
        comma, // ,
        semicolon, // ;
        parenthesis_left, // (
        parenthesis_right, // )
        brace_left, // {
        brace_right, // }
        bracket_left, // [
        bracket_right, // ]
        hash, // #

        pub fn lexeme(tag: Tag) ?[]const u8 {
            return switch (tag) {
                .invalid,
                .end_of_file,
                .unterminated_block_comment,
                .unterminated_string,
                .unterminated_character,
                .malformed_integer_literal,
                .empty_character_literal,
                .invalid_escape_sequence,
                .identifier,
                .integer_literal,
                .float_literal,
                .string_literal,
                .character_literal,
                .line_comment,
                .block_comment,
                => null,

                .keyword_import => "import",
                .keyword_as => "as",
                .keyword_extern => "extern",
                .keyword_type => "type",
                .keyword_enum => "enum",
                .keyword_struct => "struct",
                .keyword_const => "const",
                .keyword_var => "var",
                .keyword_fn => "fn",
                .keyword_if => "if",
                .keyword_else => "else",
                .keyword_while => "while",
                .keyword_for => "for",
                .keyword_match => "match",
                .keyword_break => "break",
                .keyword_continue => "continue",
                .keyword_yield => "yield",
                .keyword_return => "return",
                .keyword_new => "new",
                .keyword_move => "move",
                .keyword_self => "self",
                .keyword_pub => "pub",
                .keyword_exp => "exp",
                .keyword_true => "true",
                .keyword_false => "false",
                .keyword_interface => "interface",
                .keyword_macro => "macro",
                .keyword_is => "is",
                .keyword_to => "to",

                .plus => "+",
                .plus_equal => "+=",
                .minus => "-",
                .minus_equal => "-=",
                .arrow => "->",
                .asterisk => "*",
                .asterisk_equal => "*=",
                .slash => "/",
                .slash_equal => "/=",
                .percent => "%",
                .percent_equal => "%=",
                .shift_left => "<<",
                .shift_left_equal => "<<=",
                .shift_right => ">>",
                .shift_right_equal => ">>=",
                .ampersand => "&",
                .ampersand_ampersand => "&&",
                .ampersand_equal => "&=",
                .pipe => "|",
                .pipe_pipe => "||",
                .pipe_equal => "|=",
                .caret => "^",
                .caret_equal => "^=",
                .tilde => "~",
                .bang => "!",
                .bang_equal => "!=",
                .equal => "=",
                .equal_equal => "==",
                .angle_left => "<",
                .angle_left_equal => "<=",
                .angle_right => ">",
                .angle_right_equal => ">=",
                .colon => ":",
                .colon_colon => "::",
                .dot => ".",
                .dot_dot => "..",
                .ellipsis => "...",
                .comma => ",",
                .semicolon => ";",
                .parenthesis_left => "(",
                .parenthesis_right => ")",
                .brace_left => "{",
                .brace_right => "}",
                .bracket_left => "[",
                .bracket_right => "]",
                .hash => "#",
            };
        }

        /// User-facing message for error tags, null for ordinary tokens.
        pub fn errorMessage(tag: Tag) ?[]const u8 {
            return switch (tag) {
                .invalid => "invalid character",
                .unterminated_block_comment => "unterminated block comment",
                .unterminated_string => "unterminated string literal",
                .unterminated_character => "unterminated character literal",
                .malformed_integer_literal => "integer literal has a radix prefix but no digits",
                .empty_character_literal => "empty character literal",
                .invalid_escape_sequence => "unrecognized escape sequence",
                else => null,
            };
        }
    };

    pub const keywords = std.StaticStringMap(Tag).initComptime(.{
        .{ "import", .keyword_import },
        .{ "as", .keyword_as },
        .{ "extern", .keyword_extern },
        .{ "type", .keyword_type },
        .{ "enum", .keyword_enum },
        .{ "struct", .keyword_struct },
        .{ "const", .keyword_const },
        .{ "var", .keyword_var },
        .{ "fn", .keyword_fn },
        .{ "if", .keyword_if },
        .{ "else", .keyword_else },
        .{ "while", .keyword_while },
        .{ "for", .keyword_for },
        .{ "match", .keyword_match },
        .{ "break", .keyword_break },
        .{ "continue", .keyword_continue },
        .{ "yield", .keyword_yield },
        .{ "return", .keyword_return },
        .{ "new", .keyword_new },
        .{ "move", .keyword_move },
        .{ "self", .keyword_self },
        .{ "pub", .keyword_pub },
        .{ "exp", .keyword_exp },
        .{ "true", .keyword_true },
        .{ "false", .keyword_false },
        .{ "interface", .keyword_interface },
        .{ "macro", .keyword_macro },
        .{ "is", .keyword_is },
        .{ "to", .keyword_to },
    });

    /// Source slice of this token.
    pub fn slice(token: Token, buffer: []const u8) []const u8 {
        return buffer[token.location.start..token.location.end];
    }
};

pub const Tokenizer = struct {
    buffer: []const u8,
    index: usize,
    // the formatter lexes with comments as tokens; the compiler does not
    emit_comments: bool = false,

    pub fn init(buffer: []const u8) Tokenizer {
        // skip a UTF-8 byte order mark if present
        const start: usize = if (std.mem.startsWith(u8, buffer, "\xEF\xBB\xBF")) 3 else 0;
        return .{ .buffer = buffer, .index = start };
    }

    /// Returns the next token. After the end of input, repeatedly returns
    /// `.end_of_file` tokens with an empty location at `buffer.len`.
    pub fn next(self: *Tokenizer) Token {
        const comment = self.skipTrivia() catch {
            // the error token spans the comment opener up to the end of input
            const token: Token = .{
                .tag = .unterminated_block_comment,
                .location = .{ .start = self.index, .end = self.buffer.len },
            };
            self.index = self.buffer.len;
            return token;
        };
        if (comment) |token| return token;

        const start = self.index;
        if (start >= self.buffer.len) {
            return .{ .tag = .end_of_file, .location = .{ .start = start, .end = start } };
        }

        const first_byte = self.buffer[start];
        const tag: Token.Tag = switch (first_byte) {
            'a'...'z', 'A'...'Z', '_' => return self.identifierOrKeyword(),
            '0'...'9' => return self.number(),
            '"' => return self.stringLiteral(),
            '\'' => return self.characterLiteral(),

            '+' => self.singleOrDouble('=', .plus_equal, .plus),
            '-' => result: {
                if (self.peekAt(1) == '=') break :result self.take(2, .minus_equal);
                if (self.peekAt(1) == '>') break :result self.take(2, .arrow);
                break :result self.take(1, .minus);
            },
            '*' => self.singleOrDouble('=', .asterisk_equal, .asterisk),
            '/' => self.singleOrDouble('=', .slash_equal, .slash),
            '%' => self.singleOrDouble('=', .percent_equal, .percent),
            '<' => result: {
                if (self.peekAt(1) == '<') {
                    if (self.peekAt(2) == '=') break :result self.take(3, .shift_left_equal);
                    break :result self.take(2, .shift_left);
                }
                if (self.peekAt(1) == '=') break :result self.take(2, .angle_left_equal);
                break :result self.take(1, .angle_left);
            },
            '>' => result: {
                if (self.peekAt(1) == '>') {
                    if (self.peekAt(2) == '=') break :result self.take(3, .shift_right_equal);
                    break :result self.take(2, .shift_right);
                }
                if (self.peekAt(1) == '=') break :result self.take(2, .angle_right_equal);
                break :result self.take(1, .angle_right);
            },
            '&' => result: {
                if (self.peekAt(1) == '&') break :result self.take(2, .ampersand_ampersand);
                if (self.peekAt(1) == '=') break :result self.take(2, .ampersand_equal);
                break :result self.take(1, .ampersand);
            },
            '|' => result: {
                if (self.peekAt(1) == '|') break :result self.take(2, .pipe_pipe);
                if (self.peekAt(1) == '=') break :result self.take(2, .pipe_equal);
                break :result self.take(1, .pipe);
            },
            '^' => self.singleOrDouble('=', .caret_equal, .caret),
            '=' => self.singleOrDouble('=', .equal_equal, .equal),
            '!' => self.singleOrDouble('=', .bang_equal, .bang),
            ':' => self.singleOrDouble(':', .colon_colon, .colon),
            '.' => result: {
                if (self.peekAt(1) == '.' and self.peekAt(2) == '.') break :result self.take(3, .ellipsis);
                if (self.peekAt(1) == '.') break :result self.take(2, .dot_dot);
                break :result self.take(1, .dot);
            },
            '~' => self.take(1, .tilde),
            ',' => self.take(1, .comma),
            ';' => self.take(1, .semicolon),
            '(' => self.take(1, .parenthesis_left),
            ')' => self.take(1, .parenthesis_right),
            '{' => self.take(1, .brace_left),
            '}' => self.take(1, .brace_right),
            '[' => self.take(1, .bracket_left),
            ']' => self.take(1, .bracket_right),
            '#' => self.take(1, .hash),

            else => result: {
                if (first_byte >= 0x80) return self.identifierOrKeyword();
                break :result self.take(1, .invalid);
            },
        };

        return .{ .tag = tag, .location = .{ .start = start, .end = self.index } };
    }

    /// Tokenize the remaining input, appending tokens (including the final
    /// `.end_of_file`) to `list`.
    pub fn tokenizeAll(self: *Tokenizer, allocator: std.mem.Allocator, list: *std.ArrayList(Token)) !void {
        while (true) {
            const token = self.next();
            try list.append(allocator, token);
            if (token.tag == .end_of_file) return;
        }
    }

    const TriviaError = error{UnterminatedBlockComment};

    /// Skips whitespace and comments (section 2.2), returning the comment
    /// as a token instead when 'emit_comments' is set. On an unterminated
    /// block comment, leaves `index` at the offending `/*` opener and errors.
    fn skipTrivia(self: *Tokenizer) TriviaError!?Token {
        while (self.index < self.buffer.len) {
            switch (self.buffer[self.index]) {
                ' ', '\t', '\n', '\r', 0x0B, 0x0C => self.index += 1,
                '/' => switch (self.peekAt(1) orelse 0) {
                    '/' => {
                        const start = self.index;
                        self.index += 2;
                        while (self.index < self.buffer.len and self.buffer[self.index] != '\n') {
                            self.index += 1;
                        }
                        if (self.emit_comments) {
                            return .{ .tag = .line_comment, .location = .{ .start = start, .end = self.index } };
                        }
                    },
                    '*' => {
                        const start = self.index;
                        try self.skipBlockComment();
                        if (self.emit_comments) {
                            return .{ .tag = .block_comment, .location = .{ .start = start, .end = self.index } };
                        }
                    },
                    else => return null,
                },
                else => return null,
            }
        }
        return null;
    }

    fn skipBlockComment(self: *Tokenizer) TriviaError!void {
        const opener = self.index;
        self.index += 2;
        var depth: usize = 1;
        while (self.index < self.buffer.len) {
            if (self.buffer[self.index] == '/' and self.peekAt(1) == '*') {
                depth += 1;
                self.index += 2;
            } else if (self.buffer[self.index] == '*' and self.peekAt(1) == '/') {
                depth -= 1;
                self.index += 2;
                if (depth == 0) return;
            } else {
                self.index += 1;
            }
        }
        self.index = opener;
        return error.UnterminatedBlockComment;
    }

    fn identifierOrKeyword(self: *Tokenizer) Token {
        const start = self.index;
        while (self.index < self.buffer.len) : (self.index += 1) {
            switch (self.buffer[self.index]) {
                'a'...'z', 'A'...'Z', '0'...'9', '_' => {},
                else => |byte| if (byte < 0x80) break,
            }
        }
        const text = self.buffer[start..self.index];
        const tag = Token.keywords.get(text) orelse .identifier;
        return .{ .tag = tag, .location = .{ .start = start, .end = self.index } };
    }

    fn number(self: *Tokenizer) Token {
        const start = self.index;

        if (self.buffer[start] == '0') {
            switch (self.peekAt(1) orelse 0) {
                'x', 'b', 'o' => |prefix| {
                    self.index += 2;
                    const digit_start = self.index;
                    while (self.index < self.buffer.len and isRadixDigit(prefix, self.buffer[self.index])) {
                        self.index += 1;
                    }
                    const tag: Token.Tag = if (self.index == digit_start) .malformed_integer_literal else .integer_literal;
                    return .{ .tag = tag, .location = .{ .start = start, .end = self.index } };
                },
                else => {},
            }
        }

        while (self.index < self.buffer.len and isDecimalDigit(self.buffer[self.index])) {
            self.index += 1;
        }

        // a float needs a digit after the dot -- '1.' is an integer then
        // member access, '1..2' an integer then a range (section 2.6)
        var is_float = false;
        if (self.index < self.buffer.len and self.buffer[self.index] == '.' and isDecimalDigit(self.peekAt(1) orelse 0)) {
            is_float = true;
            self.index += 1;
            while (self.index < self.buffer.len and isDecimalDigit(self.buffer[self.index])) {
                self.index += 1;
            }
        }
        // an exponent makes a float even without a dot ('1e9'); it only
        // counts when digits follow, so '1e' stays an integer then an
        // identifier (section 2.6)
        if (self.index < self.buffer.len and (self.buffer[self.index] == 'e' or self.buffer[self.index] == 'E')) {
            var lookahead = self.index + 1;
            if (lookahead < self.buffer.len and (self.buffer[lookahead] == '+' or self.buffer[lookahead] == '-')) {
                lookahead += 1;
            }
            if (lookahead < self.buffer.len and isDecimalDigit(self.buffer[lookahead])) {
                is_float = true;
                self.index = lookahead;
                while (self.index < self.buffer.len and isDecimalDigit(self.buffer[self.index])) {
                    self.index += 1;
                }
            }
        }

        if (is_float) {
            return .{ .tag = .float_literal, .location = .{ .start = start, .end = self.index } };
        }
        return .{ .tag = .integer_literal, .location = .{ .start = start, .end = self.index } };
    }

    fn stringLiteral(self: *Tokenizer) Token {
        return self.quoted('"', .string_literal, .unterminated_string);
    }

    fn characterLiteral(self: *Tokenizer) Token {
        const token = self.quoted('\'', .character_literal, .unterminated_character);
        // a character literal needs at least one character between the quotes
        if (token.tag == .character_literal and token.location.end - token.location.start < 3) {
            return .{ .tag = .empty_character_literal, .location = token.location };
        }
        return token;
    }

    /// Scans a quote-delimited literal with backslash escapes. The tokenizer
    /// validates termination and the escape introducer set (section 2.6);
    /// escape payloads (\xHH digits, \u{...} contents) are checked by a later
    /// phase. A literal is unterminated at the end of input or at a raw
    /// newline. An unrecognized escape reports the escape's own span.
    fn quoted(self: *Tokenizer, quote: u8, terminated: Token.Tag, unterminated: Token.Tag) Token {
        const start = self.index;
        var bad_escape: ?Token.Location = null;
        self.index += 1;
        while (self.index < self.buffer.len) {
            const byte = self.buffer[self.index];
            if (byte == quote) {
                self.index += 1;
                if (bad_escape) |location| {
                    return .{ .tag = .invalid_escape_sequence, .location = location };
                }
                return .{ .tag = terminated, .location = .{ .start = start, .end = self.index } };
            }
            if (byte == '\n') break;
            if (byte == '\\') {
                self.index += 1;
                if (self.index >= self.buffer.len) break;
                switch (self.buffer[self.index]) {
                    'n', 'r', 't', '0', '\\', '\'', '"', 'x', 'X', 'u', 'U' => {},
                    else => if (bad_escape == null) {
                        bad_escape = .{ .start = self.index - 1, .end = self.index + 1 };
                    },
                }
            }
            self.index += 1;
        }
        return .{ .tag = unterminated, .location = .{ .start = start, .end = self.index } };
    }

    fn peekAt(self: *const Tokenizer, offset: usize) ?u8 {
        const target_index = self.index + offset;
        return if (target_index < self.buffer.len) self.buffer[target_index] else null;
    }

    fn take(self: *Tokenizer, length: usize, tag: Token.Tag) Token.Tag {
        self.index += length;
        return tag;
    }

    fn singleOrDouble(self: *Tokenizer, second: u8, double: Token.Tag, single: Token.Tag) Token.Tag {
        if (self.peekAt(1) == second) return self.take(2, double);
        return self.take(1, single);
    }

    fn isDecimalDigit(byte: u8) bool {
        return byte >= '0' and byte <= '9';
    }

    fn isRadixDigit(prefix: u8, byte: u8) bool {
        return switch (prefix) {
            'x' => switch (byte) {
                '0'...'9', 'a'...'f', 'A'...'F' => true,
                else => false,
            },
            'b' => byte == '0' or byte == '1',
            'o' => byte >= '0' and byte <= '7',
            else => unreachable,
        };
    }
};

fn expectTokens(source: []const u8, expected: []const Token.Tag) !void {
    var tokenizer = Tokenizer.init(source);
    for (expected) |expected_tag| {
        const token = tokenizer.next();
        try std.testing.expectEqual(expected_tag, token.tag);
    }
    try std.testing.expectEqual(Token.Tag.end_of_file, tokenizer.next().tag);
}

test "empty and whitespace-only input" {
    try expectTokens("", &.{});
    try expectTokens("  \t\r\n\x0b\x0c ", &.{});
}

test "newlines are plain whitespace" {
    try expectTokens("a\nb", &.{ .identifier, .identifier });
    try expectTokens("a\n\n  \n\t\nb", &.{ .identifier, .identifier });
    try expectTokens("a\n// comment between\n\nb", &.{ .identifier, .identifier });
}

test "semicolons terminate statements" {
    try expectTokens("a; b;", &.{ .identifier, .semicolon, .identifier, .semicolon });
}

test "keywords and identifiers" {
    try expectTokens("fn main importer _x x1 Self", &.{
        .keyword_fn, .identifier, .identifier, .identifier, .identifier, .identifier,
    });
    try expectTokens("import as extern type enum struct const var if else while for match break return new move self pub exp true false interface macro is to", &.{
        .keyword_import, .keyword_as,    .keyword_extern,    .keyword_type,  .keyword_enum,
        .keyword_struct, .keyword_const, .keyword_var,       .keyword_if,    .keyword_else,
        .keyword_while,  .keyword_for,   .keyword_match,     .keyword_break, .keyword_return,
        .keyword_new,    .keyword_move,  .keyword_self,      .keyword_pub,   .keyword_exp,
        .keyword_true,   .keyword_false, .keyword_interface, .keyword_macro, .keyword_is,
        .keyword_to,
    });
}

test "unicode identifiers" {
    // "hello" with an accented e, then a greek lambda joined to two han characters
    try expectTokens("h\xc3\xa9llo \xce\xbb\xe5\x8f\x98\xe9\x87\x8f", &.{ .identifier, .identifier });
}

test "integer literals" {
    try expectTokens("0 42 0xDEADbeef 0b1010 0o777", &.{
        .integer_literal, .integer_literal, .integer_literal, .integer_literal, .integer_literal,
    });
    // a radix prefix with no digits is malformed
    try expectTokens("0x 0b2", &.{ .malformed_integer_literal, .malformed_integer_literal, .integer_literal });
}

test "float literals" {
    try expectTokens("1.5 0.0", &.{ .float_literal, .float_literal });
    // '.5' is not a float, the spec requires leading digits; '1.' needs
    // a digit after the dot, so it is member access on an integer
    try expectTokens(".5", &.{ .dot, .integer_literal });
    try expectTokens("10.", &.{ .integer_literal, .dot });
    try expectTokens("1.foo", &.{ .integer_literal, .dot, .identifier });
    // exponents, signed either way, with or without a dot; 'e' with no
    // digits is an identifier, not an exponent
    try expectTokens("6.02e23 1E-9 2.5E+3 9e+0", &.{ .float_literal, .float_literal, .float_literal, .float_literal });
    try expectTokens("1e 1e+", &.{ .integer_literal, .identifier, .integer_literal, .identifier, .plus });
}

test "string literals" {
    try expectTokens(
        \\"hello" "esc \" \n \\ \x41 \u{1F600}" ""
    , &.{ .string_literal, .string_literal, .string_literal });

    var tokenizer = Tokenizer.init("\"unterminated");
    try std.testing.expectEqual(Token.Tag.unterminated_string, tokenizer.next().tag);
}

test "invalid escape sequences" {
    try expectTokens("\"bad \\q escape\"", &.{.invalid_escape_sequence});
    try expectTokens("'\\q'", &.{.invalid_escape_sequence});
    // the error spans just the escape, not the whole literal
    var tokenizer = Tokenizer.init("\"ab\\qcd\"");
    const token = tokenizer.next();
    try std.testing.expectEqual(Token.Tag.invalid_escape_sequence, token.tag);
    try std.testing.expectEqual(@as(usize, 3), token.location.start);
    try std.testing.expectEqual(@as(usize, 5), token.location.end);
}

test "character literals" {
    try expectTokens("'a' '\\n' '\\'' 'abcdefgh' '\\u{1F600}'", &.{
        .character_literal, .character_literal, .character_literal, .character_literal, .character_literal,
    });
    // an empty character literal is its own error
    try expectTokens("''", &.{.empty_character_literal});

    var tokenizer = Tokenizer.init("'x");
    try std.testing.expectEqual(Token.Tag.unterminated_character, tokenizer.next().tag);
}

test "operators longest match" {
    try expectTokens("+ += - -= -> * *= / /= % %=", &.{
        .plus,          .plus_equal,     .minus, .minus_equal, .arrow,
        .asterisk,      .asterisk_equal, .slash, .slash_equal, .percent,
        .percent_equal,
    });
    try expectTokens("< <= << <<= > >= >> >>=", &.{
        .angle_left,  .angle_left_equal,  .shift_left,  .shift_left_equal,
        .angle_right, .angle_right_equal, .shift_right, .shift_right_equal,
    });
    try expectTokens("& && &= | || |= ^ ^= ~ ! != = ==", &.{
        .ampersand,   .ampersand_ampersand, .ampersand_equal,
        .pipe,        .pipe_pipe,           .pipe_equal,
        .caret,       .caret_equal,         .tilde,
        .bang,        .bang_equal,          .equal,
        .equal_equal,
    });
    try expectTokens(": :: . ... , ; ( ) { } [ ] #", &.{
        .colon,        .colon_colon,      .dot,               .ellipsis,   .comma,
        .semicolon,    .parenthesis_left, .parenthesis_right, .brace_left, .brace_right,
        .bracket_left, .bracket_right,    .hash,
    });
    // adjacent runs resolve longest-first
    try expectTokens("<<=>>=&&&|||", &.{
        .shift_left_equal, .shift_right_equal, .ampersand_ampersand, .ampersand, .pipe_pipe, .pipe,
    });
}

test "comments" {
    try expectTokens("a // line comment\nb", &.{ .identifier, .identifier });
    try expectTokens("a // comment at end of input", &.{.identifier});
    try expectTokens("a /* block */ b", &.{ .identifier, .identifier });
    try expectTokens("a /* nested /* deeper /* more */ */ still outer */ b", &.{ .identifier, .identifier });
    // newlines inside a block comment are part of the comment
    try expectTokens("a /* spans\ntwo lines */ b", &.{ .identifier, .identifier });

    var tokenizer = Tokenizer.init("a /* never closed /* nested */");
    try std.testing.expectEqual(Token.Tag.identifier, tokenizer.next().tag);
    try std.testing.expectEqual(Token.Tag.unterminated_block_comment, tokenizer.next().tag);
    try std.testing.expectEqual(Token.Tag.end_of_file, tokenizer.next().tag);
}

test "spread vs float vs member access" {
    // '1...' lexes as integer then spread, not '1.' float then '..'
    try expectTokens("1...", &.{ .integer_literal, .ellipsis });
    try expectTokens("a.b", &.{ .identifier, .dot, .identifier });
}

test "realistic snippet" {
    const source =
        \\pub fn area(self s: &Shape) -> f32 {
        \\    var x: *[u32] = new [0 : 120];
        \\    const a = #if (x.length() >= 2) break 50 else break 100;
        \\    return a is f32 && true;
        \\}
    ;
    try expectTokens(source, &.{
        .keyword_pub,       .keyword_fn,      .identifier,        .parenthesis_left,    .keyword_self,
        .identifier,        .colon,           .ampersand,         .identifier,          .parenthesis_right,
        .arrow,             .identifier,      .brace_left,        .keyword_var,         .identifier,
        .colon,             .asterisk,        .bracket_left,      .identifier,          .bracket_right,
        .equal,             .keyword_new,     .bracket_left,      .integer_literal,     .colon,
        .integer_literal,   .bracket_right,   .semicolon,         .keyword_const,       .identifier,
        .equal,             .hash,            .keyword_if,        .parenthesis_left,    .identifier,
        .dot,               .identifier,      .parenthesis_left,  .parenthesis_right,   .angle_right_equal,
        .integer_literal,   .parenthesis_right, .keyword_break,   .integer_literal,     .keyword_else,
        .keyword_break,     .integer_literal, .semicolon,         .keyword_return,      .identifier,
        .keyword_is,        .identifier,      .ampersand_ampersand, .keyword_true,      .semicolon,
        .brace_right,
    });
}

test "token slice and locations" {
    const source = "fn add";
    var tokenizer = Tokenizer.init(source);
    const first_token = tokenizer.next();
    try std.testing.expectEqualStrings("fn", first_token.slice(source));
    const second_token = tokenizer.next();
    try std.testing.expectEqualStrings("add", second_token.slice(source));
    try std.testing.expectEqual(@as(usize, 3), second_token.location.start);
}

test "utf-8 byte order mark is skipped" {
    try expectTokens("\xEF\xBB\xBFpub", &.{.keyword_pub});
}

test "invalid bytes" {
    try expectTokens("a $ b", &.{ .identifier, .invalid, .identifier });
}
