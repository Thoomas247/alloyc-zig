//! The Alloy source formatter ('alloyc fmt'): a token-stream normalizer.
//! It re-lexes the source with comments as tokens, then re-emits every
//! token with canonical spacing and brace-depth indentation while
//! PRESERVING the author's line breaks (capped at one blank line) and
//! comments verbatim. Because only whitespace is touched, formatting can
//! never change meaning; a final re-lex asserts the token stream is
//! byte-identical before the result is accepted.

const std = @import("std");
const tokenizer_module = @import("tokenizer.zig");
const Token = tokenizer_module.Token;
const Tokenizer = tokenizer_module.Tokenizer;

pub const FormatError = error{ OutOfMemory, MalformedSource };

const indent_width = 4;

pub fn format(allocator: std.mem.Allocator, source: []const u8) FormatError![]u8 {
    const tokens = try lex(allocator, source);
    defer allocator.free(tokens);

    var emitter = Emitter.init(allocator, source);
    defer emitter.deinit();
    for (tokens) |token| try emitter.emit(token);
    const result = try emitter.finish();
    errdefer allocator.free(result);

    // only whitespace may change: the token streams must match exactly
    const reformatted = try lex(allocator, result);
    defer allocator.free(reformatted);
    if (reformatted.len != tokens.len) return error.MalformedSource;
    for (tokens, reformatted) |original, emitted| {
        if (original.tag != emitted.tag) return error.MalformedSource;
        if (!std.mem.eql(u8, tokenText(original, source), tokenText(emitted, result))) return error.MalformedSource;
    }
    return result;
}

// the state carried from one emitted token to the next: the nesting
// depth, the previous token, and the small context flags that spacing
// decisions depend on
const Emitter = struct {
    allocator: std.mem.Allocator,
    source: []const u8,
    out: std.ArrayList(u8) = .empty,
    depth: usize = 0,
    previous: ?Token = null,
    // whether the previously emitted '&', '*', or '-' acted as a prefix
    // operator or type modifier, so its operand glues to it
    previous_was_prefix: bool = false,
    // between 'type Name' and its '=', a ':' introduces interface markers
    // and keeps a space before it, unlike annotation colons
    in_type_header: bool = false,
    angle_depth: usize = 0,
    // innermost open group per depth: an array fill or size colon directly
    // inside '[' ']' spaces both sides ('[value : count]'), unlike
    // annotation colons
    group_stack: std.ArrayList(Token.Tag) = .empty,
    // the source's own line ending is kept, so a CRLF file stays CRLF and
    // '--check' can pass on it
    newline: []const u8,

    fn init(allocator: std.mem.Allocator, source: []const u8) Emitter {
        return .{
            .allocator = allocator,
            .source = source,
            .newline = if (std.mem.indexOf(u8, source, "\r\n") != null) "\r\n" else "\n",
        };
    }

    fn deinit(self: *Emitter) void {
        self.out.deinit(self.allocator);
        self.group_stack.deinit(self.allocator);
    }

    fn emit(self: *Emitter, token: Token) FormatError!void {
        const newlines = if (self.previous) |before| countNewlines(self.source[before.location.end..token.location.start]) else 0;

        if (closesGroup(token.tag)) {
            self.depth -|= 1;
            _ = self.group_stack.pop();
        }

        if (self.previous == null) {
            try self.indent(self.depth);
        } else if (newlines == 0) {
            if (self.spacedAfterPrevious(token)) try self.out.append(self.allocator, ' ');
        } else {
            try self.out.appendSlice(self.allocator, self.newline);
            if (newlines >= 2) try self.out.appendSlice(self.allocator, self.newline);
            try self.indent(self.depth + self.continuation(token));
        }

        try self.out.appendSlice(self.allocator, tokenText(token, self.source));
        if (opensGroup(token.tag)) {
            self.depth += 1;
            try self.group_stack.append(self.allocator, token.tag);
        }
        self.track(token);
    }

    // whether a token on the same line as the previous one takes a space
    fn spacedAfterPrevious(self: *const Emitter, token: Token) bool {
        const implements_colon = token.tag == .colon and self.in_type_header and self.angle_depth == 0;
        const bracket_colon = token.tag == .colon and self.group_stack.items.len != 0 and
            self.group_stack.items[self.group_stack.items.len - 1] == .bracket_left;
        return implements_colon or bracket_colon or pairSpace(self.previous.?, token, self.previous_was_prefix);
    }

    // a line broken mid-statement continues one level deeper; a closing
    // token never does - it aligns with its own group, whatever ended the
    // line above it ('Busy: i32' + newline + '}')
    fn continuation(self: *const Emitter, token: Token) usize {
        if (closesGroup(token.tag)) return 0;
        return switch (self.previous.?.tag) {
            // an opener already deepened the indent, so the line after it
            // starts the new level, not a continuation of the old
            .semicolon, .comma, .brace_left, .parenthesis_left, .bracket_left, .brace_right, .line_comment, .block_comment => 0,
            else => 1,
        };
    }

    // updates the context flags once a token has been emitted
    fn track(self: *Emitter, token: Token) void {
        switch (token.tag) {
            .keyword_type => self.in_type_header = true,
            .equal, .brace_left => self.in_type_header = false,
            .angle_left => self.angle_depth += 1,
            .angle_right => self.angle_depth -|= 1,
            .shift_right => self.angle_depth -|= 2,
            // a comparison's '<' never closes: the statement or group
            // ending settles the depth so a later 'type X : I' keeps its
            // marker spacing
            .semicolon => {
                self.in_type_header = false;
                self.angle_depth = 0;
            },
            .brace_right, .parenthesis_right, .bracket_right => self.angle_depth = 0,
            else => {},
        }
        self.previous_was_prefix = isPrefixCandidate(token.tag) and (self.previous == null or !endsValue(self.previous.?.tag));
        self.previous = token;
    }

    fn indent(self: *Emitter, depth: usize) FormatError!void {
        try self.out.appendSlice(self.allocator, (" " ** (indent_width * 16))[0..@min(depth, 16) * indent_width]);
    }

    // the finished text, ending in one line break
    fn finish(self: *Emitter) FormatError![]u8 {
        try self.out.appendSlice(self.allocator, self.newline);
        return self.out.toOwnedSlice(self.allocator);
    }
};

// a line comment on a CRLF line lexes with its '\r' attached; the
// formatter owns line endings, so the comment text stops before it
fn tokenText(token: Token, source: []const u8) []const u8 {
    const text = token.slice(source);
    if (token.tag == .line_comment) return std.mem.trimEnd(u8, text, "\r");
    return text;
}

fn lex(allocator: std.mem.Allocator, source: []const u8) FormatError![]Token {
    var tokens: std.ArrayList(Token) = .empty;
    defer tokens.deinit(allocator);
    var scanner = Tokenizer.init(source);
    scanner.emit_comments = true;
    while (true) {
        const token = scanner.next();
        if (token.tag == .end_of_file) break;
        if (Token.Tag.errorMessage(token.tag) != null) return error.MalformedSource;
        try tokens.append(allocator, token);
    }
    return tokens.toOwnedSlice(allocator);
}

fn countNewlines(gap: []const u8) usize {
    var count: usize = 0;
    for (gap) |byte| {
        if (byte == '\n') count += 1;
    }
    return count;
}

fn opensGroup(tag: Token.Tag) bool {
    return tag == .parenthesis_left or tag == .bracket_left or tag == .brace_left;
}

fn closesGroup(tag: Token.Tag) bool {
    return tag == .parenthesis_right or tag == .bracket_right or tag == .brace_right;
}

// tokens that can END a value, making a following '-', '&', or '*' binary
fn endsValue(tag: Token.Tag) bool {
    return switch (tag) {
        .identifier,
        .integer_literal,
        .float_literal,
        .string_literal,
        .character_literal,
        .keyword_self,
        .keyword_true,
        .keyword_false,
        .parenthesis_right,
        .bracket_right,
        .brace_right,
        .angle_right,
        => true,
        else => false,
    };
}

fn isPrefixCandidate(tag: Token.Tag) bool {
    return tag == .ampersand or tag == .asterisk or tag == .minus;
}

// whether the two tags are ambiguous at the token level (generic angle
// brackets vs comparison, lambda pipes vs bitwise or), so the author's
// original adjacency is kept
fn preservesOriginal(tag: Token.Tag) bool {
    return switch (tag) {
        .angle_left, .angle_right, .shift_right, .pipe => true,
        else => false,
    };
}

fn pairSpace(before: Token, current: Token, before_was_prefix: bool) bool {
    const a = before.tag;
    const b = current.tag;

    if (preservesOriginal(a) or preservesOriginal(b)) {
        return before.location.end != current.location.start;
    }
    // a block comment on the same line keeps one space around it
    if (a == .block_comment or b == .line_comment or b == .block_comment) return true;

    switch (b) {
        .comma, .semicolon, .parenthesis_right, .bracket_right, .dot_dot, .colon => return false,
        // member access glues to a value; a struct-init '.field' after
        // '{' or ',' and an implied '::Variant' keep their space, but an
        // opening delimiter still glues ('f(::Some(1))', 'arr[::First]')
        .dot, .colon_colon => return !endsValue(a) and a != .parenthesis_left and a != .bracket_left,
        else => {},
    }
    switch (a) {
        .parenthesis_left, .bracket_left, .dot, .dot_dot, .colon_colon, .hash, .tilde, .bang, .ellipsis => return false,
        else => {},
    }
    // a prefix '&', '*', or '-' glues to its operand
    if (before_was_prefix and isPrefixCandidate(a)) return false;
    if (a == .brace_left and b == .brace_right) return false;
    if (a == .brace_left or b == .brace_right) return true;
    // calls and grouping after a value glue to it; keywords do not
    if (b == .parenthesis_left) return !endsValue(a);
    // indexing glues to its subject
    if (b == .bracket_left) return !endsValue(a);
    return true;
}

test "implied variants glue to opening delimiters" {
    const source = "fn f() { g( ::Some(1), x); h(y, ::None); }\n";
    const formatted = try format(std.testing.allocator, source);
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings("fn f() { g(::Some(1), x); h(y, ::None); }\n", formatted);
}

test "formatting normalizes spacing and preserves line structure" {
    const source =
        "fn   twice( x :i64 )->i64{\n" ++
        "\n" ++
        "\n" ++
        "    return x*2 ;\n" ++
        "}\n" ++
        "fn main()->i32{\n" ++
        "  var total=twice( 4 );// doubled\n" ++
        "      return total to i32;\n" ++
        "}\n";
    const formatted = try format(std.testing.allocator, source);
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings(
        "fn twice(x: i64) -> i64 {\n" ++
            "\n" ++
            "    return x * 2;\n" ++
            "}\n" ++
            "fn main() -> i32 {\n" ++
            "    var total = twice(4); // doubled\n" ++
            "    return total to i32;\n" ++
            "}\n",
        formatted,
    );
}

test "formatting is idempotent and keeps ambiguous adjacency" {
    const source =
        "type Pair = struct { left: i64 };\n" ++
        "fn make<T: Number>(seed: T) -> Vec<T> {\n" ++
        "    const flag = 1 < 2;\n" ++
        "    var p: *var Pair = new Pair { .left = -4 };\n" ++
        "    const f = |p| fn(x: i64) -> i64 { return x & 3; };\n" ++
        "    /* keep me */\n" ++
        "    for ([..4]) |i| {\n" ++
        "        p.left += i to i64;\n" ++
        "    }\n" ++
        "    return make_vec(seed);\n" ++
        "}\n";
    const once = try format(std.testing.allocator, source);
    defer std.testing.allocator.free(once);
    const twice = try format(std.testing.allocator, once);
    defer std.testing.allocator.free(twice);
    try std.testing.expectEqualStrings(once, twice);
    try std.testing.expect(std.mem.indexOf(u8, once, "Vec<T>") != null);
    try std.testing.expect(std.mem.indexOf(u8, once, "1 < 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, once, "*var Pair") != null);
    try std.testing.expect(std.mem.indexOf(u8, once, ".left = -4") != null);
    try std.testing.expect(std.mem.indexOf(u8, once, "/* keep me */") != null);
    try std.testing.expect(std.mem.indexOf(u8, once, "[..4]) |i|") != null);
}

test "a comparison does not swallow later marker spacing, and CRLF is kept" {
    // '<' with no closing '>' once left the angle depth stuck, dropping
    // the space before a later 'type X : Marker'
    const source = "fn f() { const flag = 1 < 2; }\ntype Pair : Marker = struct { x: i64 };\n";
    const formatted = try format(std.testing.allocator, source);
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings(source, formatted);
    // a CRLF file stays CRLF, and a line comment's '\r' never leaks into
    // the comment text
    const crlf = "fn f() { // hi\r\n    return 1;\r\n}\r\n";
    const kept = try format(std.testing.allocator, crlf);
    defer std.testing.allocator.free(kept);
    try std.testing.expectEqualStrings(crlf, kept);
}

test "closing tokens never take continuation indent" {
    const source =
        "type State = enum {\n" ++
        "    Idle,\n" ++
        "    Busy: i32\n" ++
        "};\n" ++
        "fn main() -> i32 {\n" ++
        "    const total = add(\n" ++
        "        1,\n" ++
        "        2\n" ++
        "    );\n" ++
        "    return total;\n" ++
        "}\n";
    const formatted = try format(std.testing.allocator, source);
    defer std.testing.allocator.free(formatted);
    try std.testing.expectEqualStrings(source, formatted);
}

test "formatting refuses sources with lexical errors" {
    try std.testing.expectError(error.MalformedSource, format(std.testing.allocator, "fn main() { /* open"));
}
