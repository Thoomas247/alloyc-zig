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

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    var depth: usize = 0;
    var previous: ?Token = null;
    // whether the previously emitted '&', '*', or '-' acted as a prefix
    // operator or type modifier, so its operand glues to it
    var previous_was_prefix = false;
    // between 'type Name' and its '=', a ':' introduces interface markers
    // and keeps a space before it, unlike annotation colons
    var in_type_header = false;
    var angle_depth: usize = 0;
    // innermost open group per depth: an array fill or size colon directly
    // inside '[' ']' spaces both sides ('[value : count]'), unlike
    // annotation colons
    var group_stack: std.ArrayList(Token.Tag) = .empty;
    defer group_stack.deinit(allocator);

    for (tokens) |token| {
        const newlines = if (previous) |before| countNewlines(source[before.location.end..token.location.start]) else 0;

        if (closesGroup(token.tag)) {
            depth -|= 1;
            _ = group_stack.pop();
        }

        if (previous == null) {
            try emitIndent(allocator, &out, depth);
        } else if (newlines == 0) {
            const implements_colon = token.tag == .colon and in_type_header and angle_depth == 0;
            const bracket_colon = token.tag == .colon and group_stack.items.len != 0 and
                group_stack.items[group_stack.items.len - 1] == .bracket_left;
            const spaced = implements_colon or bracket_colon or pairSpace(previous.?, token, previous_was_prefix);
            if (spaced) try out.append(allocator, ' ');
        } else {
            try out.append(allocator, '\n');
            if (newlines >= 2) try out.append(allocator, '\n');
            // a line broken mid-statement continues one level deeper
            const continuation: usize = switch (previous.?.tag) {
                .semicolon, .comma, .brace_left, .brace_right, .line_comment, .block_comment => 0,
                else => 1,
            };
            try emitIndent(allocator, &out, depth + continuation);
        }

        try out.appendSlice(allocator, token.slice(source));
        if (opensGroup(token.tag)) {
            depth += 1;
            try group_stack.append(allocator, token.tag);
        }

        switch (token.tag) {
            .keyword_type => in_type_header = true,
            .equal, .semicolon, .brace_left => in_type_header = false,
            .angle_left => angle_depth += 1,
            .angle_right => angle_depth -|= 1,
            .shift_right => angle_depth -|= 2,
            else => {},
        }
        previous_was_prefix = isPrefixCandidate(token.tag) and (previous == null or !endsValue(previous.?.tag));
        previous = token;
    }
    try out.append(allocator, '\n');

    const result = try out.toOwnedSlice(allocator);
    errdefer allocator.free(result);
    // only whitespace may change: the token streams must match exactly
    const reformatted = try lex(allocator, result);
    defer allocator.free(reformatted);
    if (reformatted.len != tokens.len) return error.MalformedSource;
    for (tokens, reformatted) |original, emitted| {
        if (original.tag != emitted.tag) return error.MalformedSource;
        if (!std.mem.eql(u8, original.slice(source), emitted.slice(result))) return error.MalformedSource;
    }
    return result;
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

fn emitIndent(allocator: std.mem.Allocator, out: *std.ArrayList(u8), depth: usize) !void {
    try out.appendSlice(allocator, (" " ** (indent_width * 16))[0..@min(depth, 16) * indent_width]);
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
        // '{' or ',' and an implied '::Variant' keep their space
        .dot, .colon_colon => return !endsValue(a),
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

test "formatting refuses sources with lexical errors" {
    try std.testing.expectError(error.MalformedSource, format(std.testing.allocator, "fn main() { /* open"));
}
