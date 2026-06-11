//! User-facing compiler diagnostics. Every error carries a source span and
//! renders as a clang-style message: path, line and column, the offending
//! source line, and a caret marking the span.

const std = @import("std");
const Token = @import("tokenizer.zig").Token;

pub const Diagnostic = struct {
    path: []const u8,
    source: []const u8,
    span: Token.Location,
    message: []const u8,

    pub fn render(diagnostic: Diagnostic, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const position = locate(diagnostic.source, diagnostic.span.start);
        try writer.print("{s}:{d}:{d}: error: {s}\n", .{
            diagnostic.path,
            position.line,
            position.column,
            diagnostic.message,
        });
        try writer.print("    {s}\n    ", .{diagnostic.source[position.line_start..position.line_end]});
        for (diagnostic.source[position.line_start..diagnostic.span.start]) |byte| {
            // tabs are copied so the caret stays aligned with the source line
            try writer.writeByte(if (byte == '\t') '\t' else ' ');
        }
        try writer.writeByte('^');
        const span_end_in_line = @max(@min(diagnostic.span.end, position.line_end), diagnostic.span.start + 1);
        try writer.splatByteAll('~', span_end_in_line - diagnostic.span.start - 1);
        try writer.writeByte('\n');
    }
};

const Position = struct {
    line: usize,
    column: usize,
    line_start: usize,
    line_end: usize,
};

// translates a byte offset into a 1-based line and column, byte-counted
fn locate(source: []const u8, offset: usize) Position {
    const clamped_offset = @min(offset, source.len);
    var line: usize = 1;
    var line_start: usize = 0;
    for (source[0..clamped_offset], 0..) |byte, position| {
        if (byte == '\n') {
            line += 1;
            line_start = position + 1;
        }
    }
    var line_end = clamped_offset;
    while (line_end < source.len and source[line_end] != '\n') {
        line_end += 1;
    }
    if (line_end > line_start and source[line_end - 1] == '\r') {
        line_end -= 1;
    }
    return .{
        .line = line,
        .column = clamped_offset - line_start + 1,
        .line_start = line_start,
        .line_end = line_end,
    };
}

test "locate finds line and column" {
    const source = "first\nsecond line\nthird";
    const position = locate(source, 13);
    try std.testing.expectEqual(@as(usize, 2), position.line);
    try std.testing.expectEqual(@as(usize, 8), position.column);
    try std.testing.expectEqual(@as(usize, 6), position.line_start);
    try std.testing.expectEqual(@as(usize, 17), position.line_end);
}

test "render shows the source line with a caret" {
    const diagnostic: Diagnostic = .{
        .path = "test.alloy",
        .source = "var x = \"abc\nconst y = 1",
        .span = .{ .start = 8, .end = 12 },
        .message = "unterminated string literal",
    };
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try diagnostic.render(&output.writer);
    try std.testing.expectEqualStrings(
        "test.alloy:1:9: error: unterminated string literal\n" ++
            "    var x = \"abc\n" ++
            "            ^~~~\n",
        output.writer.buffered(),
    );
}
