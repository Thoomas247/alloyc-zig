//! The stdio message framing shared by the language server and the debug
//! adapter: JSON-RPC and the Debug Adapter Protocol both send
//! 'Content-Length: N' headers, a blank line, then exactly N payload bytes.

const std = @import("std");
const Io = std.Io;

// one framed message; the payload is allocated in the given arena
pub fn readMessage(reader: *Io.Reader, arena: std.mem.Allocator) ![]u8 {
    var content_length: ?usize = null;
    while (true) {
        // takeDelimiter advances PAST the newline; the exclusive
        // variant would leave it buffered and desync the framing
        const raw_line = (try reader.takeDelimiter('\n')) orelse return error.EndOfStream;
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (line.len == 0) break;
        const prefix = "Content-Length:";
        if (std.ascii.startsWithIgnoreCase(line, prefix)) {
            const digits = std.mem.trim(u8, line[prefix.len..], " ");
            content_length = std.fmt.parseInt(usize, digits, 10) catch null;
        }
    }
    const length = content_length orelse return error.MissingContentLength;
    const body = try arena.alloc(u8, length);
    try reader.readSliceAll(body);
    return body;
}

pub fn send(writer: *Io.Writer, payload: []const u8) !void {
    try writer.print("Content-Length: {d}\r\n\r\n", .{payload.len});
    try writer.writeAll(payload);
    try writer.flush();
}

// a bare 'null' literal needs a concrete optional type to stringify; any
// other value passes through with its own type
pub fn jsonNullable(value: anytype) JsonNullable(@TypeOf(value)) {
    return value;
}

fn JsonNullable(comptime T: type) type {
    return if (T == @TypeOf(null)) ?u8 else T;
}
