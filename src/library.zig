//! The .alloylib container (section 5.4): a library ships its complete
//! source, version-stamped so a future compiler can always fall back to
//! recompiling the embedded source. Precompiled cache sections join the
//! format later; the source is the artifact, caches are an optimization.
//!
//! Layout, all integers little-endian:
//!   magic "ALLOYLIB", u16 format version
//!   length-prefixed compiler version, length-prefixed library name
//!   u32 member count, then per member three length-prefixed fields:
//!   the module key within the library ('' for the entry module), the
//!   original file path (for diagnostics), and the source text

const std = @import("std");

pub const format_magic = "ALLOYLIB";
pub const format_version: u16 = 1;

pub const Member = struct {
    // '' marks the entry module; internal modules carry their import key
    key: []const u8,
    path: []const u8,
    source: []const u8,
};

pub const Library = struct {
    compiler_version: []const u8,
    name: []const u8,
    members: []const Member,
};

pub const UnpackError = error{ OutOfMemory, MalformedLibrary, UnsupportedFormatVersion };

pub fn pack(allocator: std.mem.Allocator, compiler_version: []const u8, name: []const u8, members: []const Member) ![]u8 {
    var buffer: std.ArrayList(u8) = .empty;
    try buffer.appendSlice(allocator, format_magic);
    try appendInteger(u16, &buffer, allocator, format_version);
    try appendSlice(&buffer, allocator, compiler_version);
    try appendSlice(&buffer, allocator, name);
    try appendInteger(u32, &buffer, allocator, @intCast(members.len));
    for (members) |member| {
        try appendSlice(&buffer, allocator, member.key);
        try appendSlice(&buffer, allocator, member.path);
        try appendSlice(&buffer, allocator, member.source);
    }
    return buffer.toOwnedSlice(allocator);
}

/// The returned library borrows `bytes`; the caller keeps both alive
/// together (typically in the compilation arena).
pub fn unpack(allocator: std.mem.Allocator, bytes: []const u8) UnpackError!Library {
    var cursor: usize = 0;
    const magic = try take(bytes, &cursor, format_magic.len);
    if (!std.mem.eql(u8, magic, format_magic)) return error.MalformedLibrary;
    const version = try readInteger(u16, bytes, &cursor);
    if (version != format_version) return error.UnsupportedFormatVersion;
    const compiler_version = try readSlice(bytes, &cursor);
    const name = try readSlice(bytes, &cursor);
    const member_count = try readInteger(u32, bytes, &cursor);
    const members = try allocator.alloc(Member, member_count);
    errdefer allocator.free(members);
    for (members) |*member| {
        member.* = .{
            .key = try readSlice(bytes, &cursor),
            .path = try readSlice(bytes, &cursor),
            .source = try readSlice(bytes, &cursor),
        };
    }
    if (cursor != bytes.len) return error.MalformedLibrary;
    return .{ .compiler_version = compiler_version, .name = name, .members = members };
}

fn appendInteger(comptime T: type, buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, value: T) !void {
    var encoded: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &encoded, value, .little);
    try buffer.appendSlice(allocator, &encoded);
}

fn appendSlice(buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, data: []const u8) !void {
    try appendInteger(u32, buffer, allocator, @intCast(data.len));
    try buffer.appendSlice(allocator, data);
}

fn take(bytes: []const u8, cursor: *usize, length: usize) UnpackError![]const u8 {
    if (bytes.len - cursor.* < length) return error.MalformedLibrary;
    const slice = bytes[cursor.*..][0..length];
    cursor.* += length;
    return slice;
}

fn readInteger(comptime T: type, bytes: []const u8, cursor: *usize) UnpackError!T {
    const raw = try take(bytes, cursor, @sizeOf(T));
    return std.mem.readInt(T, raw[0..@sizeOf(T)], .little);
}

fn readSlice(bytes: []const u8, cursor: *usize) UnpackError![]const u8 {
    const length = try readInteger(u32, bytes, cursor);
    return take(bytes, cursor, length);
}

test "a library round-trips through pack and unpack" {
    const members = [_]Member{
        .{ .key = "", .path = "mathx.alloy", .source = "exp fn twice(x: i64) -> i64 { return x * 2; }" },
        .{ .key = "inner", .path = "inner.alloy", .source = "pub fn helper() { }" },
    };
    const bytes = try pack(std.testing.allocator, "0.1.0", "mathx", &members);
    defer std.testing.allocator.free(bytes);
    const unpacked = try unpack(std.testing.allocator, bytes);
    defer std.testing.allocator.free(unpacked.members);
    try std.testing.expectEqualStrings("mathx", unpacked.name);
    try std.testing.expectEqualStrings("0.1.0", unpacked.compiler_version);
    try std.testing.expectEqual(@as(usize, 2), unpacked.members.len);
    try std.testing.expectEqualStrings("inner", unpacked.members[1].key);
    try std.testing.expectEqualStrings(members[0].source, unpacked.members[0].source);
}

test "unpack rejects corrupt containers" {
    try std.testing.expectError(error.MalformedLibrary, unpack(std.testing.allocator, "NOTALIB!"));
    const members = [_]Member{.{ .key = "", .path = "a.alloy", .source = "fn main() { }" }};
    const bytes = try pack(std.testing.allocator, "0.1.0", "a", &members);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(error.MalformedLibrary, unpack(std.testing.allocator, bytes[0 .. bytes.len - 1]));
}
