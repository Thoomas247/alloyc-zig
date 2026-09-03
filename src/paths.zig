//! Path normalization shared by the language server and the debug adapter:
//! forward slashes and a lowercased drive letter, so client and compiler
//! paths compare equal byte for byte.

const std = @import("std");

pub fn normalizeInPlace(path: []u8) void {
    for (path) |*byte| {
        if (byte.* == '\\') byte.* = '/';
    }
    if (path.len >= 2 and path[1] == ':') {
        path[0] = std.ascii.toLower(path[0]);
    }
}

// a normalized copy
pub fn normalized(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const copy = try allocator.dupe(u8, path);
    normalizeInPlace(copy);
    return copy;
}
