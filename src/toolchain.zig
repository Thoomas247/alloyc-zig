//! Host toolchain knowledge shared by the driver and the servers: how
//! large a source file or library container may be, and where clang lives.

const std = @import("std");
const builtin = @import("builtin");

// a source file larger than this is refused
pub const source_read_limit = 10 * 1024 * 1024;
// a packed .alloylib container (section 6.4)
pub const library_read_limit = 64 * 1024 * 1024;

// the suffix of a native executable on the host ('.exe' on Windows)
pub const executable_extension = if (builtin.os.tag == .windows) ".exe" else "";

// PATH first, then (on Windows) the conventional local LLVM install
// directories
pub const clang_candidates: []const []const u8 = if (builtin.os.tag == .windows) &.{
    "clang",
    "C:\\LLVM.bak18\\bin\\clang.exe",
    "C:\\LLVM\\bin\\clang.exe",
} else &.{"clang"};

// resolution order: the configured override ($ALLOY_CLANG), then PATH,
// then (on Windows) the conventional local LLVM install directories
pub fn findClang(arena: std.mem.Allocator, io: std.Io, configured: ?[]const u8) ?[]const u8 {
    if (configured) |path| {
        if (path.len != 0) return path;
    }
    for (clang_candidates) |candidate| {
        const result = std.process.run(arena, io, .{
            .argv = &.{ candidate, "--version" },
        }) catch continue;
        if (result.term == .exited and result.term.exited == 0) return candidate;
    }
    return null;
}
