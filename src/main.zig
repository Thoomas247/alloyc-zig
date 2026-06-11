const std = @import("std");
const Io = std.Io;

const alloyc = @import("alloyc");

const usage =
    \\usage: alloyc <file.alloy>
    \\       alloyc run <file.alloy>
    \\       alloyc build <file.alloy> [-o <output>] [--emit-llvm] [--release]
    \\
;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();

    _ = args.skip();

    const first_argument = args.next() orelse {
        std.debug.print(usage, .{});
        std.process.exit(1);
    };
    const run_mode = std.mem.eql(u8, first_argument, "run");
    const build_mode = std.mem.eql(u8, first_argument, "build");
    const entrypoint_file_path = if (run_mode or build_mode)
        args.next() orelse {
            std.debug.print(usage, .{});
            std.process.exit(1);
        }
    else
        first_argument;

    var output_path: ?[]const u8 = null;
    var emit_llvm = false;
    var release = false;
    if (build_mode) {
        while (args.next()) |argument| {
            if (std.mem.eql(u8, argument, "-o")) {
                output_path = args.next() orelse {
                    std.debug.print("error: '-o' needs an output path\n", .{});
                    std.process.exit(1);
                };
            } else if (std.mem.eql(u8, argument, "--emit-llvm")) {
                emit_llvm = true;
            } else if (std.mem.eql(u8, argument, "--release")) {
                release = true;
            } else {
                std.debug.print("error: unknown option '{s}'\n{s}", .{ argument, usage });
                std.process.exit(1);
            }
        }
    }

    const source = readFile(init.io, allocator, entrypoint_file_path) catch |err| {
        std.debug.print("error: cannot read '{s}': {s}\n", .{ entrypoint_file_path, errorDescription(err) });
        std.process.exit(1);
    };
    defer allocator.free(source);

    var compilation = alloyc.Compilation.init(allocator);
    defer compilation.deinit();

    _ = try compilation.addModule(entrypoint_file_path, source);

    var loader_io = init.io;
    const loader: alloyc.ModuleLoader = .{
        .context = @ptrCast(&loader_io),
        .function = loadImportedModule,
    };
    const success = try compilation.run(loader);
    if (!success) {
        reportDiagnostics(&compilation);
        std.process.exit(1);
    }

    if (run_mode) {
        var output: std.Io.Writer.Allocating = .init(allocator);
        defer output.deinit();
        const exit_code = compilation.interpret(&output.writer) catch |err| switch (err) {
            error.RuntimeFault => {
                std.debug.print("{s}", .{output.writer.buffered()});
                std.debug.print("runtime fault: {s}\n", .{compilation.fault orelse "unknown"});
                std.process.exit(1);
            },
            else => return err,
        };
        std.debug.print("{s}", .{output.writer.buffered()});
        std.process.exit(@truncate(@as(u64, @bitCast(exit_code))));
    }

    if (build_mode) {
        const ir_text = (try compilation.generate(release)) orelse {
            reportDiagnostics(&compilation);
            std.process.exit(1);
        };
        try buildExecutable(init, entrypoint_file_path, output_path, ir_text, emit_llvm, release);
        return;
    }

    // debug dump of the parse stage output
    for (compilation.modules.items) |module| {
        const tree = module.ast orelse continue;
        std.debug.print("{s}: {d} import(s), {d} definition(s)\n", .{
            module.path,
            tree.module.imports.len,
            tree.module.definitions.len,
        });
        for (tree.module.imports) |import| {
            std.debug.print("  import {s}\n", .{import.path[import.path.len - 1].slice(module.source)});
        }
        for (tree.module.definitions) |definition| {
            const name = switch (definition.kind) {
                .type_def => |def| def.name,
                .fn_def => |def| def.name,
                .extern_def => |def| def.name,
                .interface_def => |def| def.name,
                .macro_def => |def| def.name,
            };
            std.debug.print("  {t} {s}\n", .{ definition.kind, name.slice(module.source) });
        }
    }
}

fn reportDiagnostics(compilation: *const alloyc.Compilation) void {
    // the block scopes the lock so the unlock flushes before exit
    var buffer: [4096]u8 = undefined;
    const stderr = std.debug.lockStderr(&buffer);
    defer std.debug.unlockStderr();
    compilation.renderDiagnostics(&stderr.file_writer.interface) catch {};
}

// writes the IR next to the output and hands it to clang, which compiles
// and links in one step (section 5.4)
fn buildExecutable(
    init: std.process.Init,
    entrypoint_file_path: []const u8,
    explicit_output: ?[]const u8,
    ir_text: []const u8,
    emit_llvm: bool,
    release: bool,
) !void {
    // process-lifetime strings live in the init arena
    const allocator = init.arena.allocator();
    const stem = std.fs.path.stem(entrypoint_file_path);
    const executable_path = explicit_output orelse try std.fmt.allocPrint(allocator, "{s}.exe", .{stem});
    const ir_path = try std.fmt.allocPrint(allocator, "{s}.ll", .{stripExtension(executable_path)});

    try Io.Dir.cwd().writeFile(init.io, .{ .sub_path = ir_path, .data = ir_text });

    const clang_path = findClang(init) orelse {
        std.debug.print(
            "error: no clang found to link the executable; set ALLOY_CLANG or add clang to PATH (kept '{s}')\n",
            .{ir_path},
        );
        std.process.exit(1);
    };

    const optimization: []const u8 = if (release) "-O2" else "-O0";
    const result = std.process.run(init.arena.allocator(), init.io, .{
        .argv = &.{ clang_path, ir_path, "-o", executable_path, optimization, "-Wno-override-module" },
    }) catch |err| {
        std.debug.print("error: cannot run '{s}': {s}\n", .{ clang_path, @errorName(err) });
        std.process.exit(1);
    };
    const linked = result.term == .exited and result.term.exited == 0;
    if (!linked) {
        std.debug.print("error: clang failed:\n{s}\n", .{result.stderr});
        std.process.exit(1);
    }
    if (!emit_llvm) {
        Io.Dir.cwd().deleteFile(init.io, ir_path) catch {};
    }
    std.debug.print("built {s}\n", .{executable_path});
}

// resolution order: $ALLOY_CLANG, then PATH, then the conventional local
// LLVM install directories
fn findClang(init: std.process.Init) ?[]const u8 {
    if (init.environ_map.get("ALLOY_CLANG")) |configured| {
        if (configured.len != 0) return configured;
    }
    const candidates = [_][]const u8{
        "clang",
        "C:\\LLVM.bak18\\bin\\clang.exe",
        "C:\\LLVM\\bin\\clang.exe",
    };
    for (candidates) |candidate| {
        const result = std.process.run(init.arena.allocator(), init.io, .{
            .argv = &.{ candidate, "--version" },
        }) catch continue;
        if (result.term == .exited and result.term.exited == 0) return candidate;
    }
    return null;
}

fn stripExtension(path: []const u8) []const u8 {
    const extension = std.fs.path.extension(path);
    return path[0 .. path.len - extension.len];
}

fn readFile(io: Io, allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    const max_size = 10 * 1024 * 1024;
    return try Io.Dir.cwd().readFileAlloc(io, file_path, allocator, .limited(max_size));
}

// import loader searching the current directory; the compiler-executable
// directory and $ALLOY_STDLIB search paths (section 5.4) are still to come
fn loadImportedModule(context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8 {
    const io: *Io = @ptrCast(@alignCast(context.?));
    return readFile(io.*, allocator, file_path) catch |err| switch (err) {
        error.FileNotFound => null,
        else => err,
    };
}

fn errorDescription(err: anyerror) []const u8 {
    return switch (err) {
        error.FileNotFound => "no such file",
        error.AccessDenied => "access denied",
        error.IsDir => "path is a directory",
        error.StreamTooLong => "file exceeds the 10 MiB limit",
        else => @errorName(err),
    };
}
