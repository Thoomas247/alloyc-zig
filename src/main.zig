const std = @import("std");
const Io = std.Io;

const alloyc = @import("alloyc");

const usage =
    \\usage: alloyc <file.alloy>
    \\       alloyc run <file.alloy>
    \\       alloyc build <file.alloy> [-o <output>] [--emit-llvm] [--release]
    \\       alloyc lib <file.alloy> [-o <name.alloylib>]
    \\       alloyc fmt <file.alloy> [--check]
    \\       alloyc lsp
    \\       alloyc dap
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
    if (std.mem.eql(u8, first_argument, "lsp")) {
        return serveLanguageServer(init);
    }
    if (std.mem.eql(u8, first_argument, "dap")) {
        return serveDebugAdapter(init);
    }
    if (std.mem.eql(u8, first_argument, "fmt")) {
        return formatFile(init, &args);
    }
    const run_mode = std.mem.eql(u8, first_argument, "run");
    const build_mode = std.mem.eql(u8, first_argument, "build");
    const lib_mode = std.mem.eql(u8, first_argument, "lib");
    const entrypoint_file_path = if (run_mode or build_mode or lib_mode)
        args.next() orelse {
            std.debug.print(usage, .{});
            std.process.exit(1);
        }
    else
        first_argument;

    var output_path: ?[]const u8 = null;
    var emit_llvm = false;
    var release = false;
    if (build_mode or lib_mode) {
        while (args.next()) |argument| {
            if (std.mem.eql(u8, argument, "-o")) {
                output_path = args.next() orelse {
                    std.debug.print("error: '-o' needs an output path\n", .{});
                    std.process.exit(1);
                };
            } else if (build_mode and std.mem.eql(u8, argument, "--emit-llvm")) {
                emit_llvm = true;
            } else if (build_mode and std.mem.eql(u8, argument, "--release")) {
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

    var loader_context: DiskLoaderContext = .{
        .io = init.io,
        .search_bases = standardLibrarySearchBases(init),
    };
    const loader: alloyc.ModuleLoader = .{
        .context = @ptrCast(&loader_context),
        .function = loadImportedModule,
        .library = loadPackageContainer,
    };
    const success = try compilation.run(loader);
    if (!success) {
        reportDiagnostics(&compilation);
        std.process.exit(1);
    }

    if (run_mode) {
        var output: std.Io.Writer.Allocating = .init(allocator);
        defer output.deinit();
        // the argv behind std::process::arguments: the program path, then
        // everything after it on the 'alloyc run' command line
        var run_arguments: std.ArrayList([]const u8) = .empty;
        defer run_arguments.deinit(allocator);
        try run_arguments.append(allocator, entrypoint_file_path);
        while (args.next()) |argument| {
            try run_arguments.append(allocator, argument);
        }
        const environment: alloyc.Compilation.RunEnvironment = .{
            .host_io = init.io,
            .arguments = run_arguments.items,
        };
        const exit_code = compilation.interpretWithEnvironment(&output.writer, environment) catch |err| switch (err) {
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

    if (lib_mode) {
        // the standalone check above already gated on errors; publishing
        // packs the checked unit's own sources (section 5.4)
        const arena = init.arena.allocator();
        const stem = std.fs.path.stem(entrypoint_file_path);
        const library_path = output_path orelse try std.fmt.allocPrint(arena, "{s}.alloylib", .{stem});
        const library_name = stripExtension(std.fs.path.basename(library_path));
        const container = try compilation.packLibrary(arena, library_name);
        try Io.Dir.cwd().writeFile(init.io, .{ .sub_path = library_path, .data = container });
        std.debug.print("packed {s}\n", .{library_path});
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

// 'alloyc fmt <file> [--check]': canonical whitespace in place; '--check'
// reports without writing and exits 1 when the file would change
fn formatFile(init: std.process.Init, args: anytype) !void {
    const allocator = init.arena.allocator();
    const file_path = args.next() orelse {
        std.debug.print(usage, .{});
        std.process.exit(1);
    };
    var check_only = false;
    while (args.next()) |argument| {
        if (std.mem.eql(u8, argument, "--check")) {
            check_only = true;
        } else {
            std.debug.print("error: unknown option '{s}'\n{s}", .{ argument, usage });
            std.process.exit(1);
        }
    }
    const source = readFile(init.io, allocator, file_path) catch |err| {
        std.debug.print("error: cannot read '{s}': {s}\n", .{ file_path, errorDescription(err) });
        std.process.exit(1);
    };
    const formatted = alloyc.formatter.format(allocator, source) catch |err| switch (err) {
        error.MalformedSource => {
            std.debug.print("error: '{s}' has syntax errors; fix them before formatting\n", .{file_path});
            std.process.exit(1);
        },
        else => return err,
    };
    if (std.mem.eql(u8, source, formatted)) {
        std.debug.print("{s} is already formatted\n", .{file_path});
        return;
    }
    if (check_only) {
        std.debug.print("{s} needs formatting\n", .{file_path});
        std.process.exit(1);
    }
    try Io.Dir.cwd().writeFile(init.io, .{ .sub_path = file_path, .data = formatted });
    std.debug.print("formatted {s}\n", .{file_path});
}

// 'alloyc lsp': the language server speaks JSON-RPC over stdio until the
// client sends 'exit'
fn serveLanguageServer(init: std.process.Init) !void {
    var input_buffer: [64 * 1024]u8 = undefined;
    var output_buffer: [64 * 1024]u8 = undefined;
    var input = Io.File.stdin().readerStreaming(init.io, &input_buffer);
    var output = Io.File.stdout().writerStreaming(init.io, &output_buffer);
    var server = alloyc.LanguageServer.init(init.gpa, init.io, &input.interface, &output.interface);
    server.search_bases = standardLibrarySearchBases(init);
    defer server.deinit();
    try server.run();
}

// 'alloyc dap': the interpreter-backed debug adapter over stdio
fn serveDebugAdapter(init: std.process.Init) !void {
    var input_buffer: [64 * 1024]u8 = undefined;
    var output_buffer: [64 * 1024]u8 = undefined;
    var input = Io.File.stdin().readerStreaming(init.io, &input_buffer);
    var output = Io.File.stdout().writerStreaming(init.io, &output_buffer);
    var server = alloyc.DebugAdapter.init(init.gpa, init.io, &input.interface, &output.interface);
    server.search_bases = standardLibrarySearchBases(init);
    defer server.deinit();
    try server.run();
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
    var linked = false;
    if (!release) {
        // checked builds carry DWARF end to end; the MSVC linker cannot
        // keep DWARF sections, so the debug link goes through lld
        const debug_result = std.process.run(init.arena.allocator(), init.io, .{
            .argv = &.{ clang_path, ir_path, "-o", executable_path, optimization, "-Wno-override-module", "-g", "-fuse-ld=lld", "-Wl,/debug:dwarf" },
        }) catch null;
        linked = debug_result != null and debug_result.?.term == .exited and debug_result.?.term.exited == 0;
        if (!linked) {
            std.debug.print("note: debug-info link failed (lld unavailable?); linking without debug info\n", .{});
        }
    }
    if (!linked) {
        const result = std.process.run(init.arena.allocator(), init.io, .{
            .argv = &.{ clang_path, ir_path, "-o", executable_path, optimization, "-Wno-override-module" },
        }) catch |err| {
            std.debug.print("error: cannot run '{s}': {s}\n", .{ clang_path, @errorName(err) });
            std.process.exit(1);
        };
        linked = result.term == .exited and result.term.exited == 0;
        if (!linked) {
            std.debug.print("error: clang failed:\n{s}\n", .{result.stderr});
            std.process.exit(1);
        }
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

// standard library search bases beyond the current directory (section
// 5.4): the compiler-executable's directory, then $ALLOY_STDLIB (the
// directory CONTAINING the std/ folder)
fn standardLibrarySearchBases(init: std.process.Init) []const []const u8 {
    const arena = init.arena.allocator();
    var bases: std.ArrayList([]const u8) = .empty;
    if (executableDirectory(arena)) |directory| {
        bases.append(arena, directory) catch return &.{};
    }
    if (init.environ_map.get("ALLOY_STDLIB")) |configured| {
        if (configured.len != 0) bases.append(arena, configured) catch return &.{};
    }
    return bases.toOwnedSlice(arena) catch &.{};
}

// the running executable's directory, read from the process image path
fn executableDirectory(arena: std.mem.Allocator) ?[]const u8 {
    if (@import("builtin").os.tag != .windows) return null;
    const image = std.os.windows.peb().ProcessParameters.ImagePathName;
    const buffer = image.Buffer orelse return null;
    const wide = buffer[0 .. image.Length / 2];
    const utf8 = std.unicode.utf16LeToUtf8Alloc(arena, wide) catch return null;
    return std.fs.path.dirname(utf8);
}

const DiskLoaderContext = struct {
    io: Io,
    search_bases: []const []const u8,
};

// import loader: the current directory first, then - for std:: modules -
// the standard search bases (section 5.4)
fn loadImportedModule(context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8 {
    const loader: *DiskLoaderContext = @ptrCast(@alignCast(context.?));
    if (readFile(loader.io, allocator, file_path)) |source| {
        return source;
    } else |err| if (err != error.FileNotFound) return err;
    if (!std.mem.startsWith(u8, file_path, "std/")) return null;
    for (loader.search_bases) |base| {
        const joined = try std.fs.path.join(allocator, &.{ base, file_path });
        defer allocator.free(joined);
        if (readFile(loader.io, allocator, joined)) |source| {
            return source;
        } else |err| if (err != error.FileNotFound) return err;
    }
    return null;
}

// 'pkg::name' resolves to the local 'pkg' folder first; the trusted
// registry download (section 5.4) is still to come
fn loadPackageContainer(context: ?*anyopaque, allocator: std.mem.Allocator, package_name: []const u8) anyerror!?[]const u8 {
    const loader: *DiskLoaderContext = @ptrCast(@alignCast(context.?));
    var buffer: [512]u8 = undefined;
    const container_path = std.fmt.bufPrint(&buffer, "pkg/{s}.alloylib", .{package_name}) catch return null;
    return readFile(loader.io, allocator, container_path) catch |err| switch (err) {
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
