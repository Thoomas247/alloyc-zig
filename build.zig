const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // the library module: the whole compiler minus the CLI, so the tests
    // and any embedder build it once
    const mod = b.addModule("alloyc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "alloyc",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "alloyc", .module = mod },
            },
        }),
    });
    b.installArtifact(exe);

    // the standard library ships next to the compiler (section 6.4): the
    // executable-directory search base finds it from any working directory.
    // Its sources live in the spec submodule, shared with the self-hosted
    // implementation, so every install re-copies the submodule's copy over
    // zig-out/bin/std - edit std in the spec repository, never in zig-out
    b.installDirectory(.{
        .source_dir = b.path("spec/std"),
        .install_dir = .bin,
        .install_subdir = "std",
    });

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // every test lives in the library module (root.zig pulls each file's
    // tests in through refAllDecls); main.zig has none of its own
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
}
