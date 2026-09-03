//! The Alloy debug adapter ('alloyc dap'): the Debug Adapter Protocol over
//! stdio, executing the program through the tree-walking interpreter. A
//! statement hook drives breakpoints and stepping; while paused, the
//! adapter answers stack, scope, and variable requests from the live
//! interpreter state, then resumes when the client says so. Because the
//! interpreter is also the comptime engine, this debugger sees exactly the
//! semantics 'alloyc run' executes.

const std = @import("std");
const Io = std.Io;
const compilation_module = @import("compilation.zig");
const Compilation = compilation_module.Compilation;
const ModuleLoader = compilation_module.ModuleLoader;
const Interpreter = @import("interpreter.zig").Interpreter;
const ast = @import("ast.zig");
const tokenizer_module = @import("tokenizer.zig");
const Token = tokenizer_module.Token;
const Parser = @import("parser.zig").Parser;
const rpc = @import("rpc.zig");
const paths = @import("paths.zig");
const toolchain = @import("toolchain.zig");

pub const Server = struct {
    gpa: std.mem.Allocator,
    io: Io,
    reader: *Io.Reader,
    writer: *Io.Writer,
    // long-lived session state: breakpoints, program path, output buffer
    session_arena: std.heap.ArenaAllocator,
    message_arena: std.heap.ArenaAllocator,
    sequence: usize = 1,
    // normalized source path to requested breakpoints
    breakpoints: std.StringHashMapUnmanaged([]const Breakpoint) = .empty,
    // extra directories searched for std/ imports (the executable's
    // directory, $ALLOY_STDLIB) after the program's own tree
    search_bases: []const []const u8 = &.{},
    // compound-value handles live for one pause; ids start at handle_base
    pause_arena: std.heap.ArenaAllocator,
    variable_handles: std.ArrayList(*Interpreter.Value) = .empty,
    program_path: ?[]const u8 = null,
    configuration_done: bool = false,
    launched: bool = false,
    disconnected: bool = false,
    // live only while the program runs
    machine: ?*Interpreter = null,
    unit: ?*Compilation = null,
    step: StepMode = .running,
    step_depth: usize = 0,
    program_output: ?*Io.Writer.Allocating = null,
    output_flushed: usize = 0,

    const StepMode = enum { running, step_in, step_over, step_out };

    const Breakpoint = struct {
        line: u32,
        condition: ?[]const u8,
    };

    const handle_base: i64 = 10000;

    pub fn init(gpa: std.mem.Allocator, io: Io, reader: *Io.Reader, writer: *Io.Writer) Server {
        return .{
            .gpa = gpa,
            .io = io,
            .reader = reader,
            .writer = writer,
            .session_arena = std.heap.ArenaAllocator.init(gpa),
            .message_arena = std.heap.ArenaAllocator.init(gpa),
            .pause_arena = std.heap.ArenaAllocator.init(gpa),
        };
    }

    pub fn deinit(self: *Server) void {
        if (self.unit) |unit| {
            unit.deinit();
            self.gpa.destroy(unit);
        }
        self.breakpoints.deinit(self.gpa);
        self.session_arena.deinit();
        self.message_arena.deinit();
        self.pause_arena.deinit();
    }

    pub fn run(self: *Server) !void {
        while (!self.disconnected) {
            _ = self.message_arena.reset(.retain_capacity);
            const body = rpc.readMessage(self.reader, self.message_arena.allocator()) catch |err| switch (err) {
                error.EndOfStream => return,
                else => return err,
            };
            try self.dispatch(body);
        }
    }

    fn respond(self: *Server, request_seq: i64, command: []const u8, body: anytype) !void {
        const arena = self.message_arena.allocator();
        const payload = try std.json.Stringify.valueAlloc(arena, .{
            .seq = self.nextSequence(),
            .type = "response",
            .request_seq = request_seq,
            .success = true,
            .command = command,
            .body = rpc.jsonNullable(body),
        }, .{});
        try rpc.send(self.writer, payload);
    }

    fn sendEvent(self: *Server, name: []const u8, body: anytype) !void {
        const arena = self.message_arena.allocator();
        const payload = try std.json.Stringify.valueAlloc(arena, .{
            .seq = self.nextSequence(),
            .type = "event",
            .event = name,
            .body = rpc.jsonNullable(body),
        }, .{});
        try rpc.send(self.writer, payload);
    }

    fn nextSequence(self: *Server) usize {
        const id = self.sequence;
        self.sequence += 1;
        return id;
    }

    fn dispatch(self: *Server, body: []const u8) !void {
        const arena = self.message_arena.allocator();
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch return;
        if (parsed != .object) return;
        const message = parsed.object;
        const command_value = message.get("command") orelse return;
        if (command_value != .string) return;
        const command = command_value.string;
        const request_seq = if (message.get("seq")) |seq| (if (seq == .integer) seq.integer else 0) else 0;
        const arguments = message.get("arguments") orelse std.json.Value.null;
        try self.handle(request_seq, command, arguments);
    }

    // returns only after responding; commands that resume execution set
    // state the pause loop reads
    fn handle(self: *Server, request_seq: i64, command: []const u8, arguments: std.json.Value) !void {
        const arena = self.message_arena.allocator();

        if (std.mem.eql(u8, command, "initialize")) {
            try self.respond(request_seq, command, .{
                .supportsConfigurationDoneRequest = true,
            });
            try self.sendEvent("initialized", null);
            return;
        }
        if (std.mem.eql(u8, command, "setBreakpoints")) {
            const path = stringPath(arguments, &.{ "source", "path" }) orelse {
                return self.respond(request_seq, command, .{ .breakpoints = &[_]u8{} });
            };
            const session = self.session_arena.allocator();
            var requested: std.ArrayList(Breakpoint) = .empty;
            if (arguments == .object) {
                if (arguments.object.get("breakpoints")) |list| {
                    if (list == .array) {
                        for (list.array.items) |entry| {
                            if (entry != .object) continue;
                            const line = entry.object.get("line") orelse continue;
                            if (line != .integer or line.integer < 1) continue;
                            const condition = if (stringPath(entry, &.{"condition"})) |text|
                                try session.dupe(u8, text)
                            else
                                null;
                            try requested.append(arena, .{ .line = @intCast(line.integer), .condition = condition });
                        }
                    }
                }
            }
            const key = try paths.normalized(session, path);
            const stored = try session.dupe(Breakpoint, requested.items);
            try self.breakpoints.put(self.gpa, key, stored);
            const Verified = struct { verified: bool, line: u32 };
            var verified: std.ArrayList(Verified) = .empty;
            for (requested.items) |breakpoint| try verified.append(arena, .{ .verified = true, .line = breakpoint.line });
            try self.respond(request_seq, command, .{ .breakpoints = verified.items });
            return;
        }
        if (std.mem.eql(u8, command, "launch")) {
            const program = stringPath(arguments, &.{"program"}) orelse {
                return self.respond(request_seq, command, null);
            };
            self.program_path = try self.session_arena.allocator().dupe(u8, program);
            try self.respond(request_seq, command, null);
            self.launched = true;
            try self.maybeStart();
            return;
        }
        if (std.mem.eql(u8, command, "configurationDone")) {
            try self.respond(request_seq, command, null);
            self.configuration_done = true;
            try self.maybeStart();
            return;
        }
        if (std.mem.eql(u8, command, "threads")) {
            try self.respond(request_seq, command, .{
                .threads = &[_]struct { id: u32, name: []const u8 }{.{ .id = 1, .name = "main" }},
            });
            return;
        }
        if (std.mem.eql(u8, command, "stackTrace")) {
            try self.stackTrace(request_seq, command);
            return;
        }
        if (std.mem.eql(u8, command, "scopes")) {
            const frame_id = integerPath(arguments, "frameId") orelse 0;
            try self.respond(request_seq, command, .{
                .scopes = &[_]struct {
                    name: []const u8,
                    variablesReference: i64,
                    expensive: bool,
                }{.{ .name = "Locals", .variablesReference = frame_id, .expensive = false }},
            });
            return;
        }
        if (std.mem.eql(u8, command, "variables")) {
            try self.variables(request_seq, command, arguments);
            return;
        }
        if (std.mem.eql(u8, command, "evaluate")) {
            try self.evaluate(request_seq, command, arguments);
            return;
        }
        if (std.mem.eql(u8, command, "continue")) {
            self.step = .running;
            try self.respond(request_seq, command, .{ .allThreadsContinued = true });
            return;
        }
        if (std.mem.eql(u8, command, "next")) {
            self.step = .step_over;
            self.step_depth = if (self.machine) |machine| machine.debug_stack.items.len else 0;
            try self.respond(request_seq, command, null);
            return;
        }
        if (std.mem.eql(u8, command, "stepIn")) {
            self.step = .step_in;
            try self.respond(request_seq, command, null);
            return;
        }
        if (std.mem.eql(u8, command, "stepOut")) {
            self.step = .step_out;
            self.step_depth = if (self.machine) |machine| machine.debug_stack.items.len else 0;
            try self.respond(request_seq, command, null);
            return;
        }
        if (std.mem.eql(u8, command, "disconnect")) {
            try self.respond(request_seq, command, null);
            self.disconnected = true;
            return;
        }
        // anything else acknowledges without a body
        try self.respond(request_seq, command, null);
    }

    fn maybeStart(self: *Server) !void {
        if (!self.launched or !self.configuration_done or self.machine != null) return;
        const program = self.program_path orelse return;
        try self.execute(program);
    }

    const DiskLoader = struct {
        io: Io,
        program_directory: []const u8,
        search_bases: []const []const u8,

        fn readRelative(loader: *DiskLoader, allocator: std.mem.Allocator, base: []const u8, relative: []const u8) ?[]const u8 {
            const joined = std.fs.path.join(allocator, &.{ base, relative }) catch return null;
            defer allocator.free(joined);
            return Io.Dir.cwd().readFileAlloc(loader.io, joined, allocator, .limited(toolchain.source_read_limit)) catch null;
        }

        // the working directory first, then the program's directory, then
        // - for std/ imports - the configured search bases (section 6.4)
        fn loadModule(context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8 {
            const loader: *DiskLoader = @ptrCast(@alignCast(context.?));
            if (Io.Dir.cwd().readFileAlloc(loader.io, file_path, allocator, .limited(toolchain.source_read_limit))) |source| {
                return source;
            } else |err| if (err != error.FileNotFound) return null;
            if (loader.readRelative(allocator, loader.program_directory, file_path)) |source| return source;
            if (!std.mem.startsWith(u8, file_path, "std/")) return null;
            for (loader.search_bases) |base| {
                if (loader.readRelative(allocator, base, file_path)) |source| return source;
            }
            return null;
        }

        fn loadPackage(context: ?*anyopaque, allocator: std.mem.Allocator, package_name: []const u8) anyerror!?[]const u8 {
            const loader: *DiskLoader = @ptrCast(@alignCast(context.?));
            var buffer: [512]u8 = undefined;
            const path = std.fmt.bufPrint(&buffer, "pkg/{s}.alloylib", .{package_name}) catch return null;
            return Io.Dir.cwd().readFileAlloc(loader.io, path, allocator, .limited(toolchain.library_read_limit)) catch |err| switch (err) {
                error.FileNotFound => null,
                else => null,
            };
        }
    };

    fn execute(self: *Server, program: []const u8) !void {
        const session = self.session_arena.allocator();
        const source = Io.Dir.cwd().readFileAlloc(self.io, program, session, .limited(toolchain.source_read_limit)) catch {
            try self.sendEvent("output", .{ .category = "stderr", .output = "cannot read the program file\n" });
            return self.finish(1);
        };
        const unit = try self.gpa.create(Compilation);
        unit.* = Compilation.init(self.gpa);
        unit.comptime_io = self.io;
        self.unit = unit;
        _ = try unit.addModule(try session.dupe(u8, program), source);
        var disk_loader: DiskLoader = .{
            .io = self.io,
            .program_directory = std.fs.path.dirname(program) orelse ".",
            .search_bases = self.search_bases,
        };
        const loader: ModuleLoader = .{
            .context = @ptrCast(&disk_loader),
            .function = DiskLoader.loadModule,
            .library = DiskLoader.loadPackage,
        };
        const compiled = unit.run(loader) catch false;
        if (!compiled) {
            var rendered: Io.Writer.Allocating = .init(session);
            unit.renderDiagnostics(&rendered.writer) catch {};
            try self.sendEvent("output", .{ .category = "stderr", .output = rendered.writer.buffered() });
            return self.finish(1);
        }

        var output: Io.Writer.Allocating = .init(session);
        self.program_output = &output;
        self.output_flushed = 0;
        const machine = try session.create(Interpreter);
        machine.* = Interpreter.init(
            session,
            unit.views,
            &unit.merged.?,
            &unit.expression_types,
            &unit.call_targets,
            &unit.call_type_bindings,
            &unit.comptime_values,
            &unit.cast_shapes,
            &unit.type_targets,
            &output.writer,
        );
        machine.host_io = self.io;
        machine.pierced_results = if (unit.checker) |checked| &checked.pierced_results else null;
        machine.debug_hook = .{ .context = @ptrCast(self), .on_statement = onStatementHook };
        self.machine = machine;
        // stop on the first statement so the client can inspect entry
        self.step = .step_in;

        const exit_code: i64 = machine.run() catch |err| switch (err) {
            error.RuntimeFault => fault: {
                const message = machine.fault_message orelse "unknown fault";
                const line = try std.fmt.allocPrint(self.message_arena.allocator(), "runtime fault: {s}\n", .{message});
                try self.sendEvent("output", .{ .category = "stderr", .output = line });
                break :fault 1;
            },
            error.Return => 0,
            // a disconnect while paused unwinds the program through the
            // hook; there is no client left to report an exit to
            error.WriteFailed => if (self.disconnected) {
                self.machine = null;
                return;
            } else return err,
            error.OutOfMemory => return err,
            else => 1,
        };
        try self.flushProgramOutput();
        self.machine = null;
        try self.finish(exit_code);
    }

    fn finish(self: *Server, exit_code: i64) !void {
        try self.sendEvent("exited", .{ .exitCode = exit_code });
        try self.sendEvent("terminated", null);
    }

    fn flushProgramOutput(self: *Server) !void {
        const output = self.program_output orelse return;
        const buffered = output.writer.buffered();
        if (buffered.len > self.output_flushed) {
            try self.sendEvent("output", .{ .category = "stdout", .output = buffered[self.output_flushed..] });
            self.output_flushed = buffered.len;
        }
    }

    fn onStatementHook(context: *anyopaque, machine: *Interpreter, statement: *const ast.Statement) Interpreter.Error!void {
        const self: *Server = @ptrCast(@alignCast(context));
        self.onStatement(machine, statement) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.WriteFailed,
        };
    }

    fn onStatement(self: *Server, machine: *Interpreter, statement: *const ast.Statement) !void {
        // a block delegates to its children, which fire the hook themselves;
        // stopping on both would pause twice per line
        if (statement.* == .block) return;
        const offset = self.statementOffset(statement);
        if (machine.debug_stack.items.len != 0) {
            const top = &machine.debug_stack.items[machine.debug_stack.items.len - 1];
            top.offset = offset;
        }
        const depth = machine.debug_stack.items.len;
        const reason: ?[]const u8 = switch (self.step) {
            .step_in => "step",
            .step_over => if (depth <= self.step_depth) "step" else null,
            .step_out => if (depth < self.step_depth) "step" else null,
            .running => null,
        } orelse breakpoint: {
            if (machine.debug_stack.items.len == 0) break :breakpoint null;
            const frame = machine.debug_stack.items[machine.debug_stack.items.len - 1];
            const view = machine.views[frame.view_index];
            const line = lineOf(view.source, offset);
            const arena = self.message_arena.allocator();
            const normalized = paths.normalized(arena, view.path) catch break :breakpoint null;
            var iterator = self.breakpoints.iterator();
            while (iterator.next()) |entry| {
                if (!pathsMatch(entry.key_ptr.*, normalized)) continue;
                for (entry.value_ptr.*) |wanted| {
                    if (wanted.line != line) continue;
                    if (wanted.condition) |condition| {
                        // an unevaluable condition stops rather than
                        // silently skipping the breakpoint
                        if (!self.conditionHolds(machine, condition)) continue;
                    }
                    break :breakpoint "breakpoint";
                }
            }
            break :breakpoint null;
        };
        const stop_reason = reason orelse return;
        self.step = .running;
        _ = self.pause_arena.reset(.retain_capacity);
        self.variable_handles = .empty;
        try self.flushProgramOutput();
        try self.sendEvent("stopped", .{
            .reason = stop_reason,
            .threadId = 1,
            .allThreadsStopped = true,
        });
        try self.pauseLoop();
        // a disconnect while paused stops the program instead of letting it
        // run on with nobody listening
        if (self.disconnected) return error.Disconnected;
    }

    // while paused, requests answer against the live interpreter until a
    // resume command arrives
    fn pauseLoop(self: *Server) !void {
        while (!self.disconnected and self.step == .running) {
            _ = self.message_arena.reset(.retain_capacity);
            const body = rpc.readMessage(self.reader, self.message_arena.allocator()) catch |err| switch (err) {
                error.EndOfStream => {
                    self.disconnected = true;
                    return;
                },
                else => return err,
            };
            const arena = self.message_arena.allocator();
            const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch continue;
            if (parsed != .object) continue;
            const command_value = parsed.object.get("command") orelse continue;
            if (command_value != .string) continue;
            const command = command_value.string;
            const request_seq = if (parsed.object.get("seq")) |seq| (if (seq == .integer) seq.integer else 0) else 0;
            const arguments = parsed.object.get("arguments") orelse std.json.Value.null;
            try self.handle(request_seq, command, arguments);
            // a resume command flips the step mode and ends the pause
            if (std.mem.eql(u8, command, "continue") or std.mem.eql(u8, command, "next") or
                std.mem.eql(u8, command, "stepIn") or std.mem.eql(u8, command, "stepOut")) return;
        }
    }

    fn stackTrace(self: *Server, request_seq: i64, command: []const u8) !void {
        const arena = self.message_arena.allocator();
        const StackFrame = struct {
            id: usize,
            name: []const u8,
            source: struct { path: []const u8 },
            line: u32,
            column: u32,
        };
        var frames: std.ArrayList(StackFrame) = .empty;
        if (self.machine) |machine| {
            var index = machine.debug_stack.items.len;
            while (index > 0) {
                index -= 1;
                const frame = machine.debug_stack.items[index];
                const view = machine.views[frame.view_index];
                frames.append(arena, .{
                    .id = index + 1,
                    .name = frame.name,
                    .source = .{ .path = view.path },
                    .line = lineOf(view.source, frame.offset),
                    .column = 1,
                }) catch return error.OutOfMemory;
            }
        }
        try self.respond(request_seq, command, .{
            .stackFrames = frames.items,
            .totalFrames = frames.items.len,
        });
    }

    const Variable = struct {
        name: []const u8,
        value: []const u8,
        variablesReference: i64 = 0,
    };

    // a compound value earns a handle so the client can expand it
    fn referenceFor(self: *Server, cell: *Interpreter.Value) !i64 {
        if (!isCompound(cell.*)) return 0;
        try self.variable_handles.append(self.pause_arena.allocator(), cell);
        return handle_base + @as(i64, @intCast(self.variable_handles.items.len - 1));
    }

    fn appendVariable(self: *Server, arena: std.mem.Allocator, results: *std.ArrayList(Variable), name: []const u8, cell: *Interpreter.Value) !void {
        try results.append(arena, .{
            .name = name,
            .value = try renderValue(arena, cell.*, 0),
            .variablesReference = try self.referenceFor(cell),
        });
    }

    fn variables(self: *Server, request_seq: i64, command: []const u8, arguments: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        var results: std.ArrayList(Variable) = .empty;
        const reference = integerPath(arguments, "variablesReference") orelse 0;
        collect: {
            const machine = self.machine orelse break :collect;
            if (reference >= handle_base) {
                const index: usize = @intCast(reference - handle_base);
                if (index >= self.variable_handles.items.len) break :collect;
                try self.expandValue(arena, &results, self.variable_handles.items[index]);
                break :collect;
            }
            // frame ids are 1-based from the bottom of the call stack; the
            // scope floors slice each call's frames out of the shared stack
            if (reference < 1 or reference > @as(i64, @intCast(machine.debug_stack.items.len))) break :collect;
            const frame_index: usize = @intCast(reference - 1);
            const floor = machine.debug_stack.items[frame_index].scope_floor;
            const ceiling = if (frame_index + 1 < machine.debug_stack.items.len)
                machine.debug_stack.items[frame_index + 1].scope_floor
            else
                machine.scopes.items.len;
            var seen: std.StringHashMapUnmanaged(void) = .empty;
            var index = ceiling;
            while (index > floor) {
                index -= 1;
                const frame = machine.scopes.items[index];
                var bindings = frame.bindings.iterator();
                while (bindings.next()) |entry| {
                    if (entry.key_ptr.len == 0) continue;
                    if (seen.contains(entry.key_ptr.*)) continue;
                    try seen.put(arena, entry.key_ptr.*, {});
                    try self.appendVariable(arena, &results, entry.key_ptr.*, entry.value_ptr.*);
                }
            }
        }
        try self.respond(request_seq, command, .{ .variables = results.items });
    }

    // the children of a compound value: fields, elements, a payload, or
    // a pointee
    fn expandValue(self: *Server, arena: std.mem.Allocator, results: *std.ArrayList(Variable), cell: *Interpreter.Value) !void {
        switch (cell.*) {
            .struct_value => |instance| {
                for (instance.fields) |*field| {
                    try self.appendVariable(arena, results, field.name, &field.value);
                }
            },
            .slice, .array, .heap_array => {
                const instance = switch (cell.*) {
                    .slice => |instance| instance,
                    .array => |instance| instance,
                    .heap_array => |instance| instance orelse return,
                    else => unreachable,
                };
                for (instance.elements, 0..) |*element, index| {
                    if (index == 100) break;
                    const name = try std.fmt.allocPrint(arena, "[{d}]", .{index});
                    try self.appendVariable(arena, results, name, element);
                }
            },
            .enum_value => |instance| {
                if (instance.payload) |*payload| {
                    try self.appendVariable(arena, results, "payload", payload);
                }
            },
            .pointer => |target| {
                if (target) |alive| try self.appendVariable(arena, results, "*", alive);
            },
            .reference => |target| try self.appendVariable(arena, results, "*", target),
            else => {},
        }
    }

    // parses a watch or condition text and evaluates it against the
    // paused interpreter with a side-effect-free mini evaluator: locals,
    // member and index chains, literals, comparisons, arithmetic, and
    // logic; anything else (calls in particular) is unsupported
    fn evaluateText(self: *Server, machine: *Interpreter, text: []const u8) ?Interpreter.Value {
        const arena = self.message_arena.allocator();
        var tokens: std.ArrayList(Token) = .empty;
        var scanner = tokenizer_module.Tokenizer.init(text);
        while (true) {
            const token = scanner.next();
            tokens.append(arena, token) catch return null;
            if (token.tag == .end_of_file) break;
        }
        var parser = Parser.init(arena, text, tokens.items);
        const expression = parser.parseFreestandingExpression() catch return null;
        return miniEval(machine, expression, text) catch null;
    }

    fn conditionHolds(self: *Server, machine: *Interpreter, condition: []const u8) bool {
        const value = self.evaluateText(machine, condition) orelse return true;
        return switch (value) {
            .bool_value => |flag| flag,
            .integer => |integer| integer.value != 0,
            else => true,
        };
    }

    fn evaluate(self: *Server, request_seq: i64, command: []const u8, arguments: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        const text = stringPath(arguments, &.{"expression"}) orelse
            return self.respond(request_seq, command, .{ .result = "<no expression>", .variablesReference = @as(i64, 0) });
        const machine = self.machine orelse
            return self.respond(request_seq, command, .{ .result = "<not running>", .variablesReference = @as(i64, 0) });
        const value = self.evaluateText(machine, text) orelse
            return self.respond(request_seq, command, .{ .result = "<cannot evaluate>", .variablesReference = @as(i64, 0) });
        // the result lives past this message so expansion handles stay valid
        const cell = try self.pause_arena.allocator().create(Interpreter.Value);
        cell.* = value;
        try self.respond(request_seq, command, .{
            .result = try renderValue(arena, value, 0),
            .variablesReference = try self.referenceFor(cell),
        });
    }

    fn statementOffset(self: *Server, statement: *const ast.Statement) usize {
        const checker = if (self.unit) |unit| unit.checker orelse return 0 else return 0;
        return switch (statement.*) {
            .block => |statements| if (statements.len != 0) self.statementOffset(statements[0]) else 0,
            .var_def => |var_def| var_def.name.location.start,
            .assign => |assign| checker.expressionSpan(assign.target).start,
            .expression => |expression| checker.expressionSpan(expression).start,
            .break_stmt => |break_stmt| break_stmt.keyword.location.start,
            .continue_stmt => |continue_stmt| continue_stmt.keyword.location.start,
            .yield_stmt => |yield_stmt| yield_stmt.keyword.location.start,
            .return_stmt => |return_stmt| return_stmt.keyword.location.start,
        };
    }
};

fn isCompound(value: Interpreter.Value) bool {
    return switch (value) {
        .struct_value, .slice, .array, .reference => true,
        .heap_array => |instance| instance != null,
        .pointer => |target| target != null,
        .enum_value => |instance| instance.payload != null,
        else => false,
    };
}

const EvalError = error{Unsupported};

// follows references and live pointers so member access and arithmetic
// see the pointee, mirroring pointee transparency (section 5.2)
fn pierceValue(value: Interpreter.Value) EvalError!Interpreter.Value {
    return switch (value) {
        .reference => |target| pierceValue(target.*),
        .pointer => |target| if (target) |alive| pierceValue(alive.*) else error.Unsupported,
        else => value,
    };
}

fn miniEval(machine: *Interpreter, expression: *const ast.Expression, text: []const u8) EvalError!Interpreter.Value {
    switch (expression.*) {
        .grouped => |inner| return miniEval(machine, inner, text),
        .integer_literal => |token| {
            const value = std.fmt.parseInt(i64, token.slice(text), 0) catch return error.Unsupported;
            return .{ .integer = .{ .value = value, .primitive = .i64 } };
        },
        .bool_literal => |literal| return .{ .bool_value = literal.value },
        .path => |path| {
            if (path.len != 1) return error.Unsupported;
            const cell = lookupWatchCell(machine, path[0].slice(text)) orelse return error.Unsupported;
            return cell.*;
        },
        .member => |member| {
            const base = try pierceValue(try miniEval(machine, member.object, text));
            if (base != .struct_value) return error.Unsupported;
            const name = member.name.slice(text);
            for (base.struct_value.fields) |field| {
                if (std.mem.eql(u8, field.name, name)) return field.value;
            }
            return error.Unsupported;
        },
        .index => |index| {
            const base = try pierceValue(try miniEval(machine, index.object, text));
            const subscript = try pierceValue(try miniEval(machine, index.subscript, text));
            if (subscript != .integer) return error.Unsupported;
            const instance = switch (base) {
                .slice => |instance| instance,
                .array => |instance| instance,
                .heap_array => |instance| instance orelse return error.Unsupported,
                else => return error.Unsupported,
            };
            const position = subscript.integer.value;
            if (position < 0 or position >= instance.elements.len) return error.Unsupported;
            return instance.elements[@intCast(position)];
        },
        .unary => |unary| {
            const operand = try pierceValue(try miniEval(machine, unary.operand, text));
            return switch (unary.operator.tag) {
                .bang => if (operand == .bool_value) .{ .bool_value = !operand.bool_value } else error.Unsupported,
                .minus => if (operand == .integer) .{ .integer = .{ .value = -operand.integer.value, .primitive = operand.integer.primitive } } else error.Unsupported,
                else => error.Unsupported,
            };
        },
        .binary => |binary| {
            const left = try pierceValue(try miniEval(machine, binary.left, text));
            if (binary.operator.tag == .ampersand_ampersand or binary.operator.tag == .pipe_pipe) {
                if (left != .bool_value) return error.Unsupported;
                if (binary.operator.tag == .ampersand_ampersand and !left.bool_value) return .{ .bool_value = false };
                if (binary.operator.tag == .pipe_pipe and left.bool_value) return .{ .bool_value = true };
                const right = try pierceValue(try miniEval(machine, binary.right, text));
                if (right != .bool_value) return error.Unsupported;
                return .{ .bool_value = right.bool_value };
            }
            const right = try pierceValue(try miniEval(machine, binary.right, text));
            if (left == .bool_value and right == .bool_value) {
                return switch (binary.operator.tag) {
                    .equal_equal => .{ .bool_value = left.bool_value == right.bool_value },
                    .bang_equal => .{ .bool_value = left.bool_value != right.bool_value },
                    else => error.Unsupported,
                };
            }
            if (left != .integer or right != .integer) return error.Unsupported;
            const a = left.integer.value;
            const b = right.integer.value;
            return switch (binary.operator.tag) {
                .equal_equal => .{ .bool_value = a == b },
                .bang_equal => .{ .bool_value = a != b },
                .angle_left => .{ .bool_value = a < b },
                .angle_left_equal => .{ .bool_value = a <= b },
                .angle_right => .{ .bool_value = a > b },
                .angle_right_equal => .{ .bool_value = a >= b },
                .plus => .{ .integer = .{ .value = a +% b, .primitive = left.integer.primitive } },
                .minus => .{ .integer = .{ .value = a -% b, .primitive = left.integer.primitive } },
                .asterisk => .{ .integer = .{ .value = a *% b, .primitive = left.integer.primitive } },
                .percent => if (b != 0) .{ .integer = .{ .value = @rem(a, b), .primitive = left.integer.primitive } } else error.Unsupported,
                else => error.Unsupported,
            };
        },
        else => return error.Unsupported,
    }
}

// the innermost call's bindings, innermost scope first
fn lookupWatchCell(machine: *Interpreter, name: []const u8) ?*Interpreter.Value {
    const floor = if (machine.debug_stack.items.len != 0)
        machine.debug_stack.items[machine.debug_stack.items.len - 1].scope_floor
    else
        0;
    var index = machine.scopes.items.len;
    while (index > floor) {
        index -= 1;
        if (machine.scopes.items[index].bindings.get(name)) |cell| return cell;
    }
    return null;
}

fn renderValue(arena: std.mem.Allocator, value: Interpreter.Value, depth: usize) error{OutOfMemory}![]const u8 {
    if (depth > 2) return "...";
    switch (value) {
        .void_value => return "void",
        .integer => |integer| return std.fmt.allocPrint(arena, "{d}", .{integer.value}),
        .float => |float| return std.fmt.allocPrint(arena, "{d}", .{float.value}),
        .bool_value => |flag| return if (flag) "true" else "false",
        .slice, .array, .heap_array => {
            const instance = switch (value) {
                .slice => |instance| instance,
                .array => |instance| instance,
                .heap_array => |instance| instance orelse return "moved",
                else => unreachable,
            };
            // byte sequences render as text
            if (instance.elements.len != 0 and instance.elements[0] == .integer) {
                if (instance.elements[0].integer.primitive) |primitive| {
                    if (primitive == .u8) {
                        var text: std.ArrayList(u8) = .empty;
                        try text.append(arena, '"');
                        for (instance.elements) |element| {
                            const byte: u8 = @intCast(@max(0, @min(255, element.integer.value)));
                            if (std.ascii.isPrint(byte)) try text.append(arena, byte);
                        }
                        try text.append(arena, '"');
                        return text.toOwnedSlice(arena);
                    }
                }
            }
            var rendered: std.ArrayList(u8) = .empty;
            try rendered.append(arena, '[');
            for (instance.elements, 0..) |element, index| {
                if (index == 4) {
                    try rendered.appendSlice(arena, ", ...");
                    break;
                }
                if (index != 0) try rendered.appendSlice(arena, ", ");
                try rendered.appendSlice(arena, try renderValue(arena, element, depth + 1));
            }
            try rendered.append(arena, ']');
            return rendered.toOwnedSlice(arena);
        },
        .struct_value => |instance| {
            var rendered: std.ArrayList(u8) = .empty;
            if (instance.type_name.len != 0) try rendered.appendSlice(arena, instance.type_name);
            try rendered.appendSlice(arena, "{ ");
            for (instance.fields, 0..) |field, index| {
                if (index != 0) try rendered.appendSlice(arena, ", ");
                try rendered.appendSlice(arena, field.name);
                try rendered.appendSlice(arena, ": ");
                try rendered.appendSlice(arena, try renderValue(arena, field.value, depth + 1));
            }
            try rendered.appendSlice(arena, " }");
            return rendered.toOwnedSlice(arena);
        },
        .enum_value => |instance| {
            if (instance.payload) |payload| {
                return std.fmt.allocPrint(arena, "::{s}({s})", .{ instance.variant, try renderValue(arena, payload, depth + 1) });
            }
            return std.fmt.allocPrint(arena, "::{s}", .{instance.variant});
        },
        .pointer => |target| {
            const alive = target orelse return "null";
            return std.fmt.allocPrint(arena, "*{s}", .{try renderValue(arena, alive.*, depth + 1)});
        },
        .reference => |target| {
            return std.fmt.allocPrint(arena, "&{s}", .{try renderValue(arena, target.*, depth + 1)});
        },
        .closure => return "closure",
        .function => return "fn",
        .type_value => return "#Type",
    }
}

// module paths may be relative while the client sends absolute paths;
// a suffix match on path components bridges the two
fn pathsMatch(breakpoint_path: []const u8, module_path: []const u8) bool {
    if (std.mem.eql(u8, breakpoint_path, module_path)) return true;
    if (std.mem.endsWith(u8, breakpoint_path, module_path)) {
        const boundary = breakpoint_path.len - module_path.len;
        return boundary == 0 or breakpoint_path[boundary - 1] == '/';
    }
    if (std.mem.endsWith(u8, module_path, breakpoint_path)) {
        const boundary = module_path.len - breakpoint_path.len;
        return boundary == 0 or module_path[boundary - 1] == '/';
    }
    return false;
}

fn lineOf(source: []const u8, offset: usize) u32 {
    const clamped = @min(offset, source.len);
    var line: u32 = 1;
    for (source[0..clamped]) |byte| {
        if (byte == '\n') line += 1;
    }
    return line;
}

fn stringPath(value: std.json.Value, keys: []const []const u8) ?[]const u8 {
    var current = value;
    for (keys) |key| {
        if (current != .object) return null;
        current = current.object.get(key) orelse return null;
    }
    if (current != .string) return null;
    return current.string;
}

fn integerPath(value: std.json.Value, key: []const u8) ?i64 {
    if (value != .object) return null;
    const field = value.object.get(key) orelse return null;
    if (field != .integer) return null;
    return field.integer;
}

test "the adapter stops, walks frames, expands values, and evaluates watches" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    const program_source =
        "type Pair = struct { left: i64, right: i64 };\n" ++
        "fn twice(x: i64) -> i64 {\n" ++
        "    return x * 2;\n" ++
        "}\n" ++
        "fn main() -> i32 {\n" ++
        "    const pair = Pair { .left = 1, .right = 2 };\n" ++
        "    var total = twice(4);\n" ++
        "    for ([..4]) |i| {\n" ++
        "        total += i to i64;\n" ++
        "    }\n" ++
        "    return (total + pair.left) to i32;\n" ++
        "}\n";
    try Io.Dir.cwd().createDirPath(io, ".zig-cache/alloyc-dap-tests");
    const program_path = ".zig-cache/alloyc-dap-tests/probe.alloy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = program_path, .data = program_source });

    const Conditional = struct { line: u32, condition: ?[]const u8 = null };
    const messages = [_][]const u8{
        "{\"seq\":1,\"type\":\"request\",\"command\":\"initialize\",\"arguments\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .seq = 2,
            .type = "request",
            .command = "setBreakpoints",
            .arguments = .{
                .source = .{ .path = program_path },
                .breakpoints = &[_]Conditional{
                    .{ .line = 3 },
                    .{ .line = 9, .condition = "i == 2" },
                },
            },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .seq = 3,
            .type = "request",
            .command = "launch",
            .arguments = .{ .program = program_path },
        }, .{}),
        "{\"seq\":4,\"type\":\"request\",\"command\":\"configurationDone\",\"arguments\":{}}",
        // entry stop: run to the breakpoint inside twice
        "{\"seq\":5,\"type\":\"request\",\"command\":\"continue\",\"arguments\":{\"threadId\":1}}",
        // inside twice: both frames visible, caller locals reachable
        "{\"seq\":6,\"type\":\"request\",\"command\":\"stackTrace\",\"arguments\":{\"threadId\":1}}",
        "{\"seq\":7,\"type\":\"request\",\"command\":\"variables\",\"arguments\":{\"variablesReference\":2}}",
        "{\"seq\":8,\"type\":\"request\",\"command\":\"variables\",\"arguments\":{\"variablesReference\":1}}",
        "{\"seq\":9,\"type\":\"request\",\"command\":\"variables\",\"arguments\":{\"variablesReference\":10000}}",
        "{\"seq\":10,\"type\":\"request\",\"command\":\"evaluate\",\"arguments\":{\"expression\":\"x * 3\",\"frameId\":2}}",
        // run to the conditional breakpoint (fires only when i == 2)
        "{\"seq\":11,\"type\":\"request\",\"command\":\"continue\",\"arguments\":{\"threadId\":1}}",
        "{\"seq\":12,\"type\":\"request\",\"command\":\"variables\",\"arguments\":{\"variablesReference\":1}}",
        "{\"seq\":13,\"type\":\"request\",\"command\":\"continue\",\"arguments\":{\"threadId\":1}}",
        "{\"seq\":14,\"type\":\"request\",\"command\":\"disconnect\",\"arguments\":{}}",
    };
    var frames: std.ArrayList(u8) = .empty;
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    const transcript = output.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"event\":\"initialized\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"reason\":\"breakpoint\"") != null);
    // both frames named, innermost first
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"twice\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"main\"") != null);
    // the callee's parameter and the CALLER's locals both inspect
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"x\",\"value\":\"4\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"pair\",\"value\":\"Pair{ left: 1, right: 2 }\",\"variablesReference\":10000") != null);
    // expanding the struct handle lists its fields
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"left\",\"value\":\"1\"") != null);
    // watch evaluation over the paused frame
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"result\":\"12\"") != null);
    // the conditional breakpoint fired exactly once, at i == 2
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"i\",\"value\":\"2\"") != null);
    const stops = std.mem.count(u8, transcript, "\"reason\":\"breakpoint\"");
    try std.testing.expectEqual(@as(usize, 2), stops);
    // 8 + 0+1+2+3 + pair.left = 15
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"exitCode\":15") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"event\":\"terminated\"") != null);
}

test "a disconnect while paused stops the program" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;

    const program_source =
        "fn main() -> i32 {\n" ++
        "    const started = 1;\n" ++
        "    return started + 1;\n" ++
        "}\n";
    try Io.Dir.cwd().createDirPath(io, ".zig-cache/alloyc-dap-tests");
    const program_path = ".zig-cache/alloyc-dap-tests/disconnect.alloy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = program_path, .data = program_source });
    // the entry stop pauses before the first statement; disconnecting
    // there must not let the program run to completion
    const messages = [_][]const u8{
        "{\"seq\":1,\"type\":\"request\",\"command\":\"initialize\",\"arguments\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .seq = 2,
            .type = "request",
            .command = "launch",
            .arguments = .{ .program = program_path },
        }, .{}),
        "{\"seq\":3,\"type\":\"request\",\"command\":\"configurationDone\",\"arguments\":{}}",
        "{\"seq\":4,\"type\":\"request\",\"command\":\"disconnect\",\"arguments\":{}}",
    };
    var frames: std.ArrayList(u8) = .empty;
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    const transcript = output.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"reason\":\"step\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"command\":\"disconnect\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"event\":\"exited\"") == null);
}
