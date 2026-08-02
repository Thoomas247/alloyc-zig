//! The Alloy language server ('alloyc lsp'): JSON-RPC over stdio per the
//! Language Server Protocol. Each edit re-runs the real compiler pipeline
//! in process and KEEPS the checked compilation alive, so diagnostics,
//! completion, hover, references, rename, symbols, semantic tokens, and
//! signature help all answer from the same tables the compiler built.
//! Documents sync incrementally; positions are UTF-16 as the protocol
//! demands.

const std = @import("std");
const Io = std.Io;
const compilation_module = @import("compilation.zig");
const Compilation = compilation_module.Compilation;
const ModuleLoader = compilation_module.ModuleLoader;
const resolution = @import("resolution.zig");
const tokenizer_module = @import("tokenizer.zig");
const Token = tokenizer_module.Token;
const ast = @import("ast.zig");
const formatter = @import("formatter.zig");

// the '#Type' reflection methods (section 3.4), offered after '.' on a
// value the tooling pass typed as a '#Type'
const type_description_completions = [_]struct { name: []const u8, detail: []const u8 }{
    .{ .name = "name", .detail = "() -> &[u8]" },
    .{ .name = "is_struct", .detail = "() -> bool" },
    .{ .name = "is_enum", .detail = "() -> bool" },
    .{ .name = "is_primitive", .detail = "() -> bool" },
    .{ .name = "is_interface", .detail = "() -> bool" },
    .{ .name = "implements_interface", .detail = "(other: #Type) -> bool" },
    .{ .name = "equals", .detail = "(other: #Type) -> bool" },
    .{ .name = "add_member", .detail = "(name: &[u8], type: #Type)" },
    .{ .name = "remove_member", .detail = "(name: &[u8])" },
    .{ .name = "member_names", .detail = "() -> &[&[u8]]" },
    .{ .name = "member_types", .detail = "() -> &[#Type]" },
};

// the declaration-only macros of std::macros (section 6.4), offered after
// '#' even when the import is not written yet
const builtin_macro_completions = [_]struct { name: []const u8, detail: []const u8 }{
    .{ .name = "type_of", .detail = "macro type_of(value) - std::macros" },
    .{ .name = "struct_type", .detail = "macro struct_type() - std::macros" },
    .{ .name = "enum_type", .detail = "macro enum_type() - std::macros" },
    .{ .name = "implementers_of", .detail = "macro implementers_of(target) - std::macros" },
    .{ .name = "name_of", .detail = "macro name_of(value) - std::macros" },
};

const keyword_completions = [_][]const u8{
    "import", "as",    "extern", "type",  "enum",   "struct",    "const", "var",
    "fn",     "if",    "else",   "while", "for",    "match",     "break", "yield",
    "return", "new",   "move",   "self",  "pub",    "exp",       "true",  "false",
    "interface", "macro", "is",  "to",
};

// the semantic token legend registered at initialize; indexes match
const semantic_token_types = [_][]const u8{ "function", "type", "interface", "macro", "variable", "parameter", "typeParameter" };

pub const Server = struct {
    gpa: std.mem.Allocator,
    io: Io,
    reader: *Io.Reader,
    writer: *Io.Writer,
    // whole document texts keyed by normalized file system path
    documents: std.StringHashMapUnmanaged(Document) = .empty,
    // URIs whose last publish carried diagnostics, so they can be cleared
    published: std.StringHashMapUnmanaged(void) = .empty,
    // the last compilation whose merge succeeded, kept alive so requests
    // can consult its views, side tables, and reference table
    analysis: ?*Compilation = null,
    // the latest symbol extraction; strings live in the analysis arena
    // and stay valid until the next successful analysis
    symbols: std.ArrayList(SymbolInfo) = .empty,
    analysis_arena: std.heap.ArenaAllocator,
    message_arena: std.heap.ArenaAllocator,
    shutdown_requested: bool = false,
    // extra directories searched for std/ imports (the executable's
    // directory, $ALLOY_STDLIB) after the document's own tree
    search_bases: []const []const u8 = &.{},
    // import-relative view paths to the absolute file the loader found,
    // so navigation targets open the real file; reset per analysis
    resolved_paths: std.StringHashMapUnmanaged([]const u8) = .empty,

    const Document = struct {
        uri: []const u8,
        text: []const u8,
    };

    const SymbolInfo = struct {
        name: []const u8,
        // the declaration's source line, shown in hover and completion
        detail: []const u8,
        completion_kind: u32,
        definition: *const ast.Definition,
        view_index: usize,
        uri: []const u8,
        range: Range,
    };

    pub fn init(gpa: std.mem.Allocator, io: Io, reader: *Io.Reader, writer: *Io.Writer) Server {
        return .{
            .gpa = gpa,
            .io = io,
            .reader = reader,
            .writer = writer,
            .analysis_arena = std.heap.ArenaAllocator.init(gpa),
            .message_arena = std.heap.ArenaAllocator.init(gpa),
        };
    }

    pub fn deinit(self: *Server) void {
        var documents = self.documents.iterator();
        while (documents.next()) |entry| {
            self.gpa.free(entry.key_ptr.*);
            self.gpa.free(entry.value_ptr.uri);
            self.gpa.free(entry.value_ptr.text);
        }
        self.documents.deinit(self.gpa);
        var published = self.published.keyIterator();
        while (published.next()) |key| self.gpa.free(key.*);
        self.published.deinit(self.gpa);
        self.symbols.deinit(self.gpa);
        var resolved = self.resolved_paths.iterator();
        while (resolved.next()) |entry| {
            self.gpa.free(entry.key_ptr.*);
            self.gpa.free(entry.value_ptr.*);
        }
        self.resolved_paths.deinit(self.gpa);
        self.dropAnalysis();
        self.analysis_arena.deinit();
        self.message_arena.deinit();
    }

    fn dropAnalysis(self: *Server) void {
        if (self.analysis) |unit| {
            unit.deinit();
            self.gpa.destroy(unit);
            self.analysis = null;
        }
    }

    pub fn run(self: *Server) !void {
        while (true) {
            _ = self.message_arena.reset(.retain_capacity);
            const body = self.readMessage() catch |err| switch (err) {
                error.EndOfStream => return,
                else => return err,
            };
            const proceed = try self.dispatch(body);
            if (!proceed) return;
        }
    }

    // one framed message: 'Content-Length: N' headers, a blank line, then
    // exactly N payload bytes
    fn readMessage(self: *Server) ![]u8 {
        const arena = self.message_arena.allocator();
        var content_length: ?usize = null;
        while (true) {
            // takeDelimiter advances PAST the newline; the exclusive
            // variant would leave it buffered and desync the framing
            const raw_line = (try self.reader.takeDelimiter('\n')) orelse return error.EndOfStream;
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
        try self.reader.readSliceAll(body);
        return body;
    }

    fn send(self: *Server, payload: []const u8) !void {
        try self.writer.print("Content-Length: {d}\r\n\r\n", .{payload.len});
        try self.writer.writeAll(payload);
        try self.writer.flush();
    }

    fn respond(self: *Server, id: std.json.Value, result: anytype) !void {
        const arena = self.message_arena.allocator();
        // a bare 'null' literal needs a concrete optional type to stringify
        const concrete = if (@TypeOf(result) == @TypeOf(null)) @as(?u8, null) else result;
        const payload = try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = id,
            .result = concrete,
        }, .{});
        try self.send(payload);
    }

    fn respondError(self: *Server, id: std.json.Value, code: i64, message: []const u8) !void {
        const arena = self.message_arena.allocator();
        const payload = try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = id,
            .@"error" = .{ .code = code, .message = message },
        }, .{});
        try self.send(payload);
    }

    fn notify(self: *Server, method: []const u8, params: anytype) !void {
        const arena = self.message_arena.allocator();
        const payload = try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = method,
            .params = params,
        }, .{});
        try self.send(payload);
    }

    // returns false when the client asked the server to exit
    fn dispatch(self: *Server, body: []const u8) !bool {
        const arena = self.message_arena.allocator();
        const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch return true;
        if (parsed != .object) return true;
        const message = parsed.object;
        const method_value = message.get("method") orelse return true;
        if (method_value != .string) return true;
        const method = method_value.string;
        const id = message.get("id") orelse std.json.Value.null;
        const params = message.get("params") orelse std.json.Value.null;

        if (std.mem.eql(u8, method, "initialize")) {
            try self.respond(id, .{
                .capabilities = .{
                    .positionEncoding = "utf-16",
                    .textDocumentSync = .{ .openClose = true, .change = 2 },
                    .completionProvider = .{ .triggerCharacters = &[_][]const u8{ ":", "." } },
                    .hoverProvider = true,
                    .definitionProvider = true,
                    .referencesProvider = true,
                    .renameProvider = true,
                    .documentSymbolProvider = true,
                    .workspaceSymbolProvider = true,
                    .documentFormattingProvider = true,
                    .signatureHelpProvider = .{ .triggerCharacters = &[_][]const u8{ "(", "," } },
                    .semanticTokensProvider = .{
                        .legend = .{
                            .tokenTypes = &semantic_token_types,
                            .tokenModifiers = &[_][]const u8{},
                        },
                        .full = true,
                    },
                },
                .serverInfo = .{ .name = "alloyc", .version = compilation_module.compiler_version },
            });
            return true;
        }
        if (std.mem.eql(u8, method, "initialized")) return true;
        if (std.mem.eql(u8, method, "shutdown")) {
            self.shutdown_requested = true;
            try self.respond(id, null);
            return true;
        }
        if (std.mem.eql(u8, method, "exit")) return false;

        if (std.mem.eql(u8, method, "textDocument/didOpen")) {
            const document = objectPath(params, &.{"textDocument"}) orelse return true;
            const uri = stringField(document, "uri") orelse return true;
            const text = stringField(document, "text") orelse return true;
            try self.openDocument(uri, text);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/didChange")) {
            try self.applyChanges(params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/didClose")) {
            const document = objectPath(params, &.{"textDocument"}) orelse return true;
            const uri = stringField(document, "uri") orelse return true;
            self.closeDocument(uri);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/didSave")) return true;

        if (std.mem.eql(u8, method, "textDocument/completion")) {
            try self.completion(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/hover")) {
            try self.hover(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/definition")) {
            try self.definition(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/references")) {
            try self.references(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/rename")) {
            try self.rename(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/documentSymbol")) {
            try self.documentSymbols(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "workspace/symbol")) {
            try self.workspaceSymbols(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/semanticTokens/full")) {
            try self.semanticTokens(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/signatureHelp")) {
            try self.signatureHelp(id, params);
            return true;
        }
        if (std.mem.eql(u8, method, "textDocument/formatting")) {
            try self.formatting(id, params);
            return true;
        }

        // unknown requests get MethodNotFound; unknown notifications are
        // silently ignored per the protocol
        if (id != .null) {
            try self.respondError(id, -32601, "method not found");
        }
        return true;
    }

    fn openDocument(self: *Server, uri: []const u8, text: []const u8) !void {
        const arena = self.message_arena.allocator();
        const path = normalizedPathFromUri(arena, uri) orelse return;
        const entry = try self.documents.getOrPut(self.gpa, path);
        if (entry.found_existing) {
            self.gpa.free(entry.value_ptr.text);
            entry.value_ptr.text = try self.gpa.dupe(u8, text);
        } else {
            entry.key_ptr.* = try self.gpa.dupe(u8, path);
            entry.value_ptr.* = .{
                .uri = try self.gpa.dupe(u8, uri),
                .text = try self.gpa.dupe(u8, text),
            };
        }
        try self.analyze(entry.key_ptr.*, entry.value_ptr.*);
    }

    // incremental sync: each change is a range splice against the text as
    // updated by the previous change; a change without a range replaces
    fn applyChanges(self: *Server, params: std.json.Value) !void {
        const document = objectPath(params, &.{"textDocument"}) orelse return;
        const uri = stringField(document, "uri") orelse return;
        const arena = self.message_arena.allocator();
        const path = normalizedPathFromUri(arena, uri) orelse return;
        const entry = self.documents.getPtr(path) orelse return;
        if (params != .object) return;
        const changes = params.object.get("contentChanges") orelse return;
        if (changes != .array) return;
        for (changes.array.items) |change| {
            const text = stringField(change, "text") orelse continue;
            const range = if (change == .object) change.object.get("range") else null;
            if (range == null or range.? != .object) {
                self.gpa.free(entry.text);
                entry.text = try self.gpa.dupe(u8, text);
                continue;
            }
            const start = positionFieldOffset(entry.text, range.?, "start") orelse continue;
            const end = positionFieldOffset(entry.text, range.?, "end") orelse continue;
            if (end < start) continue;
            const updated = try std.mem.concat(self.gpa, u8, &.{ entry.text[0..start], text, entry.text[end..] });
            self.gpa.free(entry.text);
            entry.text = updated;
        }
        const key = self.documents.getKey(path).?;
        try self.analyze(key, entry.*);
    }

    fn closeDocument(self: *Server, uri: []const u8) void {
        const arena = self.message_arena.allocator();
        const path = normalizedPathFromUri(arena, uri) orelse return;
        const entry = self.documents.fetchRemove(path) orelse return;
        self.gpa.free(entry.key);
        self.gpa.free(entry.value.uri);
        self.gpa.free(entry.value.text);
    }

    const LoaderContext = struct {
        server: *Server,
        base_directory: []const u8,

        // one probe of base/relative: open buffers shadow the disk; a hit
        // records where the import-relative path actually lives
        fn readFromBase(loader: *LoaderContext, allocator: std.mem.Allocator, base: []const u8, relative: []const u8) anyerror!?[]const u8 {
            const absolute = try joinNormalized(allocator, base, relative);
            const source: ?[]const u8 = if (loader.server.documents.get(absolute)) |open_document|
                try allocator.dupe(u8, open_document.text)
            else
                Io.Dir.cwd().readFileAlloc(loader.server.io, absolute, allocator, .limited(10 * 1024 * 1024)) catch null;
            if (source != null) try loader.server.recordResolvedPath(relative, absolute);
            return source;
        }
    };

    fn recordResolvedPath(self: *Server, relative: []const u8, absolute: []const u8) !void {
        const entry = try self.resolved_paths.getOrPut(self.gpa, relative);
        if (entry.found_existing) {
            self.gpa.free(entry.value_ptr.*);
        } else {
            entry.key_ptr.* = try self.gpa.dupe(u8, relative);
        }
        entry.value_ptr.* = try self.gpa.dupe(u8, absolute);
    }

    // view paths of imported modules are import-relative; navigation and
    // diagnostics point at the absolute file the loader found
    fn uriOfViewPath(self: *Server, allocator: std.mem.Allocator, view_path: []const u8) ![]const u8 {
        const resolved = self.resolved_paths.get(view_path) orelse view_path;
        return uriFromPath(allocator, resolved);
    }

    // the parent of a normalized directory path, null at a root
    fn parentDirectory(directory: []const u8) ?[]const u8 {
        if (directory.len == 0) return null;
        const separator = std.mem.lastIndexOfScalar(u8, directory, '/') orelse return null;
        if (separator == 0) return null;
        // stop at a drive root ('c:')
        if (separator == 2 and directory[1] == ':') return null;
        return directory[0..separator];
    }

    // imports resolve next to the entry document; open editor buffers
    // shadow the file system. std/ imports additionally search the
    // document's ancestor directories (finding a project's std/ from any
    // subfolder), then the configured search bases (section 5.4)
    fn loadModuleSource(context: ?*anyopaque, allocator: std.mem.Allocator, file_path: []const u8) anyerror!?[]const u8 {
        const loader: *LoaderContext = @ptrCast(@alignCast(context.?));
        if (try loader.readFromBase(allocator, loader.base_directory, file_path)) |source| return source;
        if (!std.mem.startsWith(u8, file_path, "std/")) return null;
        var ancestor = loader.base_directory;
        var levels: usize = 0;
        while (levels < 12) : (levels += 1) {
            ancestor = parentDirectory(ancestor) orelse break;
            if (try loader.readFromBase(allocator, ancestor, file_path)) |source| return source;
        }
        for (loader.server.search_bases) |base| {
            if (try loader.readFromBase(allocator, base, file_path)) |source| return source;
        }
        return null;
    }

    fn loadPackageBytes(context: ?*anyopaque, allocator: std.mem.Allocator, package_name: []const u8) anyerror!?[]const u8 {
        const loader: *LoaderContext = @ptrCast(@alignCast(context.?));
        const relative = try std.fmt.allocPrint(allocator, "pkg/{s}.alloylib", .{package_name});
        defer allocator.free(relative);
        const absolute = try joinNormalized(allocator, loader.base_directory, relative);
        return Io.Dir.cwd().readFileAlloc(loader.server.io, absolute, allocator, .limited(64 * 1024 * 1024)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => null,
        };
    }

    // one full pipeline run for the edited document: diagnostics publish
    // per file, and a merged unit replaces the previous analysis
    fn analyze(self: *Server, path: []const u8, document: Document) !void {
        const unit = try self.gpa.create(Compilation);
        unit.* = Compilation.init(self.gpa);
        errdefer {
            unit.deinit();
            self.gpa.destroy(unit);
        }
        // the unit outlives this message and the document buffer it came
        // from (an edit frees the old text), so it owns its entry module
        const owned_path = try unit.arena.allocator().dupe(u8, path);
        const owned_source = try unit.arena.allocator().dupe(u8, document.text);
        _ = try unit.addModule(owned_path, owned_source);

        var loader_context: LoaderContext = .{
            .server = self,
            .base_directory = directoryOf(path),
        };
        const loader: ModuleLoader = .{
            .context = @ptrCast(&loader_context),
            .function = loadModuleSource,
            .library = loadPackageBytes,
        };
        _ = unit.run(loader) catch false;

        try self.publishDiagnostics(unit, document);
        if (unit.views.len != 0) {
            self.dropAnalysis();
            self.analysis = unit;
            try self.extractSymbols(unit);
        } else {
            // the parse never reached the merge; keep the previous
            // analysis for symbol features
            unit.deinit();
            self.gpa.destroy(unit);
        }
    }

    const Range = struct {
        start: Position,
        end: Position,
    };

    const Position = struct {
        line: u32,
        character: u32,
    };

    const LspDiagnostic = struct {
        range: Range,
        severity: u32 = 1,
        source: []const u8 = "alloyc",
        message: []const u8,
    };

    fn publishDiagnostics(self: *Server, unit: *const Compilation, entry_document: Document) !void {
        const arena = self.message_arena.allocator();
        var groups: std.StringArrayHashMapUnmanaged(std.ArrayList(LspDiagnostic)) = .empty;
        for (unit.diagnostics.items) |item| {
            const group = try groups.getOrPut(arena, item.path);
            if (!group.found_existing) group.value_ptr.* = .empty;
            try group.value_ptr.append(arena, .{
                .range = .{
                    .start = positionOf(item.source, item.span.start),
                    .end = positionOf(item.source, item.span.end),
                },
                .message = item.message,
            });
        }

        var fresh: std.StringHashMapUnmanaged(void) = .empty;
        var iterator = groups.iterator();
        while (iterator.next()) |entry| {
            const uri = if (std.mem.eql(u8, entry.key_ptr.*, self.pathOfUri(entry_document.uri) orelse ""))
                entry_document.uri
            else
                try self.uriOfViewPath(arena, entry.key_ptr.*);
            try self.notify("textDocument/publishDiagnostics", .{
                .uri = uri,
                .diagnostics = entry.value_ptr.items,
            });
            try fresh.put(arena, uri, {});
        }
        // the entry document always gets a publish, clearing stale markers
        if (!fresh.contains(entry_document.uri)) {
            try self.notify("textDocument/publishDiagnostics", .{
                .uri = entry_document.uri,
                .diagnostics = &[_]LspDiagnostic{},
            });
        }
        // clear any URI that carried diagnostics last time but not now
        var stale = self.published.keyIterator();
        while (stale.next()) |key| {
            if (fresh.contains(key.*) or std.mem.eql(u8, key.*, entry_document.uri)) continue;
            try self.notify("textDocument/publishDiagnostics", .{
                .uri = key.*,
                .diagnostics = &[_]LspDiagnostic{},
            });
        }
        var old = self.published.keyIterator();
        while (old.next()) |key| self.gpa.free(key.*);
        self.published.clearRetainingCapacity();
        var fresh_keys = fresh.keyIterator();
        while (fresh_keys.next()) |key| {
            try self.published.put(self.gpa, try self.gpa.dupe(u8, key.*), {});
        }
    }

    // the message arena resets between requests, so this reconstruction is
    // only valid within one dispatch
    fn pathOfUri(self: *Server, uri: []const u8) ?[]const u8 {
        return normalizedPathFromUri(self.message_arena.allocator(), uri);
    }

    fn extractSymbols(self: *Server, unit: *const Compilation) !void {
        _ = self.analysis_arena.reset(.retain_capacity);
        const arena = self.analysis_arena.allocator();
        self.symbols.clearRetainingCapacity();
        for (unit.views, 0..) |view, view_index| {
            for (view.module.definitions) |*module_definition| {
                const name_token = definitionName(module_definition);
                try self.symbols.append(self.gpa, .{
                    .name = try arena.dupe(u8, name_token.slice(view.source)),
                    .detail = try arena.dupe(u8, definitionDetail(view.source, name_token.location.start, module_definition)),
                    .completion_kind = switch (module_definition.kind) {
                        .fn_def, .extern_def => 3,
                        .type_def => 7,
                        .interface_def => 8,
                        .macro_def => 3,
                    },
                    .definition = module_definition,
                    .view_index = view_index,
                    .uri = try self.uriOfViewPath(arena, view.path),
                    .range = .{
                        .start = positionOf(view.source, name_token.location.start),
                        .end = positionOf(view.source, name_token.location.end),
                    },
                });
            }
        }
    }

    // the view whose module is the given document, re-analyzing when the
    // last analysis was rooted elsewhere and never loaded it
    fn viewIndexOfPath(self: *Server, path: []const u8) !?usize {
        if (self.analysis) |unit| {
            for (unit.views, 0..) |view, index| {
                if (std.mem.eql(u8, view.path, path)) return index;
            }
        }
        const key = self.documents.getKey(path) orelse return null;
        const document = self.documents.get(path).?;
        try self.analyze(key, document);
        if (self.analysis) |unit| {
            for (unit.views, 0..) |view, index| {
                if (std.mem.eql(u8, view.path, path)) return index;
            }
        }
        return null;
    }

    const CompletionItem = struct {
        label: []const u8,
        kind: u32,
        detail: ?[]const u8 = null,
    };

    fn completion(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        var items: std.ArrayList(CompletionItem) = .empty;

        request: {
            const location = self.requestLocation(params) orelse break :request;
            var prefix_start = location.offset;
            while (prefix_start > 0 and isIdentifierByte(location.text[prefix_start - 1])) prefix_start -= 1;
            if (prefix_start >= 1 and location.text[prefix_start - 1] == '.') {
                // member access, unless this is a '..' range
                if (prefix_start >= 2 and location.text[prefix_start - 2] == '.') break :request;
                try self.memberCompletion(arena, &items, location, prefix_start - 1);
                return self.respond(id, items.items);
            }
            if (prefix_start >= 2 and location.text[prefix_start - 1] == ':' and location.text[prefix_start - 2] == ':') {
                try self.qualifiedCompletion(arena, &items, location, prefix_start - 2);
                return self.respond(id, items.items);
            }
            // '#' invokes a macro: offer every visible macro, and the
            // built-ins from std::macros even before the import is written
            if (prefix_start >= 1 and location.text[prefix_start - 1] == '#') {
                var offered: std.StringHashMapUnmanaged(void) = .empty;
                for (self.symbols.items) |symbol| {
                    if (symbol.definition.kind != .macro_def) continue;
                    try offered.put(arena, symbol.name, {});
                    try items.append(arena, .{
                        .label = symbol.name,
                        .kind = symbol.completion_kind,
                        .detail = symbol.detail,
                    });
                }
                for (builtin_macro_completions) |builtin| {
                    if (offered.contains(builtin.name)) continue;
                    try items.append(arena, .{
                        .label = builtin.name,
                        .kind = 3,
                        .detail = builtin.detail,
                    });
                }
                return self.respond(id, items.items);
            }
        }

        for (keyword_completions) |keyword| {
            try items.append(arena, .{ .label = keyword, .kind = 14 });
        }
        for (self.symbols.items) |symbol| {
            try items.append(arena, .{
                .label = symbol.name,
                .kind = symbol.completion_kind,
                .detail = symbol.detail,
            });
        }
        try self.respond(id, items.items);
    }

    // completion after 'receiver.': the receiver's type comes from the
    // checker's recorded type of the nearest preceding use of that name
    fn memberCompletion(self: *Server, arena: std.mem.Allocator, items: *std.ArrayList(Server.CompletionItem), location: RequestLocation, dot_offset: usize) !void {
        // resolve the view FIRST: it may re-analyze and replace the unit
        const view_index = (self.viewIndexOfPath(location.path) catch null) orelse return;
        const unit = self.analysis orelse return;
        const checker = unit.checker orelse return;
        const receiver = wordAt(location.text, dot_offset -| 1) orelse return;
        const view = unit.views[view_index];

        // '#TypeName.' reflects the type (section 3.4): the receiver needs
        // no prior typed use, the name and the leading '#' are enough
        var word_start = dot_offset;
        while (word_start > 0 and isIdentifierByte(location.text[word_start - 1])) word_start -= 1;
        if (word_start > 0 and location.text[word_start - 1] == '#') {
            const names_type = for (self.symbols.items) |symbol| {
                if (!std.mem.eql(u8, symbol.name, receiver)) continue;
                switch (symbol.definition.kind) {
                    .type_def, .interface_def => break true,
                    else => {},
                }
            } else false;
            if (names_type) {
                for (type_description_completions) |method| {
                    try items.append(arena, .{ .label = method.name, .kind = 2, .detail = method.detail });
                }
                return;
            }
        }

        var paths: std.ArrayList(*const ast.Expression) = .empty;
        try collectPathExpressions(arena, view.module, &paths);
        var best: ?*const ast.Expression = null;
        var best_start: usize = 0;
        for (paths.items) |candidate| {
            const segments = candidate.path;
            if (segments.len != 1) continue;
            if (!std.mem.eql(u8, segments[0].slice(view.source), receiver)) continue;
            const start = segments[0].location.start;
            if (start >= dot_offset) continue;
            if (best == null or start > best_start) {
                best = candidate;
                best_start = start;
            }
        }
        const node = best orelse return;
        const recorded = unit.expression_types.get(node) orelse checker.place_types.get(node) orelse return;
        const pierced = checker.pierce(recorded) catch return;
        const resolved = checker.resolveAlias(pierced) catch return;

        // a '#Type' value exposes the reflection methods (section 3.4)
        if (resolved.* == .type_description) {
            for (type_description_completions) |method| {
                try items.append(arena, .{ .label = method.name, .kind = 2, .detail = method.detail });
            }
            return;
        }
        if (checker.structuralFieldsOf(resolved) catch null) |fields| {
            for (fields) |field| {
                try items.append(arena, .{
                    .label = field.name,
                    .kind = 5,
                    .detail = field.field_type.render(arena) catch null,
                });
            }
        }
        // extensions whose self receiver names the resolved type
        if (resolved.* == .declared) {
            for (self.symbols.items) |symbol| {
                if (symbol.definition.kind != .fn_def) continue;
                const fn_def = symbol.definition.kind.fn_def;
                const parameters = fn_def.function.parameters;
                if (parameters.len == 0 or !parameters[0].is_self) continue;
                const self_source = unit.views[symbol.view_index].source;
                const self_name = selfTypeName(parameters[0].parameter_type, self_source) orelse continue;
                if (!std.mem.eql(u8, self_name, resolved.declared.name)) continue;
                try items.append(arena, .{ .label = symbol.name, .kind = 2, .detail = symbol.detail });
            }
        }
    }

    // completion after 'prefix::': an enum type's variants, or a module
    // alias's visible definitions
    fn qualifiedCompletion(self: *Server, arena: std.mem.Allocator, items: *std.ArrayList(Server.CompletionItem), location: RequestLocation, colon_offset: usize) !void {
        // resolve the view FIRST: it may re-analyze and replace the unit
        const view_index = (self.viewIndexOfPath(location.path) catch null) orelse return;
        const unit = self.analysis orelse return;
        const merged = unit.merged orelse return;
        const prefix = wordAt(location.text, colon_offset -| 1) orelse return;

        // 'Enum::' lists the variants
        for (self.symbols.items) |symbol| {
            if (!std.mem.eql(u8, symbol.name, prefix)) continue;
            if (symbol.definition.kind != .type_def) continue;
            const type_def = symbol.definition.kind.type_def;
            if (type_def.base.* == .enum_type) {
                const enum_source = unit.views[symbol.view_index].source;
                for (type_def.base.enum_type) |member| {
                    try items.append(arena, .{ .label = member.name.slice(enum_source), .kind = 20 });
                }
                return;
            }
            // an aliased or synthesised enum ('type T = #...') has no
            // syntactic members; the checker resolved its variants
            const checker = unit.checker orelse continue;
            const declared = checker.declaredTypeOf(symbol.definition, symbol.view_index) catch continue;
            const body = (checker.enumBody(declared) catch continue) orelse continue;
            for (body.variants) |variant| {
                const detail: ?[]const u8 = if (variant.payload) |payload|
                    payload.render(arena) catch null
                else
                    null;
                try items.append(arena, .{ .label = variant.name, .kind = 20, .detail = detail });
            }
            return;
        }

        // 'alias::' lists the module's reachable definitions
        const key = merged.aliases[view_index].get(prefix) orelse return;
        const target_index = merged.module_keys.get(key) orelse return;
        const target = unit.views[target_index];
        const cross_library = !resolution.sameLibrary(unit.views[view_index].library, target.library);
        for (target.module.definitions) |*module_definition| {
            const visible = if (cross_library)
                module_definition.visibility == .exported
            else
                module_definition.visibility != .private;
            if (!visible) continue;
            const name_token = definitionName(module_definition);
            try items.append(arena, .{
                .label = name_token.slice(target.source),
                .kind = switch (module_definition.kind) {
                    .fn_def, .extern_def, .macro_def => 3,
                    .type_def => 7,
                    .interface_def => 8,
                },
                .detail = declarationLine(target.source, name_token.location.start),
            });
        }
    }

    fn hover(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        // a local or expression use with a recorded type wins, so locals
        // that shadow globals hover correctly
        if (self.typedNodeAtPosition(params)) |found| {
            const rendered_type = found.recorded.render(arena) catch null;
            if (rendered_type) |type_text| {
                const rendered = try std.fmt.allocPrint(arena, "```alloy\n{s}: {s}\n```", .{ found.word, type_text });
                return self.respond(id, .{
                    .contents = .{ .kind = "markdown", .value = rendered },
                });
            }
        }
        // a call's checked target resolves overloads and associated
        // functions exactly (String::empty vs Vector::empty), so it wins
        // over matching globals by bare name
        if (self.callTargetAtPosition(params)) |target| {
            for (self.symbols.items) |symbol| {
                if (symbol.definition != target.definition) continue;
                const rendered = try std.fmt.allocPrint(arena, "```alloy\n{s}\n```", .{symbol.detail});
                return self.respond(id, .{
                    .contents = .{ .kind = "markdown", .value = rendered },
                });
            }
        }
        const word = self.wordAtRequestPosition(params) orelse return self.respond(id, null);
        for (self.symbols.items) |symbol| {
            if (!std.mem.eql(u8, symbol.name, word)) continue;
            const rendered = try std.fmt.allocPrint(arena, "```alloy\n{s}\n```", .{symbol.detail});
            return self.respond(id, .{
                .contents = .{ .kind = "markdown", .value = rendered },
            });
        }
        try self.respond(id, null);
    }

    // the checked target of the call whose callee name token covers the
    // cursor. Scanned backwards: a re-analysis appends fresh records last
    fn callTargetAtPosition(self: *Server, params: std.json.Value) ?resolution.Symbol {
        const location = self.requestLocation(params) orelse return null;
        const view_index = (self.viewIndexOfPath(location.path) catch null) orelse return null;
        const unit = self.analysis orelse return null;
        const checker = unit.checker orelse return null;
        const targets = checker.call_name_targets.items;
        var index = targets.len;
        while (index > 0) {
            index -= 1;
            const target = targets[index];
            if (target.view_index != view_index) continue;
            if (location.offset < target.span.start or location.offset > target.span.end) continue;
            return target.symbol;
        }
        return null;
    }

    const TypedNode = struct {
        word: []const u8,
        recorded: *const @import("types.zig").Type,
    };

    // the single-segment path expression under the cursor that the checker
    // recorded a type for
    fn typedNodeAtPosition(self: *Server, params: std.json.Value) ?TypedNode {
        const location = self.requestLocation(params) orelse return null;
        // resolve the view FIRST: it may re-analyze and replace the unit
        const view_index = (self.viewIndexOfPath(location.path) catch null) orelse return null;
        const unit = self.analysis orelse return null;
        const view = unit.views[view_index];
        // global definitions hover through the symbol table instead
        const word = wordAt(location.text, location.offset) orelse return null;
        for (self.symbols.items) |symbol| {
            if (std.mem.eql(u8, symbol.name, word)) return null;
        }
        const arena = self.message_arena.allocator();
        var paths: std.ArrayList(*const ast.Expression) = .empty;
        collectPathExpressions(arena, view.module, &paths) catch return null;
        for (paths.items) |candidate| {
            const segments = candidate.path;
            if (segments.len != 1) continue;
            const span = segments[0].location;
            if (location.offset < span.start or location.offset > span.end) continue;
            const recorded = unit.expression_types.get(candidate) orelse place: {
                const checker = unit.checker orelse break :place null;
                break :place checker.place_types.get(candidate);
            } orelse continue;
            return .{ .word = segments[0].slice(view.source), .recorded = recorded };
        }
        // declaration sites are not expressions; the checker records them
        // separately. Scanned backwards so a rebind (a capture bound as a
        // placeholder first, then with its real type) answers with the
        // refined entry.
        if (unit.checker) |checker| {
            const declarations = checker.declaration_types.items;
            var index = declarations.len;
            while (index > 0) {
                index -= 1;
                const declaration = declarations[index];
                if (declaration.view_index != view_index) continue;
                if (location.offset < declaration.span.start or location.offset > declaration.span.end) continue;
                return .{
                    .word = view.source[declaration.span.start..declaration.span.end],
                    .recorded = declaration.binding_type,
                };
            }
        }
        return null;
    }

    const LspLocation = struct {
        uri: []const u8,
        range: Range,
    };

    fn definition(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        // a local's first recorded occurrence is its declaration: bind()
        // records the declaring token before any use resolves
        if (try self.collectLocalOccurrences(arena, params)) |locations| {
            if (locations.len != 0) return self.respond(id, locations[0]);
        }
        // a '::'-qualified or implied enum variant navigates to the member
        if (try self.enumMemberDefinition(arena, params)) |member_location| {
            return self.respond(id, member_location);
        }
        // a call's checked target beats matching globals by bare name
        if (self.callTargetAtPosition(params)) |target| {
            for (self.symbols.items) |symbol| {
                if (symbol.definition != target.definition) continue;
                return self.respond(id, LspLocation{
                    .uri = symbol.uri,
                    .range = symbol.range,
                });
            }
        }
        const word = self.wordAtRequestPosition(params) orelse return self.respond(id, null);
        for (self.symbols.items) |symbol| {
            if (!std.mem.eql(u8, symbol.name, word)) continue;
            return self.respond(id, LspLocation{
                .uri = symbol.uri,
                .range = symbol.range,
            });
        }
        try self.respond(id, null);
    }

    // the cursor sits on a variant name written '::Variant' (implied) or
    // 'Enum::Variant'; the target is the enum member's declaration. A
    // module-qualified path ('option::Option') finds no type named by its
    // qualifier and falls through to the global symbol lookup
    fn enumMemberDefinition(self: *Server, arena: std.mem.Allocator, params: std.json.Value) !?LspLocation {
        const location = self.requestLocation(params) orelse return null;
        _ = (self.viewIndexOfPath(location.path) catch null) orelse return null;
        const unit = self.analysis orelse return null;
        const word = wordAt(location.text, location.offset) orelse return null;
        const word_start = @intFromPtr(word.ptr) - @intFromPtr(location.text.ptr);
        if (word_start < 2 or location.text[word_start - 1] != ':' or location.text[word_start - 2] != ':') return null;
        // a qualifier must touch the '::' directly; an implied variant
        // ('is ::Some') has none
        var qualifier: ?[]const u8 = null;
        if (word_start >= 3 and isIdentifierByte(location.text[word_start - 3])) {
            qualifier = wordAt(location.text, word_start - 3);
        }
        var found: ?LspLocation = null;
        var matches: usize = 0;
        for (unit.views) |view| {
            for (view.module.definitions) |*module_definition| {
                const type_def = switch (module_definition.kind) {
                    .type_def => |*type_def| type_def,
                    else => continue,
                };
                if (qualifier) |name| {
                    if (!std.mem.eql(u8, type_def.name.slice(view.source), name)) continue;
                }
                const members = switch (type_def.base.*) {
                    .enum_type => |members| members,
                    else => continue,
                };
                for (members) |member| {
                    if (!std.mem.eql(u8, member.name.slice(view.source), word)) continue;
                    matches += 1;
                    found = .{
                        .uri = try self.uriOfViewPath(arena, view.path),
                        .range = .{
                            .start = positionOf(view.source, member.name.location.start),
                            .end = positionOf(view.source, member.name.location.end),
                        },
                    };
                }
            }
        }
        // an implied '::Variant' must be unambiguous, mirroring the checker
        if (qualifier == null and matches != 1) return null;
        return found;
    }

    // every declaration and recorded use of the word's definitions; all
    // overloads of a function name count as one group
    fn collectOccurrences(self: *Server, arena: std.mem.Allocator, word: []const u8) ![]const LspLocation {
        const unit = self.analysis orelse return &.{};
        const merged = unit.merged orelse return &.{};
        var targets: std.ArrayList(*const ast.Definition) = .empty;
        var locations: std.ArrayList(LspLocation) = .empty;
        for (self.symbols.items) |symbol| {
            if (!std.mem.eql(u8, symbol.name, word)) continue;
            try targets.append(arena, symbol.definition);
            try locations.append(arena, .{ .uri = symbol.uri, .range = symbol.range });
        }
        if (targets.items.len == 0) return &.{};
        for (merged.references) |reference| {
            const matches = for (targets.items) |target| {
                if (reference.definition == target) break true;
            } else false;
            if (!matches) continue;
            const view = unit.views[reference.view_index];
            try locations.append(arena, .{
                .uri = try self.uriOfViewPath(arena, view.path),
                .range = .{
                    .start = positionOf(view.source, reference.span.start),
                    .end = positionOf(view.source, reference.span.end),
                },
            });
        }
        return locations.items;
    }

    // local (parameter, variable, capture, type parameter) occurrences:
    // the binding groups whose recorded spans contain the cursor, unioned
    // so a capture token renames both the inner and the outer binding
    fn collectLocalOccurrences(self: *Server, arena: std.mem.Allocator, params: std.json.Value) !?[]const LspLocation {
        const location = self.requestLocation(params) orelse return null;
        const view_index = (self.viewIndexOfPath(location.path) catch null) orelse return null;
        const unit = self.analysis orelse return null;
        const merged = unit.merged orelse return null;
        var ids: std.ArrayList(usize) = .empty;
        for (merged.locals) |local| {
            if (local.view_index != view_index) continue;
            if (location.offset < local.span.start or location.offset > local.span.end) continue;
            try ids.append(arena, local.binding_id);
        }
        if (ids.items.len == 0) return null;
        const view = unit.views[view_index];
        var locations: std.ArrayList(LspLocation) = .empty;
        for (merged.locals) |local| {
            if (local.view_index != view_index) continue;
            const matches = for (ids.items) |binding_id| {
                if (local.binding_id == binding_id) break true;
            } else false;
            if (!matches) continue;
            const range: Range = .{
                .start = positionOf(view.source, local.span.start),
                .end = positionOf(view.source, local.span.end),
            };
            // a capture token carries both a declaration and an outer use
            const duplicate = for (locations.items) |existing| {
                if (existing.range.start.line == range.start.line and
                    existing.range.start.character == range.start.character) break true;
            } else false;
            if (duplicate) continue;
            try locations.append(arena, .{
                .uri = try self.uriOfViewPath(arena, view.path),
                .range = range,
            });
        }
        return locations.items;
    }

    fn references(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        if (try self.collectLocalOccurrences(arena, params)) |locations| {
            return self.respond(id, locations);
        }
        const word = self.wordAtRequestPosition(params) orelse return self.respond(id, null);
        const locations = try self.collectOccurrences(arena, word);
        if (locations.len == 0) return self.respond(id, null);
        try self.respond(id, locations);
    }

    const TextEdit = struct {
        range: Range,
        newText: []const u8,
    };

    fn rename(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        const new_name = if (params == .object) stringField(params, "newName") else null;
        const replacement = new_name orelse return self.respondError(id, -32602, "rename needs a newName");
        if (!isValidIdentifier(replacement)) {
            return self.respondError(id, -32602, "the new name is not a valid identifier");
        }
        const locations = if (try self.collectLocalOccurrences(arena, params)) |local_occurrences|
            local_occurrences
        else global: {
            const word = self.wordAtRequestPosition(params) orelse
                return self.respondError(id, -32602, "nothing to rename here");
            break :global try self.collectOccurrences(arena, word);
        };
        if (locations.len == 0) {
            return self.respondError(id, -32602, "nothing to rename here");
        }
        const FileEdits = struct {
            textDocument: struct { uri: []const u8, version: ?u32 = null },
            edits: []const TextEdit,
        };
        var groups: std.StringArrayHashMapUnmanaged(std.ArrayList(TextEdit)) = .empty;
        for (locations) |location| {
            const group = try groups.getOrPut(arena, location.uri);
            if (!group.found_existing) group.value_ptr.* = .empty;
            try group.value_ptr.append(arena, .{ .range = location.range, .newText = replacement });
        }
        var files: std.ArrayList(FileEdits) = .empty;
        var iterator = groups.iterator();
        while (iterator.next()) |entry| {
            try files.append(arena, .{
                .textDocument = .{ .uri = entry.key_ptr.* },
                .edits = entry.value_ptr.items,
            });
        }
        try self.respond(id, .{ .documentChanges = files.items });
    }

    fn documentSymbols(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        const document = objectPath(params, &.{"textDocument"}) orelse return self.respond(id, null);
        const uri = stringField(document, "uri") orelse return self.respond(id, null);
        const path = self.pathOfUri(uri) orelse return self.respond(id, null);
        const view_index = (try self.viewIndexOfPath(path)) orelse return self.respond(id, null);
        const unit = self.analysis.?;
        const view = unit.views[view_index];
        const DocumentSymbol = struct {
            name: []const u8,
            kind: u32,
            range: Range,
            selectionRange: Range,
        };
        var results: std.ArrayList(DocumentSymbol) = .empty;
        for (view.module.definitions) |*module_definition| {
            const name_token = definitionName(module_definition);
            const range: Range = .{
                .start = positionOf(view.source, name_token.location.start),
                .end = positionOf(view.source, name_token.location.end),
            };
            try results.append(arena, .{
                .name = name_token.slice(view.source),
                .kind = documentSymbolKind(module_definition),
                .range = range,
                .selectionRange = range,
            });
        }
        try self.respond(id, results.items);
    }

    fn workspaceSymbols(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        const query = if (params == .object) stringField(params, "query") orelse "" else "";
        const SymbolInformation = struct {
            name: []const u8,
            kind: u32,
            location: LspLocation,
        };
        var results: std.ArrayList(SymbolInformation) = .empty;
        for (self.symbols.items) |symbol| {
            if (query.len != 0 and std.ascii.indexOfIgnoreCase(symbol.name, query) == null) continue;
            try results.append(arena, .{
                .name = symbol.name,
                .kind = documentSymbolKind(symbol.definition),
                .location = .{ .uri = symbol.uri, .range = symbol.range },
            });
        }
        try self.respond(id, results.items);
    }

    // classifies identifier tokens by what their name resolves to; lexical
    // coloring stays with the TextMate grammar
    fn semanticTokens(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        const document = objectPath(params, &.{"textDocument"}) orelse return self.respond(id, null);
        const uri = stringField(document, "uri") orelse return self.respond(id, null);
        const path = self.pathOfUri(uri) orelse return self.respond(id, null);
        const open_document = self.documents.get(path) orelse return self.respond(id, null);
        const text = open_document.text;

        // resolution-exact classification: every occurrence colors by what
        // the resolver actually bound it to, never by name coincidence (a
        // module or member sharing a macro's name stays uncolored). Only a
        // current analysis lines up with the open text; a stale one kept
        // past a failed parse falls back to name-based global classes so
        // colors stay stable while typing.
        const SpanClass = struct { end: usize, class: u32 };
        var span_classes: std.AutoHashMapUnmanaged(usize, SpanClass) = .empty;
        var fresh = false;
        if (self.viewIndexOfPath(path) catch null) |view_index| {
            if (self.analysis) |unit| {
                if (unit.merged) |merged| {
                    const view = unit.views[view_index];
                    if (std.mem.eql(u8, view.source, text)) {
                        fresh = true;
                        for (merged.locals) |local| {
                            if (local.view_index != view_index) continue;
                            const class: u32 = switch (local.kind) {
                                .variable => 4,
                                .parameter => 5,
                                .type_parameter => 6,
                            };
                            try span_classes.put(arena, local.span.start, .{ .end = local.span.end, .class = class });
                        }
                        // resolved uses of globals; multi-token qualified
                        // paths are left to the grammar
                        for (merged.references) |reference| {
                            if (reference.view_index != view_index) continue;
                            try span_classes.put(arena, reference.span.start, .{
                                .end = reference.span.end,
                                .class = definitionClass(reference.definition),
                            });
                        }
                        // the definitions' own name tokens
                        for (view.module.definitions) |*module_definition| {
                            const name_token = definitionName(module_definition);
                            try span_classes.put(arena, name_token.location.start, .{
                                .end = name_token.location.end,
                                .class = definitionClass(module_definition),
                            });
                        }
                    }
                }
            }
        }

        var name_classes: std.StringHashMapUnmanaged(u32) = .empty;
        if (!fresh) {
            for (self.symbols.items) |symbol| {
                const entry = try name_classes.getOrPut(arena, symbol.name);
                if (!entry.found_existing) entry.value_ptr.* = definitionClass(symbol.definition);
            }
        }

        var data: std.ArrayList(u32) = .empty;
        var scanner = tokenizer_module.Tokenizer.init(text);
        var previous_line: u32 = 0;
        var previous_character: u32 = 0;
        while (true) {
            const token = scanner.next();
            if (token.tag == .end_of_file) break;
            if (token.tag != .identifier) continue;
            const class = class: {
                if (fresh) {
                    const found = span_classes.get(token.location.start) orelse continue;
                    // a single-token match only: a qualified path's span
                    // covers several tokens and stays grammar-colored
                    if (found.end != token.location.end) continue;
                    break :class found.class;
                }
                break :class name_classes.get(token.slice(text)) orelse continue;
            };
            const position = positionOf(text, token.location.start);
            const length = utf16Length(text[token.location.start..token.location.end]);
            const delta_line = position.line - previous_line;
            const delta_character = if (delta_line == 0) position.character - previous_character else position.character;
            try data.appendSlice(arena, &.{ delta_line, delta_character, length, class, 0 });
            previous_line = position.line;
            previous_character = position.character;
        }
        try self.respond(id, .{ .data = data.items });
    }

    fn signatureHelp(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        const location = self.requestLocation(params) orelse return self.respond(id, null);
        const call = enclosingCall(location.text, location.offset) orelse return self.respond(id, null);
        const ParameterInformation = struct { label: []const u8 };
        const SignatureInformation = struct {
            label: []const u8,
            parameters: []const ParameterInformation,
        };
        var signatures: std.ArrayList(SignatureInformation) = .empty;
        var active_signature: u32 = 0;
        for (self.symbols.items) |symbol| {
            if (!std.mem.eql(u8, symbol.name, call.name)) continue;
            const source = if (self.analysis) |unit| unit.views[symbol.view_index].source else "";
            var labels: std.ArrayList(ParameterInformation) = .empty;
            switch (symbol.definition.kind) {
                .fn_def => |def| for (def.function.parameters) |parameter| {
                    if (parameter.is_self) continue;
                    try labels.append(arena, .{ .label = parameter.name.slice(source) });
                },
                .extern_def => |def| for (def.parameters) |parameter| {
                    try labels.append(arena, .{ .label = parameter.name.slice(source) });
                },
                .macro_def => |def| for (def.parameters) |parameter| {
                    try labels.append(arena, .{ .label = parameter.name.slice(source) });
                },
                else => continue,
            }
            // prefer the overload that still has room for the argument
            // being typed
            if (labels.items.len > call.commas and active_signature == 0 and signatures.items.len != 0) {
                active_signature = @intCast(signatures.items.len);
            }
            try signatures.append(arena, .{ .label = symbol.detail, .parameters = labels.items });
        }
        if (signatures.items.len == 0) return self.respond(id, null);
        try self.respond(id, .{
            .signatures = signatures.items,
            .activeSignature = active_signature,
            .activeParameter = call.commas,
        });
    }

    // whole-document formatting: one edit replacing everything, or none
    // when the text is already canonical or has syntax errors
    fn formatting(self: *Server, id: std.json.Value, params: std.json.Value) !void {
        const arena = self.message_arena.allocator();
        const document = objectPath(params, &.{"textDocument"}) orelse return self.respond(id, null);
        const uri = stringField(document, "uri") orelse return self.respond(id, null);
        const path = self.pathOfUri(uri) orelse return self.respond(id, null);
        const open_document = self.documents.get(path) orelse return self.respond(id, null);
        const formatted = formatter.format(arena, open_document.text) catch return self.respond(id, null);
        if (std.mem.eql(u8, formatted, open_document.text)) {
            return self.respond(id, &[_]TextEdit{});
        }
        try self.respond(id, &[_]TextEdit{.{
            .range = .{
                .start = .{ .line = 0, .character = 0 },
                .end = positionOf(open_document.text, open_document.text.len),
            },
            .newText = formatted,
        }});
    }

    const RequestLocation = struct {
        path: []const u8,
        text: []const u8,
        offset: usize,
    };

    fn requestLocation(self: *Server, params: std.json.Value) ?RequestLocation {
        const document = objectPath(params, &.{"textDocument"}) orelse return null;
        const uri = stringField(document, "uri") orelse return null;
        const position = objectPath(params, &.{"position"}) orelse return null;
        const line = integerField(position, "line") orelse return null;
        const character = integerField(position, "character") orelse return null;
        const path = self.pathOfUri(uri) orelse return null;
        const open_document = self.documents.get(path) orelse return null;
        return .{
            .path = path,
            .text = open_document.text,
            .offset = offsetOf(open_document.text, @intCast(line), @intCast(character)),
        };
    }

    // the identifier under the request's text document position
    fn wordAtRequestPosition(self: *Server, params: std.json.Value) ?[]const u8 {
        const location = self.requestLocation(params) orelse return null;
        return wordAt(location.text, location.offset);
    }
};

fn definitionName(definition: *const ast.Definition) Token {
    return switch (definition.kind) {
        .type_def => |def| def.name,
        .fn_def => |def| def.name,
        .extern_def => |def| def.name,
        .interface_def => |def| def.name,
        .macro_def => |def| def.name,
    };
}

// the semantic token class of a definition's kind (legend order)
fn definitionClass(definition: *const ast.Definition) u32 {
    return switch (definition.kind) {
        .fn_def, .extern_def => 0,
        .type_def => 1,
        .interface_def => 2,
        .macro_def => 3,
    };
}

fn documentSymbolKind(definition: *const ast.Definition) u32 {
    return switch (definition.kind) {
        .fn_def, .extern_def, .macro_def => 12,
        .interface_def => 11,
        .type_def => |type_def| if (type_def.base.* == .enum_type) @as(u32, 10) else 23,
    };
}

// the receiver type name an extension's self parameter declares
fn selfTypeName(parameter_type: *const ast.TypeExpression, source: []const u8) ?[]const u8 {
    var current = parameter_type;
    while (current.* == .modified) current = current.modified.child;
    if (current.* != .named) return null;
    const path = current.named.path;
    return path[path.len - 1].slice(source);
}

const EnclosingCall = struct {
    name: []const u8,
    commas: u32,
};

// scans backwards for the unbalanced '(' the cursor sits inside, counting
// top-level commas for the active parameter
fn enclosingCall(text: []const u8, offset: usize) ?EnclosingCall {
    var depth: i32 = 0;
    var commas: u32 = 0;
    var index = @min(offset, text.len);
    while (index > 0) {
        index -= 1;
        switch (text[index]) {
            ')', ']', '}' => depth += 1,
            '[', '{' => {
                if (depth == 0) return null;
                depth -= 1;
            },
            '(' => {
                if (depth > 0) {
                    depth -= 1;
                    continue;
                }
                var name_end = index;
                while (name_end > 0 and text[name_end - 1] == ' ') name_end -= 1;
                const name = wordAt(text, name_end -| 1) orelse return null;
                return .{ .name = name, .commas = commas };
            },
            ',' => {
                if (depth == 0) commas += 1;
            },
            ';' => {
                if (depth == 0) return null;
            },
            else => {},
        }
    }
    return null;
}

// gathers every path expression in a module, for type-at-position queries
fn collectPathExpressions(arena: std.mem.Allocator, module: *const ast.Module, into: *std.ArrayList(*const ast.Expression)) !void {
    for (module.definitions) |definition| {
        switch (definition.kind) {
            .fn_def => |fn_def| try collectFromStatement(arena, fn_def.function.body, into),
            .macro_def => |macro_def| if (macro_def.body) |body| try collectFromStatement(arena, body, into),
            else => {},
        }
    }
}

fn collectFromStatement(arena: std.mem.Allocator, statement: *const ast.Statement, into: *std.ArrayList(*const ast.Expression)) error{OutOfMemory}!void {
    switch (statement.*) {
        .block => |statements| for (statements) |child| try collectFromStatement(arena, child, into),
        .var_def => |var_def| try collectFromExpression(arena, var_def.value, into),
        .assign => |assign| {
            try collectFromExpression(arena, assign.target, into);
            try collectFromExpression(arena, assign.value, into);
        },
        .expression => |expression| try collectFromExpression(arena, expression, into),
        .break_stmt => |break_stmt| if (break_stmt.value) |value| try collectFromExpression(arena, value, into),
        .yield_stmt => |yield_stmt| try collectFromExpression(arena, yield_stmt.value, into),
        .return_stmt => |return_stmt| if (return_stmt.value) |value| try collectFromExpression(arena, value, into),
    }
}

fn collectFromExpression(arena: std.mem.Allocator, expression: *const ast.Expression, into: *std.ArrayList(*const ast.Expression)) error{OutOfMemory}!void {
    switch (expression.*) {
        .path => try into.append(arena, expression),
        .integer_literal, .float_literal, .string_literal, .character_literal, .bool_literal, .implied_variant => {},
        .binary => |binary| {
            try collectFromExpression(arena, binary.left, into);
            try collectFromExpression(arena, binary.right, into);
        },
        .unary => |unary| try collectFromExpression(arena, unary.operand, into),
        .cast => |cast| try collectFromExpression(arena, cast.operand, into),
        .call => |call| {
            try collectFromExpression(arena, call.callee, into);
            for (call.arguments) |argument| try collectFromExpression(arena, argument, into);
        },
        .member => |member| try collectFromExpression(arena, member.object, into),
        .index => |index| {
            try collectFromExpression(arena, index.object, into);
            try collectFromExpression(arena, index.subscript, into);
        },
        .subslice => |subslice| {
            try collectFromExpression(arena, subslice.object, into);
            if (subslice.start) |start| try collectFromExpression(arena, start, into);
            try collectFromExpression(arena, subslice.end, into);
        },
        .struct_init => |struct_init| for (struct_init.members) |member| try collectFromExpression(arena, member.value, into),
        .array_literal => |elements| for (elements) |element| try collectFromExpression(arena, element, into),
        .array_fill => |fill| {
            try collectFromExpression(arena, fill.value, into);
            try collectFromExpression(arena, fill.count, into);
        },
        .array_range => |range| {
            if (range.start) |start| try collectFromExpression(arena, start, into);
            try collectFromExpression(arena, range.end, into);
        },
        .if_expr => |if_expr| {
            try collectFromExpression(arena, if_expr.condition, into);
            try collectFromStatement(arena, if_expr.then_branch, into);
            if (if_expr.else_branch) |branch| try collectFromStatement(arena, branch, into);
        },
        .while_expr => |while_expr| {
            try collectFromExpression(arena, while_expr.condition, into);
            try collectFromStatement(arena, while_expr.body, into);
            if (while_expr.else_branch) |branch| try collectFromStatement(arena, branch, into);
        },
        .for_expr => |for_expr| {
            for (for_expr.subjects) |subject| try collectFromExpression(arena, subject, into);
            try collectFromStatement(arena, for_expr.body, into);
            if (for_expr.else_branch) |branch| try collectFromStatement(arena, branch, into);
        },
        .match_expr => |match_expr| {
            try collectFromExpression(arena, match_expr.subject, into);
            for (match_expr.arms) |arm| {
                if (arm.pattern) |pattern| try collectFromExpression(arena, pattern, into);
                try collectFromStatement(arena, arm.body, into);
            }
            if (match_expr.else_branch) |branch| try collectFromStatement(arena, branch, into);
        },
        .lambda => |lambda| try collectFromStatement(arena, lambda.function.body, into),
        .comptime_expr => |inner| try collectFromExpression(arena, inner, into),
        .grouped => |inner| try collectFromExpression(arena, inner, into),
    }
}

// the full definition text for hover: types and interfaces show their
// whole multi-line body, functions and macros their signature up to the
// body brace, externs their declaration line
fn definitionDetail(source: []const u8, name_offset: usize, definition: *const ast.Definition) []const u8 {
    var start = name_offset;
    while (start > 0 and source[start - 1] != '\n') start -= 1;
    const full_body = switch (definition.kind) {
        .type_def, .interface_def => true,
        .fn_def, .macro_def, .extern_def => false,
    };
    var depth: usize = 0;
    var index = name_offset;
    while (index < source.len) : (index += 1) {
        switch (source[index]) {
            '{' => {
                // a signature stops before the body it would otherwise drag in
                if (!full_body and depth == 0) {
                    return std.mem.trimEnd(u8, source[start..index], " \t\r\n");
                }
                depth += 1;
            },
            '}' => {
                depth -|= 1;
                if (full_body and depth == 0) {
                    var end = index + 1;
                    if (end < source.len and source[end] == ';') end += 1;
                    return source[start..end];
                }
            },
            ';' => if (depth == 0) return source[start..index],
            else => {},
        }
    }
    return declarationLine(source, name_offset);
}

// the whole source line containing the offset, trimmed, as a signature
fn declarationLine(source: []const u8, offset: usize) []const u8 {
    var line_start = offset;
    while (line_start > 0 and source[line_start - 1] != '\n') line_start -= 1;
    var line_end = offset;
    while (line_end < source.len and source[line_end] != '\n') line_end += 1;
    return std.mem.trim(u8, source[line_start..line_end], " \t\r");
}

fn wordAt(text: []const u8, offset: usize) ?[]const u8 {
    if (text.len == 0) return null;
    var start = @min(offset, text.len);
    // the cursor may sit just past the word's last character
    if (start == text.len or !isIdentifierByte(text[start])) {
        if (start == 0 or !isIdentifierByte(text[start - 1])) return null;
        start -= 1;
    }
    var end = start + 1;
    while (start > 0 and isIdentifierByte(text[start - 1])) start -= 1;
    while (end < text.len and isIdentifierByte(text[end])) end += 1;
    return text[start..end];
}

fn isIdentifierByte(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_' or byte >= 0x80;
}

fn isValidIdentifier(name: []const u8) bool {
    if (name.len == 0) return false;
    if (std.ascii.isDigit(name[0])) return false;
    for (name) |byte| {
        if (!isIdentifierByte(byte)) return false;
    }
    return true;
}

// byte offset to a protocol position: 0-based line, UTF-16 column
pub fn positionOf(text: []const u8, offset: usize) Server.Position {
    const clamped = @min(offset, text.len);
    var line: u32 = 0;
    var line_start: usize = 0;
    for (text[0..clamped], 0..) |byte, index| {
        if (byte == '\n') {
            line += 1;
            line_start = index + 1;
        }
    }
    return .{ .line = line, .character = utf16Length(text[line_start..clamped]) };
}

// protocol position back to a byte offset, clamping past-the-end columns
pub fn offsetOf(text: []const u8, line: u32, character: u32) usize {
    var current_line: u32 = 0;
    var index: usize = 0;
    while (current_line < line) {
        const newline = std.mem.indexOfScalarPos(u8, text, index, '\n') orelse return text.len;
        index = newline + 1;
        current_line += 1;
    }
    var remaining = character;
    while (remaining > 0 and index < text.len and text[index] != '\n') {
        const sequence_length = std.unicode.utf8ByteSequenceLength(text[index]) catch 1;
        const units: u32 = if (sequence_length == 4) 2 else 1;
        if (units > remaining) break;
        remaining -= units;
        index = @min(index + sequence_length, text.len);
    }
    return index;
}

fn positionFieldOffset(text: []const u8, range: std.json.Value, key: []const u8) ?usize {
    const position = objectPath(range, &.{key}) orelse return null;
    const line = integerField(position, "line") orelse return null;
    const character = integerField(position, "character") orelse return null;
    if (line < 0 or character < 0) return null;
    return offsetOf(text, @intCast(line), @intCast(character));
}

fn utf16Length(bytes: []const u8) u32 {
    var units: u32 = 0;
    var index: usize = 0;
    while (index < bytes.len) {
        const sequence_length = std.unicode.utf8ByteSequenceLength(bytes[index]) catch 1;
        units += if (sequence_length == 4) 2 else 1;
        index += sequence_length;
    }
    return units;
}

// 'file:///c%3A/dev/x.alloy' to 'c:/dev/x.alloy', percent-decoded, with
// backslashes normalized to forward slashes and the drive lowercased
pub fn normalizedPathFromUri(allocator: std.mem.Allocator, uri: []const u8) ?[]const u8 {
    const scheme = "file://";
    if (!std.mem.startsWith(u8, uri, scheme)) return null;
    const rest = uri[scheme.len..];
    var decoded: std.ArrayList(u8) = .empty;
    var index: usize = 0;
    while (index < rest.len) : (index += 1) {
        const byte = rest[index];
        if (byte == '%' and index + 2 < rest.len) {
            const value = std.fmt.parseInt(u8, rest[index + 1 .. index + 3], 16) catch {
                decoded.append(allocator, byte) catch return null;
                continue;
            };
            decoded.append(allocator, value) catch return null;
            index += 2;
        } else {
            decoded.append(allocator, if (byte == '\\') '/' else byte) catch return null;
        }
    }
    var path = decoded.items;
    // '/c:/...' drops the leading slash on windows-style drive paths
    if (path.len >= 3 and path[0] == '/' and path[2] == ':') {
        path = path[1..];
    }
    if (path.len >= 2 and path[1] == ':') {
        path[0] = std.ascii.toLower(path[0]);
    }
    return path;
}

pub fn uriFromPath(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    var encoded: std.ArrayList(u8) = .empty;
    try encoded.appendSlice(allocator, "file:///");
    const trimmed = std.mem.trimStart(u8, path, "/");
    for (trimmed) |byte| {
        switch (byte) {
            '\\' => try encoded.append(allocator, '/'),
            ' ', '%', '#', '?' => {
                var buffer: [3]u8 = undefined;
                const hex = std.fmt.bufPrint(&buffer, "%{X:0>2}", .{byte}) catch unreachable;
                try encoded.appendSlice(allocator, hex);
            },
            else => try encoded.append(allocator, byte),
        }
    }
    return encoded.toOwnedSlice(allocator);
}

fn joinNormalized(allocator: std.mem.Allocator, base: []const u8, relative: []const u8) ![]u8 {
    const joined = if (base.len == 0)
        try allocator.dupe(u8, relative)
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, relative });
    for (joined) |*byte| {
        if (byte.* == '\\') byte.* = '/';
    }
    if (joined.len >= 2 and joined[1] == ':') {
        joined[0] = std.ascii.toLower(joined[0]);
    }
    return joined;
}

fn directoryOf(path: []const u8) []const u8 {
    const slash = std.mem.lastIndexOfScalar(u8, path, '/') orelse return "";
    return path[0..slash];
}

fn objectPath(value: std.json.Value, keys: []const []const u8) ?std.json.Value {
    var current = value;
    for (keys) |key| {
        if (current != .object) return null;
        current = current.object.get(key) orelse return null;
    }
    if (current != .object) return null;
    return current;
}

fn stringField(value: std.json.Value, key: []const u8) ?[]const u8 {
    if (value != .object) return null;
    const field = value.object.get(key) orelse return null;
    if (field != .string) return null;
    return field.string;
}

fn integerField(value: std.json.Value, key: []const u8) ?i64 {
    if (value != .object) return null;
    const field = value.object.get(key) orelse return null;
    if (field != .integer) return null;
    return field.integer;
}

test "positions map byte offsets to utf-16 lines and columns" {
    const text = "first\nsecond line\nthird";
    const position = positionOf(text, 13);
    try std.testing.expectEqual(@as(u32, 1), position.line);
    try std.testing.expectEqual(@as(u32, 7), position.character);
    try std.testing.expectEqual(@as(usize, 13), offsetOf(text, 1, 7));
    // a column past the line end clamps to the newline
    try std.testing.expectEqual(@as(usize, 5), offsetOf(text, 0, 99));
}

test "uris round-trip through normalized paths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const path = normalizedPathFromUri(arena.allocator(), "file:///C%3A/Dev/zig/alloyc/main.alloy").?;
    try std.testing.expectEqualStrings("c:/Dev/zig/alloyc/main.alloy", path);
    const uri = try uriFromPath(arena.allocator(), "c:/Dev/zig alloyc/main.alloy");
    try std.testing.expectEqualStrings("file:///c:/Dev/zig%20alloyc/main.alloy", uri);
}

test "the server answers references, rename, symbols, tokens, help, and hover" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const uri = "file:///c%3A/probe/session.alloy";
    const source =
        "fn helper(value: i64) -> i64 { return value + 1; }\n" ++
        "fn main() -> i32 {\n" ++
        "    var total = helper(4);\n" ++
        "    return total to i32;\n" ++
        "}\n";

    var frames: std.ArrayList(u8) = .empty;
    const messages = [_][]const u8{
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didOpen",
            .params = .{ .textDocument = .{ .uri = uri, .languageId = "alloy", .version = 1, .text = source } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 2,
            .method = "textDocument/references",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 2, .character = 17 } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 3,
            .method = "textDocument/rename",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 2, .character = 17 }, .newName = "boost" },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 4,
            .method = "textDocument/documentSymbol",
            .params = .{ .textDocument = .{ .uri = uri } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 5,
            .method = "textDocument/semanticTokens/full",
            .params = .{ .textDocument = .{ .uri = uri } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 6,
            .method = "textDocument/signatureHelp",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 2, .character = 23 } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 7,
            .method = "textDocument/hover",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 3, .character = 12 } },
        }, .{}),
        // an incremental splice replaces the callee with an unknown name
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didChange",
            .params = .{
                .textDocument = .{ .uri = uri, .version = 2 },
                .contentChanges = &[_]struct {
                    range: Server.Range,
                    text: []const u8,
                }{.{
                    .range = .{ .start = .{ .line = 2, .character = 16 }, .end = .{ .line = 2, .character = 22 } },
                    .text = "missing",
                }},
            },
        }, .{}),
    };
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, std.testing.io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    const transcript = output.writer.buffered();
    // references: the declaration on line 0 and the call on line 2
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"id\":2,\"result\":[") != null);
    // rename produced a workspace edit carrying the new name
    try std.testing.expect(std.mem.indexOf(u8, transcript, "documentChanges") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"newText\":\"boost\"") != null);
    // the outline names both functions
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"helper\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"main\"") != null);
    // semantic tokens carry classified identifiers
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"data\":[0,3,6,0,0") != null);
    // signature help names the parameter and the active argument
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"activeParameter\":0") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"label\":\"value\"") != null);
    // hover on a local shows its checked type
    try std.testing.expect(std.mem.indexOf(u8, transcript, "total: i64") != null);
    // the incremental change re-ran the pipeline and found the new error
    try std.testing.expect(std.mem.indexOf(u8, transcript, "use of undeclared identifier 'missing'") != null);
}

test "declaration sites hover and locals classify as semantic tokens" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const uri = "file:///c%3A/probe/locals.alloy";
    const source =
        "fn helper(value: i64) -> i64 {\n" ++
        "    var total: i64 = value;\n" ++
        "    return total;\n" ++
        "}\n";

    var frames: std.ArrayList(u8) = .empty;
    const messages = [_][]const u8{
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didOpen",
            .params = .{ .textDocument = .{ .uri = uri, .languageId = "alloy", .version = 1, .text = source } },
        }, .{}),
        // hover the parameter's declaration name, not a use
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 2,
            .method = "textDocument/hover",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 0, .character = 11 } },
        }, .{}),
        // hover the variable's declaration name
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 3,
            .method = "textDocument/hover",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 1, .character = 9 } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 4,
            .method = "textDocument/semanticTokens/full",
            .params = .{ .textDocument = .{ .uri = uri } },
        }, .{}),
    };
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, std.testing.io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    const transcript = output.writer.buffered();
    // both declaration names hover with their checked types
    try std.testing.expect(std.mem.indexOf(u8, transcript, "value: i64") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "total: i64") != null);
    // 'helper' classifies as function (0), 'value' as parameter (5), then
    // 'total' and the following uses as variable (4) / parameter (5)
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"data\":[0,3,6,0,0,0,7,5,5,0,1,8,5,4,0,0,13,5,5,0,1,11,5,4,0]") != null);
}

test "a '#TypeName.' receiver completes the reflection methods" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const uri = "file:///c%3A/probe/reflect_completion.alloy";
    const good =
        "type Point = struct { x: i32 };\n" ++
        "fn main() -> i32 { return 0; }\n";
    // typing '#Point.' mid-edit: no prior typed use of 'Point' exists,
    // the '#' plus the type name alone must be enough
    const typing =
        "type Point = struct { x: i32 };\n" ++
        "fn main() -> i32 {\n" ++
        "    const t = #Point.\n" ++
        "    return 0;\n" ++
        "}\n";

    var frames: std.ArrayList(u8) = .empty;
    const messages = [_][]const u8{
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didOpen",
            .params = .{ .textDocument = .{ .uri = uri, .languageId = "alloy", .version = 1, .text = good } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didChange",
            .params = .{
                .textDocument = .{ .uri = uri, .version = 2 },
                .contentChanges = &[_]struct { text: []const u8 }{.{ .text = typing }},
            },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 2,
            .method = "textDocument/completion",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 2, .character = 21 } },
        }, .{}),
    };
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, std.testing.io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    const transcript = output.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"label\":\"member_names\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"label\":\"add_member\"") != null);
}

test "hover on a call resolves the checked target, not the first name" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const uri = "file:///c%3A/probe/associated.alloy";
    // Left::make and Right::make share a bare name; the call target
    // decides, never declaration order
    const source =
        "type Left = struct { value: i64 };\n" ++
        "type Right = struct { value: i64 };\n" ++
        "fn Left::make() -> Left { return Left { .value = 1 }; }\n" ++
        "fn Right::make() -> Right { return Right { .value = 2 }; }\n" ++
        "fn main() -> i32 {\n" ++
        "    const r = Right::make();\n" ++
        "    return r.value to i32;\n" ++
        "}\n";

    var frames: std.ArrayList(u8) = .empty;
    const messages = [_][]const u8{
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didOpen",
            .params = .{ .textDocument = .{ .uri = uri, .languageId = "alloy", .version = 1, .text = source } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 2,
            .method = "textDocument/hover",
            .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = 5, .character = 23 } },
        }, .{}),
    };
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, std.testing.io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    const transcript = output.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, transcript, "fn Right::make") != null);
}

test "semantic tokens color by resolution, not by name" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const uri = "file:///c%3A/probe/resolution_colors.alloy";
    // the struct member shares the macro's name: it must stay uncolored
    const source =
        "macro build(x: i64) { return x; }\n" ++
        "type Pair = struct { build: i64 };\n" ++
        "fn main() -> i32 { return 0; }\n";

    var frames: std.ArrayList(u8) = .empty;
    const messages = [_][]const u8{
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didOpen",
            .params = .{ .textDocument = .{ .uri = uri, .languageId = "alloy", .version = 1, .text = source } },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 2,
            .method = "textDocument/semanticTokens/full",
            .params = .{ .textDocument = .{ .uri = uri } },
        }, .{}),
    };
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, std.testing.io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    // 'build' macro name (3), 'x' parameter declaration and use (5),
    // 'Pair' type name (1), 'main' function name (0) - and NO token for
    // the 'build' struct member, which only shares the macro's name
    const transcript = output.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"data\":[0,6,5,3,0,0,6,1,5,0,0,17,1,5,0,1,5,4,1,0,1,3,4,0,0]") != null);
}

test "the outline survives an edit that fails to parse" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const uri = "file:///c%3A/probe/outline.alloy";
    const source = "fn helper() -> i64 { return 1; }\n";

    var frames: std.ArrayList(u8) = .empty;
    const messages = [_][]const u8{
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}",
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didOpen",
            .params = .{ .textDocument = .{ .uri = uri, .languageId = "alloy", .version = 1, .text = source } },
        }, .{}),
        // typing a half-written definition: the parse never reaches the
        // merge, so the previous analysis stays in service
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .method = "textDocument/didChange",
            .params = .{
                .textDocument = .{ .uri = uri, .version = 2 },
                .contentChanges = &[_]struct { text: []const u8 }{.{ .text = source ++ "fn probe(" }},
            },
        }, .{}),
        try std.json.Stringify.valueAlloc(arena, .{
            .jsonrpc = "2.0",
            .id = 2,
            .method = "textDocument/documentSymbol",
            .params = .{ .textDocument = .{ .uri = uri } },
        }, .{}),
    };
    for (messages) |message| {
        try frames.print(arena, "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    }

    var reader = Io.Reader.fixed(frames.items);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, std.testing.io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    // the stale analysis must still own its source: a borrowed document
    // buffer is freed by the edit and the name comes back as garbage
    const transcript = output.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"id\":2,\"result\":[") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"name\":\"helper\"") != null);
}

test "words extract around a cursor" {
    const text = "return twice(21);";
    try std.testing.expectEqualStrings("twice", wordAt(text, 9).?);
    try std.testing.expectEqualStrings("twice", wordAt(text, 12).?);
    try std.testing.expectEqual(@as(?[]const u8, null), wordAt("a + b", 2));
}

test "enclosing calls resolve the callee and active parameter" {
    const text = "fn main() { total(1, add(2, 3), }";
    const outer = enclosingCall(text, 31).?;
    try std.testing.expectEqualStrings("total", outer.name);
    try std.testing.expectEqual(@as(u32, 2), outer.commas);
    const inner = enclosingCall(text, 28).?;
    try std.testing.expectEqualStrings("add", inner.name);
    try std.testing.expectEqual(@as(u32, 1), inner.commas);
}

test "the server publishes diagnostics for an opened document" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const source = "fn main() -> i32 { return missing(); }";
    const document = try std.json.Stringify.valueAlloc(arena.allocator(), .{
        .jsonrpc = "2.0",
        .method = "textDocument/didOpen",
        .params = .{ .textDocument = .{
            .uri = "file:///c%3A/probe/main.alloy",
            .languageId = "alloy",
            .version = 1,
            .text = source,
        } },
    }, .{});
    const initialize = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}";
    const input = try std.fmt.allocPrint(arena.allocator(), "Content-Length: {d}\r\n\r\n{s}Content-Length: {d}\r\n\r\n{s}", .{
        initialize.len, initialize, document.len, document,
    });

    var reader = Io.Reader.fixed(input);
    var output: Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    var server = Server.init(std.testing.allocator, std.testing.io, &reader, &output.writer);
    defer server.deinit();
    try server.run();

    const transcript = output.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, transcript, "\"capabilities\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "textDocument/publishDiagnostics") != null);
    try std.testing.expect(std.mem.indexOf(u8, transcript, "use of undeclared identifier 'missing'") != null);
}
