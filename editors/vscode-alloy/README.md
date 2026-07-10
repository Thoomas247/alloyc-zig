# Alloy for VS Code

Syntax highlighting plus language-server features (inline errors, completion,
hover, go-to-definition) for the Alloy programming language. The extension
launches `alloyc lsp`, so everything the compiler knows, the editor knows.

## Setup

1. Build the compiler and put it on PATH (or note its full path):

   ```
   zig build
   ```

2. Install the extension's one dependency and package or side-load it:

   ```
   cd editors/vscode-alloy
   npm install
   ```

   - Development: open this folder in VS Code and press F5 (Run Extension).
   - Install: `npx @vscode/vsce package` then
     `code --install-extension alloy-language-0.1.0.vsix`.

3. If `alloyc` is not on PATH, set `alloy.serverPath` in the VS Code
   settings to the executable's full path.

Diagnostics refresh on every keystroke: the server re-runs the real
compiler front end (tokenize, parse, resolve, check) in process on each
change. Imports resolve next to the open file, and open editor buffers
shadow the file system, so unsaved edits across files stay consistent.
Package imports load from the `pkg/` folder next to the entry file.

## Other editors

The server is plain LSP over stdio, so any LSP client works.

Neovim (0.11+):

```lua
vim.filetype.add({ extension = { alloy = "alloy" } })
vim.lsp.config("alloyc", {
    cmd = { "alloyc", "lsp" },
    filetypes = { "alloy" },
    root_markers = { ".git" },
})
vim.lsp.enable("alloyc")
```

Helix (`languages.toml`):

```toml
[language-server.alloyc]
command = "alloyc"
args = ["lsp"]

[[language]]
name = "alloy"
scope = "source.alloy"
file-types = ["alloy"]
language-servers = ["alloyc"]
```

## Current capabilities

- Publish diagnostics (full pipeline: syntax, resolution, type, flow, and
  move analysis errors), cleared and re-published per edit, including in
  imported files. Document sync is incremental.
- Completion: keywords and every top-level definition; after `receiver.`,
  the receiver's checked type drives field and extension completion; after
  `Enum::` the variants, after `alias::` the module's visible definitions.
- Hover: locals and expressions show their checked type; top-level symbols
  show their declaration line.
- Go-to-definition, find-all-references, and rename — for top-level
  definitions across modules and into libraries, and for locals,
  parameters, and captures (both come from the resolver's own resolution
  tables, so they are exact, not text matches).
- Document outline and workspace symbol search.
- Semantic tokens classifying identifiers by what they resolve to
  (function, type, interface, macro); lexical coloring stays with the
  TextMate grammar and works without the server.
- Signature help with overloads and active-parameter tracking.
- Document formatting via the canonical formatter (`alloyc fmt` on the
  command line): whitespace-only normalization that preserves the author's
  line breaks and comments, verified token-identical before applying.

Not yet: code actions, formatting-on-range.

## Debugging

Two debuggers, no setup beyond this extension:

**Interpreter debugger (built in).** Press F5 on an `.alloy` file (or add a
`"type": "alloy"` launch configuration) and the extension runs `alloyc dap`:
breakpoints, step in/over/out, call stacks, locals with rendered values
(structs, enums with payloads, slices as text), and program output in the
debug console. Because it drives the same interpreter as `alloyc run`, it
also reflects comptime semantics exactly.

**Native debugging via DWARF.** Checked `alloyc build` executables carry
DWARF debug info: source breakpoints, statement stepping, Alloy function
names in stacks, and variable inspection with real types work in any DWARF
debugger. Optional LLDB formatters in `lldb/alloy_formatters.py` add slice
text previews, heap-array lengths (read from the hidden prefix), and the
concrete type behind an interface object (resolved through its
`alloy.type.<Name>` identity symbol) — load them via CodeLLDB's
`initCommands`:

```json
"initCommands": ["command script import ${workspaceFolder}/editors/vscode-alloy/lldb/alloy_formatters.py"]
```

With the CodeLLDB extension installed:

```json
{
    "type": "lldb",
    "request": "launch",
    "name": "Debug Alloy",
    "program": "${workspaceFolder}/main.exe",
    "preLaunchTask": "alloyc build"
}
```

Breakpoints, statement stepping, call stacks, and variable inspection all
work: locals, parameters, and captures carry DWARF types mirroring the
language's C-compatible layouts, so struct fields expand in the variables
pane. Slices show as `{data, length}` records; enums and interface objects
currently display as opaque records of their layout size.
