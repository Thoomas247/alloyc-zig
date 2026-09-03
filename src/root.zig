//! By convention, root.zig is the root source file when making a package.
const std = @import("std");

pub const tokenizer = @import("tokenizer.zig");
pub const Token = tokenizer.Token;
pub const Tokenizer = tokenizer.Tokenizer;

pub const diagnostic = @import("diagnostic.zig");
pub const Diagnostic = diagnostic.Diagnostic;

pub const ast = @import("ast.zig");
pub const Ast = ast.Ast;

pub const parser = @import("parser.zig");
pub const Parser = parser.Parser;

pub const compilation = @import("compilation.zig");
pub const Compilation = compilation.Compilation;
pub const Module = compilation.Module;
pub const ModuleLoader = compilation.ModuleLoader;

pub const resolution = @import("resolution.zig");
pub const Resolver = resolution.Resolver;

pub const types = @import("types.zig");
pub const Type = types.Type;

pub const checker = @import("checker.zig");
pub const Checker = checker.Checker;

pub const interpreter = @import("interpreter.zig");
pub const Interpreter = interpreter.Interpreter;

pub const codegen = @import("codegen.zig");
pub const Codegen = codegen.Codegen;

pub const lsp = @import("lsp.zig");
pub const LanguageServer = lsp.Server;

pub const formatter = @import("formatter.zig");

pub const dap = @import("dap.zig");
pub const DebugAdapter = dap.Server;

pub const library = @import("library.zig");

pub const rpc = @import("rpc.zig");
pub const paths = @import("paths.zig");
pub const toolchain = @import("toolchain.zig");

// the conformance suite has no declarations to export; naming it here is
// what pulls its tests into 'zig build test'
pub const conformance_tests = @import("conformance_tests.zig");

test {
    std.testing.refAllDecls(@This());
}
