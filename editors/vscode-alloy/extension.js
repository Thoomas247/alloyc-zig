// The Alloy VS Code extension: launches 'alloyc lsp' over stdio and lets
// vscode-languageclient wire diagnostics, completion, hover, and
// go-to-definition into the editor.
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

function activate(context) {
    const serverPath = vscode.workspace.getConfiguration("alloy").get("serverPath", "alloyc");
    const serverOptions = {
        command: serverPath,
        args: ["lsp"],
        options: {},
    };
    const clientOptions = {
        documentSelector: [{ scheme: "file", language: "alloy" }],
    };
    client = new LanguageClient("alloy", "Alloy Language Server", serverOptions, clientOptions);
    context.subscriptions.push({ dispose: () => client && client.stop() });
    client.start();

    // the interpreter-backed debugger: 'alloyc dap' over stdio
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterDescriptorFactory("alloy", {
            createDebugAdapterDescriptor() {
                return new vscode.DebugAdapterExecutable(serverPath, ["dap"]);
            },
        }),
    );
}

function deactivate() {
    return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
