from fasthtml.common import *
from functools import cache
import os
import json


TARGET = os.getenv("TARGET", "debug")

global_style = """
#editor {
    height: 50vh;
    width: 50vw;
}
"""

terminal_init = """
    const terminal = new Terminal();
    const fitAddon = new FitAddon.FitAddon();
    terminal.loadAddon(fitAddon);
    window.terminal = terminal;
    console.log("Terminal initialized");
"""

print_script = """
window.customPrint = function(text) {
  console.log(text);
  if (window.terminal) {
    window.terminal.writeln(text);
  } else {
    console.warn("Terminal not initialized");
  }
};
createModule().then((Module) => {
  Module.print = window.customPrint;
  Module.printErr = window.customPrint;
  window.Module = Module;
  console.log("Initial module created");
});
"""

bind_terminal = """
    window.terminal.open(document.getElementById('output'));
    fitAddon.fit();
"""

gelu_kernel = """// Start editing here to see the results.
// Warning: You are in vim mode.
@group(0) @binding(0) var<storage, read_write> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] + 1;
    }
}
"""
# Adapted from: https://github.com/AnswerDotAI/fasthtml-example/tree/baa67c5b2ca4d4a9ba091b9f9b72b8a2de384a37/code_editor

def Toolbar():
    return Div(
        Select(
            Option("WGSL", value="wgsl"),
            Option("C++", value="c++"),
            id="language",
            cls="mr-2 p-2 border rounded"
        ),
        cls="bg-gray-200 p-4 shadow-md flex items-center w-full"
    )

def editor_script(initial_content: str) -> str:
    with open("code_editor.js", 'r') as file:
        file_content = file.read()
    initial_content = json.dumps(initial_content)
    return(f"""{file_content}
document.addEventListener('DOMContentLoaded', () => {{
    initEditor({initial_content});
    updateEditor("");
}});""")


def CodeEditor(initial_content: str):
    return (
        Div(
            Div(
                Div(id="editor", style="height: 90vh; width: 100vw;"),
                Script("""
                    me().on('contextmenu', ev => {
                        ev.preventDefault()
                        me('#context-menu').send('show', {x: ev.pageX, y: ev.pageY})
                    })
                """),
                # cls="flex-grow w-full"
                    style="height: 100vh; overflow: hidden;"
            ),
            # cls="flex flex-col h-screen w-full", style="height: 100vh; overflow: hidden;"
            style="height: 100vh; overflow: hidden;"
        ),
        Script(editor_script(initial_content))
    )

# TODO(avh) : Global state handling of terminal binding, module creation, etc.
# could be improved

HDRS = (
    picolink,
    # ace code editor
    Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js"),
    # xterm terminal for output
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/xterm/css/xterm.css"),
    Script(src="https://cdn.jsdelivr.net/npm/xterm/lib/xterm.js"),
    Script(src="https://cdn.jsdelivr.net/npm/xterm-addon-fit/lib/xterm-addon-fit.js"),
    Script(terminal_init),
    Script(src="/build/run.js"),  # gpu.cpp runtime
    Script(print_script),
    Style(global_style),
    Link(rel="stylesheet", href="https://unpkg.com/tippy.js@6/dist/tippy.css"),
    Script(src="https://unpkg.com/@popperjs/core@2"),
    Script(src="https://unpkg.com/tippy.js@6"),
    *Socials(
        title="gpu.cpp gpu puzzles",
        description="",
        site_name="gpucpp.answer.ai",
        twitter_site="@answerdotai",
        image="",
        url="https://gpucpp.answer.ai",
    ),
)

if TARGET == "release":
    app = FastHTML(hdrs=HDRS)
else:
    app = FastHTMLWithLiveReload(hdrs=HDRS)

rt = app.route


@app.get("/build/run.js")
async def serve_wasm(fname: str, ext: str):
    return FileResponse(f"build/run.js")


@app.get("/build/run.wasm")
async def serve_wasm(fname: str, ext: str):
    return FileResponse(f"build/run.wasm")


def output():
    return Div(
        "Output",
        id="output",
        style="width: 50vw; height:100vh; background-color: #444; float: right;",
    ), Script(bind_terminal)


@rt("/")
def get():
    return (
        Title("WGSL Editor"),
        Body(
            Div(
                Div(
                    CodeEditor(initial_content=gelu_kernel),
                    style="width: 50vw; height:100vh; background-color: #333; float: left;",
                ),
                output(),
            ),
            style="height: 100vh; overflow: hidden;",
        ),
    )


if __name__ == "__main__":
    run_uv(
        fname=None,
        app="app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
