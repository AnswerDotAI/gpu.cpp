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

bind_terminal = """
    window.terminal.open(document.getElementById('output'));
    fitAddon.fit();
"""

header = """
// Start editing here to see the results.
// Warning: You are in vim mode.
@group(0) @binding(0) var<storage, read_write> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}})
""".strip()

gelu_kernel = """
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] + 1;
    }
}
""".strip()

def controls():
    # left and right buttons
    return Div(
        Div(
            Button(
                "<-",
                cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded",
                id="prev",
            ),
            "Puzzle 1: Map",
            Button(
                "->",
                cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded",
                id="next",
            ),
            cls="flex justify-between space-x-2",
        ),
        cls="mb-4",
        style="text-align: center;",
    )


def dispatchInputs():
    return Div(
        Div(
            Div("Workgroup Size", cls="font-bold mb-1 text-center"),
            Div(
                Div(
                    Input(
                        type="number",
                        id="workgroup_x",
                        cls="w-1/3 p-2 border rounded",
                        value="256",
                        placeholder="X",
                        style="width: 8vw",
                    ),
                    Input(
                        type="number",
                        id="workgroup_y",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Y",
                        style="width: 8vw",
                    ),
                    Input(
                        type="number",
                        id="workgroup_z",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Z",
                        style="width: 8vw",
                    ),
                    cls="flex justify-between space-x-2",
                ),
                cls="flex justify-between space-x-2",
            ),
            cls="mb-4",
        ),
        Div(
            Div("Grid Size", cls="font-bold mb-1 text-center"),
            Div(
                Div(
                    Input(
                        type="number",
                        id="grid_x",
                        cls="w-1/3 p-2 border rounded",
                        value="256",
                        placeholder="X",
                        style="width: 8vw",
                    ),
                    Input(
                        type="number",
                        id="grid_y",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Y",
                        style="width: 8vw",
                    ),
                    Input(
                        type="number",
                        id="grid_z",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Z",
                        style="width: 8vw",
                    ),
                    cls="flex justify-between space-x-2",
                ),
                cls="flex justify-between space-x-2",
            ),
        ),
        cls="w-full max-w-md",
    )


def editor_script(initial_content: str) -> str:
    with open("code_editor.js", "r") as file:
        file_content = file.read()
    initial_content = json.dumps(initial_content)
    return f"""{file_content}
document.addEventListener('DOMContentLoaded', () => {{
    initEditor({initial_content});
    updateEditor("");
}});"""


def CodeEditor(initial_content: str):
    return (
        Div(
            Div(
                Div(id="editor", style="height: 90vh; width: 100vw;"),
                Script(
                    """
                    me().on('contextmenu', ev => {
                        ev.preventDefault()
                        me('#context-menu').send('show', {x: ev.pageX, y: ev.pageY})
                    })
                """
                ),
                # cls="flex-grow w-full"
                style="height: 50vh; overflow: hidden;",
            ),
            # cls="flex flex-col h-screen w-full", style="height: 100vh; overflow: hidden;"
            style="height: 33vh; overflow: hidden;",
        ),
        Script(editor_script(initial_content)),
    )


# TODO(avh) : Global state handling of terminal binding, module creation, etc.
# could be improved

init_app = """
document.addEventListener('DOMContentLoaded', () => {
    window.AppState = Object.create(State);
    const AppState = window.AppState;
    initializeApp();
});
""".strip()

HDRS = (
    picolink,
    # ace code editor
    Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js"),
    # xterm terminal for output
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/xterm/css/xterm.css"),
    Script(src="https://cdn.jsdelivr.net/npm/xterm/lib/xterm.js"),
    Script(src="https://cdn.jsdelivr.net/npm/xterm-addon-fit/lib/xterm-addon-fit.js"),
    Script(src="/build/run.js"),  # gpu.cpp runtime
    Style(global_style),
    Link(rel="stylesheet", href="https://unpkg.com/tippy.js@6/dist/tippy.css"),
    Script(src="https://unpkg.com/@popperjs/core@2"),
    Script(src="https://unpkg.com/tippy.js@6"),
    Script(src="/client.js"),
    Script(init_app),
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


@app.get("/client.js")
async def serve_client():
    return FileResponse(f"client.js")


@app.get("/build/run.js")
async def serve_js():
    return FileResponse(f"build/run.js")


@app.get("/build/run.wasm")
async def serve_wasm():
    return FileResponse(f"build/run.wasm")


def output():
    return (
        Div(
            "Output",
            id="output",
            style="width: 50vw; height:100vh; background-color: #444; float: right;",
        ),
    )
    # Script(bind_terminal)

rt = app.route

@rt("/")
def get():
    return (
        Title("WGSL Editor"),
        Body(
            Div(
                Div(
                    controls(),
                    dispatchInputs(),
                    "GPU Code (WGSL):",
                    Div(
                        Pre(
                            header.replace("{{workgroupSize}}", "256, 1, 1"),
                            style="font-family: monospace; font-size: 0.8rem;",
                        )
                    ),
                    CodeEditor(initial_content=gelu_kernel),
                    style="width: 50vw; height:100vh; background-color: #333; float: left;",
                ),
                output(),
            ),
            style="height: 100vh; overflow: hidden;",
        ),
    )


if __name__ == "__main__":
    serve(
        appname=None,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
