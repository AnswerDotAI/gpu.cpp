from fasthtml.common import *
from functools import cache
import os
import json


TARGET = os.getenv("TARGET", "debug")

PREAMBLE = """
// Start editing here to see the results.
// Warning: You are in vim mode.
@group(0) @binding(0) var<storage, read_write> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}})
""".strip()

INITIAL_CODE = """
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    output[i] = input[i];
}
""".strip()

CORRECT = False


def controls():
    # left and right buttons
    return (
        Div(
            Div(
                Button(
                    "<<",
                    cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded",
                    id="prev",
                ),
                # don't start a new row for div
                Div(
                    "Puzzle 1: Map",
                    id="puzzle_name",
                    style="font-size: 1.5rem; width: 15vw;",
                ),
                Button(
                    ">>",
                    cls="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded",
                    id="next",
                ),
                # no space between buttons
                style="display: flex;  align-items: center; justify-content: center;",
                # cls="flex justify-between space-x-2",
            ),
            # cls="mb-4",
            style="text-align: center;",
        ),
        Div(
            "Puzzle description",
            id="puzzle_description",
            style="font-size: 1rem; margin-top: 2rem; margin-bottom: 2rem; margin-left: 2rem; margin-right: 2rem; height: 6vh; font-size: 1.0rem;",
        ),
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
                        style="width: 5vw",
                    ),
                    Input(
                        type="number",
                        id="workgroup_y",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Y",
                        style="width: 5vw",
                    ),
                    Input(
                        type="number",
                        id="workgroup_z",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Z",
                        style="width: 5vw",
                    ),
                    cls="flex justify-between space-x-2",
                ),
                cls="flex justify-between space-x-2",
            ),
            cls="mb-4",
            style="margin-right: 2rem; margin-left: 2rem;",
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
                        style="width: 5vw",
                    ),
                    Input(
                        type="number",
                        id="grid_y",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Y",
                        style="width: 5vw",
                    ),
                    Input(
                        type="number",
                        id="grid_z",
                        cls="w-1/3 p-2 border rounded",
                        value="1",
                        placeholder="Z",
                        style="width: 5vw",
                    ),
                    cls="flex justify-between space-x-2",
                ),
                cls="flex justify-between space-x-2",
            ),
        ),
        cls="w-full max-w-md",
        style="display: flex; justify-content: center; margin-top: 2rem; margin-left: 2rem; margin-right: 2rem;",
    )


def CodeEditor(initial_code: str):
    return (
        Div(
            Div(
                Div(id="editor", style="height: 60vh; width: 100vw;"),
                style="height: 60vh; overflow: hidden;",
            ),
            # cls="flex flex-col h-screen w-full", style="height: 100vh; overflow: hidden;"
            style="height: 60vh; overflow: hidden;",
        ),
    )


def init_app(initial_code: str) -> str:
    return f"""
document.addEventListener('DOMContentLoaded', () => {{
    window.AppState = Object.create(State);
    window.customPrint = customPrint;
    const AppState = window.AppState;
    AppState.preamble = {json.dumps(PREAMBLE)};
    AppState.preamble_template = {json.dumps(PREAMBLE)};
    initializeApp({json.dumps(initial_code)});
}});
""".strip()


HDRS = (
    picolink,
    # ace code editor
    Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js"),
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/xterm/css/xterm.css"),
    Script(src="https://cdn.jsdelivr.net/npm/xterm/lib/xterm.js"),
    Script(src="https://cdn.jsdelivr.net/npm/xterm-addon-fit/lib/xterm-addon-fit.js"),
    Script(src="/build/run.js"),  # gpu.cpp runtime
    Style("#editor { height: 50vh; width: 50vw; }"),
    Link(rel="stylesheet", href="https://unpkg.com/tippy.js@6/dist/tippy.css"),
    Script(src="https://unpkg.com/@popperjs/core@2"),
    Script(src="https://unpkg.com/tippy.js@6"),
    Script(src="/client.js"),
    Script(init_app(INITIAL_CODE)),
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
            id="output",
            style="width: 50vw; height:50vh; float: right;",
        ),
        Div(
            "(Result Check)",
            style="width: 50vw; height:4rem; float: right; font-size: 2rem; text-align: center;",
            id="correct",
        ),
        Img(src="https://gpucpp.answer.ai/images/shadertui2-small-crop2-loop.gif", style="width: 50vw;"),
    )


@app.get("/")
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
                            PREAMBLE.replace("{{workgroupSize}}", "256, 1, 1"),
                            style="font-family: monospace; font-size: 1.0rem;",
                            id="preamble",
                        )
                    ),
                    CodeEditor(initial_code=INITIAL_CODE),
                    style="width: 50vw; height:100vh; float: left;",
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
