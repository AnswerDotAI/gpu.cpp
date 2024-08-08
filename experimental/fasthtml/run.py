from components.code_editor import CodeEditor
from fasthtml.common import *
from functools import cache
import os
import uvicorn

TARGET = os.getenv("TARGET", "debug")

ace_editor = Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js")
gpucpp_runtime = Script(src="/build/run.js")
gpucpp_wasm = Script(src="/build/run.wasm")
tippy_css = Link(rel="stylesheet", href="https://unpkg.com/tippy.js@6/dist/tippy.css")
tippy_js = Script(src="https://unpkg.com/@popperjs/core@2")
tippy_js2 = Script(src="https://unpkg.com/tippy.js@6")
xterm_css = Link(
    rel="stylesheet", href="https://cdn.jsdelivr.net/npm/xterm/css/xterm.css"
)
xterm_js = Script(src="https://cdn.jsdelivr.net/npm/xterm/lib/xterm.js")
xterm_fit_js = Script(
    src="https://cdn.jsdelivr.net/npm/xterm-addon-fit/lib/xterm-addon-fit.js"
)

global_style = Style(
    """
#editor {
    height: 50vh;
    width: 50vw;
}
"""
)

terminal_init = \
    """
    const terminal = new Terminal();
    const fitAddon = new FitAddon.FitAddon();
    terminal.loadAddon(fitAddon);
    // terminal.open(document.getElementById('output'));
    window.terminal = terminal;
    // fitAddon.fit();
    console.log("Terminal initialized");
"""

print_script = (
    Script(
        """
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
  console.log("Module ready");
  // window.Module.executeKernel(editor.getValue());
});
"""
    ),
)

bind_terminal = """
    window.terminal.open(document.getElementById('output'));
    fitAddon.fit();
"""

gelu_kernel = """// Start editing here to see the results.
// Warning: You are in vim mode.
@group(0) @binding(0) var<storage, read_write> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
        let i: u32 = GlobalInvocationID.x;
        if (i < arrayLength(&input)) {
            output[i] = input[i] + 1;
        }
    }
  """

# TODO(avh) : Global state handling of terminal binding, module creation, etc.
# could be improved

HDRS = (
    picolink,
    ace_editor,
    xterm_css,
    xterm_js,
    xterm_fit_js,
    Script(terminal_init),
    gpucpp_runtime,
    print_script,
    global_style,
    tippy_css,
    tippy_js,
    tippy_js2,
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


def page():
    return (
        Title("GPU Puzzles"),
        Body(
            Div(
                Div(
                    CodeEditor(initial_content=gelu_kernel),
                    style="width: 66vw; height:100vh; background-color: #333; float: left;",
                ),
                Div(
                    "Output",
                    id="output",
                    style="width: 34vw; height:100vh; background-color: #444; float: right;",
                ),
            ),
            Script(bind_terminal),
            style="height: 100vh; overflow: hidden;",
        ),
    )


@rt("/")
def get():
    return page()


rt = app.route

if __name__ == "__main__":
    run_uv(
        fname=None,
        app="app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
