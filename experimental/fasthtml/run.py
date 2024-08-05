from components.code_editor import CodeEditor
from fasthtml.common import *
from functools import cache
import os
import uvicorn

TARGET = os.getenv("TARGET", "debug")

ace_editor = Script(src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.12/ace.js")
gpucpp_runtime = Script(src="/build/run.js")    
gpucpp_wasm = Script(src="/build/run.wasm")    

global_style = Style("""
#editor {
    height: 50vh;
    width: 50vw;
}
""")

HDRS = (
        picolink,
    ace_editor,
    gpucpp_runtime,
    global_style,
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

@app.get("/build/{fname:path}.{ext:static}")
async def build(fname: str, ext: str):
    return FileResponse(f"build/{fname}.{ext}")

@app.get("/build/run.wasm")
async def serve_wasm(fname: str, ext: str):
    return FileResponse(f"build/run.wasm")

def page():
    return Title("GPU Puzzles"), Body(
        Div(
            Div(
                CodeEditor(),
                style="width: 66vw; height:100vh; background-color: #333; float: left;",
            ),
            Div(
                "Output",
                style="width: 34vw; height:100vh; background-color: #444; float: right;",
            ),
        ),
        style="height: 100vh; overflow: hidden;")


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
