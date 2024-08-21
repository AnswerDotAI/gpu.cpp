from fasthtml.common import *
from functools import cache
import os
import json


TARGET = os.getenv("TARGET", "debug")

body_style = """
    background: linear-gradient(rgba(14, 14, 14, 0.75),
                                rgba(0, 0, 0, 0.85),
                                rgba(0, 0, 0, 1.0),
                                rgba(0, 0, 0, 0.88),
                                rgba(3, 3, 3, 0.8),
                                rgba(28, 28, 28, 0.68)),
                url('https://gpucpp.answer.ai/images/shadertui2-small-crop2-loop.gif');
    background-size: cover;
    background-position: center;
    height: 100vh
"""


def button(text, id):
    return Button(
        text,
        cls="bg-blue-300 hover:bg-blue-900 text-white font-bold py-2 px-4 rounded",
        id=id,
    )


def controls():
    # left and right buttons
    return (
        Div(
            Div(
                button("<<", "prev"),
                # don't start a new row for div
                Div(
                    "Puzzle 1: Map",
                    id="puzzle_name",
                    style="font-size: 1.5rem; width: 25vw; font-weight: bold;",
                ),
                button(">>", "next"),
                style="display: flex;  align-items: center; justify-content: center;",
            ),
            style="text-align: center; margin-top: 5vh; margin-left: 2rem; margin-right: 2rem;",
        ),
        Div(
            "Puzzle description",
            id="puzzle_description",
            style="font-size: 20vh; margin-top: 5vh; margin-bottom: 5vh; margin-left: 2rem; margin-right: 2rem; height: 6vh; font-size: 1.0rem;",
        ),
    )


def init_app() -> str:
    return f"""
document.addEventListener('DOMContentLoaded', () => {{
    window.AppState = Object.create(State);
    window.customPrint = customPrint;
    const AppState = window.AppState;
    initializeApp();
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
    Script(src="https://cdn.jsdelivr.net/npm/zero-md@2.2", type="module"),
    Script(src="/client.js"),
    Script(init_app()),
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

@app.get("/assets/markdown/{fname:path}.md")
async def serve_writeups(fname: str):
    return FileResponse(f"assets/markdown/{fname}.md")

@app.get("/assets/images/{fname:path}.png")
async def serve_images(fname: str):
    return FileResponse(f"assets/images/{fname}.png")

def output():
    correctHeight = 27
    outputHeight = 100 - correctHeight + 1
    return (
        Div(
            Div(
                "(Result Check)",
                id="correct",
            ),
            button("Show Solution", id="solution"),
            style=f"width: 49vw; height:{correctHeight / 3 * 2}vh; float: right; font-size: 2rem; text-align: center; align-items: center; justify-content: center; margin-top: {correctHeight / 3}vh;",
        ),
        Div(id="output", cls="overlay", style=f"width: 49vw; height:{outputHeight}vh; float: right;"),
        Div("", id="writeup", cls="overlay hidden", style=f"width: 49vw; height: {outputHeight}vh; float: right; background-color: white; padding: 2rem; overflow-y: scroll;"),

    )


def CodeEditor():
    return (
        Div(
            Div(
                Div(id="editor", style="height: 79vh; width: 100vw;"),
                style="height: 79vh; overflow: hidden;",
            ),
            style="height: 79vh; overflow: hidden;",
        ),
    )


@app.get("/")
def get():
    return (
        Title("GPU Puzzles"),
        Body(
            Div(
                Div(
                    controls(),
                    CodeEditor(),
                    style="width: 49vw; height:100vh; float: left;",
                ),
                output(),
            ),
            style=body_style,
        ),
    )


if __name__ == "__main__":
    serve(
        appname=None,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
