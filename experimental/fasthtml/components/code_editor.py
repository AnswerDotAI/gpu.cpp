# Adapted from: https://github.com/AnswerDotAI/fasthtml-example/tree/baa67c5b2ca4d4a9ba091b9f9b72b8a2de384a37/code_editor

from fasthtml.common import *
import json

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
    with open("components/code_editor.js", 'r') as file:
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
