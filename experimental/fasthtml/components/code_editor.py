# Adapted from: https://github.com/AnswerDotAI/fasthtml-example/tree/baa67c5b2ca4d4a9ba091b9f9b72b8a2de384a37/code_editor

from fasthtml.common import *
from .toolbar import Toolbar
import json

def editor_script(initial_content: str) -> Script:
    return Script("""
let editor;
let completionTippy;
let currentCompletion = '';

function initEditor() {
    editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");
    editor.session.setMode("ace/mode/javascript");
    editor.setOptions({
        fontSize: "14px",
        showPrintMargin: false,
        // disable showing errors in gutter, Ace's WGSL parser is out of date
        showGutter: false,
        highlightActiveLine: true,
        wrap: true,
    });
    editor.setKeyboardHandler("ace/keyboard/vim");
    
    editor.setValue(""" + json.dumps(initial_content) + ");" + 
"""
    window.addEventListener('resize', function() {
        editor.resize();
    });
    document.getElementById('language').addEventListener('change', function(e) {
        let mode = "ace/mode/" + e.target.value;
        editor.session.setMode(mode);
    });

    editor.session.on('change', function(delta) {
        if (delta.action === 'insert' && (delta.lines[0] === '.' || delta.lines[0] === ' ')) {
            showCompletionSuggestion();
        }

        // Recover from errors TODO(avh): only do this if there's an error
        createModule().then((Module) => {
            // Keep your existing Module setup
            Module.print = window.customPrint;
            Module.printErr = window.customPrint;
            window.Module = Module;
            console.log("Module ready");
        });

        if (window.Module && window.Module.executeKernel) {
            console.log("Executing kernel");
            window.terminal.clear();
            window.Module.executeKernel(editor.getValue());
        } else {
            console.log("Module not ready");
        }

    });

    completionTippy = tippy(document.getElementById('editor'), {
        content: 'Loading...',
        trigger: 'manual',
        placement: 'top-start',
        arrow: true,
        interactive: true
    });

    // Override the default tab behavior
    editor.commands.addCommand({
        name: 'insertCompletion',
        bindKey: {win: 'Tab', mac: 'Tab'},
        exec: function(editor) {
            if (currentCompletion) {
                editor.insert(currentCompletion);
                currentCompletion = '';
                completionTippy.hide();
            } else {
                editor.indent();
            }
        }
    });
}

async function showCompletionSuggestion() {
    const cursorPosition = editor.getCursorPosition();
    const screenPosition = editor.renderer.textToScreenCoordinates(cursorPosition.row, cursorPosition.column);

    completionTippy.setContent('Loading...');
    completionTippy.setProps({
        getReferenceClientRect: () => ({
            width: 0,
            height: 0,
            top: screenPosition.pageY,
            bottom: screenPosition.pageY,
            left: screenPosition.pageX,
            right: screenPosition.pageX,
        })
    });
    completionTippy.show();

    try {
        const response = await fetch('/complete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                code: editor.getValue(),
                row: cursorPosition.row,
                column: cursorPosition.column
            }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentCompletion = data.completion;
        completionTippy.setContent(`${currentCompletion} (Press Tab to insert)`);
    } catch (error) {
        console.error('Error:', error);
        completionTippy.setContent('Error fetching completion');
        currentCompletion = '';
    }

    setTimeout(() => {
        if (currentCompletion) {
            completionTippy.hide();
            currentCompletion = '';
        }
    }, 5000);
}

document.addEventListener('DOMContentLoaded', initEditor);
""")

def CodeEditor(initial_content: str):
    return (
        Div(
            Toolbar(),
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
        editor_script(initial_content)
    )
