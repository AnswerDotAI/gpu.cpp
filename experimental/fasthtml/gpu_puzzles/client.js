const State = {
  preamble: "",
  editor: null,
  terminal: null,
  module: null,
  completionTippy: null,
  currentCompletion: '',
  isModuleReady: false
};

function initializeApp() {
  initializeTerminal();
  initializeEditor();
  initializeModule();
  setupEventListeners();
  console.log("App initialized");
}

function initializeTerminal() {
  AppState.terminal = new Terminal();
  const fitAddon = new FitAddon.FitAddon();
  AppState.terminal.loadAddon(fitAddon);
  AppState.terminal.open(document.getElementById('output'));
  fitAddon.fit();
  window.AppState = AppState;
  console.log("Terminal initialized");
}


function initializeEditor(initialContent) {
  AppState.editor = ace.edit("editor");
  AppState.editor.setTheme("ace/theme/monokai");
  AppState.editor.session.setMode("ace/mode/javascript");
  AppState.editor.setOptions({
    fontSize: "14px",
    showPrintMargin: false,
    showGutter: false,
    highlightActiveLine: true,
    wrap: true,
  });
  AppState.editor.setKeyboardHandler("ace/keyboard/vim");
  AppState.editor.setValue(initialContent || '');
  
  AppState.completionTippy = tippy(document.getElementById('editor'), {
    content: 'Loading...',
    trigger: 'manual',
    placement: 'top-start',
    arrow: true,
    interactive: true
  });

  console.log("Editor initialized");
}

function initializeModule() {
  createModule().then((Module) => {
    AppState.module = Module;
    AppState.module.print = customPrint;
    AppState.module.printErr = customPrint;
    AppState.isModuleReady = true;
    console.log("Module initialized");
    // Attempt to run the kernel with the initial content
    updateEditor({ action: 'insert', lines: [''] });
  }).catch(error => {
    console.error("Failed to initialize module:", error);
  });
}

function setupEventListeners() {
  AppState.editor.session.on('change', updateEditor);
  window.addEventListener('resize', () => AppState.editor.resize());
  
  AppState.editor.commands.addCommand({
    name: 'insertCompletion',
    bindKey: {win: 'Tab', mac: 'Tab'},
    exec: function(editor) {
      if (AppState.currentCompletion) {
        editor.insert(AppState.currentCompletion);
        AppState.currentCompletion = '';
        AppState.completionTippy.hide();
      } else {
        editor.indent();
      }
    }
  });
}

////////////////////////////////////////
// Printing to terminal
////////////////////////////////////////

function customPrint(text) {
  console.log(text);
  if (AppState.terminal) {
    AppState.terminal.writeln(text);
  } else {
    console.warn("Terminal not initialized");
  }
}

////////////////////////////////////////
// Code Editor
////////////////////////////////////////

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

function updateEditor(delta) {
    if (delta.action === 'insert' && (delta.lines[0] === '.' || delta.lines[0] === ' ')) {
        showCompletionSuggestion();
    }
    // Recover from errors TODO(avh): only do this if there's an error
    createModule().then((Module) => {
        // Keep your existing Module setup
        console.log("updateEditor() - Module ready");
    });
    if (AppState.module && AppState.module.executeKernel) {
        console.log("Executing kernel");
        AppState.terminal.clear();
        wgSize = [256, 1, 1];
        gridSize = [256, 1, 1];
        AppState.module.executeKernel(AppState.preamble + AppState.editor.getValue(), wgSize, gridSize);
    } else {
        console.log("updateEditor() - Module not ready");
    }
}

function initEditor(initial_code) {
    AppState.editor = ace.edit("editor");
    editor = AppState.editor;
    // editor = ace.edit("editor");
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
    editor.setValue(initial_code);
    window.addEventListener('resize', function() {
        editor.resize();
    });
    // document.getElementById('language').addEventListener('change', function(e) {
    //       let mode = "ace/mode/" + e.target.value;
    //     editor.session.setMode(mode);
    // });

    editor.session.on('change', updateEditor);

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
