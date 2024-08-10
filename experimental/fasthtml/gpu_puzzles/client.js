const State = {
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
}

function initializeTerminal() {
  AppState.terminal = new Terminal();
  const fitAddon = new FitAddon.FitAddon();
  AppState.terminal.loadAddon(fitAddon);
  AppState.terminal.open(document.getElementById('output'));
  fitAddon.fit();
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

function customPrint(text) {
  console.log(text);
  if (AppState.terminal) {
    AppState.terminal.writeln(text);
  } else {
    console.warn("Terminal not initialized");
  }
}
