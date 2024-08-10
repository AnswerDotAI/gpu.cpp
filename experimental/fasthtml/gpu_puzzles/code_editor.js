let editor;
let completionTippy;
let currentCompletion = "";

function updateEditor(delta) {
  if (
    delta.action === "insert" &&
    (delta.lines[0] === "." || delta.lines[0] === " ")
  ) {
    showCompletionSuggestion();
  }

  // Recover from errors TODO(avh): only do this if there's an error
  createModule().then((Module) => {
    // Keep your existing Module setup
    Module.print = window.customPrint;
    Module.printErr = window.customPrint;
    window.Module = Module;
    console.log("updateEditor() - Module ready");
  });

  if (window.Module && window.Module.executeKernel) {
    console.log("Executing kernel");
    window.terminal.clear();
    const wgSize = [256, 1, 1];
    const gridSize = [256, 1, 1];
    window.Module.executeKernel(editor.getValue(), wgSize, gridSize);
  } else {
    console.log("updateEditor() - Module not ready");
  }
}

function initEditor(initial_content) {
  editor = ace.edit("editor");
  editor.setTheme("ace/theme/monokai");
  editor.session.setMode("ace/mode/javascript");
  editor.setOptions({
    fontSize: "16px",
    showPrintMargin: false, // disable showing errors in gutter, Ace's WGSL parser is out of date
    showGutter: false,
    highlightActiveLine: true,
    wrap: true,
  });
  editor.setKeyboardHandler("ace/keyboard/vim");
  editor.setValue(initial_content);
  window.addEventListener("resize", function () {
    editor.resize();
  });

  editor.session.on("change", updateEditor);

  completionTippy = tippy(document.getElementById("editor"), {
    content: "Loading...",
    trigger: "manual",
    placement: "top-start",
    arrow: true,
    interactive: true,
  });

  // Override the default tab behavior
  editor.commands.addCommand({
    name: "insertCompletion",
    bindKey: { win: "Tab", mac: "Tab" },
    exec: function (editor) {
      if (currentCompletion) {
        editor.insert(currentCompletion);
        currentCompletion = "";
        completionTippy.hide();
      } else {
        editor.indent();
      }
    },
  });
}

async function showCompletionSuggestion() {
  const cursorPosition = editor.getCursorPosition();
  const screenPosition = editor.renderer.textToScreenCoordinates(
    cursorPosition.row,
    cursorPosition.column,
  );

  completionTippy.setContent("Loading...");
  completionTippy.setProps({
    getReferenceClientRect: () => ({
      width: 0,
      height: 0,
      top: screenPosition.pageY,
      bottom: screenPosition.pageY,
      left: screenPosition.pageX,
      right: screenPosition.pageX,
    }),
  });
  completionTippy.show();

  try {
    const response = await fetch("/complete", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        code: editor.getValue(),
        row: cursorPosition.row,
        column: cursorPosition.column,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    currentCompletion = data.completion;
    completionTippy.setContent(`${currentCompletion} (Press Tab to insert)`);
  } catch (error) {
    console.error("Error:", error);
    completionTippy.setContent("Error fetching completion");
    currentCompletion = "";
  }

  setTimeout(() => {
    if (currentCompletion) {
      completionTippy.hide();
      currentCompletion = "";
    }
  }, 5000);
}
