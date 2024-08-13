const State = {
  preamble_template: "",
  preamble: "",
  wgSize: [],
  gridSize: [],
  editor: null,
  terminal: null,
  module: null,
  isModuleReady: false,
  checkAnswer: false,
  isDispatchReady: true, // don't allow multiple overlapping dispatches
  puzzleIndex: 0,
};

const PuzzleSpec = [
  {
    name: "Map",
    description:
      'Implement a "kernel" (GPU function) that adds 10 to each position of vector `a` and stores it in vector `out`. You have 1 thread per position.',
  },
  {
    name: "Zip",
    description:
      "Implement a kernel that adds together each position of `a` and `b` and stores it in `out`. You have 1 thread per position.",
  },
  {
    name: "Guards",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have more threads than positions.",
  },
  {
    name: "Map 2D",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out`. Input `a` is 2D and square. You have more threads than positions.",
  },
  {
    name: "Broadcast",
    description:
      "Implement a kernel that adds `a` and `b` and stores it in `out`. Inputs `a` and `b` are vectors. You have more threads than positions.",
  },
  {
    name: "Blocks",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have fewer threads per block than the size of `a`.",
  },
  {
    name: "Blocks 2D",
    description:
      "Implement the same kernel in 2D. You have fewer threads per block than the size of `a` in both directions.",
  },
  {
    name: "Shared",
    description:
      "Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have fewer threads per block than the size of `a`. Use shared memory and `cuda.syncthreads` to ensure threads do not cross.",
  },
  {
    name: "Pooling",
    description:
      "Implement a kernel that sums together the last 3 positions of `a` and stores it in `out`. You have 1 thread per position.",
  },
  {
    name: "Dot Product",
    description:
      "Implement a kernel that computes the dot-product of `a` and `b` and stores it in `out`. You have 1 thread per position.",
  },
  {
    name: "1D Convolution",
    description:
      "Implement a kernel that computes a 1D convolution between `a` and `b` and stores it in `out`. Handle the general case.",
  },
  {
    name: "Prefix Sum",
    description:
      "Implement a kernel that computes a sum over `a` and stores it in `out`. If the size of `a` is greater than the block size, only store the sum of each block using parallel prefix sum.",
  },
  {
    name: "Axis Sum",
    description:
      "Implement a kernel that computes a sum over each column of `a` and stores it in `out`.",
  },
  {
    name: "Matrix Multiply",
    description:
      "Implement a kernel that multiplies square matrices `a` and `b` and stores the result in `out`. Optimize by using shared memory for partial dot-products.",
  },
];

////////////////////////////////////////
// Initialization
////////////////////////////////////////

function initializeApp(initial_content) {
  initializeTerminal();
  initializeEditor(initial_content);
  initializeModule();
  setupEventListeners();
  console.log("App initialized");
}

function initializeTerminal() {
  AppState.terminal = new Terminal();
  const fitAddon = new FitAddon.FitAddon();
  AppState.terminal.loadAddon(fitAddon);
  AppState.terminal.open(document.getElementById("output"));
  fitAddon.fit();
  window.AppState = AppState;
  console.log("Terminal initialized");
}

function initializeEditor(initialContent) {
  AppState.editor = ace.edit("editor");
  // AppState.editor.setTheme("ace/theme/monokai");
  AppState.editor.setTheme("ace/theme/dracula");
  AppState.editor.session.setMode("ace/mode/javascript");
  AppState.editor.setOptions({
    fontSize: "16px",
    showPrintMargin: false,
    showGutter: false,
    highlightActiveLine: true,
    wrap: true,
  });
  AppState.editor.setKeyboardHandler("ace/keyboard/vim");
  AppState.editor.setValue(initialContent || "");
  console.log("Initial content:\n", initialContent);
  console.log("Editor initialized");
}

function initializeModule() {
  createModule()
    .then((Module) => {
      AppState.module = Module;
      AppState.module.print = customPrint;
      AppState.module.printErr = customPrint;
      AppState.isModuleReady = true;
      console.log("Module initialized");
      update({ type: "init" });
    })
    .catch((error) => {
      console.error("Failed to initialize module:", error);
    });
}

function setupEventListeners() {
  AppState.editor.session.on("change", () => update({ type: "edit" }));
  window.addEventListener("resize", () => AppState.editor.resize());
  document
    .getElementById("workgroup_x")
    .addEventListener("change", () => update({ type: "wgUpdate" }));
  document
    .getElementById("workgroup_y")
    .addEventListener("change", () => update({ type: "wgUpdate" }));
  document
    .getElementById("workgroup_z")
    .addEventListener("change", () => update({ type: "wgUpdate" }));
  document
    .getElementById("grid_x")
    .addEventListener("change", () => update({ type: "gridUpdate" }));
  document
    .getElementById("grid_y")
    .addEventListener("change", () => update({ type: "gridUpdate" }));
  document
    .getElementById("grid_z")
    .addEventListener("change", () => update({ type: "gridUpdate" }));
  document.getElementById("prev").addEventListener("click", () => {
    update({ type: "selectPuzzle", value: "prev" });
  });
  document.getElementById("next").addEventListener("click", () => {
    update({ type: "selectPuzzle", value: "next" });
  });
}

////////////////////////////////////////
// Printing to output window
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
// Update
////////////////////////////////////////

function updateDispatchParams() {
  wgSize = [
    document.getElementById("workgroup_x").value,
    document.getElementById("workgroup_y").value,
    document.getElementById("workgroup_z").value,
  ];
  gridSize = [
    document.getElementById("grid_x").value,
    document.getElementById("grid_y").value,
    document.getElementById("grid_z").value,
  ];
  wgSize = wgSize.map((x) => parseInt(x));
  gridSize = gridSize.map((x) => parseInt(x));
  AppState.wgSize = wgSize;
  AppState.gridSize = gridSize;
  AppState.preamble = AppState.preamble_template.replace(
    /{{workgroupSize}}/g,
    wgSize.join(", "),
  );
  console.log("New preamble:\n", AppState.preamble);
}

async function updateEditor() {

  function waitForDispatchReady() {
    return new Promise((resolve) => {
      function checkReady() {
        if (AppState.isDispatchReady) {
          resolve();
        } else {
          console.log("Waiting...");
          setTimeout(checkReady, 100); // Check every 100ms
        }
      }
      checkReady();
    });
  }

  // Recover from errors TODO(avh): only do this if there's an error
  createModule().then((Module) => {
    console.log("updateEditor() - Module ready");
  });
  if (AppState.module && AppState.module.runCheck) {
    if (!AppState.isDispatchReady) {
      console.log("Waiting for dispatch to be ready");
      await waitForDispatchReady();
    }

    console.log("Executing kernel");
    AppState.terminal.clear();
    console.log("Code:\n", AppState.preamble + AppState.editor.getValue());
    AppState.isDispatchReady = false;
    AppState.module
      .runCheck(
        AppState.preamble + AppState.editor.getValue(),
        AppState.wgSize,
        AppState.gridSize,
      )
      .then((result) => {
        console.log("check:", result);
        AppState.checkAnswer = result;
        AppState.isDispatchReady = true;
        render();
      });
  } else {
    console.log("updateEditor() - Module not ready");
  }
}

function update(event) {
  console.log("Updating");
  if ((event.type === "selectPuzzle") & (event.value === "prev")) {
    AppState.puzzleIndex = (AppState.puzzleIndex - 1);
  }
  if ((event.type === "selectPuzzle") & (event.value === "next")) {
    AppState.puzzleIndex = (AppState.puzzleIndex + 1);
  }
  if (AppState.puzzleIndex < 0) {
    AppState.puzzleIndex = PuzzleSpec.length - 1;
  }
  if (AppState.puzzleIndex >= PuzzleSpec.length) {
    AppState.puzzleIndex = 0;
  }
  updateDispatchParams();
  updateEditor();
  render();
}

////////////////////////////////////////
// Render
////////////////////////////////////////

function render() {
  document.getElementById("preamble").innerHTML = AppState.preamble;
  // update DIV
  console.log("AppState.checkAnswer: ", AppState.checkAnswer);
  document.getElementById("correct").textContent = AppState.checkAnswer
    ? "Your answer is adequate."
    : "You are WRONG.";
  document.getElementById("puzzle_name").textContent =
    "Puzzle " +
    (AppState.puzzleIndex + 1) +
    ": " +
    PuzzleSpec[AppState.puzzleIndex].name;
  document.getElementById("puzzle_description").textContent =
    PuzzleSpec[AppState.puzzleIndex].description;
}
