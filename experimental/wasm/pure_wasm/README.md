Test of pure wasm control of webgpu w/o emscripten runtime.

First, build the wasm binary:

```
make build/run.wasm
```

Run the static server:
```
make server
```

Open browser to localhost:8000.

You should see in the developer console `WebAssembly module loaded ... Hello
World!` printed from the wasm binary.
