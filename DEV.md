# Development

## Dependency model

Dawn and its native dependencies are pinned in `tools/dependencies.sh`.
`tools/build_dawn.sh` maintains ignored working
trees under `third_party/local/dawn/` and stages the small runtime distribution
under `third_party/dawn/`. Neither directory is committed.

The same file pins the emsdk revision and Emscripten version recorded in
Dawn's DEPS. `tools/build_emdawn.sh` builds `emdawnwebgpu_pkg` from that exact
Dawn source and stages its local Emscripten port under
`third_party/emdawnwebgpu/`. The sibling `../emsdk` clone and staged package are
not committed.

The top-level build fetches the tagged pybind11 release declared in
`CMakeLists.txt`; consumers using only the C++ target do not fetch or build it.
The Python runtime test also requires NumPy.

The current Dawn revision needs compatible SPIRV-Tools and SPIRV-Headers
revisions for `TINT_BUILD_SPV_READER`; those exact revisions are pinned beside
the Dawn revision. This is dependency selection, not a Dawn source patch.

The build enables only the native platform backend (Metal or Vulkan), the WGSL
reader, and the SPIR-V reader. Dawn's Null backend is disabled because it
accepts work but intentionally does not execute it.

The Emdawn package build disables protobuf, Tint's protobuf-based IR format,
and C++ modules because none are part of the browser port. This also avoids a
false-positive Emscripten C++-module probe at the current Dawn revision.

To update Dawn and Emdawn:

1. Change the Dawn, SPIRV-Tools, and SPIRV-Headers revisions in
   `tools/dependencies.sh`.
2. Copy the emsdk revision and corresponding Emscripten version from the new
   Dawn revision's `DEPS`.
3. Run `./test --rebuild-dawn`, then `./test` on a machine with a real GPU.
4. Run `./test --rebuild-web` in Chrome.
5. Update the SPIR-V version/profile documentation if Tint's reader changed.

No gpu.cpp patches are applied to the Dawn source tree.

## Tests

`./test` is the only test entrypoint. It configures with CMake/Ninja, builds all
maintained examples, and runs two readable end-to-end stories through CTest.
The C++ story covers:

1. WGSL upload, explicit read/read-write bindings, dispatch, and readback.
2. Native `_Float16` upload and execution through an f16 WGSL pipeline.
3. Assembly and execution of a minimal WebGPU-compatible SPIR-V module.

The SPIR-V assembly is kept as text so the compiler/runtime contract is
reviewable. Avoid adding tests that merely restate Dawn validation; add a new
story only when gpu.cpp itself owns meaningful behavior.

The Python story covers NumPy upload/readback, explicit binding access,
uniform parameters, reuse of a compiled kernel, blocking waits, and native
`asyncio` dispatch and readback.

`tools/build_python_wheel.sh` packages the staged macOS 12 Dawn runtime into a
self-contained ARM64 wheel and checks its metadata. Publish that fresh wheel
with `twine upload dist/*.whl`, using the developer's standard `~/.pypirc`
credentials. Publishing from other platforms belongs in CI rather than this
local release path.

`./test --web` cross-compiles the same public core and Embind API against
Emdawnwebgpu, opens the result with `emrun`, and checks rejected invalid WGSL,
context reuse, typed-array upload, dispatch, and readback in Chrome. The
release build uses Wasm exceptions and JSPI. CI uses `./test --build-web` to
compile the same artifacts without requiring a browser GPU; the complete story
remains the local runtime contract.

## Architecture

`gpu.hpp` is the library. Public objects own Dawn's `wgpu` RAII handles, so
normal C++ lifetime rules replace the old tensor/kernel pools and manual C API
release bookkeeping.

Pipeline creation uses a validation error scope and returns ordinary C++
exceptions with Dawn's diagnostic. Permanent device callbacks never throw;
they write into `Context` error state, which foreground API calls surface.

`gpu::Future` retains both the callback result and Dawn's `wgpu::Future`.
Native waits pump `ProcessEvents`; browser waits call `WaitAny`, allowing JSPI
to suspend Wasm while JavaScript and WebGPU make progress.

Python futures retain shared ownership of their context. Their awaiter polls
`ProcessEvents` on the asyncio thread rather than moving Dawn work to an
executor thread. Async readback uses callback-owned storage, so abandoning a
future cannot leave Dawn writing into a freed NumPy buffer.

The browser Embind layer owns one persistent `Context`. Each `run()` creates
short-lived tensors and a kernel from JavaScript typed arrays, rejects
overlapping calls, and copies `readWrite` results back in place. Wasm
exceptions turn C++ validation failures into rejected JavaScript Promises.

Each dispatch creates a command encoder because WebGPU command buffers are
single-use. Pipelines and bind groups remain reusable. Readback uses a staging
buffer and returns a future whose callback owns that buffer until mapping is
complete.
