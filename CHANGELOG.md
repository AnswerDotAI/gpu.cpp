# Changelog

## 0.2.0 - 2026-08-21

This release modernizes gpu.cpp around current Dawn and adds a maintained
browser target.

### Added

- A browser backend using Emdawnwebgpu, Emscripten, JSPI, and an ES-module
  Embind API for persistent contexts and JavaScript typed arrays.
- Explicit read/read-write bindings, reusable kernels, uniform parameters, and
  asynchronous dispatch and readback through `gpu::Future`.
- Native SPIR-V input with an executable WebGPU-profile contract test.
- C++ stories for WGSL, native f16, and SPIR-V; a NumPy Python story; and a
  Chrome story covering browser errors, context reuse, dispatch, and readback.
- Reproducible scripts pinning Dawn, Emdawn, emsdk, SPIRV-Tools, and
  SPIRV-Headers to exact revisions.
- A self-contained macOS ARM64 Python wheel and direct PyPI publish workflow.

### Changed

- WebGPU objects now use Dawn's generated C++ RAII facade and normal C++ value
  semantics.
- The Python binding now follows the C++ API and preserves NumPy shape and
  `float16`, `float32`, and `int32` dtypes. Futures retain their context and
  support blocking waits or native `asyncio` dispatch and NumPy readback.
- Host-side f16 uses native `_Float16` where supported; the portable core
  treats f16 as two-byte IEEE 754 storage.
- Examples and the build are consolidated under the root CMake project and the
  single `./test` entrypoint.

### Removed

- The legacy raw WebGPU C API implementation, manual resource pools, custom
  half implementation, Haskell binding, obsolete build files, and abandoned
  experimental targets.
- The Closure compiler and Java dependency from browser builds.
