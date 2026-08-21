# gpu.cpp

gpu.cpp is a small, header-only C++20 interface for GPU compute through
[Dawn](https://dawn.googlesource.com/dawn), Google's WebGPU implementation. It
uses Metal on macOS, Vulkan on Linux, and Emdawnwebgpu in browsers.

The library deliberately stays close to WebGPU: it owns the native resources,
handles asynchronous work and errors, and removes repetitive descriptor code,
but shaders still declare their resources and execution model explicitly.

## Build and test

Install CMake, Ninja, Python 3, and a C++20 compiler. Linux also needs a Vulkan
driver; Mesa's Vulkan driver is sufficient for development and CI.

```bash
./test --rebuild-dawn  # first setup, or after changing the pinned Dawn revision
./test                 # configure, build, and run the native GPU stories
./test --rebuild-web   # first browser setup
./test --web           # build and run the browser story in Chrome
./build/hello_gpu      # run the hello-world example after building
```

`tools/build_dawn.sh` checks out exact revisions, builds a monolithic shared
Dawn library, and stages its headers, library, and `spirv-as` under
`third_party/dawn/`. During Dawn development, CMake can instead use the source
and build trees under `third_party/local/dawn/` directly.

Browser builds require Chrome with WebGPU and JSPI, plus a sibling `../emsdk`
clone:

```bash
git clone --depth 1 https://github.com/emscripten-core/emsdk.git ../emsdk
./test --rebuild-web
```

`tools/build_emdawn.sh` moves that clone to the exact emsdk revision pinned by
Dawn, installs its Emscripten SDK, and stages an Emdawn local port under
`third_party/emdawnwebgpu/`. Nothing needs to be added to the shell profile.

Native tests must have access to a real Metal or Vulkan adapter. gpu.cpp
disables Dawn's Null backend so an inaccessible GPU fails clearly rather than
producing plausible-looking no-op executions.

## Example

```cpp
#include "gpu.hpp"

#include <cstdint>
#include <vector>

using namespace gpu;

static constexpr auto twice = R"(
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&input)) {
    output[id.x] = input[id.x] * 2;
  }
}
)";

int main() {
  auto context = createContext();
  std::vector<int32_t> input{1, 2, 3, 4};
  std::vector<int32_t> output(input.size());
  auto gpuInput = createTensor(context, {input.size()}, ki32, input);
  auto gpuOutput = createTensor(context, {output.size()}, ki32);
  auto kernel = createKernel(context, WGSL{twice, 4},
                             Bindings{read(gpuInput), readWrite(gpuOutput)});

  auto dispatched = dispatchKernel(context, kernel);
  wait(context, dispatched);
  auto downloaded = toCPU(context, gpuOutput, output);
  wait(context, downloaded);
}
```

`read()` and `readWrite()` are explicit because WebGPU pipeline layouts must
agree with the shader declarations. Dawn validates the agreement when
`createKernel()` builds the pipeline.

## API shape

- `Context` owns the instance, native adapter, device, queue, and error state.
- `Tensor` owns a WebGPU buffer plus its shape and numeric type.
- `WGSL` and `SPIRV` own shader source and entry-point metadata.
- `Kernel` owns a reusable compute pipeline, bind group, and optional uniform
  parameter buffer.
- `dispatchKernel()` and `toCPU()` return `gpu::Future`; `wait()` pumps Dawn
  events natively or suspends through JSPI in a browser, and propagates
  asynchronous failures.
- `toGPU()` updates an existing tensor or a kernel's parameter buffer.

The portable core treats f16 host data as two-byte IEEE 754 storage. Native
code may include `numeric_types/half.hpp` for the compiler's `_Float16` type;
browser bindings can use `Float16Array` or raw `Uint16Array` storage.

All WebGPU handles use Dawn's generated `wgpu` C++ RAII facade. gpu.cpp does
not maintain parallel resource pools or manually release C handles.

## SPIR-V input

SPIR-V is an opt-in Dawn instance feature:

```cpp
auto context = createContext({.enableSPIRV = true});
auto kernel = createKernel(context, SPIRV{words},
                           Bindings{readWrite(output)});
```

Input must satisfy Dawn/Tint's WebGPU SPIR-V profile. The current pinned Dawn
accepts SPIR-V 1.3; `tests/write42.spvasm` is the executable contract fixture.
This is intentionally a WebGPU-facing compiler contract, not arbitrary Vulkan
SPIR-V passthrough.

## Using gpu.cpp from CMake

After staging Dawn, add this repository as a subdirectory and link the
interface target:

```cmake
add_subdirectory(path/to/gpu.cpp)
target_link_libraries(my_program PRIVATE gpucpp)
```

See [DEV.md](DEV.md) for the dependency layout and update process, and
[CHANGELOG.md](CHANGELOG.md) for release notes.

Browser consumers can supply their own Embind surface while inheriting the
Emdawnwebgpu, Wasm-exception, and JSPI settings from `gpucpp`:

```cmake
add_subdirectory(path/to/gpu.cpp EXCLUDE_FROM_ALL)
add_executable(my_web_app app.cpp)
set_target_properties(my_web_app PROPERTIES SUFFIX ".js")
target_link_libraries(my_web_app PRIVATE gpucpp)
target_link_options(my_web_app PRIVATE "--bind" "--no-entry"
  "-sMODULARIZE=1" "-sEXPORT_NAME=createModule" "-sENVIRONMENT=web"
  "-sALLOW_MEMORY_GROWTH=1")
```

The Embind functions that call `createContext()`, `wait()`, or browser-facing
helpers containing them must use `emscripten::async()` so JSPI can suspend the
Wasm stack.

## Python

Install the self-contained macOS ARM64 wheel from PyPI with `pip install
gpu-cpp`. The optional pybind11 module is also built by default when gpu.cpp is
the top-level CMake project. It accepts C-contiguous NumPy arrays and preserves
tensor shape and `float16`, `float32`, or `int32` dtype information:

```python
import numpy as np
import gpu_cpp as gpu

context = gpu.Context()
values = np.arange(4, dtype=np.float32)
gpu_values = gpu.tensor(context, values)
gpu_output = gpu.create_tensor(context, values.shape, gpu.f32)
kernel = gpu.create_kernel(
    context, gpu.WGSL(source, workgroup_size=[4, 1, 1]),
    [gpu.read(gpu_values), gpu.read_write(gpu_output)])
dispatched = gpu.dispatch_kernel(context, kernel)
gpu.wait(dispatched)
result = gpu.wait(gpu.to_numpy(context, gpu_output))
```

Futures own their context and support both blocking scripts and native
`asyncio`/notebook execution. Readback futures also own their destination array
and return it on completion:

```python
await gpu.dispatch_kernel(context, kernel)
result = await gpu.to_numpy(context, gpu_output)
```

Set `GPUCPP_BUILD_PYTHON=OFF` when embedding gpu.cpp in a CMake project that
does not need the module.

## Browser

The same `gpucpp` CMake target uses Dawn's Emdawnwebgpu port under Emscripten.
Browser builds accept WGSL, use the browser-selected WebGPU adapter, and use
JSPI for gpu.cpp's synchronous-looking `wait()` calls. SPIR-V is intentionally
native-only because browser WebGPU does not accept it.

The build also produces `gpu_cpp_web.mjs` and its Wasm file. Its persistent
context accepts ordinary JavaScript typed arrays; `readWrite` arrays are
updated in place:

```js
import createGpuCpp from "./gpu_cpp_web.mjs";

const gpu = await createGpuCpp();
const context = await gpu.createContext(false); // true enables shader-f16
const input = new Float32Array([1, 2, 3, 4]);
const output = new Float32Array(4);

await context.run({
  code: wgsl,
  workgroupSize: [4, 1, 1],
  workgroups: [1, 1, 1],
  bindings: [
    {data: input, access: "read"},
    {data: output, access: "readWrite"},
  ],
  // parameters: new Uint8Array(...), // optional uniform bytes
});
context.delete();
```

Bindings accept `Float32Array`, `Int32Array`, and either `Float16Array` or raw
IEEE-754 half bits in a `Uint16Array`. Calls on one context must be awaited
sequentially; validation and runtime failures reject the returned Promise.

## Scope

gpu.cpp targets general-purpose WebGPU compute. It is not a tensor
framework, graph compiler, or rendering engine. The goal is a concise layer for
projects that need direct control over shaders, bindings, dispatch, and data
movement without carrying raw WebGPU setup throughout their code.
