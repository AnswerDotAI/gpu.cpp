# gpu.cpp examples

This directory contains examples of how to use gpu.cpp. 

Each example is a standalone project that can be built and run independently by
running `make` from within the example directory.

Before running any of these examples, make sure you've downloaded the Dawn
native webgpu installation binary by running `make dawnlib` from the root
directory of the repository.

## Basic Examples

| Example | Description |
|---------|-------------|
| [hello_world](hello_world) | Minimal example to get started with gpu.cpp, implements a GELU neural network activation function. |
| [gpu_puzzles](gpu_puzzles) | Implementation of Sasha Rush's GPU puzzles. |
| [shadertui](shadertui) | An example of runtime live reloading of WGSL - demonstrated using a terminal shadertoy-like scii rendering. |
| [render](render) | GPU ascii rendering of a signed distance function for two rotating 3D spheres. |
| [physics](physics) | Parallel physics simulation of a double pendulum with each thread starting at a different initial condition. |
| [web](web) | A minimal example of how to use gpu.cpp to build a WebAssembly module that runs in the browser. Before building this example, make sure you've installed the emscripten sdk by following the [instructions here](https://emscripten.org/docs/getting_started/downloads.html) and run `source emsdk_env.sh` from the `emsdk/` directory that was created when you cloned the emscripten repository. |

## Advanced Examples

| Example | Description |
|---------|-------------|
| [matmul](matmul) | Tiled matrix multiplication. |
| [transpose](transpose) | Tiled matrix transpose. |
| [webgpu_from_scratch](webgpu_from_scratch) | A minimal from-scratch example of how to use WebGPU directly without this library. This is useful to understand the code internals of gpu.cpp. Note this takes a while to build as it compiles the WebGPU C API implementation. |
