# Examples (Work-in-Progress)

Here are some standalone projects exemplifying how to use this library as well
as what's under the hood.

Note some of these examples are still a work-in-progress and may not be fully
functional.

In order of beginning to advanced:

| Example | Description |
|---------|-------------|
| [hello_world](hello_world) | Minimal example to get started with gpu.cpp, implements a GELU neural network activation function. |
| [gpu_puzzles](gpu_puzzles) | (WIP) Implementation of Sasha Rush's GPU puzzles
| [render](render) | GPU rendering of a signed distance function for a 3D sphere. |
| [shadertui](shadertui) | A terminal UI taker on shadertoy demonstrating runtime live reloading of WGSL. |
| [physics](physics) | Parallel physics simulation of a double pendulum with each thread starting at a different initial condition. |
| [matmul](matmul) | Tiled matrix multiplication. |
| [transpose](transpose) | Tiled matrix transpose. |
| [webgpu_from_scratch](webgpu_from_scratch) | A minimal from-scratch example of how to use WebGPU directly without this library. This is useful to understand the code internals of gpu.cpp. Note this takes a while to build as it compiles the WebGPU C API implementation. |
