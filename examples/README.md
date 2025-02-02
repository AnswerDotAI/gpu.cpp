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

## Advanced Examples

| Example | Description |
|---------|-------------|
| [float16](float16) | Hello World example using the float16 WebGPU extension, instead of the default float32. |
| [matmul](matmul) | Tiled matrix multiplication. |
| [transpose](transpose) | Tiled matrix transpose. |
