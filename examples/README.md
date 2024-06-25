# Examples (Work-in-Progress)

Here are some standalone projects exemplifying how to use this library as well
as what's under the hood.

NOTE: These examples are a work-in-progress.

In order of beginner to advanced:

- hello_world - Minimal example to get started with gpu.cpp, implements a GELU
  neural network activation function.
- gpu_puzzles - (WIP) Implementation of Sasha Rush's GPU puzzles https://github.com/srush/GPU-Puzzles
- render - GPU rendering of a signed distance function for a 3D sphere.
- physics - Parallel physics simulation of a double pendulum with each thread starting at a different initial condition.
- transformer - (TODO) a neural network transformer block computation
- webgpu_intro - A minimal from-scratch example of how to use WebGPU directly
  without this library. This is useful to understand the code internals of
  gpu.cpp. Note this takes a while to build as it compiles the WebGPU C API
  implementation.
