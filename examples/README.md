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
| [physics](physics) | Parallel physics simulation of a double pendulum with each thread starting at a different initial condition. |
| [webgpu_intro](webgpu_intro) | A minimal from-scratch example of how to use WebGPU directly without this library. This is useful to understand the code internals of gpu.cpp. Note this takes a while to build as it compiles the WebGPU C API implementation. |

Future examples that may be added (collaborations welcome):

- Matrix multiplication worklog / tutorial
- A transformer block for a neural network (see `experimental/` WIP)
