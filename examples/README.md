# gpu.cpp examples

The root CMake project builds every example:

```bash
./test
```

- `hello_world`: minimal WGSL upload, dispatch, and readback
- `float16`: device feature selection and f16 compute
- `gpu_puzzles`: exercises and solutions for Sasha Rush's GPU puzzles
- `matmul`: progressively optimized matrix multiplication kernels
- `physics`: an interactive double-pendulum simulation
- `render`: an interactive terminal SDF renderer
- `shadertui`: live-reloaded WGSL terminal shaders
- `transpose`: naive and tiled matrix transpose kernels

`make run` runs `hello_world`. The interactive and benchmark examples are built
but not run by the test suite.
