# gpu.cpp

gpu.cpp is a minimal library for portable low-level GPU computation using
WebGPU. 

*Work-in-Progress* See [release
tasks](https://github.com/AnswerDotAI/gpu.cpp/wiki/Release-Tasks) for the
current status.

## Who is gpu.cpp for?

gpu.cpp is a lightweight library for R&D projects and products prioritizing
both low-level control of GPU computation and portability.  

To this end, gpu.cpp leverages the WebGPU API spec to provide a portable
interface to the GPU. In spite of the name, WebGPU has both native (e.g.
[Dawn](https://github.com/google/dawn/) and
[wgpu](https://github.com/gfx-rs/wgpu)) as well as [browser
implementations](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status),
it does not necessitate programs to be running on the web in a browser.

The library provides a small set of composable functions and types that make
WebGPU compute much easier to work with, while keeping abstractions minimal and
transparent.

The goal of gpu.cpp is to make integrating portable, low-level GPU computations
into projects simple and concise.

# Hello World: A GELU Kernel

Here's an GELU kernel implemented (based on the CUDA implementation of
[llm.c](https://github.com/karpathy/llm.c) as an on-device WGSL shader and
invoked from the host using this library.

```
#include <array>
#include <cstdio>
#include "gpu.h"

using namespace gpu;

// Device code (runs on the GPU) using WGSL (WebGPU Shading Language)
const char *kGELU = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + cube)));
    }
}
)";

// Host code (runs on the CPU) using C++
int main(int argc, char **argv) {
  GPUContext ctx = CreateGPUContext();
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  GPUTensor input = CreateTensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  Kernel op =
      PrepareKernel(ctx, kGELU, std::array{input}, output);
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 10; ++i) {
    fprintf(stdout, "%d : %f\n", i, outputArr[i]);
  }
  fprintf(stdout, "...\n\n");
  return 0;
}
```

In practice, an implementation of GELU (as well as some other common kernels is
available in `nn/shaders.h`, but for the sake of completeness this example
includes both the host WebGPU API code and GPU device WGSL shader
implementation.

For those curious about what happens under the hood with the raw WebGPU API,
the equivalent functionality is implemented using the raw WebGPU C API in
`examples/webgpu_intro/run.cpp`.

## Quick Start

The only dependency of this library is a WebGPU implementation. Currently we
recommend using the Dawn backend until further testing, but we plan to support
emscripten (web) and wgpu (native) backends.

The build is handled by cmake. Some useful common cmake invocations are wrapped
in the convenience Makefile. To start you can try building a terminal demo
tutorial which also tests the functionality of the library, this builds the
demo tutorial in `run.cpp`:

```
make demo
```

You should see an introductory message:
```
   ____ _____  __  __ _________  ____ 
  / __ `/ __ \/ / / // ___/ __ \/ __ \
 / /_/ / /_/ / /_/ // /__/ /_/ / /_/ /
 \__, / .___/\__,_(_)___/ .___/ .___/ 
/____/_/               /_/   /_/

================================================================================

Welcome!
--------

This program is a brief intro to the gpu.cpp library.

You can use the library by simply including the gpu.h header, starting with a
build template (see examples/hello_gpu/ for a template project that builds the
library).

  #include "gpu.h"

See `examples/hello_world/` for an examle of build scripts to run a standalone
program that uses this library.

┌──────────────────────────────┐
│ Press Enter to Continue...   │
└──────────────────────────────┘
```

The first time you build and run this, it will download the WebGPU backend
implementation (Dawn by default) and build it which may take a few minutes. The
gpu.cpp library itself is small so after building the Dawn backend the first
time, subsequent builds of the library should take seconds on most personal
computing devices.

You can build the library itself which builds a shared library that you can
link against for your own projects. This builds a library that can be used in
other C++ projects (most of the code is in `gpu.h`, plus some supporting code
in `utils/`).

```
make libgpu
```

((TODO(avh): link to a template repo that with gpu.cpp as a library already
configured.))

From there you can explore the example projects in `examples/` which illustrate
how to use gpu.cpp as a library. For example a standalone version of the hello
world gelu kernel shown above:

```
cd examples/hello_world && make run & cd ../..
```

You should see a bit of output showing the first elements of the GELU
computation from the above example:

```
Hello, gpu.cpp

0 : 0.000000
1 : 0.841192
2 : 1.954598
3 : 2.996363
4 : 3.999930
5 : 5.000000
6 : 6.000000
7 : 7.000000
8 : 8.000000
9 : 9.000000
...
```

A more extensive set of (machine learning-centric) kernels is implemented in
`utils/test_kernels.cpp`. This can be built and run (from the top level
directory) using:

```
make tests
```

For more configurability and control of the build, see the `cmake`  invocations
in the `Makefile`,  as well as the configuration in `Cmakelists.txt`.

## Motivation and Goals

Although gpu.cpp is intended for any form of general purpose GPU computation,
the project is partly motivated by emerging needs in machine learning R&D. With
large foundation models, a significant amount of R&D now occurs in the
post-training computation.

Large foundation models have become become computable objects over which custom
algorithms are implemented. Many important foundation model advances today take
this form, for example:

- Approximate Computation - quantization, sparsification, model compression, distillation
- Conditional/Branching Computation - Mixture-of-experts, Hydranets, Fast feed-forward, Early Exit
- Auxillary Computation - Q[X]oRA variants, Speculative Decoding, Constrained Decoding

Performing custom computations over compute-intensive foundation models
benefits from low-level control of the GPU. At this time, tooling for
implementing low-level GPU computation is heavily focused on CUDA as a first
class citizen.

This leaves a gap in portability, meaning R&D algorithms that work in the data
center do not get operationalized to for everyday use to run on compute that's
broadly accessible.

We created gpu.cpp as a lightweight C++ library that allows us to easily and
directly implement native low-level GPU algorithms as part of R&D and drop
implementations into code running on personal computing devices either as
native applications or in the browser without being blocked by hardware,
tooling, or runtime support.


## What gpu.cpp is not

gpu.cpp is a tool for building things for end users, it's not intended to be
used by end users directly. Although there is basic low level support for
ND-arrays and a small library implementing neural network blocks as shaders, it
is not a high-level machine learning framework or inference engine (although it
can be useful in implementing one).

It is also not strictly for the web or deploying to the web - WebGPU is
convenient API with both both native (e.g. Dawn) and browser implementations.
It uses WebGPU as a portable GPU API first and foremost, with the possibility
of running in the browser being support being a convenient bonus.

Finally, the focus of gpu.cpp is explicitly compute rather than
rendering/graphics, although it might be useful for compute shaders in graphics
projects - one of the examples is a small compute renderer, rendered to the
terminal.

## Contributing and Work-in-Progress

We welcome contributions! There's a lot of low hanging fruit - fleshing out
examples, adding to the library of useful shader building blocks, filling in
gaps in the library API. Happy to welcome collaborators to make the library
better. 
