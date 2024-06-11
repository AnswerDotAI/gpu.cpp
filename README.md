# gpu.cpp

gpu.cpp is a minimal library for portable low-level GPU computation using
WebGPU. 

*Work-in-Progress* See [release
tasks](https://github.com/AnswerDotAI/gpu.cpp/wiki/Release-Tasks) for the
current status.

## Who is gpu.cpp for?

gpu.cpp is aimed at R&D projects and products involving GPU computation
requiring low-level control of GPU device computation code and hardware
portability.  

To have both portability and low level control, the gpu.cpp leverages the
WebGPU API spec to provide a portable host interface to the GPU and WebGPU
Shading Language (WGSL) for on-device code.  

WebGPU is leveraged as a portable GPU interface with both native (e.g. Dawn and
wgpu) as well as browser implementations. WebGPU does not necessitate programs
running in the browser.

The library provides a small set of types and functions that make WebGPU
compute much easier to work with, while being transparent - WebGPU API
resources are directly accessible never more than 1 layer of indirection from
the library interface. 

We hope to make integrating portable, low-level GPU computations simple and
concise, even enjoyable.

# Hello World: A GELU Kernel

Here's an GELU kernel implemented (based on the CUDA implementation of
[llm.c](https://github.com/karpathy/llm.c) as an on-device WGSL shader and
invoked from the host using this library.

```
#include <array>
#include <cstdio>
#include "gpu.h"

using namespace gpu;

// GPU device code as a WGSL shader 
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

// Host code ot launch kernel
int main(int argc, char **argv) {
  GPUContext ctx = CreateGPUContext();
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr, outputArr;
  std::iota(begin(inputArr), end(inputArr), 0.0f); // use 1..N as dummy data
  GPUTensor input = Tensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = Tensor(ctx, {N}, kf32, outputArr.data());
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

For those curious of what things look like under the hood, and as a comparison,
the equivalent functionality is implemented using the raw WebGPU C API in
`examples/webgpu_intro/run.cpp`.

## Quick Start

The only dependency of this library is a WebGPU implementation. Currently we
recommend using the Dawn backend until further testing, but we hope to
eventually support emscripten (for web) and wgpu backends.

The build is handled by cmake. Some common cmake invocations are wrapped in the
convenience Makefile. To start you can try building a terminal demo tutorial
which also tests the functionality of the library, this builds the demo
tutorial in `run.cpp`

```
make demo
```

You can build the library itself which builds a shared library that you can link against for your own projects. This builds a library that can be used in other C++ projects (most of the code is in `gpu.h`, plus some supporting code in `utils/`).

```
make libgpu
```

((TODO(avh): link to a template repo that with gpu.cpp as a library already configured.))

From there you can explore the example projects ((TODO(avh): link)). which illustrate how to use gpu.cpp as a library. For example a standalone version of the hello world gelu kernel shown above:

```
cd examples/hello_world && make run
```

A more extensive set of (machine learning-centric) kernels is implemented in `utils/test_kernels.cpp`. This can be built and run (from the top level directory using:

```
make tests
```

For more configurability and control of the build, see the `cmake`  invocations in the `Makefile`,  as welll as the configuration in `Cmakelists.txt`.

## Motivation and Goals

Although gpu.cpp is intended for any form of general purpose GPU computation
and is not restricted to machine learning, the project is partly motivated by
emerging needs in machine learning R&D, specifically fine-grained control of
computation and algorithmic development for post-training computation.

What do we mean by post-training computation? In the past, model training has
been the primary focus of machine learning research and algorithmic advances.
Training has been the primary mechanism of control, and tooling has focused on
training compute. 

By contrast, in this era of large pre-trained foundation models, a significant
amount of R&D now occurs in the post-training computation. We can regard
foundation models as defining a contextual object over which we compose and
modify post-training computations. Many of the most important foundation model
research today can be considered composing computation and modification over a
trained model object. Some examples include (but not limited to):

- Input/Output control systems: prompt engineering, RAG, tool use / code interpreters
- Approximate Computation - quantization, sparsification, model compression, distillation
- Conditional/Branching Computation - Mixture-of-experts, Hydranets, Fast feed-forward, Early Exit
- Auxillary Computation - Q[X]oRA variants, Speculative Decoding, Constrained Decoding

This list is not intended to be comprehensive. The creation of tooling such as
gpu.cpp for low-level control of GPU compute reflects a hypothesis that there
are broad swaths of approaches to be explored that aren't folded into existing
inference engines and expressible through current compiler toolchains.

## What gpu.cpp is not

gpu.cpp is a tool for building things for end users, rather than for end users.
Although there is basic low level support for ND-arrays and kernels implementing
neural network blocks, it is not a high-level machine learning framework or
inference engine, although it can be useful building such things.

It is also not strictly for the web or deploying to the web - WebGPU is
convenient API with both both native (e.g. Dawn) and browser implementations. It
uses WebGPU as a portable GPU API first and foremost, with the possibility of
running in the browser being support being a convenient bonus.

Finally, the focus of gpu.cpp is explicitly compute rather than rendering/graphics, although it might be useful for compute shaders in graphics projects - one of the examples is a small compute renderer, rendered to the terminal.

## Contributing and Work-in-Progress

We welcome contributions! There's a lot of low hanging fruit - fleshing out
examples, adding to the library of useful shader building blocks, filling in
gaps in the library API. Happy to welcome collaborators to make the library
better. 
