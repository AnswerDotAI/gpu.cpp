# gpu.cpp

gpu.cpp is a lightweight C++ library to write portable low-level GPU code that
runs everywhere.

*Work-in-Progress*

## Who is gpu.cpp for?

For exploring the space of GPU algorithms with the ability to portably just
work on a wide range of hardware, without any intermediaries of model export,
compilation, or runtime support, gpu.cpp is a lightweight library that makes it
simple to write low-level GPU code that runs on any device with almost any GPU.

gpu.cpp is implemented using the WebGPU API specification, which is designed
for cross-platform GPU interactions. In spite of the name, WebGPU has native
(Dawn and wgpu) implementations decoupled from the web and the browser (
for additional background - see [WebGPU is
Not Just about the Web](https://www.youtube.com/watch?v=qHrx41aOTUQ))

By leveraging the WebGPU API specification as simply a portable interface to
any GPU supported by native implementations that conform to major GPU
interfaces like Metal, DirectX, and Vulkan. This means we can drop-in simple,
low-level GPU code in our C++ projects and have it run on Nvidia, Intel, AMD
GPUs, and even on Apple and Android mobile devices.

gpu.cpp provides a small but powerful set of core functions and types that make
WebGPU compute simple and concise to work with R&D and application use cases.
It keeps abstractions minimal and transparent. It has no dependencies other
than the WebGPU API implementation itself (Dawn for native).

## Hello World: A GELU Kernel

Here's an GELU kernel implemented (based on the CUDA implementation of
[llm.c](https://github.com/karpathy/llm.c)) as an on-device WGSL shader and
invoked from the host using this library.

```
#include "gpu.h"
#include <array>
#include <cstdio>
#include <future>

using namespace gpu; // CreateContext, CreateTensor, CreateKernel,
                     // CreateShader, DispatchKernel, Wait, ToCPU
                     // Tensor, Kernel, Context, Shape, kf32

static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dummy: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        // select is more stable than tanh for large x
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
               * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

int main(int argc, char **argv) {
  printf("\nHello, gpu.cpp\n\n");
  Context ctx = CreateContext();
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i) / 2.0; // dummy input data
  }
  Tensor input = CreateTensor(ctx, Shape{N}, kf32, inputArr.data());
  Tensor output = CreateTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = CreateKernel(ctx, CreateShader(kGelu, 256, kf32), TensorList{input, output},
                           /* nthreads */ {N, 1, 1});
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 32; ++i) {
    printf("out[%d] : gelu(%.2f) = %.2f\n", i, inputArr[i], outputArr[i]);
  }
  printf("...\n\n");
  return 0;
}
```

This example is available in `examples/hello_world/run.cpp`. 

For those curious about what happens under the hood with the raw WebGPU API,
the equivalent functionality is implemented using the WebGPU C API in
`examples/webgpu_intro/run.cpp`.

## Quick Start: Dependencies and Installation

The only dependency of this library is a WebGPU implementation. Currently we
recommend using the Dawn backend until further testing, but we plan to support
emscripten (web) and wgpu (native) backends.

You should have clang and cmake installed (we currently test on 3.25+). On mac,
you can install cmake using [homebrew](https://brew.sh/) with: `brew install
cmake`. On Ubuntu, you can install cmake using `apt-get` with: `sudo apt-get
install cmake`.

## Quick Start: Building and Running

The build is handled by cmake. Some useful common cmake invocations are wrapped
in the convenience Makefile. 

The first time you build and run the project, it will download the WebGPU
backend implementation (Dawn by default) and build it which may take a few
minutes. The gpu.cpp library itself is small so after building the Dawn backend
the first time, subsequent builds of the library should take seconds on most
personal computing devices.

*Using gpu.cpp as a Library*

You can build the library itself which builds a shared library that you can
link against for your own projects. This builds a library that can be used in
other C++ projects (most of the code is in `gpu.h`, plus some supporting code
in `utils/`).

```
make libgpu
```

If you are starting a new project using gpu.cpp as a library dependency, we
recommend starting by cloning the template project
[https://github.com/AnswerDotAI/gpu.cpp-template](https://github.com/AnswerDotAI/gpu.cpp-template).

*Example Demos in `examples/`*

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

*Machine Learning Kernel Implementations (WIP)*

A more extensive set of (machine learning-centric) kernels is implemented in
`utils/test_kernels.cpp`. This can be built and run (from the top level
directory) using:

```
make tests
```

For more configurability and control of the build, see the `cmake`  invocations
in the `Makefile`,  as well as the configuration in `Cmakelists.txt`.

*Resetting Build State / Removing Build Artifacts*

If you need to clean up the build artifacts, you can run:

```
make clean
```

## Troubleshooting

If you run into issues building the project, please open an issue.

## Motivation and Goals

Although gpu.cpp is intended for any form of general purpose GPU computation,
the project is partly motivated by emerging needs in machine learning R&D. 

Specifically, with large foundation models, a significant amount of R&D now
occurs in the post-training computation. Large foundation models have become
become computable objects over which custom algorithms are implemented.
Performing custom computations over compute-intensive foundation models
benefits from low-level control of the GPU. 

At this time, tooling for implementing low-level GPU computation is heavily
focused on CUDA as a first class citizen. This leaves a gap in portability,
meaning R&D algorithms that work in a research environment are difficult to
operationalize for everyday use to run on personal computing hardware that's
broadly accessible (personal workstations, laptops, mobile devices).

We created gpu.cpp as a lightweight C++ library that allows us to easily and
directly implement native low-level GPU algorithms as part of R&D and drop
implementations into code running on personal computing devices either as
native applications or in the browser without impediment by hardware, tooling,
or runtime support.

## What gpu.cpp is for

(TODO(avh))

## What gpu.cpp is not

gpu.cpp is meant for developers with basic familiarity with C++ and GPU
programming and not intended to be used by end users directly. Although there
is basic low level support for ND-arrays and a small library implementing
neural network blocks as shaders, it is not a high-level machine learning
framework or inference engine (although it can be useful in implementing one).

gpu.cpp is also not strictly for the web or deploying to the web - WebGPU is
convenient API with both both native (e.g. Dawn) and browser implementations.
It uses WebGPU as a portable GPU API first and foremost, with the possibility
of running in the browser being support being a convenient bonus.

For additional background on WebGPU as a portable native GPU API, see Elie
Michel's talk [WebGPU is Not Just about the
Web](https://www.youtube.com/watch?v=qHrx41aOTUQ).

Finally, the focus of gpu.cpp is general-purpose GPU computation rather than
rendering/graphics on the GPU, although it might be useful for compute shaders
in graphics projects - one of the examples is a small compute renderer,
rendered to the terminal.

## Limitations

(TODO(avh))

## Contributing and Work-in-Progress

We welcome contributions! There's a lot of low hanging fruit - fleshing out
examples, adding to the library of useful shader building blocks, filling in
gaps in the library API. Happy to welcome collaborators to make the library
better. 
