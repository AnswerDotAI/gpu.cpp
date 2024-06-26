# gpu.cpp

gpu.cpp is a lightweight C++ library to write portable low-level GPU code that
runs everywhere.

*Work-in-Progress*

## Who is gpu.cpp for?

gpu.cpp makes it simple to implement and integrate low-level GPU algorithms in
your code with the ability to portably just work on a wide range of hardware,
without any intermediaries of model exporting, compilation, or runtime support.
gpu.cpp is a lightweight library that makes it simple to write low-level GPU
code that runs on any device with almost any GPU.

While AI research today focuses on the frontier of datacenters, there is
immense potential for highly capable generalist models to be embedded in
computers that are personal and local.

There’s a deep stack of technologies supporting large-scale datacenter GPU
compute beginning with low level CUDA on top of which there’s a stack of
compilers and frameworks. By contrast, when we think of developing low level
GPU compute on personal devices, it’s been largely relegated to a small group
of game engine developers and machine learning compiler and inference runtime
experts. 

We created gpu.cpp as a lightweight C++ library that allows us to easily and
directly implement native low-level GPU algorithms as part of R&D and drop
implementations into code running on personal computing devices either as
native applications or in the browser without impediment by hardware, tooling,
or runtime support.

gpu.cpp is implemented using the WebGPU API specification, which is designed
for cross-platform GPU interactions. In spite of the name, WebGPU has native
(Dawn and wgpu) implementations decoupled from the web and the browser. For
additional background - see [WebGPU is Not Just about the
Web](https://www.youtube.com/watch?v=qHrx41aOTUQ))

By leveraging the WebGPU API specification as simply a portable interface to
any GPU supported by native implementations that conform to major GPU
interfaces like Metal, DirectX, and Vulkan. This means we can drop-in simple,
low-level GPU code in our C++ projects and have it run on Nvidia, Intel, AMD
GPUs, and even on Apple and Android mobile devices.

gpu.cpp can be used for projects where you want portable easily-integrated GPU
compute directly integrated into your system implementation. Some examples (but
not limited to) include:

- R&D for low-level GPU algorithms to be run on personal devices
- GPU compution for applications - audio and video digital signal processing,
  custom game engines etc.
- Direct fine-grained implementations of neural network architectures.
- Offline rendering.
- ML inference engines and runtimes.
- Parallel compute-intensive physics simulations.

gpu.cpp provides a small but powerful set of core functions and types that make
WebGPU compute simple and concise to work with R&D and application use cases.
It keeps abstractions minimal and transparent. It has no dependencies other
than an implementation of the WebGPU API (Google's Dawn in the case of native
builds).

## Hello World: A GELU Kernel

While there are many general purpose GPU computing beyond machine learning, a
simple example of a GELU kernel is a good starting point for understanding how
to use gpu.cpp.

GELU is an activation function in neural networks often used in mdoern large
language model transformer architectures. It takes as input a vector of floats
and applies the GELU function to each element of the vector. The function is
nonlinear, attenuating values below zero to near zero, approximating the y = x
identity function for largepositive values. For values near zero, smoothly
interpolates between the identity function and the zero function.

To implement the GELU function, we can think of the code in three parts:

1. The code that runs on the GPU that implements the mathematical opporation.
2. The code that runs on the CPU that sets up the GPU computation by allocating
   and preparing resources. For high performance, this code should be run
   ahead-of-time from the hot paths of the application.
3. The code that runs on the CPU that dispatches the GPU computation and
   retrieves the results. This code should be run from the hot paths of the
   application.

In this example, the GELU computation is only performed once so (2) and (3)
occur sequentially, but other examples show examples of how resource
acquisition is prepared ahead of time and dispatch occurs in the hot path.

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
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                 * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

int main(int argc, char **argv) {
  printf("\033[2J\033[1;1H");
  printf("\nHello gpu.cpp!\n");
  printf("--------------\n\n");

  Context ctx = CreateContext();
  static constexpr size_t N = 10000;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i) / 10.0; // dummy input data
  }
  Tensor input = CreateTensor(ctx, Shape{N}, kf32, inputArr.data());
  Tensor output = CreateTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = CreateKernel(ctx, CreateShader(kGelu, 256, kf32),
                           TensorList{input, output},
                           /* nthreads */ {N, 1, 1});
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 16; ++i) {
    printf("  gelu(%.2f) = %.2f\n", inputArr[i], outputArr[i]);
  }
  printf("  ...\n\n");
  printf("Computed %zu values of GELU(x)\n\n", N);
  return 0;
}
```

As shown above, the GPU code is quoted in a domain specific language called
WGSL (WebGPU Shading Language). The code is compiled and runs on the GPU.

The CPU code in `main()` sets up the GPU computation, dispatches it
asynchronously, blocks on the result with `Wait()`, and then retrieves the
result for display with `ToCPU()`. In performance critical code, you try to
keep as much of the computation on the GPU as possible and minimize the amount
of data transfer between the CPU and GPU. 

This example is available in `examples/hello_world/run.cpp`. 

## Dependencies

To build gpu.cpp, you only need:

- `clang++` compiler installed with support for C++17.
- python 3+ (to run the script which downloads the Dawn shared library) 
- `make` to build the project.

The only dependency of this library is a WebGPU implementation. Currently we
recommend using the Dawn native backend until further testing, but we plan to
support other targets and WebGPU implementations (eg the web, or possibly wgpu
as an alternative native backend to Dawn). Currently we also only support Linux
and MacOS although Windows support via WGSL is planned.

Optionally, Dawn can be built from scratch using a cmake build option, but this
is only recommended for advanced users. cmake builds take much longer than
using the provided precompiled Dawn shared library binary as it compiles the
entire WebGPU C API implementation from scratch.

## Quick Start: Building and Running

After cloning the repo, from the top-level gpu.cpp, you should be able to build
and run the hello world GELU example by typing:

`make`

The first time you build and run the project this way, it will download a
prebuilt shared library for the Dawn native WebGPU implementation automatically
(using the `setup.py` script). This places the Dawn shared library in the
`third_party/lib` directory. Afterwards you should see `libdawn.dylib` on MacOS
or `libdawn.so` on Linux. This download only occurs the first time, after which
the shared library is reused for builds.

The build process itself should take a second or two - both the implementation
code in `examples/hello_world/run.cpp` and the core library in `gpu.h` is very
small to make compilation iterations fast.

If the build and executions is successful, you should see the output of the
GELU computation:

```
Hello gpu.cpp!
--------------

  gelu(0.00) = 0.00
  gelu(0.10) = 0.05
  gelu(0.20) = 0.12
  gelu(0.30) = 0.19
  gelu(0.40) = 0.26
  gelu(0.50) = 0.35
  gelu(0.60) = 0.44
  gelu(0.70) = 0.53
  gelu(0.80) = 0.63
  gelu(0.90) = 0.73
  gelu(1.00) = 0.84
  gelu(1.10) = 0.95
  gelu(1.20) = 1.06
  gelu(1.30) = 1.17
  gelu(1.40) = 1.29
  gelu(1.50) = 1.40
  ...

Computed 10000 values of GELU(x)
```

From here you can explore the example projects in `examples/` which illustrate
how to use gpu.cpp as a library. 

If you need to clean up the build artifacts, you can run:

```
make clean
```

## Troubleshooting

If you run into issues building the project, please open an issue.

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

## Limitations and Upcoming Features

gpu.cpp is a work-in-progress and there are many features and improvements to
come. At this early stage, we expect the API design to evolve as we identify
improvements / needs from use cases.

In spite of using WebGPU we haven't tested builds/targeting the browser yet
though that is planned. 

## Acknowledgements

We use:
- [Dawn](https://dawn.googlesource.com/dawn) as the WebGPU implementation
- [webgpu-dawn-binaries](https://github.com/jspanchu/webgpu-dawn-binaries) by
  @jspanchu to build a binary artifact of Dawn.
- [webgpu-distribution](https://github.com/eliemichel/WebGPU-distribution) by
  @eliemichel for cmake builds.

## Contributing and Work-in-Progress

We welcome contributions! There's a lot of low hanging fruit - fleshing out
examples, adding to the library of useful shader building blocks, filling in
gaps in the library API. Happy to welcome collaborators to make the library
better. 
