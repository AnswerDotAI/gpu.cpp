# gpu.cpp

gpu.cpp is a lightweight library incorporating portable, low-level GPU
computations directly into the C++ code of applications and R&D projects.

It uses WebGPU as a portable native GPU API, and provides a small, simple set
of functions and types for portable GPU computation.

*** This project is a work-in-progress *** for now we recommend use to be
limited to contributing developers and early adopters.

## Hello World: A GELU Kernel

Although there are many uses for general purpose GPU computing beyond AI, a
simple example of a GELU kernel used in neural networks is nevertheless a good
starting point for understanding how to use gpu.cpp.

GELU is an activation function in neural networks often used in modern large
language model transformer architectures. It takes as input a vector of floats
and applies the GELU function to each element of the vector. The function is
nonlinear, attenuating values below zero to near zero, approximating the y = x
identity function for largepositive values. For values near zero, smoothly
interpolates between the identity function and the zero function.

To implement this, we can think of the code in three parts:

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

using namespace gpu; // createContext, createTensor, createKernel,
                     // createShader, dispatchKernel, wait, toCPU
                     // Bindings, Tensor, Kernel, Context, Shape, kf32

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
  printf("\nHello gpu.cpp!\n\n");
  Context ctx = createContext();
  static constexpr size_t N = 10000;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i) / 10.0; // dummy input data
  }
  Tensor input = createTensor(ctx, Shape{N}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, createShader(kGelu, 256, kf32),
                           Bindings{input, output},
                           /* nWorkgroups */ {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 16; ++i) {
    printf("  gelu(%.2f) = %.2f\n", inputArr[i], outputArr[i]);
  }
  printf("... \n\n  Computed %zu values of GELU(x)\n\n", N);
  return 0;
}
```

As shown above, the GPU code is quoted in a domain specific language called
WGSL (WebGPU Shading Language). In a larger project, you would store this code
in a separate file to be loaded at runtime. The WGSL code is compiled and runs
on the GPU.

The CPU code in `main()` sets up the host coordination for the GPU computation.
The ahead-of-time resource acquisition functions are prefaced with `create`,
such as `createContext`, `createTensor`, `createKernel`, `createShader`. 

The dispatch occurs asynchronously via the `dispatchKernel` invocation. `wait`
blocks until the GPU computation is complete and `toCPU` moves data from the
GPU to CPU.  This example is available in `examples/hello_world/run.cpp`. 

## Quick Start

To build gpu.cpp, you will need to have installed on your system:

- `clang++` compiler installed with support for C++17.
- `python3` and above, to run the script which downloads the Dawn shared library.
- `make` to build the project.
- *Only on Linux systems* - Vulkan drivers. If Vulkan is not installed, you can
  run `sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools` to install
  them.

The only library dependency of gpu.cpp is a WebGPU implementation. Currently we
support the Dawn native backend, but we plan to support other targets and
WebGPU implementations (web browsers or other native implementations such as
wgpu). Currently we support MacOS, Linux, and Windows (via WSL).

Optionally, Dawn can be built from scratch with gpu.cpp using the cmake build
scripts provided - see the `-cmake` targets in the Makefile. However, this is
only recommended for advanced users. cmake builds take much longer than using
the provided precompiled Dawn shared library binary as it compiles the entire
WebGPU C API implementation from scratch.

### Building and Running

After cloning the repo, from the top-level gpu.cpp, you should be able to build
and run the hello world GELU example by typing:

`make`

The first time you build and run the project this way, it will download a
prebuilt shared library for the Dawn native WebGPU implementation automatically
(using the `setup.py` script). This places the Dawn shared library in the
`third_party/lib` directory. Afterwards you should see `libdawn.dylib` on MacOS
or `libdawn.so` on Linux. This download only occurs the first time.

The build process itself should take a second or two - the core library in
`gpu.h` are kept small by design to make compilation iterations fast.

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

The binary is built in the `examples/hello_world/build/` directory as
`hello_world`. One detail if you want to run it separate from the `make` build
process - be sure to run `. source` from the top-level directory (or add the
content of `source` in your `.bashrc`). This makes the path to the Dawn shared
library build available to the binary.

From here you can explore the example projects in `examples/` which illustrate
how to use gpu.cpp as a library. 

If you need to clean up the build artifacts, you can run:

```
make clean
```

## Troubleshooting

If you run into issues building the project, please open an issue.

## Who is gpu.cpp for?

gpu.cpp is a lightweight library that makes it simple to write low-level GPU
code that runs on any device with almost any GPU. You can simply implement and
integrate low-level GPU algorithms in an application with the ability to
portably just work on a wide range of hardware, without intermediaries of model
exporting, compilation, or runtime support. 

Although gpu.cpp can be applied to any GPU compute use cases beyond AI, we also
hope it helps to explore the potential for personal and local AI. 

There is already a deep ecosystem of technologies supporting large-scale
datacenter GPU compute beginning with low level CUDA on top of which there’s a
stack of compilers and frameworks. By contrast, when we think of developing low
level GPU compute on personal devices, its use has been mostly the realm of
large efforts such as ML compiler and runtime implementations. 

We created gpu.cpp as a lightweight C++ library that allows us to easily and
directly implement native low-level GPU algorithms as part of R&D and drop
implementations into portable code running on personal computing devices.

gpu.cpp is implemented using the WebGPU API specification, which is designed
for cross-platform GPU interactions. In spite of the name, WebGPU has native
(Dawn and wgpu) implementations decoupled from the web and the browser. For
additional background - see [WebGPU is Not Just about the
Web](https://www.youtube.com/watch?v=qHrx41aOTUQ))

By leveraging the WebGPU API specification as simply a portable interface to
any GPU supported by native implementations that conform to major GPU
interfaces like Metal, DirectX, and Vulkan. This means we can incorporate
simple, low-level GPU code in our C++ projects and have it run on Nvidia,
Intel, AMD GPUs, and even Apple and Android mobile devices.

gpu.cpp can be used for projects targeting portable GPU compute directly
integrated into your project. Some examples (but not limited to) include:

- Direct code implementations of neural network architectures.
- R&D for low-level GPU algorithms to be run on personal devices.
- Parallel compute-intensive physics simulations and physics engines.
- GPU compution for applications - audio and video digital signal processing,
  custom game engines etc.
- Offline rendering.
- ML inference engines and runtimes.

gpu.cpp provides a small but powerful set of core functions and types that make
WebGPU compute simple and concise to work with R&D and application use cases.
It keeps abstractions minimal and transparent. It has no dependencies other
than an implementation of the WebGPU API (Google's Dawn in the case of native
builds).

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
