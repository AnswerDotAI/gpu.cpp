# gpu.cpp

gpu.cpp is a lightweight library that makes portable GPU compute with C++ simple.

It focuses on general purpose native GPU computation, leveraging the WebGPU
specification as a portable low-level GPU interface. This means we can drop in
GPU code in C++ projects and have it run on Nvidia, Intel, AMD, and other GPUs.
The same C++ code can work on a wide variety of laptops, workstations, mobile
devices or virtually any hardware with Vulkan, Metal, or DirectX support.

## Objectives: Lightweight, Fast Iteration, and Low Boilerplate

With gpu.cpp we want to enable a high-leverage library for individual developers and researchers to incorporate GPU computation into programs relying on nothing more than a standard C++ compiler as tooling. Our goals are:

- High power-to-weight ratio API: Provide the smallest API surface area that can cover the full range of GPU compute needs.
- Fast compile/run cycles: Ensure projects can build nearly instantaneously, compile/run cycles should be <5 seconds on a modern laptop.
- Minimal dependencies and tooling overhead: A standard clang C++ compiler should be enough, no external library dependencies beyond the WebGPU native implementation.

The implementation aims for a small API surface area with minimum boilerplate. There are a small number of library operations to carry out an broad range of low-level GPU operations. We avoid abstractions that add layers of indirection, making the mapping between the gpu.cpp library to raw WebGPU API clear when it's needed.

In this spirit of fast experimentation, we also want near-instantaneous C++ builds taking no more than a second or two even on modestly capable personal computing devices. With this in mind, we not only keep the API surface area small, but also keep the implementation small and we also provide a prebuilt binary of the Dawn native WebGPU implementation.

The core library implementation in the header-only `gpu.hpp` source code is around 1000 lines of code. In addition to enabling instantaneous, semi-interactive compilation cycles, the small implementation surface area keeps maintenance burden low and the velocity of improvements high.
We also pre-build Google's Dawn WebGPU implementation as a shared library binary. This allows builds to link the shared library with each build and incorporate Google's powerful native WebGPU implementation without paying the cost of re-compiling Dawn during development cycles.

For more advanced users and release deployments, we include `cmake` examples for building both Dawn with gpu.cpp end-to-end, but this is not required nor recommended for most users to get started.

## Quick Start: Building and Running

To build a gpu.cpp project, you will need to have installed on your system:

- `clang++` compiler installed with support for C++17.
- `python3` and above, to run the script which downloads the Dawn shared library.
- `make` to build the project.
- Only on Linux systems - Vulkan drivers. If Vulkan is not installed, you can run `sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools` to install them.

The only library dependency of gpu.cpp is a WebGPU implementation. Currently we support the Dawn native backend, but we plan to support other targets and WebGPU implementations (web browsers or other native implementations such as wgpu). Currently we support MacOS, Linux, and Windows (via WSL).

Optionally, Dawn can be built from scratch with gpu.cpp using the cmake build scripts provided - see the -cmake targets in the Makefile. However, this is recommended for advanced users only. Building Dawn dependencies with cmake takes much longer than using the precompiled Dawn shared library.

After cloning the repo, from the top-level gpu.cpp, you should be able to build and run the hello world GELU example by typing:

```
make
```

The first time you build and run the project this way, it will download a prebuilt shared library for the Dawn native WebGPU implementation automatically (using the setup.py script). This places the Dawn shared library in the `third_party/lib` directory. Afterwards you should see `libdawn.dylib` on MacOS or `libdawn.so` on Linux. This download only occurs once.

The build process itself should take a few seconds. If the build and executions is successful, you should see the output of the GELU computation:

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
  ...

Computed 10000 values of GELU(x)
```

If you need to clean up the build artifacts, you can run:

```
make clean
```

## Hello World Tutorial: A GELU Kernel

As a real-world example for how to use gpu.cpp, let's start with a practical-but-simple example of a GPU kernel from neural networks.

GELU is a non-linear embarassingly parallel operation often used in modern large language model transformer-based architectures.

It takes as input a vector of floats and applies the GELU function to each element of the vector. The function is nonlinear, attenuating values below zero to near zero, approximating the y = x identity function for large positive values. For values close to zero, GELU smoothly interpolates between the identity function and the zero function.

The GELU code below will illustrate the three main aspects of setting up a GPU computation with gpu.cpp:

1. The code that runs on the GPU (in WebGPU Shading Language, or WGSL), implementing the compute operation.

2. The code that runs on the CPU (in C++) that sets up the GPU computation by allocating and preparing resources. For high performance, this code should be run ahead-of-time from the hot paths of the application.

3. The code that runs on the CPU (in C++) that dispatches the GPU computation and retrieves the results. The key concern of hot-path dispatch code is to eliminate or minimize any unnecessary resource allocation or data movement (offloading such concerns to step 2). A secondary consideration is that GPU dispatches are asynchronous. We work with standard C++ asynchronous primitives to manage the asynchronous aspect of kernel dispatch.

Here's a GELU kernel implemented (based on the CUDA implementation in [llm.c](https://github.com/karpathy/llm.c)) as on-device WebGPU WGSL code and invoked from the host using gpu.cpp library functions and types. It can be compiled using a standard C++ compiler (we recommend Clang):

```cpp
#include <array>
#include <cstdio>
#include <future>

#include "gpu.hpp"

using namespace gpu; // createContext, createTensor, createKernel,
                     // dispatchKernel, wait, toCPU Bindings,
                     // Tensor, Kernel, Context, Shape, kf32

static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
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
  Kernel op = createKernel(ctx, {kGelu, /* 1-D workgroup size */ 256, kf32},
                           Bindings{input, output},
                           /* number of workgroups */ {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 16; ++i) {
    printf("  gelu(%.2f) = %.2f\n", inputArr[i], outputArr[i]);
  }
  return 0;
}
```

Here we see the GPU code is quoted in a domain specific language called WGSL (WebGPU Shading Language). In a larger project, you might store this code in a separate file to be loaded at runtime (see [examples/shadertui](https://github.com/AnswerDotAI/gpu.cpp/tree/main/examples/shadertui) for a demonstration of live WGSL code re-loading).

The CPU code in main() sets up the host coordination for the GPU computation.
We can think of the use of gpu.cpp library as a collection of GPU nouns and
verbs.

The "nouns" are GPU resources modeled by the type definitions of the library
and the "verbs" actions on GPU resources, modeled by the functions of the
library. The ahead-of-time resource acquisition functions are prefaced with
`create*`, such as:

- `createContext()` - constructs a reference to the GPU device context (`Context`).
- `createTensor()` - acquires a contiguous buffer on the GPU (`Tensor`).
- `createKernel()` - constructs a handle to resources for the GPU computation (`Kernel`), taking the shader code as input and the tensor resources to bind.

These resource acquisition functions are tied to resource types for interacting with the GPU:

- `Context` - a handle to the state of resources for interacting with the GPU device.
- `Tensor` - a buffer of data on the GPU.
- `KernelCode` - the code for a WGSL program that can be dispatched to the
  GPU. This is a thin wrapper around a WGSL string and also includes the
  workgroup size the code is designed to run with.
- `Kernel` - a GPU program that can be dispatched to the GPU. This accepts a
  `KernelCode` and a list of `Tensor` resources to bind for the dispatch
  computation. This takes an argument `Bindings` that is a list of `Tensor` instances and should map the bindings declared at the top of the WGSL code. In this example there's two bindings corresponding to the `input` buffer on the GPU and the `ouptut` buffer on the GPU.

In this example, the GELU computation is performed only once and the program immediately exits so preparing resources and dispatch are side-by-side. Other examples in the [examples/](https://github.com/AnswerDotAI/gpu.cpp/blob/main/examples/) directory illustrate how resource acquisition is prepared ahead of time and dispatch occurs in the hot path like a render, model inference, or simulation loop.

Besides the `create*` resource acquisition functions, there are a few more "verbs" in the gpu.cpp library for handling dispatching execution to the GPU and data movement:

- `dispatchKernel()` - dispatches a `Kernel` to the GPU for computation. This is an asynchronous operation that returns immediately.
- `wait()` - blocks until the GPU computation is complete. This is a standard C++ future/promise pattern.
- `toCPU()` - moves data from the GPU to the CPU. This is a synchronous operation that blocks until the data is copied.
- `toGPU()` - moves data from the CPU to the GPU. This is a synchronous operation that blocks until the data is copied. In this particular example, `toGPU()` is not used because there's only one data movement from CPU to GPU in the program and that happens when the `createTensor()` function is called.

This example is available in [examples/hello_world/run.cpp](https://github.com/AnswerDotAI/gpu.cpp/blob/main/examples/hello_world/run.cpp).

## Other Examples: Matrix Multiplication, Physics Sim, and SDF Rendering

You can explore the example projects in
[examples/](https://github.com/AnswerDotAI/gpu.cpp/blob/main/examples/) which
illustrate how to use gpu.cpp as a library.

After you have run `make` in the top-level directory which retrieves the prebuilt Dawn shared library, you can run each example by navigating to its directory and running `make` from the example's directory.

An example of tiled matrix multiplication is in [examples/matmul](https://github.com/AnswerDotAI/gpu.cpp/blob/main/examples/matmul/). This implements a WebGPU version of the first few kernels of Simon Boehm's [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) post. It currently runs at ~ 3.5+ TFLOPs on a Macbook Pro M1 Max laptop. Contributions to optimize this further are welcome.

A parallel physics simulation of an ensemble of double pendulums simulated in parallel with different initial conditions on the GPU is shown in [examples/physics](https://github.com/AnswerDotAI/gpu.cpp/tree/main/examples/physics).

<div align="center">
<img src="docs/images/matmul.png" alt="matmul example output" width=40%>
<img src="docs/images/pendulum.gif" alt="physics example animated gif" width=42%>
</div>

We also show some examples of signed distance function computations, rendered in the terminal as ascii. A 3D SDF of spheres is shown in [examples/render](https://github.com/AnswerDotAI/gpu.cpp/tree/main/examples/render) and a shadertoy-like live-reloading example is in [examples/shadertui](https://github.com/AnswerDotAI/gpu.cpp/tree/main/examples/shadertui).

<div align="center">
  <img src="docs/images/shadertui.gif" alt="shadertui example animated gif" width=88%>
</div>

## Who is gpu.cpp for?

gpu.cpp is aimed at enabling projects requiring portable on-device GPU computation with minimal implementation complexity and friction. Some example use cases are:

- Development of GPU algorithms to be run on personal computing devices
- Direct standalone implementations of neural network models
- Physics simulations and simulation environments
- Multimodal applications - audio and video processing
- Offline graphics rendering
- ML inference engines and runtimes
- Parallel compute intensive data processing applications

Although gpu.cpp is meant for any general purpose GPU computation and not strictly AI, one area we're interested in is pushing the limits exploring the intersection of new algorithms for post-training and on-device compute.

To date, AI research has primarily been built with CUDA as the privileged first-class target. CUDA has been dominant at large scale training and inference but at the other end of the the spectrum in the world of GPU compute on personal devices, there exists far more heterogeneity in the hardware and software stack.

GPU compute in this personal device ecosystem has been largely limited to a small group of experts such as game engine developers and engineers working directly on ML compilers or inference runtimes. Along with that, implementing against the Vulkan or even WebGPU API directly tends to be targeted mostly towards infrastructure scale efforts - game engines, production ML inference engines, large software packages.

We want to make it easier for a broader range of projects to harness the power of GPUs on personal devices. With a small amount of code, we can access the GPU at a low-level, focusing on directly implementing algorithms rather than the scaffolding and tech stack around the GPU. For example, in our AI research there's much to explore with the various forms of dynamic/conditional post-training computation - dynamic use of adapters, sparsity, model compression, realtime multimodal integrations etc.

gpu.cpp lets us implement and drop-in any algorithm with fine-grained control of data movement and GPU code, and explore outside boundaries of what is supported by existing production-oriented inference runtimes. At the same time we can write code that is portable and immediately usable on a wide variety of and GPU vendors and compute form factors - workstations, laptops, mobile, or even emerging hardware platforms such as AR/VR and robotics.

## What gpu.cpp is not

gpu.cpp is meant for developers with some familiarity with C++ and GPU programming. It is not a high-level numerical computing or machine learning framework or inference engine, though it can be used in support of such implementations.

Second, in spite of the name, WebGPU has native implementations decoupled from the web and the browser. If you find it counterintuitive, watch Elie Michel's excellent talk ["WebGPU is Not Just About the Web"](https://www.youtube.com/watch?v=qHrx41aOTUQ).

Finally, the focus of gpu.cpp is general-purpose GPU computation rather than rendering/graphics on the GPU, although it can be useful for offline rendering or video processing use cases. We may explore directions with graphics in the future, but for now our focus is GPU compute.

## Limitations and Upcoming Features

_Browser Targets_ - In spite of using WebGPU we haven't tested builds targeting the browser yet though this is a short-term priority.

_Reusable Kernel Library_ - Currently the core library is strictly the operations and types for interfacing with the WebGPU API, with some specific use case example WGSL implementations in `examples/`. Over time, as kernel implementations mature we may migrate some of the reusable operations from specific examples into a small reusable kernel library.

## Troubleshooting

If you run into issues building the project, please open an issue.

## Acknowledgements

gpu.cpp makes use of:

- [Dawn](https://dawn.googlesource.com/dawn) as the WebGPU implementation
- [webgpu-dawn-binaries](https://github.com/jspanchu/webgpu-dawn-binaries) by
  @jspanchu to build a binary artifact of Dawn.
- [webgpu-distribution](https://github.com/eliemichel/WebGPU-distribution) by
  @eliemichel for cmake builds.

Thanks also to fellow colleagues at Answer.AI team for their support, testing help, and feedback.

## Discord Community and Contributing

Join our community in the `#gpu-cpp` channel on the [AnswerDotAI Discord with this invite link](https://discord.gg/zmJVhXsC7f). Feel free to get in touch via X [@austinvhuang](https://twitter.com/austinvhuang) as well.

Feedback, issues and pull requests are welcome.

## Code Guidelines for Contributors

For contributors, here are general rules of thumb regarding the design and
style of the gpu.cpp library:

Aesthetics - Maximize Leverage and Account for Sources of Friction:

- In addition to performance, time-to-grok the codebase, compilation time,
  number of failure modes for builds are things worth optimizing for.
- Increase the implementation surface area only when there's a clear goal
  behind doing so. This maximizes leverage per unit effort, increases
  optionality in how the library can be used, and keeps compile times low.
- Taking inspiration from the time-tested horizontal extensibility
  of neural network libraries like PyTorch, to a first approximation the library
  architecture could be described as a bag of composable functions.
- Design choices general attempt to blend the composability of functional
  programming with the performance awareness of data oriented design.

Overloads and Templates:

- Prefer value-level types over type-level templates, especially for core
  implementation code. It's easy to add a more typesafe templated wrapper
  around a value type core implementation. Whereas moving templated core
  implementations from comptime to runtime leads to a more significant
  refactor.
- For comptime polymorphism, prefer trivial function overloads over templates.
  Besides compile time benefits, this reasoning about which version of a
  function is being called becomes explicit and scanable in the codebase.

Avoid Encapsulation and Methods:

- To build systems effectively, we need to construct them out of subsystems for
  which the behavior is known and thereby composable and predictable.
  Therefore, we prefer transparency and avoid encapsulation. Don't use abstract
  classes as interface specifications, the library and its function signatures
  is the interface.
- Use struct as a default over class unless there's a clear reason otherwise.
- Instead of methods, pass the "owning object" object as a reference to a
  function. In general this convention can perform any operation that a method
  can, but with more flexibility and less coupling. Using mutating functions
  generalizes more cleanly to operations that have side effects on more than
  one parameter, whereas methods priveledge the the owning class, treating the
  single variable case as a special case and making it harder to generalize to
  multiple parameters.
- Methods are usually only used for constructor/destructor/operator priveledged
  cases.
- For operations requesting GPU resources and more complex initialization, use
  factory functions following the `create[X]` convention - createTensor,
  createKernel, createContext etc.
- Use (as-trivial-as-possible) constructors for simple supporting types (mostly
  providing metadata for a dispatch) Shape, KernelCode, etc.

Ownership:

- Prefer stack allocation for ownership, use unique_ptr for ownership when the
  heap is needed. Use raw pointers only for non-owning views. Avoid shared_ptr
  unless there's a clear rationale for shared ownership.
- Use pools as a single point of control to manage sets of resources. Consider
  incorporating a pool in Context if the resource is universal enough to the
  overall API.

Separating Resource Acquisition from Hot Paths:

- In general, resource acquisition should be done ahead of time from the hot
  paths of the application. This is to ensure that the hot paths are as fast as
  possible and don't have to deal with resource allocation or data movement.
- Operations in the API should be implemented with a use in mind - typically
  either ahead-of-time resource preparation/acquisition, hot-paths, or
  non-critical testing/observability code.
