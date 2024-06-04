#include "gpu.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>

using namespace gpu;

static const char *kAsciiBanner = R"(
   ____ _____  __  __ _________  ____ 
  / __ `/ __ \/ / / // ___/ __ \/ __ \
 / /_/ / /_/ / /_/ // /__/ /_/ / /_/ /
 \__, / .___/\__,_(_)___/ .___/ .___/ 
/____/_/               /_/   /_/
)";

void wait() {
  fprintf(stdout, "┌──────────────────────────────┐\n");
  fprintf(stdout, "│ Press Enter to Continue...   │\n");
  fprintf(stdout, "└──────────────────────────────┘\n");
  getchar();
}

void section(const char *content) {
  fprintf(stdout, "\033[2J\033[1;1H"); // clear screen
  fprintf(stdout, "%s\n", kAsciiBanner);
  fprintf(stdout, "================================================================================\n");
  fprintf(stdout, "%s\n", content);
  wait();
  // fprintf(stdout, "\033[4A\033[0J"); // clear lines
}

int main(int argc, char **argv) {

  // Clear screen and print banner
  fprintf(stdout, "\033[2J\033[1;1H");

  // Creating a GPUContext

  section(R"(
Welcome!
--------

This program is a brief intro to the gpu.cpp library.

You can use the library by simply including the gpu.h header, starting with a
build template (see examples/hello_gpu/ for a template project that builds the
library).

  #include "gpu.h"

See `examples/hello_world/` for an examle of build scripts to run a standalone
program that uses this library.
)");


  section(R"(
Nouns and Verbs of gpu.cpp
--------------------------

We can think of gpu.cpp in terms of its "nouns" (types or resources) and
"verbs" (functions). 

The core nouns (resources / types) are:

- *Device State* - Interacting with the GPU state - `GPUContext` and supporting types.
  Once instantiated, the GPUContext instance is passed to most functions to provide 
  references to interact with the GPU.
- *Data* - Data that you want to pass to/from the GPU for the
  computation. These are effectively flat buffers of values (GPUArray),
  optionally with an associated shape (GPUTensor).
- *Computation* - that you want to execute on the GPU - a Kernel instance comprised of a
  Shader and references to its associated data.

The core verbs (functions) of interest are:

- *Requesting GPU Resources* - CreateGPUContext(), CreateArray() and
  CreateTensor() 
- *Ahead-of-Time Preparation of a Computation* - PrepareKernel() which both binds
  resources and compiles the kernel 
- *Asynchronous Execution of Computation* - LaunchKernel(), Wait()
- *Data Movement* - ToCPU(), ToGPU(), also CreateArray and CreateTensor have
  convenience overloads that take CPU data directly as part of instantiation.

Each of these has some supporting functions and types which we can get to
later.
)");

  section(R"(
Interfacing with the GPU  
-------------------------

The GPUContext is the main entry point for interacting with the GPU. It
represents the state of the GPU and is used to allocate resources and execute
kernels.

In your program, you can create a GPUContext like this:

  GPUContext ctx = gpu::GPUContext();

Let's try doing that in this program now.
)");

  GPUContext ctx = CreateGPUContext();
  fprintf(stdout, "\nSuccessfully created a GPUContext.\n\n");
  wait();

  section(R"(
Creating Data on the GPU
-------------------------

As a low-level library, gpu.cpp primarily deals with flat arrays of data either
on the CPU or GPU. 

The main data structure is the GPUArray which represents a flat buffer of
values on the GPU. GPUTensor is a thin wrapper around GPUArray that adds shape
metadata.

In most applications, you may prepare arrays or allocated
chunks on the CPU (eg for model weights or input data), and then 

  std::array<float, N> inputArr;
  std::array<float, N> outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  GPUTensor input = Tensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = Tensor(ctx, {N}, kf32, outputArr.data());

Let's try creating some data on the GPU now.

)");

std::array<float, 3072> inputArr;
std::array<float, 3072> outputArr;
for (int i = 0; i < 3072; ++i) {
  inputArr[i] = static_cast<float>(i); // dummy input data
}
GPUTensor input = Tensor(ctx, {3072}, kf32, inputArr.data());
GPUTensor output = Tensor(ctx, {3072}, kf32, outputArr.data());

fprintf(stdout, "\nSuccessfully created input and output tensors.\n\n");
wait();


section(R"(
Custom WGSL Compute Kernels
---------------------------

Device code in WebGPU uses the WGSL shading language. In addition to mechanisms
for invoking WGSL shaders as compute kernels as shown so far, you can write
your own WGSL shaders and use the same mechanisms to invoke them.

Here is an example of a custom WGSL shader that implements the GELU activation:

```
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    // Ensure we do not access out of bounds
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + cube)));
    }
}
```

If you are familiar with CUDA, this is pretty similar to the code you would
find in a CUDA kernel. Like a CUDA kernel, there are invocation ids that are
passed in.

The `@group(0)` and `@binding(0)` annotations are used to specify the binding
points for the input and output buffers. The `@compute` annotation specifies
that this is a compute kernel. The `@workgroup_size(256)` annotation specifies
the workgroup size for the kernel.

Workgroups are a concept in WebGPU that are similar to CUDA blocks. They are
groups of threads that can share memory and synchronize with each other. The
workgroup size is the number of threads in a workgroup.

)");

section(R"(
Preparing a kernel
------------------

TODO(avh)
)");


section(R"(
Launching a kernel
------------------

TODO(avh)
)");


  section(R"(
gpu.cpp vs. the raw WebGPU API
------------------------------

The main responsibility of the types and functions of the library is to make
it trivial to represent these common building blocks of computation

If you look at `examples/webgpu_intro/run.cpp` you can get a sense of what it's
like to interact directly with the WebGPU.
)");

  fprintf(stdout, "Goodbye!\n");
}
