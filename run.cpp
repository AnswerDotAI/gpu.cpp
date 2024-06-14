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

void runHelloGELU(GPUContext& ctx) {
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

  static constexpr size_t N = 3072;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  GPUTensor input = CreateTensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  Kernel op =
      CreateKernel(ctx, ShaderCode{kGELU, 256}, input, output);
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 10; ++i) {
    fprintf(stdout, "%d : %f\n", i, outputArr[i]);
  }
  fprintf(stdout, "...\n\n");
  wait();
}

int main(int argc, char **argv) {

  // Clear screen and print banner
  fprintf(stdout, "\033[2J\033[1;1H");

  // Creating a GPUContext

  section(R"(
Welcome!
--------

This program is a brief intro to the gpu.cpp library.

You can use the library by simply including the gpu.h header:

  #include "gpu.h"

and starting with a build template (see examples/hello_gpu/ for a template
project that builds the library).
)");

  section(R"(
Before diving into the details of the library, let's test out some code to
perform a simple GPU computation - a GELU activation function. These activation
functions are common in deep learning large language models.

The code is broken into two parts:

*The code that runs on the GPU*

.. is written in WGSL the WebGPU Shading Language. WGSL is a domain specific
language for writing GPU compute kernels approximately maps to the computations
available on the GPU. If you are familiar with CUDA, this is similar to writing
a CUDA kernel.

*The code that runs on the host (CPU)*

.. is written in C++ and uses the gpu.cpp which invokes the WebGPU C API.

We'll see a WGSL example later, for now let's see the host CPU C++ code that
uses the gpu.cpp to run the GELU activation function.
)");

  section(R"(
Here is the host CPU C++ code that uses the gpu.cpp to run the GELU activation
function:

```
#include <array>
#include <cstdio>
#include "gpu.h"

using namespace gpu;

// Device code (runs on the GPU) using WGSL (WebGPU Shading Language)
const char *kGELU = ... // we'll look at the WGSL shader code later

int main(int argc, char **argv) {
  GPUContext ctx = CreateContext();
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  GPUTensor input = CreateTensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  Kernel op =
      CreateKernel(ctx, ShaderCode{kGELU, 256}, input, output);
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 10; ++i) {
    fprintf(stdout, "%d : %f\n", i, outputArr[i]);
  }
  fprintf(stdout, "...\n\n");
  return 0;
}
```

Let's try running this.
)");

  GPUContext ctx = CreateContext();
  runHelloGELU(ctx);


  section(R"(
Design Objectives of gpu.cpp
----------------------------

1. Maximal Leverage. Maximize the space of implementations that this
   library is useful for with the least amount of implementation complexity.
   Implementation complexity. 

2. Minimize integration complexity. Whereas the integration pattern for custom
   low-level GPU algorithm code is to integrate it into an existing engine (eg
   an inference runtime, or a compiler), the objective of gpu.cpp is to enable
   adding GPU computation code inside your own project with a minimal amount of
   integration complexity.

2. High ceiling on low-level control.
    - Direct control of on-device GPU code unconstrained by fixed set of ops
    - Direct control of on-device GPU memory management
)");

  section(R"(
Separating Resource Acquisition and Dispatch
--------------------------------------------

We can think of the use of gpu.cpp library as a collection of GPU nouns and
verbs.

The "nouns" are GPU resources modeled by the type definitions of the library
and the "verbs" actions on GPU resources, modeled by the functions of the
library. 

The key functions can be further subdivided into two categories in relation to
when the GPU computation occurs: 

1) Ahead-of-time GPU Resource Preparation: these are functions that
   acquire resources and prepare state for GPU computation. These are less
   performance critical.

2) Performance critical dispatch of GPU computation: these are functions that
   dispatch GPU computation to the GPU, usually in a tight hot-path loop. 

)");

  section(R"(


*Ahead-of-time GPU Resource Preparation*


)");

  section(R"(
Preparing GPU Resources I: Resource Type Definitions
----------------------------------------------------

The main resources are:

- `GPUContext` - the state of resources for interacting with the GPU.
- `GPUTensor` - a buffer of data on the GPU.
- `ShaderCode` - the code for a shader program that can be dispatched to the
  GPU. This is a thin wrapper around a WGSL string but also includes the
  workgroup size the code is designed to run with.
- `Kernel` - a GPU program that can be dispatched to the GPU. This accepts a
  `ShaderCode` and a list of `GPUTensor` resources to bind for the dispatch
  computation.
- `MultiKernel` - a collection of kernels that can be dispatched to the GPU.

)");

  section(R"(
Preparing GPU Resources II: Acquiring GPU Resources with `Create*()` Functions
------------------------------------------------------------------------------

Resources are acquired using the `Create` functions. These are assumed to be
ahead-of-time and not performance critical.

- `GPUContext CreateContext(...)` - creates a GPU context.
- `GPUTensor CreateTensor(...)` - creates and allocates a buffer for a tensor
  on the GPU.
- `Kernel CreateKernel(...)` - creates and prepares a kernel on the GPU,
  including underlying GPU buffer data bindings and compute pipeline for the
  shader code.
- `MultiKernel CreateMultiKernel(...)` - Same as `CreateKernel`, but for
  multiple kernels to be dispatched together.

There's a few supporting types in addition to these. `Shape` is a simple type
to specify the shape of a tensor. `KernelDesc` and `MultiKernelDesc` are
effectively. `TensorPool` manages `GPUTensor` resources and is used as context
for allocating and deallocating tensors data on the GPU. In practice
`TensorPool` is managed as a member variable of `GPUContext`.

)");


  section(R"(
`CreateContext()` creates a GPUContext
--------------------------------------

Let's zoom in a bit on the invocation of these Create functions, starting with
CreateContext:

The GPUContext is the main entry point for interacting with the GPU. It
represents the state of the GPU and is used to allocate resources and execute
kernels.

In your program, you can create a GPUContext like this:

  GPUContext ctx = CreateContext();
)");

  section(R"(
`CreateTensor()` allocates GPUTensor on the GPU
-----------------------------------------------

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
  GPUTensor input = CreateTensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());

Let's try creating some data on the GPU now.

)");

std::array<float, 3072> inputArr;
std::array<float, 3072> outputArr;
for (int i = 0; i < 3072; ++i) {
  inputArr[i] = static_cast<float>(i); // dummy input data
}
GPUTensor input = CreateTensor(ctx, {3072}, kf32, inputArr.data());
GPUTensor output = CreateTensor(ctx, {3072}, kf32, outputArr.data());

fprintf(stdout, "\nSuccessfully created input and output tensors.\n\n");
wait();


section(R"(
WGSL Compute Kernels are Programs that run Computation on the GPU
------------------------------------------------------------------

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
)");

section(R"(
`CreateKernel()` is used to create a Kernel
-------------------------------------------

Reviewing our GELU example and after using `CreateTensor()` to allocate and
bind buffers for input and output data, we can use `CreateKernel()` to create a
kernel.

```
  GPUTensor input = CreateTensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  Kernel op =
      CreateKernel(ctx, ShaderCode{kGELU, 256}, input, output);
```

Note this *does not run* the kernel, it just prepares the kernel as a resource
to be dispatched later.

There are four arguments to `CreateKernel()`:
- `GPUContext` - the context for the GPU
- `ShaderCode` - the shader code for the kernel
- `GPUTensor` - the input tensor. Even though the kernel is not executed,
GPUTensor provides a handle to the buffers on the GPU to be loaded when the
kernel is run. If there's more than one input, `GPUTensors` can be used which
is an ordered collection of `GPUTensor`.
- `GPUTensor` - the output tensor. As with the input tensor, the values are not
important at this point, the underlying reference to the GPU buffer is bound to
the kernel so that when the kernel is dispatched, it will know where to write
the output data.

)");


section(R"(
Dispatching a kernel
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
  return 0;
}
