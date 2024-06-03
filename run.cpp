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
  fprintf(stdout, "────────────────────────────────────────────────────────────"
                  "───────────────────\n");
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
)");

  section(R"(
Nouns and Verbs of gpu.cpp
--------------------------

We can think of gpu.cpp in terms of its "nouns" (types or resources) and
"verbs" (functions). 

The core nouns (resources /types) are:

- *Device State* - (`GPUContext` and supporting types)
- *Data* - that you want to pass to the GPU (GPUArray and GPUTensor)
- *Computation* - that you want to execute on the GPU (Kernel)

The core verbs (functions) of interest are:

- *Requesting GPU Resources* - CreateGPUContext(), CreateArray() and CreateTensor()
- *Ahead-of-Time Compute Preparation* - PrepareKernel() which both binds
  resources and compiles the kernel
- *Asynchronous Execution* - LaunchKernel(), Wait()
- *Data Movement* - ToCPU(), ToGPU()

Each of these has some supporting functions and types which we can get to
later.
)");

  section(R"(
gpu.cpp vs. the raw WebGPU API
------------------------------

The main responsibility of the types and functions of the library is to make
it trivial to represent these common building blocks of computation

If you look at `examples/webgpu_intro/run.cpp` you can get a sense of what it's
like to interact directly with the WebGPU.
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
Custom WGSL Compute Kernels
---------------------------

TODO(avh)
)");

  fprintf(stdout, "Goodbye!\n");
}
