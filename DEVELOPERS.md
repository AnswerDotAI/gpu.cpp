# Developers

This note is for developers who want to contribute to the gpu.cpp library.

## Design Objectives

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

## Separating Resource Acquisition and Dispatch

We can think of the use of gpu.cpp as GPU resources modeled by the type
definitions of the library and actions on GPU resources, modeled by the
functions of the library. 

The key functions can be further subdivided into two categories in relation to
when the GPU computation occurs: 

1) Ahead-of-time preparation of resources and state: thess are functions that
   acquire resources and prepare state for GPU computation. These are less
   performance critical.

2) Performance critical dispatch of GPU computation: these are functions that
   dispatch GPU computation to the GPU, usually in a tight hot-path loop. 

This pattern is different from non-performance critical application code where
resource acquisition is often interleaved with computation throughout the
program execution.

This is a pattern for performance critical GPU computation that gpu.cpp is
intended for. Some example use cases that fit this are custom neural network
inference engines, render loops, simulations loops, etc.

We'll see how the functions and types of the library are organized around these
two types of actions.

## Resource Type Definitions and Acquisition

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

Resources are acquired using the `Create` functions. These are assumed to be
ahead-of-time and not performance critical.

- `GPUContext CreateGPUContext(...)` - creates a GPU context.
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

## Dispatching GPU Computation

GPU computation is launched using the `Dispatch` functions. These are assumed
to be performance critical.

- `void DispatchKernel(...)` - dispatches a single kernel to the GPU.
- `void DispatchMultiKernel(...)` - dispatches multiple kernels to the GPU.
