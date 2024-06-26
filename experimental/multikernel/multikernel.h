#ifndef _EXPERIMENTAL_H_
#define _EXPERIMENTAL_H_

#include <future>
#include <memory>
#include <vector>
#include "webgpu/webgpu.h"

#include "gpu.h"

/**
 * @brief Represents a GPU computation with multiple shaders in a single
 * compute pipeline / command buffer dispatch.
 *
 * The fields are analogous to the fields in the single-shader Kernel type ,
 * but with aggregates for each shader's data.
 */
struct MultiKernel {
  size_t numShaders;
  std::unique_ptr<WGPUBuffer[]> buffers;       // length = sum of numBuffers[]
  std::unique_ptr<size_t[]> bufferSizes;       // length = sum of numBuffers[]
  std::unique_ptr<WGPUBuffer[]> outputBuffers; // length = numShaders
  std::unique_ptr<size_t[]> outputSize;        // length = numShaders
  std::unique_ptr<size_t[]>
      numBuffers; // length = numShaders,
                  // value[i] = numInputs[i] + 1 (output) + 0 or 1
                  //    depending on whether paramSizes is > 0 or not.
                  //    paramSizes = 0 means no params buffer
  std::unique_ptr<size_t[]> numInputs;         // length = numShaders
  std::unique_ptr<Shape[]> nWorkgroups;        // length = numShaders
  std::unique_ptr<WGPUBindGroup[]> bindGroups; // length = numShaders
                                               // persists between submission
  std::unique_ptr<WGPUComputePipeline[]>
      computePipelines;                // length = numShaders
                                       // persists between
                                       // submission
  WGPUCommandBuffer commandBuffer;     // All kernels in the multiKernel
  WGPUComputePipeline computePipeline; // TODO(avh): decide how to handle
                                       // compute multiKernels for multikernel
  WGPUBuffer readbackBuffer; // Readback buffer for the final output buffer
  CallbackDataDyn callbackData;
  std::promise<void> promise;
  std::future<void> future;
};

/**
 * @brief Input arguments to CreateMultiKernel to construct a Kernel instance.
 */
struct MultiKernelDesc {
  size_t numShaders;
  const ShaderCode *shader; // pointer to (dynamic) array of length = numShaders
  const Tensor *inputs;     // length = sum of numInputs[]
  const size_t *numInputs;  // length = numShaders
  const Tensor *output;     // length = numShaders
  const void *params;       // length = numShaders
                            // use void* so params can be different
                            // types for each shader
  const size_t *paramSizes; // length = numShaders
  const Shape *nThreads;    // length = numShaders
};


void ResetMultiCommandBuffer(WGPUDevice &device, MultiKernel &multiKernel) {
  WGPUCommandEncoder commandEncoder =
      wgpuDeviceCreateCommandEncoder(device, nullptr);
  for (size_t shaderIdx = 0; shaderIdx < multiKernel.numShaders; ++shaderIdx) {
    WGPUComputePassEncoder computePassEncoder =
        wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);
    wgpuComputePassEncoderSetPipeline(computePassEncoder,
                                      multiKernel.computePipelines[shaderIdx]);
    wgpuComputePassEncoderSetBindGroup(
        computePassEncoder, 0, multiKernel.bindGroups[shaderIdx], 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        computePassEncoder, multiKernel.nWorkgroups[shaderIdx][0],
        multiKernel.nWorkgroups[shaderIdx][1],
        multiKernel.nWorkgroups[shaderIdx][2]);
    wgpuComputePassEncoderEnd(computePassEncoder);
  }
  LOG(kDefLog, kInfo, "Finish command encoder");
  multiKernel.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
  check(multiKernel.commandBuffer, "Create command buffer", __FILE__, __LINE__);
}

/**
 * @brief Factory function to create a multi-kernel on the GPU. This is similar
 * to CreateKernel but allows multiple shaders to be in the same command buffer.
 *
 * Note - this interface is experimental and likely to change in the future.
 *
 * @param[in] ctx Context instance to manage the multiKernel
 * @param[in] desc A description / specification of the multiKernel to be
 * instantiated. This is analogous to the input parameters of CreateKernel, but
 * since there are more complex data structures involved, it is more convenient
 * to pass a struct.
 * @return MultiKernel instance representing the created
 * multiKernel
 * @example MultiKernel multiKernel = CreateMultiKernel(ctx, desc);
 */
MultiKernel CreateMultiKernel(Context &ctx, const MultiKernelDesc &desc) {
  WGPUDevice device = ctx.device;
  WGPUQueue queue = ctx.queue;
  MultiKernel multiKernel;
  multiKernel.numShaders = desc.numShaders;
  size_t totalBuffers = 0;
  multiKernel.numBuffers = std::make_unique<size_t[]>(desc.numShaders);
  multiKernel.numInputs = std::make_unique<size_t[]>(desc.numShaders);
  multiKernel.outputBuffers = std::make_unique<WGPUBuffer[]>(desc.numShaders);
  multiKernel.outputSize = std::make_unique<size_t[]>(desc.numShaders);
  // Calculate total number of buffers
  for (size_t i = 0; i < desc.numShaders; ++i) {
    multiKernel.numInputs[i] = desc.numInputs[i];
    multiKernel.numBuffers[i] = desc.numInputs[i] + 1; // +1 for output buffer
    if (desc.paramSizes[i] > 0) {
      // == 0 means shader does not have a parameter input
      multiKernel.numBuffers[i] += 1; // +1 for params buffer
    }
    totalBuffers += multiKernel.numBuffers[i];
  }
  multiKernel.buffers = std::make_unique<WGPUBuffer[]>(totalBuffers);
  multiKernel.bufferSizes = std::make_unique<size_t[]>(totalBuffers);
  // Create command encoder for all kernels
  WGPUCommandEncoder commandEncoder =
      wgpuDeviceCreateCommandEncoder(device, nullptr);
  size_t bufferIndex = 0;
  // Iterate over all shaders in the multiKernel
  // make and allocate computePipeline per shader
  multiKernel.computePipelines =
      std::make_unique<WGPUComputePipeline[]>(desc.numShaders);
  multiKernel.bindGroups = std::make_unique<WGPUBindGroup[]>(desc.numShaders);
  multiKernel.nWorkgroups = std::make_unique<Shape[]>(desc.numShaders);
  for (size_t shaderIdx = 0; shaderIdx < desc.numShaders; ++shaderIdx) {
    // Create buffers and bind group for each shader
    size_t outputIndex = desc.numInputs[shaderIdx];
    size_t paramIndex =
        desc.paramSizes[shaderIdx] > 0 ? desc.numInputs[shaderIdx] + 1 : -1;
    // Create layout entries for input buffers
    LOG(kDefLog, kInfo, "Create the bind group layout");
    std::vector<WGPUBindGroupLayoutEntry> bgLayoutEntries(
        multiKernel.numBuffers[shaderIdx]);
    for (size_t i = 0; i < multiKernel.numBuffers[shaderIdx]; ++i) {
      LOG(kDefLog, kInfo, "Create layout entry for buffer %d", i);
      LOG(kDefLog, kInfo, "i %d outputIndex %d i == paramIndex ? %d", i,
          outputIndex, i == paramIndex);
      bgLayoutEntries[i] = WGPUBindGroupLayoutEntry{
          .binding = static_cast<uint32_t>(i),
          .visibility = WGPUShaderStage_Compute,
          .buffer = WGPUBufferBindingLayout{
              .type = (i != paramIndex) ? WGPUBufferBindingType_Storage
                                        : WGPUBufferBindingType_Uniform,
              .minBindingSize = i < outputIndex ? desc.inputs[i].data.size
                                : i == outputIndex
                                    ? desc.output[shaderIdx].data.size
                                    : desc.paramSizes[shaderIdx],
          }};
    }
    WGPUBindGroupLayoutDescriptor bgLayoutDesc = {
        .entryCount = static_cast<uint32_t>(bgLayoutEntries.size()),
        .entries = bgLayoutEntries.data()};
    WGPUBindGroupLayout bgLayout =
        wgpuDeviceCreateBindGroupLayout(device, &bgLayoutDesc);
    LOG(kDefLog, kInfo, "Create input and output buffers");
    for (size_t inputIndex = 0; inputIndex < desc.numInputs[shaderIdx];
         ++inputIndex) {
      multiKernel.buffers[bufferIndex] = desc.inputs[inputIndex].data.buffer;
      multiKernel.bufferSizes[bufferIndex] = desc.inputs[inputIndex].data.size;
      bufferIndex++;
    }
    // Set up output buffer
    multiKernel.outputBuffers[shaderIdx] = desc.output[shaderIdx].data.buffer;
    multiKernel.outputSize[shaderIdx] = desc.output[shaderIdx].data.size;
    multiKernel.buffers[bufferIndex] = multiKernel.outputBuffers[shaderIdx];
    multiKernel.bufferSizes[bufferIndex] = multiKernel.outputSize[shaderIdx];
    bufferIndex++;
    // Set up params buffer if required
    if (desc.paramSizes[shaderIdx] > 0) {
      WGPUBufferDescriptor paramsBufferDesc = {
          .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
          .size = desc.paramSizes[shaderIdx],
          .mappedAtCreation = false,
      };
      LOG(kDefLog, kInfo, "Create the params buffer at bufferIndex %d",
          bufferIndex);
      multiKernel.buffers[bufferIndex] =
          wgpuDeviceCreateBuffer(device, &paramsBufferDesc);
      multiKernel.bufferSizes[bufferIndex] = desc.paramSizes[shaderIdx];
      bufferIndex++;
      LOG(kDefLog, kInfo, "Params buffer written");
    } else {
      LOG(kDefLog, kInfo, "No params buffer needed");
    }
    {
      std::vector<WGPUBindGroupEntry> bindGroupEntries(
          multiKernel.numBuffers[shaderIdx]);
      LOG(kDefLog, kInfo, "Number of buffers: %d",
          multiKernel.numBuffers[shaderIdx]);
      for (size_t i = 0; i < multiKernel.numBuffers[shaderIdx]; ++i) {
        bindGroupEntries[i] = WGPUBindGroupEntry{
            .binding = static_cast<uint32_t>(i),
            .buffer = multiKernel.buffers[i + bufferIndex -
                                          multiKernel.numBuffers[shaderIdx]],
            .offset = 0,
            .size = multiKernel.bufferSizes[i + bufferIndex -
                                            multiKernel.numBuffers[shaderIdx]]};
      }
      WGPUBindGroupDescriptor bindGroupDesc = {
          .layout = bgLayout,
          .entryCount = static_cast<uint32_t>(bindGroupEntries.size()),
          .entries = bindGroupEntries.data()};
      multiKernel.bindGroups[shaderIdx] =
          wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
    }
    WGPUPipelineLayoutDescriptor multiKernelLayoutDesc = {
        .bindGroupLayoutCount = 1, .bindGroupLayouts = &bgLayout};
    WGPUPipelineLayout multiKernelLayout =
        wgpuDeviceCreatePipelineLayout(device, &multiKernelLayoutDesc);
    // Create shader module
    WGPUShaderModuleWGSLDescriptor wgslDesc = {
        .code = desc.shader[shaderIdx].data.c_str(),
    };
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    WGPUShaderModuleDescriptor shaderModuleDesc = {
        .nextInChain = &wgslDesc.chain, .label = "shader"};
    WGPUShaderModule shaderModule =
        wgpuDeviceCreateShaderModule(device, &shaderModuleDesc);
    WGPUComputePipelineDescriptor computePipelineDesc = {
        .layout = multiKernelLayout,
        .compute = {.module = shaderModule, .entryPoint = "main"}};
    multiKernel.computePipelines[shaderIdx] =
        wgpuDeviceCreateComputePipeline(device, &computePipelineDesc);
    multiKernel.nWorkgroups[shaderIdx] = {
        (desc.nThreads[shaderIdx][0] +
         (desc.shader[shaderIdx].workgroupSize[0] - 1)) /
            desc.shader[shaderIdx].workgroupSize[0],
        (desc.nThreads[shaderIdx][1] +
         (desc.shader[shaderIdx].workgroupSize[1] - 1)) /
            desc.shader[shaderIdx].workgroupSize[1],
        (desc.nThreads[shaderIdx][2] +
         (desc.shader[shaderIdx].workgroupSize[2] - 1)) /
            desc.shader[shaderIdx].workgroupSize[2]};
  }


  ResetMultiCommandBuffer(device, multiKernel);
  check(multiKernel.commandBuffer, "Create command buffer", __FILE__, __LINE__);
  {
    WGPUBufferDescriptor readbackBufferDescriptor = {
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = multiKernel.outputSize[multiKernel.numShaders - 1],
    };
    multiKernel.readbackBuffer =
        wgpuDeviceCreateBuffer(device, &readbackBufferDescriptor);
  }
  multiKernel.promise = std::promise<void>();
  multiKernel.future = multiKernel.promise.get_future();
  multiKernel.callbackData =
      CallbackDataDyn{multiKernel.readbackBuffer,
                      multiKernel.outputSize[multiKernel.numShaders - 1],
                      nullptr, &multiKernel.promise};
  ctx.kernelPool.multiData.insert(&multiKernel);
  return multiKernel;
}

/**
 * @brief Asynchronously submits a multi-kernel to the GPU queue for execution.
 *
 * Note - as with CreateMultiKernel, this interface is experimental and likely
 * to change in the future.
 *
 * @param[in] ctx Context instance to manage the multiKernel
 * @param[in] multiKernel MultiKernel instance to dispatch
 * @example DispatchMultiKernel(ctx, multiKernel);
 */
void DispatchMultiKernel(Context &ctx, MultiKernel &multiKernel) {
  wgpuQueueSubmit(ctx.queue, 1, &multiKernel.commandBuffer);
  wgpuQueueOnSubmittedWorkDone(
      ctx.queue,
      [](WGPUQueueWorkDoneStatus status, void *callbackData) {
        LOG(kDefLog, kInfo, "QueueOnSubmittedWorkDone status: %d",
            WGPUQueueWorkDoneStatus_Success == status);
        check(status == WGPUQueueWorkDoneStatus_Success,
              "Check queue work success", __FILE__, __LINE__);
        const auto *data = static_cast<CallbackDataDyn *>(callbackData);
        data->promise->set_value();
      },
      &multiKernel.callbackData);
}


void TestMultiKernel1(Context &ctx) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  static constexpr size_t B = 6;    // batch size
  static constexpr size_t T = 8;    // token index
  static constexpr size_t C = 3072; // input channels
  std::array<float, B * T * C> inputArr;
  std::array<float, B * T * C> outputArr;
  std::mt19937 gen(31415);
  randint(inputArr, gen, 0, 3);
  Tensor input = CreateTensor(ctx, {B, T, C}, kf32, inputArr.data());
  Tensor output = CreateTensor(ctx, {B, T, C}, kf32, outputArr.data());
  auto shader = CreateShader(kShaderSoftmax1, 256, kf32);
  constexpr size_t size = sizeof(SoftmaxParam);
  auto param = SoftmaxParam{B * T, C};
  std::array<size_t, 1> numInputs = {1};
  // First test with the degenerate case of a 1-shader multi kernel
  std::unique_ptr<Shape[]> nThreads(new Shape[1]);
  nThreads[0] = {B * T, 1, 1};
  MultiKernelDesc desc{
      .numShaders = 1,
      .shader = &shader,
      .inputs = &input,
      .numInputs = numInputs.data(),
      .output = &output,
      .params = &param,
      .paramSizes = &size,
      .nThreads = nThreads.get(),
  };
  MultiKernel pipeline = CreateMultiKernel(ctx, desc);
  DispatchMultiKernel(ctx, pipeline);
  Wait(ctx, pipeline.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output").c_str());
  LOG(kDefLog, kInfo, "Done with MultiKernel Test 1");
}

void TestMultiKernel2(Context &ctx) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  static constexpr size_t B = 6;    // batch size
  static constexpr size_t T = 8;    // token index
  static constexpr size_t C = 3072; // input channels
  std::array<float, B * T * C> inputArr;
  std::array<float, B * T * C> outputArr;
  std::mt19937 gen(31415);
  randint(inputArr, gen, 0, 3);
  std::array<Tensor, 2> inputs;
  std::array<Tensor, 2> outputs;
  std::array<SoftmaxParam, 2> params;
  inputs[0] = CreateTensor(ctx, {B, T, C}, kf32, inputArr.data());
  outputs[0] = CreateTensor(ctx, {B, T, C}, kf32, outputArr.data());
  params[0] = SoftmaxParam{B * T, C};
  inputs[1] = CreateTensor(ctx, {B, T, C}, kf32, inputArr.data());
  outputs[1] = CreateTensor(ctx, {B, T, C}, kf32, outputArr.data());
  params[1] = SoftmaxParam{B * T, C};
  std::array<ShaderCode, 2> shaders = {
      CreateShader(kShaderSoftmax1, 256, kf32),
      CreateShader(kShaderSoftmax1, 256, kf32)};
  std::array<size_t, 2> numInputs = {1, 1};
  std::array<size_t, 2> paramSizes = {sizeof(SoftmaxParam),
                                      sizeof(SoftmaxParam)};
  // First test with the degenerate case of a 1-shader multi kernel
  std::unique_ptr<Shape[]> nThreads(new Shape[2]);
  nThreads[0] = {B * T, 1, 1};
  nThreads[1] = {B * T, 1, 1};
  MultiKernelDesc desc{
      .numShaders = 2,
      .shader = shaders.data(),
      .inputs = inputs.data(),
      .numInputs = numInputs.data(),
      .output = outputs.data(),
      .params = params.data(),
      .paramSizes = paramSizes.data(),
      .nThreads = nThreads.get(),
  };
  MultiKernel pipeline = CreateMultiKernel(ctx, desc);
  DispatchMultiKernel(ctx, pipeline);
  Wait(ctx, pipeline.future);
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  ToCPU(ctx, outputs[0], outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output 0").c_str());
  ToCPU(ctx, outputs[1], outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output 1").c_str());
  LOG(kDefLog, kInfo, "Done with MultiKernel Test 2");
}

#endif  // _EXPERIMENTAL_H_
