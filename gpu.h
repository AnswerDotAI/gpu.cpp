#ifndef GPU_H
#define GPU_H

#include "spdlog/spdlog.h"
#include <array>
#include <cassert>
#include <cstring>
#include <future>
#include <memory>
#include <type_traits>

#include "webgpu/webgpu.h"

namespace gpu {

#ifdef NDEBUG
static constexpr bool kDebug = false;
#else
static constexpr bool kDebug = true;
#endif

static constexpr size_t kMaxRank = 8;

struct GPUContext;

struct ShaderCode {
  std::string code;
  size_t wgSize; // workgroup size
};

struct Shape {
  std::array<size_t, kMaxRank> data = {0};
  size_t rank = 0;

  Shape() = default;

  Shape(std::initializer_list<size_t> dims) {
    assert(dims.size() <= kMaxRank);
    std::copy(dims.begin(), dims.end(), data.begin());
    rank = dims.size();
  }

  size_t &operator[](size_t index) {
    assert(index < rank);
    return data[index];
  }

  const size_t &operator[](size_t index) const {
    assert(index < rank);
    return data[index];
  }
};

size_t size(const Shape &shape) {
  size_t numels = 1;
  for (size_t i = 0; i < shape.rank; i++) {
    numels *= shape.data[i];
  }
  return numels;
}

struct GPUArray {
  WGPUBuffer buffer;
  WGPUBufferUsageFlags usage;
  size_t size;
};

struct GPUTensor {
  GPUArray data;
  Shape shape;
};

struct TensorPool {
  TensorPool(GPUContext *ctx) : ctx(ctx), data(){};
  GPUContext *ctx;
  std::unordered_map<WGPUBuffer, GPUTensor> data;
  ~TensorPool();
};

struct GPUContext {
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice device;
  WGPUQueue queue;
  TensorPool pool = TensorPool(this);
  ~GPUContext() {
    spdlog::info("Destroying context");
    if (queue) {
      wgpuQueueRelease(queue);
      wgpuInstanceProcessEvents(instance);
    } else {
      spdlog::warn("Queue is null");
    }
    if (device) {
      wgpuDeviceRelease(device);
      wgpuInstanceProcessEvents(instance);
    } else {
      spdlog::warn("Device is null");
    }
    if (adapter) {
      wgpuAdapterRelease(adapter);
      wgpuInstanceProcessEvents(instance);
    } else {
      spdlog::warn("Adapter is null");
    }
    if (instance) {
      wgpuInstanceRelease(instance);
    } else {
      spdlog::warn("Instance is null");
    }
  }
};

enum NumType { kf32 };

/* Tensor factory function */
GPUTensor Tensor(TensorPool &pool, const Shape &shape, NumType dtype,
                 WGPUBufferUsageFlags usage = WGPUBufferUsage_Storage |
                                              WGPUBufferUsage_CopyDst |
                                              WGPUBufferUsage_CopySrc) {
  spdlog::trace("Creating tensor");
  size_t numElements = 1;
  for (size_t dim = 0; dim < shape.rank; dim++) {
    numElements *= shape.data[dim];
  }
  size_t size = dtype == kf32 ? sizeof(float) * numElements : 0;
  WGPUBufferDescriptor bufferDesc = {
      .usage = usage,
      .size = size,
  };
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(pool.ctx->device, &bufferDesc);
  pool.data[buffer] = GPUTensor{
      .data = GPUArray{.buffer = buffer, .usage = usage, .size = size},
      .shape = shape,
  };
  wgpuDeviceCreateBuffer(pool.ctx->device, &bufferDesc);
  return pool.data[buffer];
}

/* Syntactic sugar - take in ctx instead of pool*/
GPUTensor Tensor(GPUContext &ctx, const Shape &shape, NumType dtype) {
  return Tensor(ctx.pool, shape, dtype,
                WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                    WGPUBufferUsage_CopySrc);
}

/* With Value Initialization (pointer) */
GPUTensor Tensor(GPUContext &ctx, const Shape &shape, NumType dtype,
                 float *data) {
  GPUTensor tensor = Tensor(ctx.pool, shape, dtype,
                            WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                                WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

void FreeTensor(TensorPool &pool, GPUTensor tensor) {
  wgpuBufferRelease(tensor.data.buffer);
  pool.data.erase(tensor.data.buffer);
}

TensorPool::~TensorPool() {
  // Need to get keys in a separate iteration, otherwise iterator is getting
  // invalidated during erase.
  std::vector<WGPUBuffer> keys;
  for (auto &pair : data) {
    keys.push_back(pair.first);
  }
  for (auto &key : keys) {
    FreeTensor(*this, data[key]);
    spdlog::trace("Freed tensor");
  }
}

struct CallbackDataDyn {
  WGPUBuffer buffer;
  size_t bufferSize;
  float *output;
  std::promise<void> *promise;
};

struct Kernel {
  std::unique_ptr<WGPUBuffer[]> buffers;
  std::unique_ptr<size_t[]> bufferSizes;
  WGPUBuffer outputBuffer;
  size_t outputSize;
  size_t numBuffers;
  size_t numInputs;
  WGPUCommandBuffer commandBuffer;
  WGPUBuffer readbackBuffer;
  CallbackDataDyn callbackData;
  std::promise<void> promise;
  std::future<void> future;
};

struct NoParam {};

template <typename T> constexpr bool IsNoParam = std::is_same_v<T, NoParam>;

inline void check(bool condition, const char *message,
                  const char *file = "unkown", int line = -1) {
  if constexpr (kDebug) {
    if (!condition) {
      spdlog::error("Error in file {} line {}:\n{}", file, line, message);
      exit(1);
    } else {
      spdlog::trace("Success in file {} line {}:\n{}", file, line, message);
    }
  }
}

void showDeviceInfo(WGPUAdapter &adapter) {
  WGPUAdapterProperties properties;
  wgpuAdapterGetProperties(adapter, &properties);
  printf("Device Name: %s\n", properties.name);
  printf("Vendor ID: %u\n", properties.vendorID);
  printf("Device ID: %u\n", properties.deviceID);
  WGPULimits limits;
  WGPUSupportedLimits supportedLimits;
  wgpuAdapterGetLimits(adapter, &supportedLimits);
}

GPUContext CreateGPUContext(bool quietLogging = true,
                            const WGPUInstanceDescriptor &desc = {},
                            const WGPURequestAdapterOptions &adapterOpts = {},
                            WGPUDeviceDescriptor devDescriptor = {}) {
  if (quietLogging) {
    // TODO(avh): don't step on global logger
    auto logger = spdlog::default_logger();
    logger->set_level(spdlog::level::err);
  }
  GPUContext context;
  {
    context.instance = wgpuCreateInstance(&desc);
    check(context.instance, "Initialize WebGPU", __FILE__, __LINE__);
  }
  spdlog::info("Requesting adapter");
  {
    struct AdapterData {
      WGPUAdapter adapter = nullptr;
      bool requestEnded = false;
    };
    AdapterData adapterData;
    auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status,
                                    WGPUAdapter adapter, char const *message,
                                    void *pUserData) {
      AdapterData &adapterData = *reinterpret_cast<AdapterData *>(pUserData);
      check(status == WGPURequestAdapterStatus_Success,
            "Request WebGPU adapter", __FILE__, __LINE__);
      adapterData.adapter = adapter;
      adapterData.requestEnded = true;
    };
    wgpuInstanceRequestAdapter(context.instance, &adapterOpts,
                               onAdapterRequestEnded, (void *)&adapterData);
    assert(adapterData.requestEnded);
    context.adapter = adapterData.adapter;
  }
  spdlog::info("Requesting device");
  {
    struct DeviceData {
      WGPUDevice device = nullptr;
      bool requestEnded = false;
    };
    DeviceData devData;
    auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status,
                                   WGPUDevice device, char const *message,
                                   void *pUserData) {
      DeviceData &devData = *reinterpret_cast<DeviceData *>(pUserData);
      check(status == WGPURequestDeviceStatus_Success,
            "Could not get WebGPU device.", __FILE__, __LINE__);
      spdlog::info("Device Request succeeded {}", static_cast<void *>(device));
      devData.device = device;
      devData.requestEnded = true;
    };

    devDescriptor.deviceLostCallbackInfo = {
        .callback =
            [](WGPUDevice const *device, WGPUDeviceLostReason reason,
               char const *message, void *userdata) {
              if (reason != WGPUDeviceLostReason_Destroyed) {
                spdlog::error("Device lost (code {}):\n{}",
                              static_cast<int>(reason), message);
              } else {
                spdlog::info("Device destroyed: {}", message);
              }
            },
    };

    wgpuAdapterRequestDevice(context.adapter, &devDescriptor,
                             onDeviceRequestEnded, (void *)&devData);
    assert(devData.requestEnded);
    context.device = devData.device;
    wgpuDeviceSetUncapturedErrorCallback(
        context.device,
        [](WGPUErrorType type, char const *message, void *devData) {
          spdlog::error("Device uncaptured error: {}", message);
        },
        nullptr);
  }
  // Queue
  context.queue = wgpuDeviceGetQueue(context.device);
  return context;
}

/* Populates Op with the readbackBuffer as well as the commandBuffer */
void PrepareCommandBuffer(GPUContext &ctx, const ShaderCode &shader,
                          WGPUBindGroup &bindGroup,
                          WGPUBindGroupLayout &bgLayout,
                          uint32_t bufferSize, // readback buffer size
                          size_t N,            // readback buffer # elements
                          Kernel &op) {
  WGPUDevice device = ctx.device;
  spdlog::info("Create the readback buffer");
  {
    WGPUBufferDescriptor readbackBufferDescriptor = {
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = bufferSize,
    };
    op.readbackBuffer =
        wgpuDeviceCreateBuffer(device, &readbackBufferDescriptor);
  }
  spdlog::info("Create the compute pipeline");
  WGPUComputePipeline computePipeline;
  {
    WGPUPipelineLayout pipelineLayout;
    WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bgLayout,
    };
    pipelineLayout =
        wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);
    WGPUShaderModuleWGSLDescriptor wgslDesc = {
        .code = shader.code.c_str(),
    };
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    WGPUShaderModuleDescriptor shaderModuleDesc = {};
    shaderModuleDesc.nextInChain = &wgslDesc.chain;
    shaderModuleDesc.label = "shader";
    WGPUComputePipelineDescriptor computePipelineDesc = {};
    computePipelineDesc.layout = pipelineLayout;
    computePipelineDesc.compute.module =
        wgpuDeviceCreateShaderModule(device, &shaderModuleDesc);
    computePipelineDesc.compute.entryPoint = "main";
    computePipeline =
        wgpuDeviceCreateComputePipeline(device, &computePipelineDesc);
    check(computePipeline, "Create compute pipeline", __FILE__, __LINE__);
  }
  spdlog::info("Create the command encoder");
  {
    // After beginning the compute pass, use
    // wgpuComputePassEncoderInsertDebugMarker instead of
    // wgpuCommandEncoderInsertDebugMarker o/w the command encoder will be
    // locked after wgpuComputePassEncoderEnd.
    WGPUCommandEncoder commandEncoder;
    WGPUComputePassEncoder computePassEncoder;
    commandEncoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    computePassEncoder =
        wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);
    wgpuComputePassEncoderSetPipeline(computePassEncoder, computePipeline);
    wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, bindGroup, 0,
                                       nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        computePassEncoder, (N + (shader.wgSize - 1)) / shader.wgSize, 1, 1);
    wgpuComputePassEncoderEnd(computePassEncoder);
    wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, op.outputBuffer, 0,
                                         op.readbackBuffer, 0, bufferSize);
    op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
    check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
  }
}

void Wait(GPUContext &ctx, std::future<void> &future) {
  while (future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    wgpuInstanceProcessEvents(ctx.instance);
  }
}

/* Copy from GPU to CPU.

  Combines subset of PrepareCommandBuffer, but there is no compute pipeline +
  execution of LaunchKernel. A more performant version of
  this would prepare the command buffer once and reuse it for multiple
  readbacks. This version is for one-offs in non-hot paths.
*/
void ToCPU(GPUContext &ctx, GPUTensor &tensor, float *data, size_t bufferSize) {
  WGPUDevice device = ctx.device;
  struct CopyOp {
    WGPUCommandBuffer commandBuffer;
    WGPUBuffer readbackBuffer;
    std::promise<void> promise;
    std::future<void> future;
    CallbackDataDyn callbackData;
  };
  CopyOp op;
  op.future = op.promise.get_future();
  {
    WGPUBufferDescriptor readbackBufferDescriptor = {
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = bufferSize,
    };
    op.readbackBuffer =
        wgpuDeviceCreateBuffer(device, &readbackBufferDescriptor);
  }
  {
    WGPUCommandEncoder commandEncoder;
    WGPUComputePassEncoder computePassEncoder;
    commandEncoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, tensor.data.buffer, 0,
                                         op.readbackBuffer, 0, bufferSize);
    op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
    check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
  }
  wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);
  CallbackDataDyn callbackData = {op.readbackBuffer, bufferSize, data,
                                  &op.promise};
  wgpuQueueOnSubmittedWorkDone(
      ctx.queue,
      [](WGPUQueueWorkDoneStatus status, void *callbackData) {
        spdlog::info("QueueOnSubmittedWorkDone status: {}",
                     WGPUQueueWorkDoneStatus_Success == status);
        check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done",
              __FILE__, __LINE__);
        const auto *data = static_cast<CallbackDataDyn *>(callbackData);
        wgpuBufferMapAsync(
            data->buffer, WGPUMapMode_Read, 0, data->bufferSize,
            [](WGPUBufferMapAsyncStatus status, void *captureData) {
              const auto *data = static_cast<CallbackDataDyn *>(captureData);
              check(status == WGPUBufferMapAsyncStatus_Success,
                    "Map readbackBuffer", __FILE__, __LINE__);
              const void *mappedData = wgpuBufferGetConstMappedRange(
                  data->buffer, /*offset=*/0, data->bufferSize);
              check(mappedData, "Get mapped range", __FILE__, __LINE__);
              memcpy(data->output, mappedData, data->bufferSize);
              wgpuBufferUnmap(data->buffer);
              data->promise->set_value();
            },
            callbackData);
      },
      &callbackData);
  Wait(ctx, op.future);
}

void ToGPU(GPUContext &ctx, GPUTensor &tensor, const float *data) {
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
}

template <size_t N>
void ToCPU(GPUContext &ctx, GPUTensor &tensor, std::array<float, N> data) {
  ToCPU(ctx, tensor, data.data(), sizeof(data));
}

// TODO(avh): add a version that takes multiple kernels
template <typename ParamsType = NoParam>
Kernel PrepareKernel(GPUContext &ctx, const ShaderCode &shader,
                 const GPUTensor *inputs, size_t numInputs,
                 const GPUTensor &output,
                 const ParamsType &params = ParamsType{}) {
  WGPUDevice device = ctx.device;
  WGPUQueue queue = ctx.queue;
  Kernel op;
  // Calculate the total number of buffers
  size_t numBuffers = numInputs + 1; // numInputs + 1 output
  size_t outputIndex = numInputs;    // index of the output buffer within
                                     // op.buffers, opbufferSizes and
                                     // bgLayoutEntries

  size_t paramIndex;
  // paramIndex is undefined
  // unless ParamsType
  // is not NoParam
  if constexpr (!IsNoParam<ParamsType>) {
    numBuffers += 1;            // parameters buffer
    paramIndex = numInputs + 1; // == numBuffers - 1
    assert(outputIndex == numBuffers - 2);
    assert(paramIndex == numBuffers - 1);
  }

  op.buffers = std::make_unique<WGPUBuffer[]>(numBuffers);
  op.bufferSizes = std::make_unique<size_t[]>(numBuffers);
  op.numBuffers = numBuffers;
  op.numInputs = numInputs;
  size_t paramsBufferSize = sizeof(ParamsType);
  spdlog::info("Create the bind group layout");
  std::vector<WGPUBindGroupLayoutEntry> bgLayoutEntries(numBuffers);
  // Create layout entries for input buffers
  for (size_t i = 0; i < numInputs; ++i) {
    bgLayoutEntries[i] = WGPUBindGroupLayoutEntry{
        .binding = static_cast<uint32_t>(i),
        .visibility = WGPUShaderStage_Compute,
        .buffer =
            WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Storage,
                .minBindingSize = inputs[i].data.size,
            },
    };
  }
  // Create layout entry for output buffer
  bgLayoutEntries[outputIndex] = WGPUBindGroupLayoutEntry{
      .binding = static_cast<uint32_t>(outputIndex),
      .visibility = WGPUShaderStage_Compute,
      .buffer =
          WGPUBufferBindingLayout{
              .type = WGPUBufferBindingType_Storage,
              .minBindingSize = output.data.size,
          },
  };
  if constexpr (!IsNoParam<ParamsType>) {
    spdlog::info("Create layout entry for the params buffer");
    // Create layout entry for the params buffer
    bgLayoutEntries[paramIndex] = WGPUBindGroupLayoutEntry{
        .binding = static_cast<uint32_t>(paramIndex),
        .visibility = WGPUShaderStage_Compute,
        .buffer =
            WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Uniform,
                .minBindingSize = paramsBufferSize,
            },
    };
  }

  WGPUBindGroupLayoutDescriptor bgLayoutDesc = {
      .entryCount = static_cast<uint32_t>(bgLayoutEntries.size()),
      .entries = bgLayoutEntries.data(),
  };
  WGPUBindGroupLayout bgLayout =
      wgpuDeviceCreateBindGroupLayout(device, &bgLayoutDesc);

  spdlog::info("Create input and output buffers");
  for (size_t i = 0; i < numInputs; ++i) {
    op.buffers[i] = inputs[i].data.buffer;
    op.bufferSizes[i] = inputs[i].data.size;
  }
  // Set up the output buffer
  op.buffers[outputIndex] = output.data.buffer;
  op.outputBuffer = op.buffers[outputIndex];
  op.outputSize = output.data.size;
  op.bufferSizes[outputIndex] = output.data.size;
  // Create a buffer for the Params struct
  if constexpr (!IsNoParam<ParamsType>) {
    WGPUBufferDescriptor paramsBufferDesc = {
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size = paramsBufferSize,
        .mappedAtCreation = false,
    };
    op.buffers[paramIndex] = wgpuDeviceCreateBuffer(device, &paramsBufferDesc);
    op.bufferSizes[paramIndex] = paramsBufferSize;

    spdlog::info("Create the params buffer");
    spdlog::info("param index {}", paramIndex);
    // spdlog::info("buffers {}", op.buffers[paramIndex]);
    // spdlog::info("params {}", params);
    // spdlog::info("paramsBufferSize {}", paramsBufferSize);
    wgpuQueueWriteBuffer(queue, op.buffers[paramIndex], 0, &params,
                         paramsBufferSize);
    spdlog::info("Params buffer written");
  } else {
    spdlog::info("No params buffer needed");
  }

  spdlog::info("Create the bind group");
  std::vector<WGPUBindGroupEntry> bindGroupEntries(numBuffers);
  for (size_t i = 0; i <= numInputs; ++i) { // <= for output buffer
    bindGroupEntries[i] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(i),
        .buffer = op.buffers[i],
        .offset = 0,
        .size = op.bufferSizes[i],
    };
  }
  if constexpr (!IsNoParam<ParamsType>) {
    bindGroupEntries[paramIndex] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(numBuffers - 1),
        .buffer = op.buffers[paramIndex],
        .offset = 0,
        .size = paramsBufferSize,
    };
  }
  WGPUBindGroupDescriptor bindGroupDesc = {
      .layout = bgLayout,
      .entryCount = static_cast<uint32_t>(bindGroupEntries.size()),
      .entries = bindGroupEntries.data(),
  };
  WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

  spdlog::info("Initializing promise and future");
  op.promise = std::promise<void>();
  op.future = op.promise.get_future();

  spdlog::info("Preparing command buffer");
  size_t outN = size(output.shape);

  PrepareCommandBuffer(ctx, shader, bindGroup, bgLayout,
                           op.bufferSizes[outputIndex], outN, op);

  // Write the params data to the params buffer
  if (!IsNoParam<ParamsType>) {
    // Write the params data to the params buffer
    wgpuQueueWriteBuffer(ctx.queue, op.buffers[op.numInputs + 1], 0, &params,
                         sizeof(ParamsType));
  }

  spdlog::info("Exiting PrepareKernel");
  return op;
}

/*
 * PrepareKernel with array of inputs (convienence function)
 */
template <typename ParamsType = NoParam, size_t numInputs>
Kernel PrepareKernel(GPUContext &ctx, const ShaderCode &shader,
                 const std::array<GPUTensor, numInputs> &inputs,
                 const GPUTensor &output,
                 const ParamsType &params = ParamsType{}) {
  return PrepareKernel(ctx, shader, inputs.data(), numInputs, output, params);
}

void LaunchKernel(GPUContext &ctx, Kernel &op) {

  // Total size of the output buffer in bytes
  uint32_t bufferSizeOut = static_cast<uint32_t>(op.outputSize);

  // Submit the command buffer
  wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);

  op.callbackData =
      CallbackDataDyn{op.readbackBuffer, op.outputSize, nullptr, &op.promise};

  // Set up the callback for when the work is done
  wgpuQueueOnSubmittedWorkDone(
      ctx.queue,
      [](WGPUQueueWorkDoneStatus status, void *callbackData) {
        spdlog::info("QueueOnSubmittedWorkDone status: {}",
                     WGPUQueueWorkDoneStatus_Success == status);
        check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done",
              __FILE__, __LINE__);
        const auto *data = static_cast<CallbackDataDyn *>(callbackData);
        data->promise->set_value();
      },
      &op.callbackData);
}

} // namespace gpu

#endif // GPU_H
