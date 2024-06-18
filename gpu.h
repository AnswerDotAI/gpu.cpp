#ifndef GPU_H
#define GPU_H

#include <array>
#include <cassert>
#include <cstring>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "utils/logging.h"
#include "webgpu/webgpu.h"

namespace gpu {

static Logger kGpuLog = {stdout, "", kInfo};

#ifdef NDEBUG
static constexpr bool kDebug = false;
#else
static constexpr bool kDebug = true;
#endif

// Maximum rank of a tensor so we don't have to dynamically allocate memory for
// the shape of a tensor.
static constexpr size_t kMaxRank = 8;

struct Context;

struct Array {
  WGPUBuffer buffer;
  WGPUBufferUsageFlags usage;
  size_t size;
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

struct Tensor {
  Array data;
  Shape shape;
};

template <std::size_t N> struct Tensors {
  std::array<Tensor, N> data;
  Tensors(std::initializer_list<Tensor> init) {
    std::copy(init.begin(), init.end(), data.begin());
  }
  Tensor &operator[](std::size_t index) { return data[index]; }
  const Tensor &operator[](std::size_t index) const { return data[index]; }
};

template <std::size_t N> Tensors(std::array<Tensor, N>) -> Tensors<N>;

// Deduction guide for Tensors
template <typename... Args> Tensors(Args...) -> Tensors<sizeof...(Args)>;

struct TensorPool {
  TensorPool(Context *ctx) : ctx(ctx), data() {};
  Context *ctx;
  std::unordered_map<WGPUBuffer, Tensor> data;
  ~TensorPool();
};

enum NumType { kf32 };

std::string ToString(NumType type) {
  switch (type) {
  case kf32:
    return "f32";
  default:
    log(kDefLog, kError, "Invalid NumType in string conversion.");
    return "unknown";
  }
}

std::string ToString(const Shape &shape) {
  std::string str;
  for (size_t i = 0; i < shape.rank; i++) {
    str += std::to_string(shape.data[i]);
    if (i < shape.rank - 1) {
      str += ", ";
    }
  }
  return str;
}

struct ShaderCode {
  ShaderCode(const std::string &data, size_t workgroupSize = 256,
             NumType precision = kf32)
      : data(data), workgroupSize({workgroupSize, 1, 1}), precision(precision) {
  }

  ShaderCode(const std::string &data, const Shape &workgroupSize,
             NumType precision = kf32)
      : data(data), workgroupSize(workgroupSize), precision(precision) {}
  std::string data;
  Shape workgroupSize;
  NumType precision;
};

struct CallbackDataDyn {
  WGPUBuffer buffer; // managed by owning Kernel
  size_t bufferSize;
  float *output; // non-owning, only for target memory in ToCPU, not used for
                 // kernel invocations
  std::promise<void> *promise;
};

struct KernelDesc {
  const ShaderCode shader;
  const Tensor *inputs;
  size_t numInputs;
  const Tensor output;
  const void *params;
  const size_t paramSize;
};

struct Kernel {
  std::unique_ptr<WGPUBuffer[]> buffers; // non-owning
  std::unique_ptr<size_t[]> bufferSizes;
  WGPUBuffer outputBuffer; // non-owning
  size_t outputSize;
  size_t numBuffers;
  size_t numInputs;
  Shape nWorkgroups;
  WGPUBindGroup bindGroup;             // persists between submission
  WGPUComputePipeline computePipeline; // persists between submission
  WGPUCommandBuffer commandBuffer;     // destroyed upon submission
  WGPUBuffer readbackBuffer;
  CallbackDataDyn callbackData;
  std::promise<void> promise;
  std::future<void> future;
};

struct MultiKernelDesc {
  size_t numShaders;
  const ShaderCode *shader; // pointer to (dynamic) array of length = numShaders
  const Tensor *inputs;  // length = sum of numInputs[]
  const size_t *numInputs;  // length = numShaders
  const Tensor *output;  // length = numShaders
  const void *params;       // length = numShaders
                            // use void* so params can be different
                            // types for each shader
  const size_t *paramSizes; // length = numShaders
  const Shape *nThreads;    // length = numShaders
};

// TODO(avh): implement equivalent of CreateKernel for MultiKernel with
// MultiKernelDesc as input argument
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

// make Kernel hashable by std::set
bool operator<(const Kernel &lhs, const Kernel &rhs) {
  return lhs.commandBuffer < rhs.commandBuffer;
}

struct KernelPool {
  KernelPool(Context *ctx) : ctx(ctx), data() {}
  Context *ctx;
  std::set<Kernel *> data;
  std::set<MultiKernel *> multiData;
  ~KernelPool() {
    // Note : commandBuffer is destroyed upon queue submission,
    // explicitly destroying readback and callback buffers
    // produces runtime errors.
    data.clear();
    multiData.clear();
  }
};

struct Context {
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice device;
  WGPUQueue queue;
  TensorPool pool = TensorPool(this);
  KernelPool kernelPool = KernelPool(this);
  ~Context() {
    log(kDefLog, kInfo, "Destroying context");
    if (queue) {
      wgpuQueueRelease(queue);
      wgpuInstanceProcessEvents(instance);
    } else {
      log(kDefLog, kWarn, "Queue is null");
    }
    if (device) {
      wgpuDeviceRelease(device);
      wgpuInstanceProcessEvents(instance);
    } else {
      log(kDefLog, kWarn, "Device is null");
    }
    if (adapter) {
      wgpuAdapterRelease(adapter);
      wgpuInstanceProcessEvents(instance);
    } else {
      log(kDefLog, kWarn, "Adapter is null");
    }
    if (instance) {
      wgpuInstanceRelease(instance);
    } else {
      log(kDefLog, kWarn, "Instance is null");
    }
    log(kDefLog, kInfo, "Destroyed context");
  }
};

/* Tensor factory function */
/**
 * @brief 
 * @param[in] 
 * @return 
 * @example 
 */
Tensor CreateTensor(TensorPool &pool, WGPUDevice &device, const Shape &shape,
                       NumType dtype,
                       WGPUBufferUsageFlags usage = WGPUBufferUsage_Storage |
                                                    WGPUBufferUsage_CopyDst |
                                                    WGPUBufferUsage_CopySrc) {
  log(kDefLog, kInfo, "Creating tensor");
  size_t numElements = 1;
  for (size_t dim = 0; dim < shape.rank; dim++) {
    numElements *= shape.data[dim];
  }
  size_t size = dtype == kf32 ? sizeof(float) * numElements : 0;
  WGPUBufferDescriptor bufferDesc = {
      .usage = usage,
      .size = size,
  };
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
  pool.data[buffer] = Tensor{
      .data = Array{.buffer = buffer, .usage = usage, .size = size},
      .shape = shape,
  };
  wgpuDeviceCreateBuffer(device, &bufferDesc);
  return pool.data[buffer];
}

/* Syntactic sugar - take in ctx instead of pool*/
Tensor CreateTensor(Context &ctx, const Shape &shape, NumType dtype) {
  return CreateTensor(ctx.pool, ctx.device, shape, dtype,
                      WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                          WGPUBufferUsage_CopySrc);
}

/* With Value Initialization (pointer) */
Tensor CreateTensor(Context &ctx, const Shape &shape, NumType dtype,
                       float *data) {
  Tensor tensor =
      CreateTensor(ctx.pool, ctx.device, shape, dtype,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

void FreeTensor(TensorPool &pool, Tensor tensor) {
  if (tensor.data.buffer) {
    wgpuBufferRelease(tensor.data.buffer);
  } else {
    log(kDefLog, kWarn, "Tried to free tensor with null buffer");
  }
  if (pool.data.find(tensor.data.buffer) != pool.data.end()) {
    pool.data.erase(tensor.data.buffer);
  } else {
    log(kDefLog, kWarn, "Tried to free tensor that was not in pool");
  }
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
    log(kDefLog, kTrace, "Freed tensor");
  }
}

void ReplaceAll(std::string &str, const std::string &from,
                const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

ShaderCode CreateShader(const char *shaderRaw,
                        const Shape &workgroupSize = {256, 1, 1},
                        NumType precision = kf32) {
  std::string codeString(shaderRaw);
  ReplaceAll(codeString, "{{workgroupSize}}", ToString(workgroupSize));
  ReplaceAll(codeString, "{{precision}}", ToString(precision));
  log(kDefLog, kInfo, "Shader code:\n%s", codeString.c_str());
  return ShaderCode{codeString, workgroupSize};
}

ShaderCode CreateShader(const char *shaderRaw, size_t workgroupSize,
                        NumType precision = kf32) {
  return CreateShader(shaderRaw, Shape{workgroupSize, 1, 1}, precision);
}

struct NoParam {};

template <typename T> constexpr bool IsNoParam = std::is_same_v<T, NoParam>;

inline void check(bool condition, const char *message,
                  const char *file = "unkown", int line = -1) {
  if constexpr (kDebug) {
    if (!condition) {
      log(kDefLog, kError, "Error in file %s line %d:\n%s", file, line,
          message);
      exit(1);
    } else {
      log(kDefLog, kTrace, "Success in file %s line %d:\n%s", file, line,
          message);
    }
  }
}

Context CreateContext(bool quietLogging = true,
                         const WGPUInstanceDescriptor &desc = {},
                         const WGPURequestAdapterOptions &adapterOpts = {},
                         WGPUDeviceDescriptor devDescriptor = {}) {
  if (quietLogging) {
    kDefLog.level = kError;
  }
  Context context;
  {
    context.instance = wgpuCreateInstance(&desc);
    check(context.instance, "Initialize WebGPU", __FILE__, __LINE__);
  }
  log(kDefLog, kInfo, "Requesting adapter");
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
  log(kDefLog, kInfo, "Requesting device");
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
      log(kDefLog, kInfo, "Device Request succeeded %s",
          static_cast<void *>(device));
      devData.device = device;
      devData.requestEnded = true;
    };

#ifdef WEBGPU_BACKEND_DAWN
    devDescriptor.deviceLostCallbackInfo = {
        .callback =
            [](WGPUDevice const *device, WGPUDeviceLostReason reason,
               char const *message, void *userdata) {
              if (reason != WGPUDeviceLostReason_Destroyed) {
                log(kDefLog, kError, "Device lost (code %d):\n%s", reason,
                    message);
              } else {
                log(kDefLog, kInfo, "Device destroyed: %s", message);
              }
            },
    };
#endif

    wgpuAdapterRequestDevice(context.adapter, &devDescriptor,
                             onDeviceRequestEnded, (void *)&devData);
    assert(devData.requestEnded);
    context.device = devData.device;
    wgpuDeviceSetUncapturedErrorCallback(
        context.device,
        [](WGPUErrorType type, char const *message, void *devData) {
          log(kDefLog, kError, "Device uncaptured error: %s", message);
        },
        nullptr);
  }
  // Queue
  context.queue = wgpuDeviceGetQueue(context.device);
  return context;
}

void Wait(Context &ctx, std::future<void> &future) {
  while (future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    wgpuInstanceProcessEvents(ctx.instance);
  }
}

/* Copy from GPU to CPU.
  A more performant version of this would prepare the command buffer once and
  reuse it for multiple readbacks. This version is a convenience implementation
  for non-hot paths.
*/
void ToCPU(Context &ctx, Tensor &tensor, float *data, size_t bufferSize) {
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
        log(kDefLog, kInfo, "QueueOnSubmittedWorkDone status == success ? %d",
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
// Convenience wrapper for array outputs
template <size_t N>
void ToCPU(Context &ctx, Tensor &tensor, std::array<float, N> data) {
  ToCPU(ctx, tensor, data.data(), sizeof(data));
}
void ToGPU(Context &ctx, const void *data, WGPUBuffer buffer, size_t size) {
  wgpuQueueWriteBuffer(ctx.queue, buffer, 0, data, size);
}
void ToGPU(Context &ctx, const float *data, Tensor &tensor) {
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
}
// Separate this out since WGPUCommandBuffer is destroyed upon submission
void ResetCommandBuffer(WGPUDevice &device, const Shape &nThreads, Kernel &op) {
  log(kDefLog, kInfo, "Create command buffer 0x%x", op.commandBuffer);
  {
    WGPUCommandEncoder commandEncoder =
        wgpuDeviceCreateCommandEncoder(device, nullptr);
    WGPUComputePassEncoder computePassEncoder =
        wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);
    wgpuComputePassEncoderSetPipeline(computePassEncoder, op.computePipeline);
    wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, op.bindGroup, 0,
                                       nullptr);
    log(kDefLog, kInfo, "Dispatching workgroups for number of threads = %s",
        ToString(nThreads).c_str());
    wgpuComputePassEncoderDispatchWorkgroups(
        computePassEncoder, op.nWorkgroups[0], op.nWorkgroups[1],
        op.nWorkgroups[2]);
    wgpuComputePassEncoderEnd(computePassEncoder);
    op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
    check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
  }
  op.promise = std::promise<void>();
  op.future = op.promise.get_future();
}
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
        computePassEncoder,
        multiKernel.nWorkgroups[shaderIdx][0],
        multiKernel.nWorkgroups[shaderIdx][1],
        multiKernel.nWorkgroups[shaderIdx][2]);
    wgpuComputePassEncoderEnd(computePassEncoder);
  }
  log(kDefLog, kInfo, "Finish command encoder");
  multiKernel.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
  check(multiKernel.commandBuffer, "Create command buffer", __FILE__, __LINE__);
}
Kernel CreateKernel(Context &ctx, const ShaderCode &shader,
                    const Tensor *inputs, size_t numInputs,
                    const Tensor &output, const Shape &nThreads,
                    const void *params, size_t paramsSize) {
  assert(nThreads.rank == 3);
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
  // unless ParamsType != NoParam
  if (paramsSize > 0) {
    numBuffers += 1;            // parameters buffer
    paramIndex = numInputs + 1; // == numBuffers - 1
    assert(outputIndex == numBuffers - 2);
    assert(paramIndex == numBuffers - 1);
  }
  op.buffers = std::make_unique<WGPUBuffer[]>(numBuffers);
  op.bufferSizes = std::make_unique<size_t[]>(numBuffers);
  op.numBuffers = numBuffers;
  op.numInputs = numInputs;
  log(kDefLog, kInfo, "Create the bind group layout");
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
  if (paramsSize > 0) {
    log(kDefLog, kInfo, "Create layout entry for the params buffer");
    // Create layout entry for the params buffer
    bgLayoutEntries[paramIndex] = WGPUBindGroupLayoutEntry{
        .binding = static_cast<uint32_t>(paramIndex),
        .visibility = WGPUShaderStage_Compute,
        .buffer =
            WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Uniform,
                .minBindingSize = paramsSize,
            },
    };
  }

  WGPUBindGroupLayoutDescriptor bgLayoutDesc = {
      .entryCount = static_cast<uint32_t>(bgLayoutEntries.size()),
      .entries = bgLayoutEntries.data(),
  };
  WGPUBindGroupLayout bgLayout =
      wgpuDeviceCreateBindGroupLayout(device, &bgLayoutDesc);

  log(kDefLog, kInfo, "Create input and output buffers");
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
  if (paramsSize > 0) {
    WGPUBufferDescriptor paramsBufferDesc = {
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size = paramsSize,
        .mappedAtCreation = false,
    };
    op.buffers[paramIndex] = wgpuDeviceCreateBuffer(device, &paramsBufferDesc);
    op.bufferSizes[paramIndex] = paramsSize;

    wgpuQueueWriteBuffer(queue, op.buffers[paramIndex], 0, params, paramsSize);
    log(kDefLog, kInfo, "Params buffer written");
  } else {
    log(kDefLog, kInfo, "No params buffer needed");
  }

  log(kDefLog, kInfo, "Create the bind group");
  std::vector<WGPUBindGroupEntry> bindGroupEntries(numBuffers);
  for (size_t i = 0; i <= numInputs; ++i) { // <= for output buffer
    bindGroupEntries[i] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(i),
        .buffer = op.buffers[i],
        .offset = 0,
        .size = op.bufferSizes[i],
    };
  }
  if (paramsSize > 0) {
    bindGroupEntries[paramIndex] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(numBuffers - 1),
        .buffer = op.buffers[paramIndex],
        .offset = 0,
        .size = paramsSize,
    };
  }
  WGPUBindGroupDescriptor bindGroupDesc = {
      .layout = bgLayout,
      .entryCount = static_cast<uint32_t>(bindGroupEntries.size()),
      .entries = bindGroupEntries.data(),
  };
  op.bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

  log(kDefLog, kInfo, "Create the readback buffer");
  {
    WGPUBufferDescriptor readbackBufferDescriptor = {
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = op.bufferSizes[outputIndex],
    };
    op.readbackBuffer =
        wgpuDeviceCreateBuffer(device, &readbackBufferDescriptor);
  }

  log(kDefLog, kInfo, "Create the compute multiKernel");
  {
    WGPUPipelineLayout multiKernelLayout;
    WGPUPipelineLayoutDescriptor multiKernelLayoutDesc = {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bgLayout,
    };
    multiKernelLayout =
        wgpuDeviceCreatePipelineLayout(device, &multiKernelLayoutDesc);
    WGPUShaderModuleWGSLDescriptor wgslDesc = {
        .code = shader.data.c_str(),
    };
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    WGPUShaderModuleDescriptor shaderModuleDesc = {};
    shaderModuleDesc.nextInChain = &wgslDesc.chain;
    shaderModuleDesc.label = "shader";
    WGPUComputePipelineDescriptor computePipelineDesc = {};
    computePipelineDesc.layout = multiKernelLayout;
    computePipelineDesc.compute.module =
        wgpuDeviceCreateShaderModule(device, &shaderModuleDesc);
    computePipelineDesc.compute.entryPoint = "main";
    op.computePipeline =
        wgpuDeviceCreateComputePipeline(device, &computePipelineDesc);
    check(op.computePipeline, "Create compute multiKernel", __FILE__, __LINE__);
  }
  op.nWorkgroups = {
      (nThreads[0] + (shader.workgroupSize[0] - 1)) / shader.workgroupSize[0],
      (nThreads[1] + (shader.workgroupSize[1] - 1)) / shader.workgroupSize[1],
      (nThreads[2] + (shader.workgroupSize[2] - 1)) / shader.workgroupSize[2]};
  ResetCommandBuffer(device, nThreads, op);
  op.callbackData = {op.readbackBuffer, op.outputSize, nullptr, &op.promise};
  ctx.kernelPool.data.insert(&op);
  log(kDefLog, kInfo, "Exiting CreateKernel");
  return op;
}

// Convenience wrapper: params type is statically templated instead of void*
template <typename ParamsType = NoParam>
Kernel CreateKernel(Context &ctx, const ShaderCode &shader,
                    const Tensor *inputs, size_t numInputs,
                    const Tensor &output, const Shape &nThreads,
                    const ParamsType &params = ParamsType{}) {
  if constexpr (!IsNoParam<ParamsType>) {
    log(kDefLog, kInfo, "Using params of size %d bytes", sizeof(ParamsType));
    return CreateKernel(ctx, shader, inputs, numInputs, output, nThreads,
                        reinterpret_cast<const void *>(&params),
                        sizeof(ParamsType));
  } else {
    log(kDefLog, kInfo, "No params");
    return CreateKernel(ctx, shader, inputs, numInputs, output, nThreads,
                        nullptr, 0);
  }
}

// Convenience wrapper: inputs is Tensors static collection instead of a
// pointer
template <typename ParamsType = NoParam, size_t numInputs>
Kernel CreateKernel(Context &ctx, const ShaderCode &shader,
                    const Tensors<numInputs> &inputs,
                    const Tensor &output, const Shape &nThreads,
                    const ParamsType &params = ParamsType{}) {
  // first .data gets the array, second .data() gets the pointer
  return CreateKernel<ParamsType>(ctx, shader, inputs.data.data(), numInputs,
                                  output, nThreads, params);
}

// Convenience wrapper: specialization for single input passed by reference
template <typename ParamsType = NoParam>
Kernel CreateKernel(Context &ctx, const ShaderCode &shader,
                    const Tensor &input, const Tensor &output,
                    const Shape &nThreads,
                    const ParamsType &params = ParamsType{}) {
  return CreateKernel(ctx, shader, &input, 1, output, nThreads, params);
}

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
    log(kDefLog, kInfo, "Create the bind group layout");
    std::vector<WGPUBindGroupLayoutEntry> bgLayoutEntries(
        multiKernel.numBuffers[shaderIdx]);
    for (size_t i = 0; i < multiKernel.numBuffers[shaderIdx]; ++i) {
      log(kDefLog, kInfo, "Create layout entry for buffer %d", i);
      log(kDefLog, kInfo, "i %d outputIndex %d i == paramIndex ? %d", i,
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

    log(kDefLog, kInfo, "Create input and output buffers");
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
      log(kDefLog, kInfo, "Create the params buffer at bufferIndex %d",
          bufferIndex);
      multiKernel.buffers[bufferIndex] =
          wgpuDeviceCreateBuffer(device, &paramsBufferDesc);
      multiKernel.bufferSizes[bufferIndex] = desc.paramSizes[shaderIdx];
      bufferIndex++;
      log(kDefLog, kInfo, "Params buffer written");
    } else {
      log(kDefLog, kInfo, "No params buffer needed");
    }
    {
      std::vector<WGPUBindGroupEntry> bindGroupEntries(
          multiKernel.numBuffers[shaderIdx]);
      log(kDefLog, kInfo, "Number of buffers: %d",
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
      (desc.nThreads[shaderIdx][0] + (desc.shader[shaderIdx].workgroupSize[0] - 1)) /
          desc.shader[shaderIdx].workgroupSize[0],
      (desc.nThreads[shaderIdx][1] + (desc.shader[shaderIdx].workgroupSize[1] - 1)) /
          desc.shader[shaderIdx].workgroupSize[1],
      (desc.nThreads[shaderIdx][2] + (desc.shader[shaderIdx].workgroupSize[2] - 1)) /
          desc.shader[shaderIdx].workgroupSize[2]
    };
  }
  ResetMultiCommandBuffer(device, multiKernel);

  check(multiKernel.commandBuffer, "Create command buffer", __FILE__, __LINE__);

  log(kDefLog, kInfo, "Create the readback buffer");
  {
    WGPUBufferDescriptor readbackBufferDescriptor = {
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = multiKernel.outputSize[multiKernel.numShaders - 1],
    };
    multiKernel.readbackBuffer =
        wgpuDeviceCreateBuffer(device, &readbackBufferDescriptor);
  }

  // Set up promise and future for asynchronous handling
  multiKernel.promise = std::promise<void>();
  multiKernel.future = multiKernel.promise.get_future();

  multiKernel.callbackData =
      CallbackDataDyn{multiKernel.readbackBuffer,
                      multiKernel.outputSize[multiKernel.numShaders - 1],
                      nullptr, &multiKernel.promise};

  ctx.kernelPool.multiData.insert(&multiKernel);

  return multiKernel;
}

void DispatchKernel(Context &ctx, Kernel &op) {
  // Submit the command buffer
  wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);
  wgpuQueueOnSubmittedWorkDone(
      ctx.queue,
      [](WGPUQueueWorkDoneStatus status, void *callbackData) {
        log(kDefLog, kInfo, "QueueOnSubmittedWorkDone status success ? %d",
            WGPUQueueWorkDoneStatus_Success == status);
        check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done",
              __FILE__, __LINE__);
        const auto *data = static_cast<CallbackDataDyn *>(callbackData);
        data->promise->set_value();
      },
      &op.callbackData);
}

void DispatchMultiKernel(Context &ctx, MultiKernel &multiKernel) {
  wgpuQueueSubmit(ctx.queue, 1, &multiKernel.commandBuffer);
  wgpuQueueOnSubmittedWorkDone(
      ctx.queue,
      [](WGPUQueueWorkDoneStatus status, void *callbackData) {
        log(kDefLog, kInfo, "QueueOnSubmittedWorkDone status: %d",
            WGPUQueueWorkDoneStatus_Success == status);
        check(status == WGPUQueueWorkDoneStatus_Success,
              "Check queue work success", __FILE__, __LINE__);
        const auto *data = static_cast<CallbackDataDyn *>(callbackData);
        data->promise->set_value();
      },
      &multiKernel.callbackData);
}

} // namespace gpu

#endif // GPU_H
