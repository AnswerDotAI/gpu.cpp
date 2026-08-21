#ifndef GPU_HPP
#define GPU_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include "webgpu/webgpu.h"
#include "webgpu/webgpu_cpp.h"
#else
// Include Dawn's extended C header before the generated C++ facade. This makes
// the opt-in SPIR-V source descriptor available without dropping down to C.
#include "dawn/webgpu.h"
#include "dawn/webgpu_cpp.h"
#endif

#include "utils/logging.hpp"

namespace gpu {

struct Shape {
  static constexpr size_t kMaxRank = 8;
  std::array<size_t, kMaxRank> data{};
  size_t rank = 0;

  Shape() = default;
  Shape(std::initializer_list<size_t> dimensions) : rank(dimensions.size()) {
    if (rank > kMaxRank) throw std::invalid_argument("tensor rank exceeds 8");
    std::copy(dimensions.begin(), dimensions.end(), data.begin());
  }

  size_t &operator[](size_t index) {
    if (index >= rank) throw std::out_of_range("shape index");
    return data[index];
  }
  const size_t &operator[](size_t index) const {
    if (index >= rank) throw std::out_of_range("shape index");
    return data[index];
  }
};

inline size_t size(const Shape &shape) {
  size_t result = 1;
  for (size_t i = 0; i < shape.rank; ++i) result *= shape[i];
  return result;
}

inline size_t ceilDiv(size_t value, size_t divisor) {
  if (!divisor) throw std::invalid_argument("division by zero");
  return value / divisor + (value % divisor != 0);
}

inline Shape ceilDiv(const Shape &value, const Shape &divisor) {
  if (value.rank != divisor.rank)
    throw std::invalid_argument("shape ranks differ");
  Shape result = value;
  for (size_t i = 0; i < value.rank; ++i)
    result[i] = ceilDiv(value[i], divisor[i]);
  return result;
}

enum NumType { kf16, kf32, ki32 };

inline size_t sizeBytes(NumType type) {
  switch (type) {
  case kf16: return sizeof(uint16_t);
  case kf32: return sizeof(float);
  case ki32: return sizeof(int32_t);
  }
  throw std::invalid_argument("unknown numeric type");
}

inline std::string toString(NumType type) {
  switch (type) {
  case kf16: return "f16";
  case kf32: return "f32";
  case ki32: return "i32";
  }
  throw std::invalid_argument("unknown numeric type");
}

inline std::string toString(const Shape &shape) {
  std::string result;
  for (size_t i = 0; i < shape.rank; ++i) {
    if (i) result += ", ";
    result += std::to_string(shape[i]);
  }
  return result;
}

inline void replaceAll(std::string &text, std::string_view from,
                       std::string_view to) {
  if (from.empty()) return;
  for (size_t pos = 0; (pos = text.find(from, pos)) != std::string::npos;
       pos += to.size())
    text.replace(pos, from.size(), to);
}

inline void replaceAll(
    std::string &text,
    std::initializer_list<std::pair<std::string_view, std::string_view>> values) {
  for (const auto &[from, to] : values) replaceAll(text, from, to);
}

struct WGSL {
  std::string code;
  Shape workgroupSize{256, 1, 1};
  std::string label = "kernel";
  std::string entryPoint = "main";

  WGSL(std::string code = {}, size_t workgroupSize = 256,
       NumType precision = kf32)
      : WGSL(std::move(code), Shape{workgroupSize, 1, 1}, precision) {}

  WGSL(std::string code, Shape workgroupSize, NumType precision = kf32)
      : code(std::move(code)), workgroupSize(workgroupSize) {
    if (this->workgroupSize.rank != 3)
      throw std::invalid_argument("workgroup size must have three dimensions");
    if (precision == kf16) this->code = "enable f16;\n" + this->code;
    replaceAll(this->code, "{{workgroupSize}}", toString(this->workgroupSize));
    replaceAll(this->code, "{{precision}}", toString(precision));
  }

  WGSL(std::string code, Shape workgroupSize, NumType precision,
       const Shape &totalWorkgroups)
      : WGSL(std::move(code), workgroupSize, precision) {
    replaceAll(this->code, "{{totalWorkgroups}}", toString(totalWorkgroups));
  }
};

struct SPIRV {
  std::vector<uint32_t> code;
  std::string label = "kernel";
  std::string entryPoint = "main";
};

using Shader = std::variant<WGSL, SPIRV>;

struct Tensor {
  wgpu::Buffer buffer;
  Shape shape;
  NumType type = kf32;
  size_t bytes = 0;
};

struct TensorView {
  const Tensor &tensor;
  size_t offset = 0;
  size_t bytes = 0;
};

struct Binding {
  wgpu::Buffer buffer;
  size_t offset;
  size_t bytes;
  wgpu::BufferBindingType type;
};

inline Binding read(const Tensor &tensor) {
  return {tensor.buffer, 0, tensor.bytes,
          wgpu::BufferBindingType::ReadOnlyStorage};
}

inline Binding readWrite(const Tensor &tensor) {
  return {tensor.buffer, 0, tensor.bytes, wgpu::BufferBindingType::Storage};
}

inline Binding read(const TensorView &view) {
  if (view.offset > view.tensor.bytes)
    throw std::out_of_range("tensor view starts beyond its buffer");
  const size_t bytes = view.bytes ? view.bytes : view.tensor.bytes - view.offset;
  if (bytes > view.tensor.bytes - view.offset)
    throw std::out_of_range("tensor view exceeds its buffer");
  return {view.tensor.buffer, view.offset, bytes,
          wgpu::BufferBindingType::ReadOnlyStorage};
}

inline Binding readWrite(const TensorView &view) {
  auto binding = read(view);
  binding.type = wgpu::BufferBindingType::Storage;
  return binding;
}

template <size_t N> struct Bindings {
  std::array<Binding, N> data;

  template <typename... T>
    requires(sizeof...(T) == N && (std::same_as<std::decay_t<T>, Binding> && ...))
  explicit Bindings(T &&...bindings)
      : data{std::forward<T>(bindings)...} {}

  const Binding &operator[](size_t index) const { return data.at(index); }
};

template <typename... T> Bindings(T &&...) -> Bindings<sizeof...(T)>;

struct ContextOptions {
  wgpu::PowerPreference powerPreference = wgpu::PowerPreference::HighPerformance;
#if defined(__APPLE__)
  wgpu::BackendType backend = wgpu::BackendType::Metal;
#elif defined(__linux__)
  wgpu::BackendType backend = wgpu::BackendType::Vulkan;
#else
  wgpu::BackendType backend = wgpu::BackendType::Undefined;
#endif
  wgpu::FeatureLevel featureLevel = wgpu::FeatureLevel::Core;
  std::vector<wgpu::FeatureName> requiredFeatures;
  std::optional<wgpu::Limits> requiredLimits;
  bool enableSPIRV = false;
};

namespace detail {

inline std::string string(wgpu::StringView value) {
  return std::string(static_cast<std::string_view>(value));
}

struct ErrorState {
  std::mutex mutex;
  std::string message;

  void set(std::string value) {
    std::lock_guard lock(mutex);
    message = std::move(value);
  }

  std::string take() {
    std::lock_guard lock(mutex);
    return std::exchange(message, {});
  }
};

inline constexpr wgpu::CallbackMode completionMode() {
#if defined(__EMSCRIPTEN__)
  return wgpu::CallbackMode::WaitAnyOnly;
#else
  return wgpu::CallbackMode::AllowProcessEvents;
#endif
}

inline constexpr wgpu::CallbackMode persistentMode() {
#if defined(__EMSCRIPTEN__)
  return wgpu::CallbackMode::AllowSpontaneous;
#else
  return wgpu::CallbackMode::AllowProcessEvents;
#endif
}

template <typename T>
T await(const wgpu::Instance &instance, std::future<T> &result,
        wgpu::Future event) {
#if defined(__EMSCRIPTEN__)
  if (instance.WaitAny(event, UINT64_MAX) != wgpu::WaitStatus::Success)
    throw std::runtime_error("could not wait for WebGPU operation");
#else
  (void)event;
  while (result.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready) {
    instance.ProcessEvents();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
#endif
  return result.get();
}

} // namespace detail

struct Context {
  std::shared_ptr<detail::ErrorState> errors =
      std::make_shared<detail::ErrorState>();
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Queue queue;

  Context() = default;
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = default;
  Context &operator=(Context &&) = default;

  void check() {
#if !defined(__EMSCRIPTEN__)
    instance.ProcessEvents();
#endif
    if (auto message = errors->take(); !message.empty())
      throw std::runtime_error(message);
  }
};

class Future {
  std::future<void> result;
  wgpu::Future event;

public:
  Future() = default;
  Future(std::future<void> result, wgpu::Future event)
      : result(std::move(result)), event(event) {}
  Future(const Future &) = delete;
  Future &operator=(const Future &) = delete;
  Future(Future &&) = default;
  Future &operator=(Future &&) = default;

  friend void wait(Context &, Future &);
};

inline Context createContext(const ContextOptions &options = {}) {
  Context context;
  std::vector<wgpu::InstanceFeatureName> instanceFeatures;
#if defined(__EMSCRIPTEN__)
  if (options.enableSPIRV)
    throw std::invalid_argument("SPIR-V input is unavailable in browsers");
  instanceFeatures.push_back(wgpu::InstanceFeatureName::TimedWaitAny);
#else
  if (options.enableSPIRV)
    instanceFeatures.push_back(wgpu::InstanceFeatureName::ShaderSourceSPIRV);
#endif

  wgpu::InstanceDescriptor instanceDescriptor{};
  instanceDescriptor.requiredFeatureCount = instanceFeatures.size();
  instanceDescriptor.requiredFeatures = instanceFeatures.data();
  context.instance = wgpu::CreateInstance(&instanceDescriptor);
  if (!context.instance) throw std::runtime_error("could not create WebGPU instance");

  wgpu::RequestAdapterOptions adapterOptions{};
  adapterOptions.featureLevel = options.featureLevel;
  adapterOptions.powerPreference = options.powerPreference;
  adapterOptions.backendType = options.backend;
  auto adapterPromise = std::make_shared<std::promise<wgpu::Adapter>>();
  auto adapterFuture = adapterPromise->get_future();
  auto adapterEvent = context.instance.RequestAdapter(
      &adapterOptions, detail::completionMode(),
      [adapterPromise](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                       wgpu::StringView message) {
        if (status == wgpu::RequestAdapterStatus::Success)
          adapterPromise->set_value(std::move(adapter));
        else
          adapterPromise->set_exception(std::make_exception_ptr(
              std::runtime_error("could not request WebGPU adapter: " +
                                 detail::string(message))));
      });
  context.adapter =
      detail::await(context.instance, adapterFuture, adapterEvent);

  wgpu::DeviceDescriptor deviceDescriptor{};
  deviceDescriptor.requiredFeatureCount = options.requiredFeatures.size();
  deviceDescriptor.requiredFeatures = options.requiredFeatures.data();
  deviceDescriptor.requiredLimits = options.requiredLimits
                                        ? &*options.requiredLimits
                                        : nullptr;
  deviceDescriptor.SetUncapturedErrorCallback(
      [](const wgpu::Device &, wgpu::ErrorType, wgpu::StringView message,
         detail::ErrorState *errors) {
        errors->set(detail::string(message));
      }, context.errors.get());
  deviceDescriptor.SetDeviceLostCallback(
      detail::persistentMode(),
      [](const wgpu::Device &, wgpu::DeviceLostReason reason,
         wgpu::StringView message, detail::ErrorState *errors) {
        if (reason != wgpu::DeviceLostReason::Destroyed)
          errors->set("WebGPU device lost: " + detail::string(message));
      }, context.errors.get());

  auto devicePromise = std::make_shared<std::promise<wgpu::Device>>();
  auto deviceFuture = devicePromise->get_future();
  auto deviceEvent = context.adapter.RequestDevice(
      &deviceDescriptor, detail::completionMode(),
      [devicePromise](wgpu::RequestDeviceStatus status, wgpu::Device device,
                      wgpu::StringView message) {
        if (status == wgpu::RequestDeviceStatus::Success)
          devicePromise->set_value(std::move(device));
        else
          devicePromise->set_exception(std::make_exception_ptr(
              std::runtime_error("could not request WebGPU device: " +
                                 detail::string(message))));
      });
  context.device = detail::await(context.instance, deviceFuture, deviceEvent);
  context.queue = context.device.GetQueue();
  return context;
}

inline Tensor createTensor(Context &context, const Shape &shape, NumType type) {
  Tensor tensor{.shape = shape, .type = type,
                .bytes = size(shape) * sizeBytes(type)};
  if (!tensor.bytes) throw std::invalid_argument("tensor cannot be empty");
  wgpu::BufferDescriptor descriptor{};
  descriptor.label = "gpu.cpp tensor";
  descriptor.size = tensor.bytes;
  descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                     wgpu::BufferUsage::CopyDst;
  tensor.buffer = context.device.CreateBuffer(&descriptor);
  context.check();
  return tensor;
}

template <typename T>
Tensor createTensor(Context &context, const Shape &shape, NumType type,
                    std::span<const T> values) {
  if (sizeof(T) != sizeBytes(type))
    throw std::invalid_argument("host and tensor element sizes differ");
  Tensor tensor{.shape = shape, .type = type,
                .bytes = size(shape) * sizeBytes(type)};
  if (!tensor.bytes) throw std::invalid_argument("tensor cannot be empty");
  if (values.size_bytes() != tensor.bytes)
    throw std::invalid_argument("host data size does not match tensor shape");
  wgpu::BufferDescriptor descriptor{};
  descriptor.label = "gpu.cpp initialized tensor";
  descriptor.size = tensor.bytes;
  descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                     wgpu::BufferUsage::CopyDst;
  descriptor.mappedAtCreation = true;
  tensor.buffer = context.device.CreateBuffer(&descriptor);
  auto *mapped = tensor.buffer.GetMappedRange(0, tensor.bytes);
  if (!mapped) throw std::runtime_error("could not map initialized tensor");
  std::memcpy(mapped, values.data(), tensor.bytes);
  tensor.buffer.Unmap();
  context.check();
  return tensor;
}

template <typename T>
Tensor createTensor(Context &context, const Shape &shape, NumType type,
                    const std::vector<T> &values) {
  return createTensor(context, shape, type, std::span<const T>(values));
}

template <typename T, size_t N>
Tensor createTensor(Context &context, const Shape &shape, NumType type,
                    const std::array<T, N> &values) {
  return createTensor(context, shape, type, std::span<const T>(values));
}

template <typename T>
void toGPU(Context &context, const Tensor &tensor, std::span<const T> values) {
  if (sizeof(T) != sizeBytes(tensor.type))
    throw std::invalid_argument("host and tensor element sizes differ");
  if (values.size_bytes() != tensor.bytes)
    throw std::invalid_argument("host data size does not match tensor");
  context.queue.WriteBuffer(tensor.buffer, 0, values.data(), values.size_bytes());
  context.check();
}

template <typename T>
void toGPU(Context &context, const Tensor &tensor, const std::vector<T> &values) {
  toGPU(context, tensor, std::span<const T>(values));
}

template <typename T, size_t N>
void toGPU(Context &context, const Tensor &tensor,
           const std::array<T, N> &values) {
  toGPU(context, tensor, std::span<const T>(values));
}

struct Kernel {
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroup bindGroup;
  Shape workgroups{1, 1, 1};
  wgpu::Buffer parameters;
  size_t parameterBytes = 0;
};

namespace detail {

inline wgpu::ShaderModule createShaderModule(Context &context,
                                              const Shader &shader) {
  return std::visit(
      [&](const auto &source) -> wgpu::ShaderModule {
        using T = std::decay_t<decltype(source)>;
        wgpu::ShaderModuleDescriptor descriptor{};
        descriptor.label = std::string_view(source.label);
        if constexpr (std::same_as<T, WGSL>) {
          wgpu::ShaderSourceWGSL chained{wgpu::ShaderSourceWGSL::Init{
              nullptr, std::string_view(source.code)}};
          descriptor.nextInChain = &chained;
          return context.device.CreateShaderModule(&descriptor);
        } else {
#if defined(__EMSCRIPTEN__)
          throw std::invalid_argument("SPIR-V input is unavailable in browsers");
#else
          if (source.code.size() > UINT32_MAX)
            throw std::invalid_argument("SPIR-V module is too large");
          wgpu::ShaderSourceSPIRV chained{wgpu::ShaderSourceSPIRV::Init{
              nullptr, static_cast<uint32_t>(source.code.size()),
              source.code.data()}};
          descriptor.nextInChain = &chained;
          return context.device.CreateShaderModule(&descriptor);
#endif
        }
      },
      shader);
}

inline std::string entryPoint(const Shader &shader) {
  return std::visit([](const auto &source) { return source.entryPoint; }, shader);
}

inline std::string label(const Shader &shader) {
  return std::visit([](const auto &source) { return source.label; }, shader);
}

inline Kernel createKernel(Context &context, const Shader &shader,
                           std::span<const Binding> bindings,
                           const Shape &workgroups,
                           std::span<const std::byte> parameters) {
  if (workgroups.rank != 3)
    throw std::invalid_argument("dispatch size must have three dimensions");

  context.device.PushErrorScope(wgpu::ErrorFilter::Validation);
  auto module = createShaderModule(context, shader);

  std::vector<wgpu::BindGroupLayoutEntry> layoutEntries(bindings.size());
  std::vector<wgpu::BindGroupEntry> groupEntries(bindings.size());
  for (size_t i = 0; i < bindings.size(); ++i) {
    layoutEntries[i].binding = i;
    layoutEntries[i].visibility = wgpu::ShaderStage::Compute;
    layoutEntries[i].buffer.type = bindings[i].type;
    layoutEntries[i].buffer.minBindingSize = bindings[i].bytes;
    groupEntries[i].binding = i;
    groupEntries[i].buffer = bindings[i].buffer;
    groupEntries[i].offset = bindings[i].offset;
    groupEntries[i].size = bindings[i].bytes;
  }

  Kernel kernel{.workgroups = workgroups};
  if (!parameters.empty()) {
    const size_t alignedBytes = (parameters.size() + 15) & ~size_t(15);
    wgpu::BufferDescriptor bufferDescriptor{};
    bufferDescriptor.label = "gpu.cpp kernel parameters";
    bufferDescriptor.size = alignedBytes;
    bufferDescriptor.usage =
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    kernel.parameters = context.device.CreateBuffer(&bufferDescriptor);
    kernel.parameterBytes = parameters.size();
    context.queue.WriteBuffer(kernel.parameters, 0, parameters.data(),
                              parameters.size());

    wgpu::BindGroupLayoutEntry layout{};
    layout.binding = layoutEntries.size();
    layout.visibility = wgpu::ShaderStage::Compute;
    layout.buffer.type = wgpu::BufferBindingType::Uniform;
    layout.buffer.minBindingSize = alignedBytes;
    layoutEntries.push_back(layout);

    wgpu::BindGroupEntry entry{};
    entry.binding = groupEntries.size();
    entry.buffer = kernel.parameters;
    entry.size = alignedBytes;
    groupEntries.push_back(entry);
  }

  wgpu::BindGroupLayoutDescriptor bindLayoutDescriptor{};
  bindLayoutDescriptor.label = "gpu.cpp bind group layout";
  bindLayoutDescriptor.entryCount = layoutEntries.size();
  bindLayoutDescriptor.entries = layoutEntries.data();
  auto bindLayout = context.device.CreateBindGroupLayout(&bindLayoutDescriptor);

  wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{};
  pipelineLayoutDescriptor.label = "gpu.cpp pipeline layout";
  pipelineLayoutDescriptor.bindGroupLayoutCount = 1;
  pipelineLayoutDescriptor.bindGroupLayouts = &bindLayout;
  auto pipelineLayout =
      context.device.CreatePipelineLayout(&pipelineLayoutDescriptor);

  const auto entry = entryPoint(shader);
  const auto pipelineLabel = label(shader);
  wgpu::ComputePipelineDescriptor pipelineDescriptor{};
  pipelineDescriptor.label = std::string_view(pipelineLabel);
  pipelineDescriptor.layout = pipelineLayout;
  pipelineDescriptor.compute.module = module;
  pipelineDescriptor.compute.entryPoint = std::string_view(entry);
  kernel.pipeline = context.device.CreateComputePipeline(&pipelineDescriptor);

  wgpu::BindGroupDescriptor bindGroupDescriptor{};
  bindGroupDescriptor.label = "gpu.cpp bind group";
  bindGroupDescriptor.layout = bindLayout;
  bindGroupDescriptor.entryCount = groupEntries.size();
  bindGroupDescriptor.entries = groupEntries.data();
  kernel.bindGroup = context.device.CreateBindGroup(&bindGroupDescriptor);

  auto errorPromise = std::make_shared<std::promise<std::string>>();
  auto errorFuture = errorPromise->get_future();
  auto errorEvent = context.device.PopErrorScope(
      detail::completionMode(),
      [errorPromise](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type,
                     wgpu::StringView message) {
        if (status != wgpu::PopErrorScopeStatus::Success)
          errorPromise->set_value("could not read WebGPU validation result: " +
                                  string(message));
        else if (type != wgpu::ErrorType::NoError)
          errorPromise->set_value(string(message));
        else
          errorPromise->set_value({});
      });
  if (auto error = await(context.instance, errorFuture, errorEvent);
      !error.empty())
    throw std::runtime_error(error);
  context.check();
  return kernel;
}

} // namespace detail

inline Kernel createKernel(
    Context &context, const Shader &shader, std::span<const Binding> bindings,
    const Shape &workgroups = {1, 1, 1},
    std::span<const std::byte> parameters = {}) {
  return detail::createKernel(context, shader, bindings, workgroups, parameters);
}

template <size_t N>
Kernel createKernel(Context &context, const Shader &shader,
                    const Bindings<N> &bindings,
                    const Shape &workgroups = {1, 1, 1}) {
  return detail::createKernel(context, shader, bindings.data, workgroups, {});
}

template <size_t N, typename Parameters>
  requires std::is_trivially_copyable_v<Parameters>
Kernel createKernel(Context &context, const Shader &shader,
                    const Bindings<N> &bindings, const Shape &workgroups,
                    const Parameters &parameters) {
  const auto *data = reinterpret_cast<const std::byte *>(&parameters);
  return detail::createKernel(context, shader, bindings.data, workgroups,
                              {data, sizeof(parameters)});
}

template <typename Parameters>
  requires std::is_trivially_copyable_v<Parameters>
void toGPU(Context &context, const Parameters &parameters, Kernel &kernel) {
  if (sizeof(parameters) != kernel.parameterBytes)
    throw std::invalid_argument("kernel parameter size changed");
  context.queue.WriteBuffer(kernel.parameters, 0, &parameters,
                            sizeof(parameters));
  context.check();
}

inline Future dispatchKernel(Context &context, const Kernel &kernel) {
  auto encoder = context.device.CreateCommandEncoder();
  auto pass = encoder.BeginComputePass();
  pass.SetPipeline(kernel.pipeline);
  pass.SetBindGroup(0, kernel.bindGroup);
  pass.DispatchWorkgroups(kernel.workgroups[0], kernel.workgroups[1],
                          kernel.workgroups[2]);
  pass.End();
  auto commands = encoder.Finish();
  if (!commands) throw std::runtime_error("could not encode WebGPU dispatch");
  context.queue.Submit(1, &commands);

  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  auto event = context.queue.OnSubmittedWorkDone(
      detail::completionMode(),
      [promise](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
        if (status == wgpu::QueueWorkDoneStatus::Success)
          promise->set_value();
        else
          promise->set_exception(std::make_exception_ptr(std::runtime_error(
              "WebGPU dispatch failed: " + detail::string(message))));
      });
  return {std::move(future), event};
}

inline void wait(Context &context, Future &future) {
  detail::await(context.instance, future.result, future.event);
  context.check();
}

template <typename T>
Future toCPU(Context &context, const Tensor &tensor, std::span<T> output) {
  if (sizeof(T) != sizeBytes(tensor.type))
    throw std::invalid_argument("host and tensor element sizes differ");
  if (output.size_bytes() != tensor.bytes)
    throw std::invalid_argument("host output size does not match tensor");

  wgpu::BufferDescriptor descriptor{};
  descriptor.label = "gpu.cpp readback";
  descriptor.size = tensor.bytes;
  descriptor.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  auto readback = context.device.CreateBuffer(&descriptor);
  auto encoder = context.device.CreateCommandEncoder();
  encoder.CopyBufferToBuffer(tensor.buffer, 0, readback, 0, tensor.bytes);
  auto commands = encoder.Finish();
  if (!commands) throw std::runtime_error("could not encode WebGPU readback");
  context.queue.Submit(1, &commands);

  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  auto event = readback.MapAsync(
      wgpu::MapMode::Read, 0, tensor.bytes,
      detail::completionMode(),
      [promise, readback, output](wgpu::MapAsyncStatus status,
                                  wgpu::StringView message) {
        if (status != wgpu::MapAsyncStatus::Success) {
          promise->set_exception(std::make_exception_ptr(std::runtime_error(
              "WebGPU readback failed: " + detail::string(message))));
          return;
        }
        std::memcpy(static_cast<void *>(output.data()),
                    readback.GetConstMappedRange(0, output.size_bytes()),
                    output.size_bytes());
        readback.Unmap();
        promise->set_value();
      });
  return {std::move(future), event};
}

template <typename T>
Future toCPU(Context &context, const Tensor &tensor, std::vector<T> &output) {
  return toCPU(context, tensor, std::span<T>(output));
}

template <typename T, size_t N>
Future toCPU(Context &context, const Tensor &tensor,
             std::array<T, N> &output) {
  return toCPU(context, tensor, std::span<T>(output));
}

} // namespace gpu

#endif
