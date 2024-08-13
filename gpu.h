#ifndef GPU_H
#define GPU_H

#include <array>
#include <cassert>
#include <cstring>
#include <future>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility> // std::pair
#include <vector>

#include "webgpu/webgpu.h"

#include "numeric_types/half.h"
#include "utils/logging.h"

#ifdef __EMSCRIPTEN__
#include "emscripten/emscripten.h"
#endif

namespace gpu {

/**
 * @brief Represents a buffer of values on the GPU.
 */
struct Array {
  WGPUBuffer buffer;
  WGPUBufferUsageFlags usage;
  size_t size; // in bytes
};

/**
 * @brief Represents the shape of a tensor.
 *
 * The rank of the tensor is the
 * number of dimensions in the shape. The data array stores the size of each
 * dimension. For now, we limit the rank to 8 to avoid dynamic allocation.
 *
 * @code
 * Shape shape = {256, 256};
 * @endcode
 */
struct Shape {
  static constexpr size_t kMaxRank = 8; // Maximum rank of a tensor, avoids
                                        // dynamic allocation for shape data
  std::array<size_t, kMaxRank> data = {0};
  size_t rank = 0;
  inline Shape() = default;
  inline Shape(std::initializer_list<size_t> dims) {
    assert(dims.size() <= kMaxRank);
    std::copy(dims.begin(), dims.end(), data.begin());
    rank = dims.size();
  }
  inline size_t &operator[](size_t index) {
    assert(index < rank);
    return data[index];
  }
  inline const size_t &operator[](size_t index) const {
    assert(index < rank);
    return data[index];
  }
};

/**
 * @brief Returns the number of elements in a tensor with the given shape,
 * which is equal to the product of the dimensions.
 * @param[in] shape Shape of the tensor
 * @return Number of elements in the tensor
 *
 * @code
 * size({256, 256}) -> 65536
 * @endcode
 */
inline size_t size(const Shape &shape) {
  size_t numels = 1;
  for (size_t i = 0; i < shape.rank; i++) {
    numels *= shape.data[i];
  }
  return numels;
}

/**
 * @brief Represents a tensor on the GPU, which is a buffer of values with a
 * shape.
 *
 * @code
 * Tensor tensor = createTensor(ctx, {256, 256}, kf32);
 * @endcode
 */
struct Tensor {
  Array data;
  Shape shape;
};

/**
 * @brief Represents a non-owning view into a tensor specifying an offset and a
 * subspan. This is useful for specifying a slice of a tensor on the GPU
 * without copying the data.
 *
 * @code
 * TensorView view = {tensor, 0, 256};
 * @endcode
 */
struct TensorView {
  Tensor data; // non-owning view
  size_t offset = 0;
  size_t span = 0;
};

/**
 * @brief Represents an ordered collection of WGPUBuffers (wrapped as tensors,
 * non-overlapping views, or arrays) for the purpose of binding them to a
 * kernel operation to make them accessible to the GPU kernel.
 *
 * The ordering of the bindings should match the binding indices in the WGSL
 * code.
 */
template <std::size_t N> struct Bindings {
  std::array<Tensor, N> data;
  std::array<size_t, N> viewOffsets;
  std::array<size_t, N> viewSpans;
  Bindings(const std::initializer_list<Tensor> &init) {
    std::copy(begin(init), end(init), begin(data));
    std::fill(begin(viewOffsets), end(viewOffsets), 0);
    for (size_t i = 0; i < N; ++i) {
      viewSpans[i] = data[i].data.size;
    }
  }

  Bindings(const std::initializer_list<TensorView> &init) {
    size_t i = 0;
    for (const auto &tv : init) {
      data[i] = tv.data;
      viewOffsets[i] = tv.offset;
      viewSpans[i] = tv.span;
      ++i;
    }
  }

  Bindings(const std::initializer_list<Array> &init) {
    std::copy(begin(init), end(init), begin(data));
    std::fill(begin(viewOffsets), end(viewOffsets), 0);
    for (size_t i = 0; i < N; ++i) {
      viewSpans[i] = data[i].size;
    }
  }

  Tensor &operator[](std::size_t index) { return data[index]; }
  const Tensor &operator[](std::size_t index) const { return data[index]; }
};

/**
 * @brief Deduction guide for Bindings
 */
template <std::size_t N> Bindings(std::array<Tensor, N>) -> Bindings<N>;
template <typename... Args> Bindings(Args...) -> Bindings<sizeof...(Args)>;

struct Context; // Forward declaration so that TensorPool can have a pointer to
                // Context

/**
 * @brief Represents a pool of tensors to manage GPU resources. The pool is
 * responsible for managing the lifetime of the tensors and freeing them when
 * the pool is destroyed.
 *
 * Most users do not need to interact with the TensorPool type, as there is a
 * member instance in the Context struct to simplify lifetime management of GPU
 * resources.
 */
struct TensorPool {
  inline TensorPool(Context *ctx) : ctx(ctx), data(){};
  Context *ctx;
  std::unordered_map<WGPUBuffer, Tensor> data;
  ~TensorPool();
};

enum NumType {
  kf16, // (experimental)
  kf32
};

/**
 * @brief Returns the number of bytes of a number type.
 */
inline size_t sizeBytes(const NumType &type) {
  switch (type) {
  case kf16:
    return sizeof(uint16_t);
  case kf32:
    return sizeof(float);
  default:
    LOG(kDefLog, kError, "Invalid NumType in size calculation.");
    return 0;
  }
}

/**
 * @brief Converts NumType to string.
 */
inline std::string toString(NumType type) {
  switch (type) {
  case kf16:
    return "f16";
  case kf32:
    return "f32";
  default:
    LOG(kDefLog, kError, "Invalid NumType in string conversion.");
    return "unknown";
  }
}

/**
 * @brief Converts Shape to string. The string formatting is meant to be
 * slotted into WGSL code (hence no additional parentheses or brackets).
 */
inline std::string toString(const Shape &shape) {
  std::string str;
  for (size_t i = 0; i < shape.rank; i++) {
    str += std::to_string(shape.data[i]);
    if (i < shape.rank - 1) {
      str += ", ";
    }
  }
  return str;
}

/**
 * @brief Converts size_t to string. Wraps std::to_string for consistency,
 * instead of having to remember to switch between std::to_string and toString
 * depending on the type.
 */
inline std::string toString(size_t value) { return std::to_string(value); }

/**
 * @brief simple in-place string replacement helper function for substituting
 * placeholders in a WGSL string template.
 *
 * Note this is not meant to be used in performance-critical code paths and
 * should be used ahead-of-time before any performance-critical codepath to
 * preprocess WGSL code strings.
 *
 * @param[in] str String to mutate with substitution replacements.
 * @param[in] from Substring to replace
 * @param[in] to Substring to replace with
 *
 * @code
 * replaceAll(str, "{{workgroupSize}}", "256");
 * @endcode
 */
inline void replaceAll(std::string &str, const std::string &from,
                       const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

/**
 * @brief KernelCode is the representation of WGSL GPU code with template
 * substitutions applied. It is a type around the code string with additional
 * metadata for workgroup size and precision since they are specified in the
 * WGSL code. Additionally, label and entryPoint are used by `createKernel()`
 * to specify the label and entry point of the kernel.
 */
struct KernelCode {
  /**
   * @brief Constructor to create a code object from a template
   * string and optional workgroup size and precision.
   *
   * @param[in] pData Shader template string with placeholders
   * @param[in] workgroupSize Shape of the workgroup. Unlike tensor shapes which
   * can be of arbitrary rank, workgroup size is always of rank 3 corresponding
   * to x y and z. workgroupSize is stored as a field in the KernelCode instance
   * that is returned by createShader().
   * @param[in] precision Data type precision to be substituted for
   * {{precision}} in the WGSL code. As with workgroupSize, precision is stored
   * as a field in the KernelCode instance that is returned by createShader().
   * @code
   * KernelCode code = {kShaderText, {256, 1, 1}, kf32};
   * @endcode
   */
  inline KernelCode(const std::string &pData = "", size_t workgroupSize = 256,
                    NumType precision = kf32)
      : data(pData), workgroupSize({workgroupSize, 1, 1}),
        precision(precision) {
    if (precision == kf16) {
      data = "enable f16;\n" + data;
    }
    replaceAll(data, "{{workgroupSize}}", toString({workgroupSize, 1, 1}));
    replaceAll(data, "{{precision}}", toString(precision));
    LOG(kDefLog, kTrace, "Shader code:\n%s", data.c_str());
  }

  /**
   * @brief Overload of the constructor to create a code object from a
   * template string and workgroup size. Unlike the main factory function,
   * this overload takes a single size_t workgroupSize parameter instead of a
   * 3D shape for the workgroup size and instantiates a 3D shape with the
   * workgroupSize in the x dimension and 1 in the y and z dimensions.
   *
   * @param[in] pData Shader template string with placeholders
   * @param[in] workgroupSize Workgroup size in the x dimension
   * @param[in] precision Data type precision for the shader
   *
   * @code
   * KernelCode code = {kPuzzle1, 256, kf32};
   * @endcode
   */

  inline KernelCode(const std::string &pData,
                    const Shape &workgroupSize = {256, 1, 1},
                    NumType precision = kf32)
      : data(pData), workgroupSize(workgroupSize), precision(precision) {
    if (precision == kf16) {
      data = "enable f16;\n" + data;
    }
    replaceAll(data, "{{workgroupSize}}", toString(workgroupSize));
    replaceAll(data, "{{precision}}", toString(precision));
    LOG(kDefLog, kInfo, "Shader code:\n%s", data.c_str());
  }
  std::string data;
  Shape workgroupSize;
  NumType precision = kf32;
  std::string label = "kernel";
  std::string entryPoint = "main";
};

/**
 * @brief Overload of the string replacement helper function to replace
 * multiple substrings in a string with multiple replacements.
 *
 * @param[in] str String to mutate with substitution replacements.
 * @param[in] reps Vector of pairs of substrings to replace and their
 * replacements.
 *
 * @code
 * replaceAll(str, {{"{{workgroupSize}}", "256"}, {"{{precision}}",
 * @endcode
 * "f32"}});
 */
inline void
replaceAll(std::string &str,
           const std::vector<std::pair<std::string, std::string>> &reps) {
  for (const auto &rep : reps) {
    replaceAll(str, rep.first, rep.second);
  }
}

/**
 * @brief Used for on-done callback data for asynchronous operations sduch as
 * kernel launching.
 */
struct CallbackData {
  WGPUBuffer buffer; // managed by owning Kernel
  size_t bufferSize;
  void *output; // non-owning, only for target memory in toCPU, not used for
                // kernel invocations
  std::promise<void> *promise;
  std::future<void> *future;
};

/**
 * @brief Staging buffer and callback data for copying data between the GPU and
 * CPU.
 */
struct CopyData {
  WGPUCommandBuffer commandBuffer;
  WGPUBuffer readbackBuffer;
  std::promise<void> promise;
  std::future<void> future;
};

/**
 * @brief Represents handles + metadata for a reusable kernel on the GPU.
 * The struct members can be divided into "consumed upon dispatch"
 * (commandBuffer) and reusable ahead-of-time setup (all other members).
 */
struct Kernel {
  std::unique_ptr<WGPUBuffer[]> buffers; // non-owning
  std::unique_ptr<size_t[]> bufferSizes;
  size_t numBindings;
  Shape nWorkgroups;
  WGPUBindGroup bindGroup;             // persists between submission
  WGPUComputePipeline computePipeline; // persists between submission
  WGPUCommandBuffer commandBuffer;     // destroyed upon submission
};

/**
 * @brief Operator implementation to make the Kernel type hashable.
 * @param[in] lhs First Kernel instance to compare
 * @param[in] rhs Second Kernel instance to compare
 * @return True if lhs < rhs, false otherwise
 */
inline bool operator<(const Kernel &lhs, const Kernel &rhs) {
  return lhs.commandBuffer < rhs.commandBuffer;
}

/**
 * @brief A pool of kernels to manage GPU resources. For simple use cases this
 * is instantiated as a member in the Context struct although it's possible to
 * have multiple resource pools of kernels in more complex scenarios.
 */
struct KernelPool {
  inline KernelPool(Context *ctx) : ctx(ctx), data() {}
  Context *ctx;
  std::set<Kernel *> data;
  inline ~KernelPool() {
    // Note : Some kernel resources such as commandBuffer are harvested by
    // queue submission, explicitly destroying readback and callback buffers
    // produces runtime errors.
    data.clear();
  }
};

inline void processEvents(const WGPUInstance &instance) {
#ifdef __EMSCRIPTEN__
  emscripten_sleep(0);
#else
  wgpuInstanceProcessEvents(instance);
#endif
}

/**
 * @brief Represents a GPU context, aggregates WebGPU API handles to interact
 * with the GPU including the instance, adapter, device, and queue.
 *
 * Additionally contains a TensorPool and KernelPool for managing GPU resources
 * to simplify lifetime management of GPU resources.
 */
struct Context {
  WGPUInstance instance;
  WGPUAdapter adapter;
  WGPUDevice device;
  WGPUQueue queue;
  TensorPool pool = TensorPool(this);
  KernelPool kernelPool = KernelPool(this);
  ~Context() {
    LOG(kDefLog, kTrace, "Destroying context");
    if (queue) {
      wgpuQueueRelease(queue);
    } else {
      LOG(kDefLog, kWarn, "Queue is null");
    }
    if (device) {
      wgpuDeviceRelease(device);
      processEvents(instance);
    } else {
      LOG(kDefLog, kWarn, "Device is null");
    }
    if (adapter) {
      wgpuAdapterRelease(adapter);
      processEvents(instance);
    } else {
      LOG(kDefLog, kWarn, "Adapter is null");
    }
    if (instance) {
      wgpuInstanceRelease(instance);
    } else {
      LOG(kDefLog, kWarn, "Instance is null");
    }
    LOG(kDefLog, kInfo, "Context destroyed");
  }
};

/**
 * @brief Tensor factory function to create a tensor (a Tensor type is simply
 * an Array with an N-dimensional  Shape specification) on the GPU. The tensor
 * is created with the given shape, data type, and usage flags, added to the
 * TensorPool, and returned.
 *
 * This is the core implementation which takes the minimal set of parameters in
 * terms of the raw WebGPU API, and is used by the other createTensor overloads
 * which provide more ergonomic interfaces.
 *
 * @param[in] pool TensorPool instance to manage the tensor
 * @param[in] device WGPUDevice instance to create the tensor on
 * @param[in] shape Shape of the tensor
 * @param[in] dtype Data type of the tensor (e.g. kf32)
 * @param[in] usage Usage flags for the tensor buffer
 * @return Tensor instance representing the created tensor
 *
 * @code
 * Tensor tensor = createTensor(pool, device, {256, 256}, kf32);
 * @endcode
 */
inline Tensor
createTensor(TensorPool &pool, WGPUDevice &device, const Shape &shape,
             NumType dtype,
             WGPUBufferUsageFlags usage = WGPUBufferUsage_Storage |
                                          WGPUBufferUsage_CopyDst |
                                          WGPUBufferUsage_CopySrc) {
  LOG(kDefLog, kTrace, "Creating tensor");
  size_t numElements = size(shape);
  size_t size = sizeBytes(dtype) * numElements;
  WGPUBufferDescriptor bufferDesc = {
      .usage = usage,
      .size = size,
  };
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
  pool.data[buffer] = Tensor{
      .data = Array{.buffer = buffer, .usage = usage, .size = size},
      .shape = shape,
  };
  return pool.data[buffer];
}

/**
 * @brief Overload of the tensor factory function to instantiate a tensor on
 * the GPU with a given shape and data type.
 *
 * Instead of taking the TensoPool and raw WebGPU API WGPUDevice and
 * WGPUBufferUsageFlags arguments, this is a convenience wrapper around the
 * core createTensor function which has default usage flags for a storage
 * buffer, and also takes in the Context object.
 *
 * instance instead of the narrower TensorPool object.
 * @param[in] ctx Context instance to manage the tensor
 * @param[in] shape Shape of the tensor
 * @param[in] dtype Data type of the tensor (e.g. kf32)
 * @return Tensor instance representing the created tensor
 *
 * @code
 * Tensor tensor = createTensor(ctx, {256, 256}, kf32);
 * @endcode
 */
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype) {
  return createTensor(ctx.pool, ctx.device, shape, dtype);
}

/**
 * @brief Overload of the tensor factory function to instantiate a tensor on
 * the GPU with a given shape, data type. This overload also takes initial
 * float* data to populate the tensor with.
 *
 * The data is assumed to be of size equal to the product of the dimensions in
 * the shape, and is copied to the GPU buffer.
 *
 * @param[in] ctx Context instance to manage the tensor
 * @param[in] shape Shape of the tensor
 * @param[in] dtype Data type of the tensor (e.g. kf32)
 * @param[in] data Initial data to populate the tensor with
 * @return Tensor instance representing the created tensor
 *
 * @code
 * Tensor tensor = createTensor(ctx, {256, 256}, kf32, data);
 * @endcode
 */
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           float *data) {
  assert(dtype == kf32);
  Tensor tensor =
      createTensor(ctx.pool, ctx.device, shape, dtype,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

/**
 * @brief Overload of the tensor factory function to instantiate a tensor on
 * the GPU with a given shape, data type. This overload also takes initial
 * half* data to populate the tensor with.
 *
 * The data is assumed to be of size equal to the product of the dimensions in
 * the shape, and is copied to the GPU buffer.
 *
 * @param[in] ctx Context instance to manage the tensor
 * @param[in] shape Shape of the tensor
 * @param[in] dtype Data type of the tensor (e.g. kf32)
 * @param[in] data Initial data to populate the tensor with
 * @return Tensor instance representing the created tensor
 *
 * @code
 * Tensor tensor = createTensor(ctx, {256, 256}, kf32, data);
 * @endcode
 */
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           half *data) {
  assert(dtype == kf16);
  Tensor tensor =
      createTensor(ctx.pool, ctx.device, shape, dtype,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

/**
 * @brief Frees a tensor resource and updates the tensor pool.
 *
 * Only needed if the use case requires manually managing resource lifetimes of
 * GPU tensors. For simple use cases, the TensorPool destructor will
 * automatically free all tensors.
 *
 * @param[in] pool TensorPool instance to manage the tensor
 * @param[in] tensor Tensor instance to free
 *
 * @code
 * FreeTensor(pool, tensor);
 * @endcode
 */
inline void FreeTensor(TensorPool &pool, Tensor tensor) {
  if (tensor.data.buffer) {
    wgpuBufferRelease(tensor.data.buffer);
  } else {
    LOG(kDefLog, kWarn, "Tried to free tensor with null buffer");
  }
  if (pool.data.find(tensor.data.buffer) != pool.data.end()) {
    pool.data.erase(tensor.data.buffer);
  } else {
    LOG(kDefLog, kWarn, "Tried to free tensor that was not in pool");
  }
}

/**
 * @brief Destructor for TensorPool which frees all tensors in the pool.
 */
inline TensorPool::~TensorPool() {
  // Need to get keys in a separate iteration, otherwise iterator is getting
  // invalidated during erase.
  std::vector<WGPUBuffer> keys;
  for (auto &pair : data) {
    keys.push_back(pair.first);
  }
  for (auto &key : keys) {
    FreeTensor(*this, data[key]);
    LOG(kDefLog, kTrace, "Freed tensor");
  }
}

/**
 * @brief Checks a condition and logs an error message if the condition is
 * false. In debug mode, it will also exit the program with an error code.
 * @param[in] condition The condition to check.
 * @param[in] message The error message to log if the condition is false.
 * @param[in] file The source file where the check is performed.
 * @param[in] line The line number in the source file where the check is
 * performed.
 */
inline void check(bool condition, const char *message,
                  const char *file = "unkown", int line = -1) {
  if (!condition) {
    LOG(kDefLog, kError, "Error in file %s line %d:\n%s", file, line, message);
    exit(1);
  } else {
    LOG(kDefLog, kTrace, "Success in file %s line %d:\n%s", file, line,
        message);
  }
}

/**
 * @brief Factory function to create a GPU context, which aggregates WebGPU API
 * handles to interact with the GPU including the instance, adapter, device, and
 * queue.
 *
 * The function takes optional descriptor parameters for the instance
 * descriptor, adapter request options, and device descriptor, which are passed
 * through to the WebGPU API calls to create the instance, adapter, and device.
 *
 * If dawn is used, it also sets up an error callback for device loss.
 *
 * @param[in] desc Instance descriptor for the WebGPU instance (optional)
 * @param[in] adapterOpts Adapter request options for the WebGPU adapter
 * (optional)
 * @param[in] devDescriptor Device descriptor for the WebGPU device (optional)
 * @return Context instance representing the created GPU context
 *
 * @code
 * Context ctx = createContext();
 * @endcode
 */
inline Context createContext(const WGPUInstanceDescriptor &desc = {},
                             const WGPURequestAdapterOptions &adapterOpts = {},
                             const WGPUDeviceDescriptor &devDescriptor = {}) {
  Context context;
  {
#ifdef __EMSCRIPTEN__
    // Emscripten does not support the instance descriptor
    // and throws an assertion error if it is not nullptr.
    context.instance = wgpuCreateInstance(nullptr);
#else
    context.instance = wgpuCreateInstance(&desc);
#endif
    // check status
    check(context.instance, "Initialize WebGPU", __FILE__, __LINE__);
  }

  LOG(kDefLog, kInfo, "Requesting adapter");
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
      LOG(kDefLog, kInfo, "WGPURequestAdapterStatus_Success: %d",
          WGPURequestAdapterStatus_Success);
      LOG(kDefLog, kInfo, "WGPURequestAdapterStatus_Unavailable: %d",
          WGPURequestAdapterStatus_Unavailable);
      LOG(kDefLog, kInfo, "Status: %d", status);
#ifdef __EMSCRIPTEN__
      if (status != WGPURequestAdapterStatus_Success) {
        LOG(kDefLog, kError, "Could not get WebGPU adapter: %s", message);
        LOG(kDefLog, kError,
            "\n\nA common reason is that the browser does not have WebGPU "
            "enabled, particularly on Linux.\n"
            "- Open `chrome://flags/` in the browser and make sure "
            "\"WebGPU Support\" is enabled.\n"
        "- Chrome is launched with vulkan enabled. From the command line launch chrome as `google-chrome --enable-features=Vulkan`\n");
      }
#endif
      check(status == WGPURequestAdapterStatus_Success,
            "Request WebGPU adapter", __FILE__, __LINE__);
      adapterData.adapter = adapter;
      adapterData.requestEnded = true;
    };

    wgpuInstanceRequestAdapter(context.instance, &adapterOpts,
                               onAdapterRequestEnded, (void *)&adapterData);

    while (!adapterData.requestEnded) {
      processEvents(context.instance);
    }
    assert(adapterData.requestEnded);
    context.adapter = adapterData.adapter;
  }

  LOG(kDefLog, kInfo, "Requesting device");
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
      LOG(kDefLog, kTrace, "Device Request succeeded %x",
          static_cast<void *>(device));
      devData.device = device;
      devData.requestEnded = true;
    };
#if defined(WEBGPU_BACKEND_DAWN) && !defined(__EMSCRIPTEN__)
    devDescriptor.deviceLostCallbackInfo = {
        .callback =
            [](WGPUDevice const *device, WGPUDeviceLostReason reason,
               char const *message, void *userdata) {
              if (reason != WGPUDeviceLostReason_Destroyed) {
                LOG(kDefLog, kError, "Device lost (code %d):\n%s", reason,
                    message);
              } else {
                LOG(kDefLog, kInfo, "Device destroyed: %s", message);
              }
            },
    };
#endif
    wgpuAdapterRequestDevice(context.adapter, &devDescriptor,
                             onDeviceRequestEnded, (void *)&devData);
    LOG(kDefLog, kInfo, "Waiting for device request to end");
    while (!devData.requestEnded) {
      processEvents(context.instance);
    }
    LOG(kDefLog, kInfo, "Device request ended");
    assert(devData.requestEnded);
    context.device = devData.device;
    wgpuDeviceSetUncapturedErrorCallback(
        context.device,
        [](WGPUErrorType type, char const *message, void *devData) {
          LOG(kDefLog, kError, "Device uncaptured error: %s", message);
          throw std::runtime_error("Device uncaptured exception.");
        },
        nullptr);
  }
  context.queue = wgpuDeviceGetQueue(context.device);
  return context;
}

inline void wait(Context &ctx, std::future<void> &future) {
  while (future.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
    processEvents(ctx.instance);
  }
}

/**
 * @brief Copies data from a GPU buffer to CPU memory.
 * @param[in] ctx Context instance to manage the operation
 * @param[in] tensor Tensor instance representing the GPU buffer to copy from
 * @param[out] data Pointer to the CPU memory to copy the data to
 * @param[in] bufferSize Size of the data buffer in bytes
 * @param[in] op StagingBuffer instance to manage the operation
 *
 * @code
 * toCPU(ctx, tensor, data, bufferSize);
 * @endcode
 */
inline void toCPU(Context &ctx, Tensor &tensor, void *data, size_t bufferSize,
                  CopyData &op) {
  wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);
  CallbackData callbackData = {op.readbackBuffer, bufferSize, data, &op.promise,
                               &op.future};
  wgpuQueueOnSubmittedWorkDone(
      ctx.queue,
      [](WGPUQueueWorkDoneStatus status, void *callbackData) {
        check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done",
              __FILE__, __LINE__);
        const auto *data = static_cast<CallbackData *>(callbackData);
        wgpuBufferMapAsync(
            data->buffer, WGPUMapMode_Read, 0, data->bufferSize,
            [](WGPUBufferMapAsyncStatus status, void *captureData) {
              const auto *data = static_cast<CallbackData *>(captureData);
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
  wait(ctx, op.future);
}

/**
 * @brief Overload of the toCPU function to copy data from a GPU buffer to CPU
 * but initializes a staging buffer and promise/future for the operation for
 * you.
 *
 * For simple use cases, this overload is recommended as it abstracts away the
 * staging buffer and promise/future management. For more custom use cases
 * where the staging buffer is initialized ahead of time, use the other
 * overload.
 *
 * @param[in] ctx Context instance to manage the operation
 * @param[in] tensor Tensor instance representing the GPU buffer to copy from
 * @param[in] bufferSize Size of the data buffer in bytes
 * @param[out] data Pointer to the CPU memory to copy the data to
 */
inline void toCPU(Context &ctx, Tensor &tensor, void *data, size_t bufferSize) {
  CopyData op;
  op.future = op.promise.get_future();
  {
    WGPUBufferDescriptor readbackBufferDescriptor = {
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = bufferSize,
    };
    op.readbackBuffer =
        wgpuDeviceCreateBuffer(ctx.device, &readbackBufferDescriptor);
  }
  {
    WGPUCommandEncoder commandEncoder;
    WGPUComputePassEncoder computePassEncoder;
    commandEncoder = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, tensor.data.buffer, 0,
                                         op.readbackBuffer, 0, bufferSize);
    op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
    check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
  }
  toCPU(ctx, tensor, data, bufferSize, op);
}

/**
 * @brief Overload of the toCPU function to copy data from a GPU buffer to CPU
 * memory for an array of floats instead of a pointer to a float buffer.
 * @param[in] ctx Context instance to manage the operation
 * @param[in] tensor Tensor instance representing the GPU buffer to copy from
 * @param[out] data Array of floats to copy the data to
 *
 * @code
 * toCPU(ctx, tensor, data);
 * @endcode
 */
template <size_t N>
void toCPU(Context &ctx, Tensor &tensor, std::array<float, N> &data) {
  toCPU(ctx, tensor, data.data(), sizeof(data));
}

/**
 * @brief Copies data from CPU memory to a GPU buffer. The toGPU overloads are
 * effectively a convenience wrapper around the WebGPU API call
 * wgpuQueueWriteBuffer.
 *
 * @param[in] ctx Context instance to manage the operation
 * @param[in] data Pointer to the CPU memory to copy from
 * @param[in] buffer WGPUBuffer instance representing the GPU buffer to copy
 * to
 * @param[in] size Size of the data buffer in bytes
 *
 * @code
 * toGPU(ctx, data, buffer, size);
 * @endcode
 */
inline void toGPU(Context &ctx, const void *data, WGPUBuffer buffer,
                  size_t size) {
  wgpuQueueWriteBuffer(ctx.queue, buffer, 0, data, size);
}

/**
 * @brief Overload of the toGPU function to copy data from CPU memory to a GPU
 * taking a Tensor instance instead of a WGPUBuffer instance.
 * @param[in] ctx Context instance to manage the operation
 * @param[in] data Pointer to the CPU memory to copy from
 * @param[in] tensor Tensor instance representing the GPU buffer to copy to
 *
 * @code
 * toGPU(ctx, data, tensor);
 * @endcode
 */
inline void toGPU(Context &ctx, const float *data, Tensor &tensor) {
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
}

inline void toGPU(Context &ctx, const half *data, Tensor &tensor) {
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
}

template <typename Params>
inline void toGPU(Context &ctx, Params &params, Kernel &op) {
  // TODO(avh): Maintain params metadata in Kernel and check for consistency.
  // If a kernel does not have parameters this will quietly overwrite
  // the last buffer in the bind group with the parameters buffer.
  if (op.numBindings > 0) {
    wgpuQueueWriteBuffer(ctx.queue, op.buffers[op.numBindings - 1], 0,
                         static_cast<void *>(&params), sizeof(params));
  }
}

/**
 * @brief Resets the command buffer in preparation for a kernel dispatch.
 * Since command buffers are consumed upon submission, this function is used
 * both in the initial kernel creation and every time the kernel is to be
 * reused for a dispatch.
 * @param[in] device WGPUDevice instance to manage the operation
 * @param[in] op Kernel instance representing the kernel to reset
 *
 * @code
 * resetCommandBuffer(device, op);
 * @endcode
 */
inline void resetCommandBuffer(WGPUDevice &device, Kernel &op) {
  {
    WGPUCommandEncoder commandEncoder =
        wgpuDeviceCreateCommandEncoder(device, nullptr);
    WGPUComputePassEncoder computePassEncoder =
        wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);
    wgpuComputePassEncoderSetPipeline(computePassEncoder, op.computePipeline);
    wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, op.bindGroup, 0,
                                       nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        computePassEncoder, op.nWorkgroups[0], op.nWorkgroups[1],
        op.nWorkgroups[2]);
    wgpuComputePassEncoderEnd(computePassEncoder);
    op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
  }
}

/**
 * @brief NoParam is a no-op type used to indicate that a kernel does not have
 * any parameters.
 */
struct NoParam {};
template <typename T> constexpr bool IsNoParam = std::is_same_v<T, NoParam>;

/**
 * @brief Ceiling division.
 */
inline size_t cdiv(size_t n, size_t d) { return (n + d - 1) / d; }

/**
 * @brief cdiv for shape specification. Mostly useful for evenly dividing
 * total # threads by workgroup size dimensions.
 */
inline Shape cdiv(Shape total, Shape group) {
  assert(total.rank == group.rank);
  Shape result;
  result.rank = total.rank;
  for (size_t dim = 0; dim < total.rank; ++dim) {
    result[dim] = cdiv(total[dim], group[dim]);
  }
  return result;
}

/**
 * @brief A factory function to create a kernel on the GPU. The kernel is
 * created with the given WGSL code, input tensors, output tensor, and
 * optional parameters.
 *
 * Note that the values of the input tensors are not used here, only the
 * reference handles to the underlying buffers as well as the size of the
 * buffers.
 *
 * @param[in] ctx Context instance to manage the kernel
 * @param[in] code WGSL code for the kernel
 * @param[in] dataBindings Pointer to a span of tensors bound to the kernel
 * @param[in] numTensors Number of tensors in the dataBindings span
 * @param[in] viewOffsets Pointer to an array of view offsets for the input
 * tensors
 * @param[in] nWorkgroups Shape of the workgroup
 * @param[in] params Optional parameters for the kernel. If the kernel does
 * not have any parameters, use NoParam. This is cast as void* to allow for
 * arbitrary types to be passed as parameters.
 * @param[in] paramsSize Size of the parameters buffer in bytes.
 * @return Kernel instance representing the created kernel
 *
 * @code
 * Kernel kernel = createKernel(ctx, code, dataBindings, numInputs,
 * @endcode
 * output, nThreads, params, paramsSize);
 */
inline Kernel createKernel(Context &ctx, const KernelCode &code,
                           const Tensor *dataBindings, size_t numTensors,
                           const size_t *viewOffsets, const Shape &nWorkgroups,
                           const void *params = nullptr,
                           size_t paramsSize = 0) {
  assert(nWorkgroups.rank == 3);
  WGPUDevice device = ctx.device;
  WGPUQueue queue = ctx.queue;
  Kernel op;
  // paramIndex is the index into bgLayoutEntries for the parameters buffer If
  // there are no parameters for the kernel, paramsSize == 0 and paramIndex is
  // effectively undefined (== -1)
  size_t paramIndex = -1;
  // Note: paramIndex is undefined unless paramsSize > 0
  size_t numBindings = numTensors;
  if (paramsSize > 0) {
    numBindings++;                // parameters buffer
    paramIndex = numBindings - 1; // index of the parameters buffer within
                                  // op.buffers, op.bufferSizes and
                                  // bgLayoutEntries
  }
  op.buffers = std::make_unique<WGPUBuffer[]>(numBindings);
  op.bufferSizes = std::make_unique<size_t[]>(numBindings);
  op.numBindings = numBindings;
  std::vector<WGPUBindGroupLayoutEntry> bgLayoutEntries(numBindings);
  // Create layout entries for input buffers
  for (size_t i = 0; i < numTensors; ++i) {
    bgLayoutEntries[i] = WGPUBindGroupLayoutEntry{
        .binding = static_cast<uint32_t>(i),
        .visibility = WGPUShaderStage_Compute,
        .buffer =
            WGPUBufferBindingLayout{
                .type = WGPUBufferBindingType_Storage,
                .minBindingSize = dataBindings[i].data.size,
            },
    };
  }
  if (paramsSize > 0) {
    LOG(kDefLog, kInfo, "Create layout entry for the params buffer");
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
  for (size_t i = 0; i < numTensors; ++i) {
    op.buffers[i] = dataBindings[i].data.buffer;
    op.bufferSizes[i] = dataBindings[i].data.size;
  }
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
    LOG(kDefLog, kTrace, "Params buffer written");
  } else {
    LOG(kDefLog, kTrace, "No params buffer needed");
  }
  std::vector<WGPUBindGroupEntry> bindGroupEntries(numBindings);
  for (size_t i = 0; i < numTensors; ++i) {
    bindGroupEntries[i] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(i),
        .buffer = op.buffers[i],
        .offset = viewOffsets[i],
        .size = op.bufferSizes[i],
    };
  }
  if (paramsSize > 0) {
    LOG(kDefLog, kInfo, "Create bind group entry for the params buffer");
    LOG(kDefLog, kInfo, "paramIndex: %d", paramIndex);
    bindGroupEntries[paramIndex] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(paramIndex),
        .buffer = op.buffers[paramIndex],
        .offset = 0,
        .size = paramsSize,
    };
  }
  LOG(kDefLog, kTrace, "BG Entries Size: %d", numBindings);
  WGPUBindGroupDescriptor bindGroupDesc = {
      .layout = bgLayout,
      .entryCount = static_cast<uint32_t>(numBindings),
      .entries = bindGroupEntries.data(),
  };
  op.bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
  {
    WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bgLayout,
    };
    WGPUPipelineLayout pipelineLayout =
        wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);
    WGPUShaderModuleWGSLDescriptor wgslDesc = {
        .code = code.data.c_str(),
    };
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    WGPUShaderModuleDescriptor shaderModuleDesc = {};
    shaderModuleDesc.nextInChain = &wgslDesc.chain;
    shaderModuleDesc.label = code.label.c_str();
    WGPUComputePipelineDescriptor computePipelineDesc = {};
    computePipelineDesc.layout = pipelineLayout;
    computePipelineDesc.compute.module =
        wgpuDeviceCreateShaderModule(device, &shaderModuleDesc);
    computePipelineDesc.compute.entryPoint = code.entryPoint.c_str();
    computePipelineDesc.label = code.label.c_str();
    op.computePipeline =
        wgpuDeviceCreateComputePipeline(device, &computePipelineDesc);
  }
  /*
  op.nWorkgroups = {cdiv(nThreads[0], code.workgroupSize[0]),
                    cdiv(nThreads[1], code.workgroupSize[1]),
                    cdiv(nThreads[2], code.workgroupSize[2])};
  */
  op.nWorkgroups = {nWorkgroups[0], nWorkgroups[1], nWorkgroups[2]};
  resetCommandBuffer(device, op);
  ctx.kernelPool.data.insert(&op);
  return op;
}

/**
 * @brief Overload which wraps the createKernel factory function to create a
 * kernel on the GPU. This overload uses takes a static collection of input
 * tensors instead of a pointer and a statically determined ParamsType instead
 * of casting params to a void pointer.
 *
 * @param[in] ctx Context instance to manage the kernel
 * @param[in] code WGSL code for the kernel
 * @param[in] dataBindings A Bindings of tensors whose GPU buffers are bound
 * to the kernel as inputs and outputs.
 * @param[in] nWorkgroups Number of workgroups in the x, y, z grid, must be a
 * Shape of rank == 3.
 * @param[in] params Optional parameters for the kernel. If the kernel does
 * not have any parameters, use NoParam.
 * @return Kernel instance representing the created kernel
 *
 * @code
 * Kernel kernel = createKernel(ctx, code, tensorData, output,
 * @endcode
 * nWorkgroups, params);
 */
template <typename ParamsType = NoParam, size_t numInputs>
Kernel createKernel(Context &ctx, const KernelCode &code,
                    const Bindings<numInputs> &dataBindings,
                    const Shape &nWorkgroups,
                    const ParamsType &params = ParamsType{}) {
  if constexpr (!IsNoParam<ParamsType>) {
    // LOG(kDefLog, kTrace, "Using params of size %d bytes",
    // sizeof(ParamsType));
    return createKernel(ctx, code, dataBindings.data.data(), numInputs,
                        dataBindings.viewOffsets.data(), nWorkgroups,
                        reinterpret_cast<const void *>(&params),
                        sizeof(ParamsType));
  } else {
    // LOG(kDefLog, kTrace , "No params");
    return createKernel(ctx, code, dataBindings.data.data(), numInputs,
                        dataBindings.viewOffsets.data(), nWorkgroups, nullptr,
                        0);
  }
}

/**
 * @brief Asynchronously submits a kernel to the GPU queue for execution.
 * It also sets up a callback to notify when the kernel has finished executing
 * by setting the value of the promise in the kernel instance argument.
 *
 * dispatchKernel does *not* wait for the kernel to finish executing and
 * returns immediately. The caller can wait for the kernel to finish executing
 * by calling wait() on the future in the kernel instance.
 *
 * @param[in] ctx Context instance to manage the kernel, from which the queue
 * for the GPU is obtained
 * @param[in] kernel Kernel instance to dispatch
 * @param[in] promise Promise to set when the kernel has finished executing
 *
 * @code
 * dispatchKernel(ctx, kernel);
 * @endcode
 */
inline void dispatchKernel(Context &ctx, Kernel &kernel,
                           std::promise<void> &promise) {
  // Submit the command buffer
  wgpuQueueSubmit(ctx.queue, 1, &kernel.commandBuffer);
  wgpuQueueOnSubmittedWorkDone(
      ctx.queue,
      [](WGPUQueueWorkDoneStatus status, void *data) {
        check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done",
              __FILE__, __LINE__);
        auto *promise = static_cast<std::promise<void> *>(data);
        promise->set_value();
      },
      &promise);
}

} // namespace gpu

#endif // GPU_H
