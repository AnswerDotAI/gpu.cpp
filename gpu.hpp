#ifndef GPU_HPP
#define GPU_HPP

#include "webgpu.h"
#include <array>
#include <cassert>
#include <cstring>
#include <future>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility> // std::pair
#include <vector>

#ifdef __EMSCRIPTEN__
#include "emscripten/emscripten.h"
#endif

#include "numeric_types/half.hpp"
#include "utils/logging.hpp"

#ifdef USE_DAWN_API
#include "dawn/native/DawnNative.h"
#endif

namespace gpu {

/**
 * @brief Represents a buffer of values on the GPU.
 */
struct Array {
  WGPUBuffer buffer;
  WGPUBufferUsage usage;
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

  Bindings(const std::array<Tensor, N> &init) {
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
  inline TensorPool(Context *ctx) : ctx(ctx), data() {};
  Context *ctx;
  std::unordered_map<WGPUBuffer, Tensor> data;
  ~TensorPool();
};

enum NumType {
  kf16, // (experimental)
  kf32,
  kf64,
  ki8,
  ki16,
  ki32,
  ki64,
  ku8,
  ku16,
  ku32,
  ku64,
  kUnknown
};

/**
 * @brief Returns the number of bytes of a number type.
 */
inline size_t sizeBytes(const NumType &type, int numElements = 1) {
  switch (type) {
  case kf16:
    return sizeof(half) * numElements;
  case kf32:
    return sizeof(float) * numElements;
  case kf64:
    return sizeof(double) * numElements;
  case ki8:
    return sizeof(uint32_t) * ((numElements + 3) / 4);
  case ki16:
    return sizeof(uint32_t) * ((numElements + 1) / 2);
  case ki32:
    return sizeof(int32_t) * numElements;
  case ki64:
    return sizeof(int64_t) * numElements;
  case ku8:
    return sizeof(uint32_t) * ((numElements + 3) / 4);
  case ku16:
    return sizeof(uint32_t) * ((numElements + 1) / 2);
  case ku32:
    return sizeof(uint32_t) * numElements;
  case ku64:
    return sizeof(uint64_t) * numElements;
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
  case kf64:
    return "f64";
  case ki8:
    return "i8";
  case ki16:
    return "i16";
  case ki32:
    return "i32";
  case ki64:
    return "i64";
  case ku8:
    return "u8";
  case ku16:
    return "u16";
  case ku32:
    return "u32";
  case ku64:
    return "u64";
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
 * @brief Converts a WGPUStringView to an std::string.
 *
 * If the view's data is null, an empty string is returned. If the view's
 * length equals WGPU_STRLEN, it is assumed to be null‑terminated; otherwise,
 * the explicit length is used.
 *
 * @param strView The WGPUStringView to convert.
 * @return std::string The resulting standard string.
 */
inline std::string formatWGPUStringView(WGPUStringView strView) {
  if (!strView.data) {
    return "";
  }
  if (strView.length == WGPU_STRLEN) {
    return std::string(strView.data);
  }
  return std::string(strView.data, strView.length);
}

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
   * @brief Overload of the constructor to create a code object from a template
   * string and workgroup size. This overload takes a single size_t
   * workgroupSize parameter instead of a 3D shape for the workgroup size and
   * instantiates a 3D shape with the workgroupSize in the x dimension and 1 in
   * the y and z dimensions.
   *
   * @param[in] pData Shader template string with placeholders @param[in]
   * workgroupSize 3D Workgroup size
   * @param[in] precision Data type precision for the shader
   *
   * @code KernelCode code = {kPuzzle1, 256, kf32}; @endcode
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

  /**
   * @brief Overload of the constructor, adding totalWorkgroups parameter to
   * perform a string replacement for the total number of workgroups in the
   * kernel code.
   *
   * @param[in] pData Shader template string with placeholders
   * @param[in] workgroupSize 3D Workgroup size
   * @param[in] precision Data type precision for the shader
   * @param[in] totalWorkgroups Total number of workgroups in the kernel
   *
   * @code
   * KernelCode code = {kPuzzle1, {256, 1, 1}, kf32, {2, 2, 1}};
   * @endcode
   */
  inline KernelCode(const std::string &pData, const Shape &workgroupSize,
                    NumType precision, const Shape &totalWorkgroups)
      : data(pData), workgroupSize(workgroupSize), precision(precision) {
    if (precision == kf16) {
      data = "enable f16;\n" + data;
    }
    replaceAll(data, "{{workgroupSize}}", toString(workgroupSize));
    replaceAll(data, "{{precision}}", toString(precision));
    replaceAll(data, "{{totalWorkgroups}}", toString(totalWorkgroups));
    LOG(kDefLog, kInfo, "Shader code:\n%s", data.c_str());
  }

  /**
   * @brief Overload of the constructor, adding totalWorkgroups parameter as
   * well as the size_t 1D workgroupSize parameter.
   *
   * @param[in] pData Shader template string with placeholders
   * @param[in] workgroupSize Workgroup size in the x dimension
   * @param[in] precision Data type precision for the shader
   * @param[in] totalWorkgroups Total number of workgroups in the kernel
   *
   * @code
   * KernelCode code = {kPuzzle1, {256, 1, 1}, kf32, {2, 2, 1}};
   * @endcode
   */
  inline KernelCode(const std::string &pData, const size_t &workgroupSize,
                    NumType precision, const Shape &totalWorkgroups)
      : data(pData), workgroupSize({workgroupSize, 1, 1}),
        precision(precision) {
    if (precision == kf16) {
      data = "enable f16;\n" + data;
    }
    replaceAll(data, "{{workgroupSize}}", toString({workgroupSize, 1, 1}));
    replaceAll(data, "{{precision}}", toString(precision));
    replaceAll(data, "{{totalWorkgroups}}", toString(totalWorkgroups));
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
inline const std::string
replaceAll(std::string &str,
           const std::vector<std::pair<std::string, std::string>> &reps) {
  for (const auto &rep : reps) {
    replaceAll(str, rep.first, rep.second);
  }

  return str;
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
  std::shared_ptr<std::promise<void>> promise;
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
struct RawKernel {
  std::unique_ptr<WGPUBuffer[]> buffers; // non-owning
  std::unique_ptr<size_t[]> bufferSizes;
  size_t numBindings;
  Shape totalWorkgroups;
  WGPUBindGroup bindGroup;             // persists between submission
  WGPUComputePipeline computePipeline; // persists between submission
  WGPUCommandBuffer commandBuffer;     // destroyed upon submission
  bool used;
};

typedef std::shared_ptr<RawKernel> Kernel;

/**
 * @brief A struct to package the result of a WGSL code compilation.
 */
struct CompilationInfo {
  WGPUCompilationInfoRequestStatus status;
  std::vector<std::string> messages;
  std::vector<uint64_t> lineNums;
  std::vector<uint64_t> linePos;
  bool finished; // true if the compilation is finished
};

/**
 * @brief Operator implementation to make the Kernel type hashable.
 * @param[in] lhs First Kernel instance to compare
 * @param[in] rhs Second Kernel instance to compare
 * @return True if lhs < rhs, false otherwise
 */
inline bool operator<(const Kernel &lhs, const Kernel &rhs) {
  return lhs->commandBuffer < rhs->commandBuffer;
}

/**
 * @brief A pool of kernels to manage GPU resources. For simple use cases this
 * is instantiated as a member in the Context struct although it's possible to
 * have multiple resource pools of kernels in more complex scenarios.
 */
struct KernelPool {
  inline KernelPool(Context *ctx) : ctx(ctx), data() {}
  Context *ctx;
  std::unordered_map<std::string, Kernel> data;
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
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  TensorPool pool = TensorPool(this);
  KernelPool kernelPool = KernelPool(this);
  WGPURequestAdapterStatus adapterStatus;
  WGPURequestDeviceStatus deviceStatus;

  // Default constructor
  Context() = default;

  Context(Context &&other) noexcept
      : instance(other.instance), adapter(other.adapter), device(other.device),
        queue(other.queue),
        // Re‐initialize pools to point to *this*:
        pool(this), kernelPool(this), adapterStatus(other.adapterStatus),
        deviceStatus(other.deviceStatus) {
    LOG(kDefLog, kTrace, "Moving Context ownership");
    // Move over the resources in the pools:
    pool.data = std::move(other.pool.data);
    kernelPool.data = std::move(other.kernelPool.data);

    // Null out handles in the source so its destructor won't release them.
    other.instance = nullptr;
    other.adapter = nullptr;
    other.device = nullptr;
    other.queue = nullptr;
    // other.adapterStatus = 0;
    // other.deviceStatus = 0;
  }

  Context &operator=(Context &&other) noexcept {
    if (this != &other) {
      // Free any existing resources. In most cases, this should be a no-op
      // since we typically shouldn't have two active initialized Context
      // instances with resources acquired.
      this->~Context();
      // Then placement‐new a move‐constructed copy in-place:
      new (this) Context(std::move(other));
    }
    return *this;
  }

  ~Context() {
    LOG(kDefLog, kTrace, "Destroying context");

#ifdef __EMSCRIPTEN__
    // For WebAssembly, do NOT call processEvents during destruction
    // This prevents "Asyncify cannot be done during or after runtime exits"
    LOG(kDefLog, kTrace,
        "WebAssembly context destruction - skipping processEvents");
#endif

    if (queue) {
      wgpuQueueRelease(queue);
      queue = nullptr;
    } else {
      LOG(kDefLog, kTrace, "Queue already null");
    }

    if (device) {
      wgpuDeviceRelease(device);
      device = nullptr;
    } else {
      LOG(kDefLog, kTrace, "Device already null");
    }

    if (adapter) {
      wgpuAdapterRelease(adapter);
      adapter = nullptr;
    } else {
      LOG(kDefLog, kTrace, "Adapter already null");
    }

    if (instance) {
#ifndef __EMSCRIPTEN__
      // Only call processEvents on native platforms during cleanup
      processEvents(instance);
#endif
      wgpuInstanceRelease(instance);
      instance = nullptr;
    } else {
      LOG(kDefLog, kTrace, "Instance already null");
    }

    LOG(kDefLog, kTrace, "Context destroyed");
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
inline Tensor createTensor(TensorPool &pool, WGPUDevice &device,
                           const Shape &shape, NumType dtype,
                           WGPUBufferUsage usage = WGPUBufferUsage_Storage |
                                                   WGPUBufferUsage_CopyDst |
                                                   WGPUBufferUsage_CopySrc) {
  LOG(kDefLog, kTrace, "Creating tensor");
  size_t numElements = size(shape);
  size_t size = sizeBytes(dtype, numElements);
  WGPUBufferDescriptor bufferDesc = {
      .label = {.data = nullptr, .length = 0},
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
 * WGPUBufferUsage arguments, this is a convenience wrapper around the
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
                           const half *data) {
  assert(dtype == kf16);
  Tensor tensor =
      createTensor(ctx.pool, ctx.device, shape, dtype,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const float *data) {
  assert(dtype == kf32);
  Tensor tensor =
      createTensor(ctx.pool, ctx.device, shape, dtype,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

// Overload for double: pack each double into a float (losing precision)
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const double *data) {
  assert(dtype == kf64);
  size_t numElements = size(shape);
  // Each double (8 bytes) will be packed into 2 uint32_t values (2×4 bytes).
  std::vector<uint32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    uint64_t bits;
    std::memcpy(&bits, &data[i], sizeof(double)); // Extract raw bits.
    packed[2 * i] = static_cast<uint32_t>(bits & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<uint32_t>(bits >> 32);
  }
  // Create a tensor using the core overload that accepts a TensorPool and
  // WGPUDevice.
  Tensor tensor =
      createTensor(ctx.pool, ctx.device, shape, kf64,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);

  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       packed.size() * sizeof(uint32_t));

  return tensor;
}

inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const int32_t *data) {
  assert(dtype == ki32);
  Tensor tensor =
      createTensor(ctx.pool, ctx.device, shape, dtype,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

// Overload for int8_t: pack four 8‑bit ints into one 32‑bit integer
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const int8_t *data) {
  assert(dtype == ki8); // unsupported: pack into ki32
  size_t numElements = size(shape);
  size_t packedCount = (numElements + 3) / 4;
  std::vector<int32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 4;
    size_t shift = (i % 4) * 8;
    // pack as unsigned bits then reinterpret; shader is then responsible for
    // unpacking
    packed[idx] |= (static_cast<uint8_t>(data[i]) << shift);
  }
  Tensor tensor = createTensor(ctx, shape, ki8);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
  return tensor;
}

// Overload for int16_t: pack two 16‑bit ints into one 32‑bit integer
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const int16_t *data) {
  assert(dtype == ki16); // unsupported: pack into ki32
  size_t numElements = size(shape);
  size_t packedCount = (numElements + 1) / 2;
  std::vector<int32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 2;
    size_t shift = (i % 2) * 16;
    packed[idx] |= (static_cast<uint16_t>(data[i]) << shift);
  }
  Tensor tensor = createTensor(ctx, shape, ki16);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
  return tensor;
}

// Overload for int64_t: pack each 64‑bit int into two 32‑bit integers
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const int64_t *data) {
  assert(dtype == ki64); // unsupported: pack into two ki32s
  size_t numElements = size(shape);
  std::vector<int32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    int64_t val = data[i];
    packed[2 * i] = static_cast<int32_t>(val & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<int32_t>((val >> 32) & 0xFFFFFFFF);
  }
  Tensor tensor = createTensor(ctx, shape, ki64);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
  return tensor;
}

inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const uint32_t *data) {
  assert(dtype == ku32);
  Tensor tensor =
      createTensor(ctx.pool, ctx.device, shape, dtype,
                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, data,
                       tensor.data.size);
  return tensor;
}

// Overload for uint8_t: pack four 8‑bit integers into one 32‑bit unsigned
// integer
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const uint8_t *data) {
  assert(dtype == ku8); // unsupported: pack into ku32
  size_t numElements = size(shape);
  size_t packedCount = (numElements + 3) / 4;
  std::vector<uint32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 4;
    size_t shift = (i % 4) * 8;
    packed[idx] |= (static_cast<uint32_t>(data[i]) << shift);
  }
  Tensor tensor = createTensor(ctx, shape, ku8);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
  return tensor;
}

// Overload for uint16_t: pack two 16‑bit integers into one 32‑bit unsigned
// integer
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const uint16_t *data) {
  assert(dtype == ku16); // unsupported: pack into ku32
  size_t numElements = size(shape);
  size_t packedCount = (numElements + 1) / 2;
  std::vector<uint32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 2;
    size_t shift = (i % 2) * 16;
    packed[idx] |= (static_cast<uint32_t>(data[i]) << shift);
  }
  Tensor tensor = createTensor(ctx, shape, ku16);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
  return tensor;
}

// Overload for uint64_t: pack each 64‑bit integer into two 32‑bit unsigned
// integers
inline Tensor createTensor(Context &ctx, const Shape &shape, NumType dtype,
                           const uint64_t *data) {
  assert(dtype == ku64); // unsupported: pack into two ku32s
  size_t numElements = size(shape);
  std::vector<uint32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    uint64_t val = data[i];
    packed[2 * i] = static_cast<uint32_t>(val & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<uint32_t>(val >> 32);
  }
  Tensor tensor = createTensor(ctx, shape, ku64);
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
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
 * false.
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
  } else {
    LOG(kDefLog, kTrace, "Success in file %s line %d:\n%s", file, line,
        message);
  }
}

/**
 * @brief Pumps events until the provided future is ready.
 *
 * This helper template function continuously checks the status of the provided
 * std::future<T> until it becomes ready. On Emscripten builds, it yields
 * control to the JavaScript event loop using emscripten_sleep to allow
 * asynchronous callbacks to execute. On other platforms, it processes events
 * from the given WGPUInstance using wgpuInstanceProcessEvents. Once the future
 * is ready, its value is returned.
 *
 * @tparam T The type of the value contained in the future.
 * @param instance The WGPUInstance used to process events.
 * @param f The future to wait on.
 * @return T The value retrieved from the ready future.
 *
 * @code
 * std::future<WGPUDevice> deviceFuture = requestDeviceAsync(adapter,
 * devDescriptor); WGPUDevice device = wait(instance, deviceFuture);
 * @endcode
 */
#ifdef __EMSCRIPTEN__
// Global flag to prevent overlapping async operations in WebAssembly
static std::atomic<bool> asyncOperationInProgress{false};
#endif

template <typename T> T wait(Context &ctx, std::future<T> &f) {
#ifdef __EMSCRIPTEN__
  // Check if another async operation is in progress
  if (asyncOperationInProgress.load()) {
    LOG(kDefLog, kWarn,
        "wait(): Another async operation in progress, skipping wait");
    if constexpr (std::is_void_v<T>) {
      return; // For void functions, just return
    } else {
      return T{}; // Return default-constructed value for non-void types
    }
  }

  // Set the flag before starting async operation
  asyncOperationInProgress.store(true);

  try {
    // Poll until the future is ready
    while (f.wait_for(std::chrono::milliseconds(0)) !=
           std::future_status::ready) {
      emscripten_sleep(1);
    }

    // Handle void vs non-void return types
    if constexpr (std::is_void_v<T>) {
      f.get(); // Just call get() without storing result
      asyncOperationInProgress.store(false);
      return; // void return
    } else {
      T result = f.get();
      asyncOperationInProgress.store(false);
      return result;
    }

  } catch (...) {
    asyncOperationInProgress.store(false);
    throw;
  }
#else
  // Native implementation unchanged
  while (f.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready) {
    wgpuInstanceProcessEvents(ctx.instance);
  }

  // Handle void vs non-void for native too
  if constexpr (std::is_void_v<T>) {
    f.get();
    return;
  } else {
    return f.get();
  }
#endif
}

// Context Callbacks & Helpers

/**
 * @brief Waits for the provided std::future<T> to become ready by polling its
 * status.
 *
 * This helper template function continuously checks the status of the provided
 * std::future<T> until it is ready. On Emscripten builds, it yields control to
 * the JavaScript event loop using emscripten_sleep(1) for smooth asynchronous
 * behavior. On non-Emscripten platforms, it sleeps for a short duration (10
 * milliseconds) between checks. Once the future is ready, its value is
 * returned.
 *
 * @tparam T The type of the value contained in the future.
 * @param f The future to wait on.
 * @return T The value retrieved from the ready future.
 *
 * @code
 * std::future<Context> contextFuture = createContext();
 * Context ctx = waitForContextFuture(contextFuture);
 * @endcode
 */
template <typename T>
T waitForContextFuture(std::future<T> &f, size_t sleepTime = 10) {
#ifdef __EMSCRIPTEN__
  while (f.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready) {
    emscripten_sleep(1); // Yield back to the JS event loop.
  }
  return f.get();
#else
  while (f.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
  }
  return f.get();
#endif
}

/**
 * @brief Adapter callback function invoked upon completion of an asynchronous
 * WebGPU adapter request.
 *
 * This callback is triggered when the request for a WebGPU adapter completes.
 * It verifies whether the adapter was successfully obtained. On failure, it
 * logs an error message (in Emscripten builds) and sets an exception on the
 * associated promise. On success, it sets the value of the promise with the
 * obtained adapter. Finally, it frees the allocated memory for the promise
 * pointer.
 *
 * @param status The status of the adapter request. Expected to be
 * WGPURequestAdapterStatus_Success on success.
 * @param adapter The WGPUAdapter obtained on a successful request.
 * @param message A string view containing additional information about the
 * adapter request.
 * @param userdata1 A pointer to a heap-allocated
 * std::shared_ptr<std::promise<WGPUAdapter>>.
 * @param userdata2 Unused.
 */
inline void adapterCallback(WGPURequestAdapterStatus status,
                            WGPUAdapter adapter, WGPUStringView message,
                            void *userdata1, void * /*userdata2*/) {
  auto *promisePtr =
      reinterpret_cast<std::shared_ptr<std::promise<WGPUAdapter>> *>(userdata1);
  if (status != WGPURequestAdapterStatus_Success) {
#ifdef __EMSCRIPTEN__
    LOG(kDefLog, kError, "Could not get WebGPU adapter: %.*s",
        static_cast<int>(message.length), message.data);
#endif
    (*promisePtr)
        ->set_exception(std::make_exception_ptr(
            std::runtime_error("Request WebGPU adapter failed")));
  } else {
    (*promisePtr)->set_value(adapter);
  }
  delete promisePtr;
}

/**
 * @brief Callback function invoked upon completion of an asynchronous WebGPU
 * device request.
 *
 * This callback is triggered when the request for a WebGPU device completes. It
 * verifies that the device was successfully created. On success, the callback
 * sets the value of the associated promise; otherwise, it sets an exception.
 * After fulfilling the promise, it frees the allocated memory for the promise
 * pointer.
 *
 * @param status The status of the device request. Expected to be
 * WGPURequestDeviceStatus_Success on success.
 * @param device The WGPUDevice obtained on successful request.
 * @param message A string view containing additional information about the
 * device request.
 * @param userdata1 A pointer to a heap-allocated
 * std::shared_ptr<std::promise<WGPUDevice>>.
 * @param userdata2 Unused.
 */
inline void deviceCallback(WGPURequestDeviceStatus status, WGPUDevice device,
                           WGPUStringView message, void *userdata1,
                           void * /*userdata2*/) {
  auto *promisePtr =
      reinterpret_cast<std::shared_ptr<std::promise<WGPUDevice>> *>(userdata1);
  if (status != WGPURequestDeviceStatus_Success) {
    (*promisePtr)
        ->set_exception(std::make_exception_ptr(
            std::runtime_error("Request WebGPU device failed")));
  } else {
    LOG(kDefLog, kTrace, "Device Request succeeded %p",
        static_cast<void *>(device));
    (*promisePtr)->set_value(device);
  }
  delete promisePtr;
}

/**
 * @brief Asynchronously requests a WebGPU adapter from the given instance.
 *
 * This helper function wraps the asynchronous call to request an adapter using
 * the WebGPU API. It sets up a promise and registers an adapter callback,
 * returning a future that will eventually hold the requested WGPUAdapter.
 *
 * @param instance The WGPUInstance from which to request the adapter.
 * @param adapterOpts The options for requesting the adapter.
 * @return std::future<WGPUAdapter> A future that will eventually hold the
 * created WGPUAdapter.
 */
inline std::future<WGPUAdapter>
requestAdapterAsync(WGPUInstance instance,
                    const WGPURequestAdapterOptions &adapterOpts) {
  auto promise = std::make_shared<std::promise<WGPUAdapter>>();
  auto *promisePtr = new std::shared_ptr<std::promise<WGPUAdapter>>(promise);

  WGPURequestAdapterCallbackInfo callbackInfo{
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = adapterCallback,
      .userdata1 = promisePtr,
      .userdata2 = nullptr};
  wgpuInstanceRequestAdapter(instance, &adapterOpts, callbackInfo);
  return promise->get_future();
}

/**
 * @brief Asynchronously requests a WebGPU device from a given adapter.
 *
 * This helper function wraps the asynchronous call to request a device using
 * the WebGPU API. It sets up a promise and registers a device callback,
 * returning a future that will be fulfilled once the device is available.
 *
 * @param adapter The WGPUAdapter to request the device from.
 * @param devDescriptor The descriptor specifying the characteristics of the
 * requested device.
 * @return std::future<WGPUDevice> A future that will eventually hold the
 * created WGPUDevice.
 */
inline std::future<WGPUDevice>
requestDeviceAsync(WGPUAdapter adapter,
                   const WGPUDeviceDescriptor &devDescriptor) {
  auto promise = std::make_shared<std::promise<WGPUDevice>>();
  auto *promisePtr = new std::shared_ptr<std::promise<WGPUDevice>>(promise);

  WGPURequestDeviceCallbackInfo deviceCallbackInfo{
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = deviceCallback,
      .userdata1 = promisePtr,
      .userdata2 = nullptr};
  wgpuAdapterRequestDevice(adapter, &devDescriptor, deviceCallbackInfo);
  return promise->get_future();
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
 */
inline std::future<Context>
createContextAsync(const WGPUInstanceDescriptor &desc = {},
                   const WGPURequestAdapterOptions &adapterOpts = {},
                   const WGPUDeviceDescriptor &devDescriptor = {}) {

  auto promise = std::make_shared<std::promise<Context>>();

  // On native platforms, run our context creation in a detached thread.

  Context ctx;
  ctx.instance = wgpuCreateInstance(&desc);
  if (!ctx.instance) {
    promise->set_exception(std::make_exception_ptr(
        std::runtime_error("Failed to create WebGPU instance.")));
    return promise->get_future();
  }
  try {
    auto adapterFuture = requestAdapterAsync(ctx.instance, adapterOpts);
    ctx.adapter = wait(ctx, adapterFuture);
    ctx.adapterStatus = WGPURequestAdapterStatus_Success;
  } catch (const std::exception &ex) {
    promise->set_exception(std::make_exception_ptr(ex));
    return promise->get_future();
  }
  try {
    auto deviceFuture = requestDeviceAsync(ctx.adapter, devDescriptor);
    ctx.device = wait(ctx, deviceFuture);
    ctx.deviceStatus = WGPURequestDeviceStatus_Success;
  } catch (const std::exception &ex) {
    promise->set_exception(std::make_exception_ptr(ex));
    return promise->get_future();
  }
  ctx.queue = wgpuDeviceGetQueue(ctx.device);
  promise->set_value(std::move(ctx));

  return promise->get_future();
}

/**
 * @brief Synchronously waits for and returns the created GPU context.
 *
 * This function invokes the asynchronous createContext() factory function to
 * create a GPU context, then waits for its completion using
 * waitForContextFuture. The returned Context holds handles to the WebGPU
 * instance, adapter, device, and queue, and is used for subsequent GPU
 * operations.
 *
 * @return Context The fully initialized GPU context.
 *
 * @code
 * Context ctx = waitForContext();
 * // Now ctx can be used for GPU operations.
 * @endcode
 */
inline Context createContext(const WGPUInstanceDescriptor &desc = {},
                             const WGPURequestAdapterOptions &adapterOpts = {},
                             const WGPUDeviceDescriptor &devDescriptor = {}) {
  std::future<Context> contextFuture =
      createContextAsync(desc, adapterOpts, devDescriptor);
  return waitForContextFuture<Context>(contextFuture);
}

#ifndef __EMSCRIPTEN__
#if USE_DAWN_API
/**
 * @brief Retrieves the list of available GPU adapters from the Dawn instance.
 *
 * This function creates a Dawn instance using the provided context's instance
 * handle, then enumerates and returns the available GPU adapters as a vector.
 *
 * @param ctx The Context containing the WebGPU instance handle.
 * @return std::vector<dawn::native::Adapter> A vector of available GPU
 * adapters.
 *
 * @code
 * std::vector<dawn::native::Adapter> adapters = getAdapters(ctx);
 * @endcode
 */
inline std::vector<dawn::native::Adapter> getAdapters(Context &ctx) {
  dawn::native::Instance dawnInstance(
      reinterpret_cast<dawn::native::InstanceBase *>(ctx.instance));
  return dawnInstance.EnumerateAdapters();
}

/**
 * @brief Formats the given vector of Dawn adapters into a single concatenated
 * string.
 *
 * This function iterates over each Dawn adapter in the provided vector,
 * retrieves its description using the WebGPU API, and converts the description
 * from a WGPUStringView to an std::string using the formatWGPUStringView
 * helper. The resulting descriptions are concatenated into a single string
 * separated by newline characters.
 *
 * @param adapters A vector of Dawn adapters obtained from a WebGPU instance.
 * @return std::string A newline-delimited string listing each adapter's
 * description.
 *
 * @code
 * std::string adapterList = formatAdapters(adapters);
 * @endcode
 */
inline std::string
formatAdapters(const std::vector<dawn::native::Adapter> &adapters) {
  std::string adapterList;
  for (size_t i = 0; i < adapters.size(); ++i) {
    auto adapterPtr = adapters[i].Get();
    if (adapterPtr) {
      WGPUAdapterInfo info = {};
      wgpuAdapterGetInfo(adapterPtr, &info);
      std::string desc = formatWGPUStringView(info.description);
      adapterList += "GPU Adapter [" + std::to_string(i) + "]: " + desc + "\n";
      wgpuAdapterInfoFreeMembers(info);
    }
  }
  return adapterList;
}

/**
 * @brief Lists the available GPU adapters in the current WebGPU instance.
 *
 * This function retrieves the list of available GPU adapters using the
 * getAdapters helper function, then formats and returns the adapter
 * descriptions as a single string using the formatAdapters helper function.
 *
 * @param ctx The Context containing the WebGPU instance handle.
 * @return std::string A newline-delimited string listing each adapter's
 * description.
 *
 * @code
 * std::string adapterList = listAdapters(ctx);
 * @endcode
 */
inline std::string listAdapters(Context &ctx) {
  auto adapters = getAdapters(ctx);
  return formatAdapters(adapters);
}

/**
 * @brief Asynchronously creates a GPU context using the specified GPU index.
 *
 * This function creates a WebGPU instance, retrieves the available GPU
 * adapters, and selects the adapter at the specified index. It then requests a
 * device from the selected adapter and sets up a logging callback for device
 * errors. The function returns a future that will be fulfilled with the
 * created Context once all operations are complete.
 *
 * @param gpuIdx The index of the GPU adapter to use.
 * @param desc Instance descriptor for the WebGPU instance (optional)
 * @param devDescriptor Device descriptor for the WebGPU device (optional)
 * @return std::future<Context> A future that will eventually hold the created
 * Context.
 *
 * @code
 * std::future<Context> contextFuture = createContextByGpuIdxAsync(0);
 * Context ctx = waitForContextFuture(contextFuture);
 * @endcode
 */
inline std::future<Context>
createContextByGpuIdxAsync(int gpuIdx, const WGPUInstanceDescriptor &desc = {},
                           const WGPUDeviceDescriptor &devDescriptor = {}) {
  auto promise = std::make_shared<std::promise<Context>>();
  Context ctx;

  ctx.instance = wgpuCreateInstance(&desc);

  if (!ctx.instance) {
    promise->set_exception(std::make_exception_ptr(
        std::runtime_error("Failed to create WebGPU instance.")));
    return promise->get_future();
  }
  check(ctx.instance, "Initialize WebGPU", __FILE__, __LINE__);

  // Use helper functions to obtain and format the adapters.
  auto adapters = getAdapters(ctx);

  if (gpuIdx >= adapters.size()) {
    promise->set_exception(
        std::make_exception_ptr(std::runtime_error("Invalid GPU index.")));
    return promise->get_future();
  }
  LOG(kDefLog, kInfo, "Using GPU Adapter[%d]", gpuIdx);
  auto adapterPtr = adapters[gpuIdx].Get();
  if (adapterPtr) {
    WGPUAdapterInfo info = {};
    wgpuAdapterGetInfo(adapterPtr, &info);
    LOG(kDefLog, kInfo, "GPU(Adapter)[%d] = %s", gpuIdx,
        formatWGPUStringView(info.description).c_str());
    wgpuAdapterInfoFreeMembers(info);
  }
  ctx.adapter = reinterpret_cast<WGPUAdapter>(adapterPtr);
  dawn::native::GetProcs().adapterAddRef(ctx.adapter);

  LOG(kDefLog, kInfo, "Requesting device");
  // Request the device asynchronously (using our requestDeviceAsync helper).
  auto deviceFuture = requestDeviceAsync(ctx.adapter, devDescriptor);
  try {
    ctx.device = wait(ctx, deviceFuture);
    ctx.deviceStatus = WGPURequestDeviceStatus_Success;
  } catch (const std::exception &ex) {
    promise->set_exception(std::make_exception_ptr(ex));
    return promise->get_future();
  }

  WGPULoggingCallbackInfo loggingCallbackInfo{
      .nextInChain = nullptr,
      .callback =
          [](WGPULoggingType type, WGPUStringView message, void *userdata1,
             void *userdata2) {
            LOG(kDefLog, kError, "Device logging callback: %.*s",
                static_cast<int>(message.length), message.data);
            if (type == WGPULoggingType_Error) {
              throw std::runtime_error("Device error logged.");
            }
          },
      .userdata1 = nullptr,
      .userdata2 = nullptr};
  wgpuDeviceSetLoggingCallback(ctx.device, loggingCallbackInfo);
  ctx.queue = wgpuDeviceGetQueue(ctx.device);
  promise->set_value(std::move(ctx));
  return promise->get_future();
}

/**
 * @brief Synchronously creates a GPU context using the specified GPU index.
 *
 * This function calls the asynchronous createContextByGpuIdxAsync function to
 * create a GPU context, then waits for its completion using
 * waitForContextFuture. The returned Context holds handles to the WebGPU
 * instance, adapter, device, and queue, and is used for subsequent GPU
 * operations.
 *
 * @param gpuIdx The index of the GPU adapter to use.
 * @param desc Instance descriptor for the WebGPU instance (optional)
 * @param devDescriptor Device descriptor for the WebGPU device (optional)
 * @return Context The fully initialized GPU context.
 *
 * @code
 * Context ctx = createContextByGpuIdx(0);
 * @endcode
 */
inline Context
createContextByGpuIdx(int gpuIdx, const WGPUInstanceDescriptor &desc = {},
                      const WGPUDeviceDescriptor &devDescriptor = {}) {
  std::future<Context> contextFuture =
      createContextByGpuIdxAsync(gpuIdx, desc, devDescriptor);
  return waitForContextFuture<Context>(contextFuture);
}

#endif // USE_DAWN_API
#endif // __EMSCRIPTEN__

/**
 * @brief Callback function invoked upon completion of an asynchronous GPU
 * buffer mapping.
 *
 * This callback is triggered when the GPU buffer mapping for a readback buffer
 * is completed. It verifies that the mapping operation was successful,
 * retrieves the mapped memory, copies the data from the GPU buffer to a CPU
 * memory region, unmaps the buffer, signals the completion by fulfilling the
 * associated promise, and cleans up the allocated callback data.
 *
 * @param status The mapping status. Expected to be WGPUMapAsyncStatus_Success
 * on success.
 * @param message A string view containing additional information about the
 * mapping operation.
 * @param userdata1 A pointer to a heap-allocated CallbackData structure
 * containing the GPU buffer, buffer size, destination CPU memory pointer, and a
 * promise for signaling completion.
 * @param userdata2 Unused.
 */
inline void bufferMapCallback(WGPUMapAsyncStatus status, WGPUStringView message,
                              void *userdata1, void * /*userdata2*/) {
  const CallbackData *cbData = static_cast<CallbackData *>(userdata1);
  // Check that mapping succeeded.
  check(status == WGPUMapAsyncStatus_Success, "Map readbackBuffer", __FILE__,
        __LINE__);

  // Get the mapped memory.
  const void *mappedData =
      wgpuBufferGetConstMappedRange(cbData->buffer, 0, cbData->bufferSize);
  check(mappedData, "Get mapped range", __FILE__, __LINE__);

  // Copy the data from the mapped GPU buffer to the CPU memory.
  memcpy(cbData->output, mappedData, cbData->bufferSize);

  // Unmap the buffer.
  wgpuBufferUnmap(cbData->buffer);

  // Signal that the copy has completed.
  // Ensure you use the arrow operator on the shared_ptr to call set_value().
  cbData->promise->set_value();

  // Clean up the dynamically allocated callback data.
  delete cbData;
}

/**
 * @brief Callback function invoked when the GPU queue’s submitted work is
 * complete.
 *
 * This callback is registered with the GPU queue after submitting work. When
 * invoked, it verifies that all queued work completed successfully, and then
 * sets up the buffer mapping callback to initiate the asynchronous mapping of a
 * readback buffer. The readback buffer is mapped to access the processed data
 * on the CPU.
 *
 * @param status The status of the completed work. Expected to be
 * WGPUQueueWorkDoneStatus_Success on success.
 * @param userdata1 A pointer to a heap-allocated CallbackData structure
 * containing the readback buffer, buffer size, destination CPU memory pointer,
 * and a promise to signal completion.
 * @param userdata2 Unused.
 */
inline void queueWorkDoneCallback(WGPUQueueWorkDoneStatus status,
                                  WGPUStringView message,
                                  void *userdata1, void * /*userdata2*/) {
  const CallbackData *cbData = static_cast<CallbackData *>(userdata1);
  // Ensure the queue work finished successfully.
  check(status == WGPUQueueWorkDoneStatus_Success, "Queue work done", __FILE__,
        __LINE__);

  // Set up the buffer mapping callback information.
  WGPUBufferMapCallbackInfo mapCallbackInfo = {
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = bufferMapCallback,
      .userdata1 =
          const_cast<CallbackData *>(cbData), // Pass the callback data.
      .userdata2 = nullptr                    // No additional user data.
  };

  // Begin the asynchronous mapping of the readback buffer.
  wgpuBufferMapAsync(cbData->buffer, WGPUMapMode_Read, 0, cbData->bufferSize,
                     mapCallbackInfo);

  // cbData->buffer needs to be freed, but calling it here will cause a segmentation fault.
  // wgpuBufferRelease(cbData->buffer);
}

/**
 * @brief Copies data from a GPU buffer to CPU memory.
 * @param[in] ctx Context instance to manage the operation
 * @param[out] data Pointer to the CPU memory to copy the data to
 * @param[in] bufferSize Size of the data buffer in bytes
 * @param[in] op StagingBuffer instance to manage the operation
 * @param[in] sourceOffset Offset in the GPU buffer to start copying from.
 *
 * @code
 * toCPU(ctx, tensor, data, bufferSize);
 * @endcode
 */

// NOTE: I think this one is redundant? CopyData not used externally.
inline std::future<void> toCPUAsync(Context &ctx, void *data, size_t bufferSize,
                                    CopyData &op, size_t sourceOffset = 0) {
  // Submit the command buffer and release it.
  wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);
  wgpuCommandBufferRelease(op.commandBuffer);

  // Create a promise and get its future.
  auto promise = std::make_shared<std::promise<void>>();

  // Allocate callback data so it remains valid until the async
  // chain finishes.
  CallbackData *cbData = new CallbackData{
      op.readbackBuffer, // The GPU buffer to be read back.
      bufferSize,
      data,    // CPU memory destination.
      promise, // The promise to be signaled.
  };

  // Set up the work-done callback to initiate the buffer mapping.
  WGPUQueueWorkDoneCallbackInfo workDoneCallbackInfo = {
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = queueWorkDoneCallback,
      .userdata1 = const_cast<CallbackData *>(cbData),
      .userdata2 = nullptr};

  // Begin the asynchronous chain by registering the queue work-done callback.
  wgpuQueueOnSubmittedWorkDone(ctx.queue, workDoneCallbackInfo);

  // Release the readback buffer as it is no longer needed.
  if (op.readbackBuffer) {
    wgpuBufferRelease(op.readbackBuffer);
  }

  return promise->get_future();
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
 * @param[in] bufferSize Size to read in bytes as out data.
 * @param[out] data Pointer to the CPU memory to copy the data to
 * @param[in] sourceOffset Offset in the GPU buffer to start copying from.
 */
inline std::future<void> toCPUAsync(Context &ctx, Tensor &tensor, void *data,
                                    size_t bufferSize,
                                    size_t sourceOffset = 0) {
  // Create a promise that will later be satisfied when the async copy
  // completes.
  auto promise = std::make_shared<std::promise<void>>();

  // Create a readback buffer that will be used for copying and mapping.
  WGPUBufferDescriptor readbackBufferDescriptor = {
      .label = {.data = nullptr, .length = 0},
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
      .size = bufferSize, // Size of the readback buffer.
  };
  WGPUBuffer readbackBuffer =
      wgpuDeviceCreateBuffer(ctx.device, &readbackBufferDescriptor);

  // Create a command encoder and record a copy from the tensor GPU buffer
  WGPUCommandEncoder commandEncoder =
      wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
  wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, tensor.data.buffer,
                                       sourceOffset, readbackBuffer, 0,
                                       bufferSize);
  // Finish recording by creating a command buffer and release the encoder.
  WGPUCommandBuffer commandBuffer =
      wgpuCommandEncoderFinish(commandEncoder, nullptr);
  wgpuCommandEncoderRelease(commandEncoder);
  check(commandBuffer, "Create command buffer", __FILE__, __LINE__);

  // Submit the work to the queue and release the command buffer immediately.
  wgpuQueueSubmit(ctx.queue, 1, &commandBuffer);
  wgpuCommandBufferRelease(commandBuffer);

  // Allocate callback data
  CallbackData *cbData = new CallbackData{
      readbackBuffer, // The readback buffer to map.
      bufferSize,     // The size of the copy.
      data,           // CPU memory destination.
      promise         // The promise to signal when done.
  };

  // Set up the work-done callback. When the queue’s submitted work is
  // completed, it is routed to queueWorkDoneCallback which then starts the
  // asynchronous map.
  WGPUQueueWorkDoneCallbackInfo workDoneCallbackInfo = {
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = queueWorkDoneCallback,
      .userdata1 = cbData,
      .userdata2 = nullptr,
  };

  // Register the callback. The async chain continues inside
  // queueWorkDoneCallback.
  wgpuQueueOnSubmittedWorkDone(ctx.queue, workDoneCallbackInfo);

  return promise->get_future();
}

inline std::future<void> toCPUAsync(Context &ctx, WGPUBuffer buffer, void *data,
                                    size_t bufferSize,
                                    size_t sourceOffset = 0) {

  // Create an operation structure (here we reuse CopyData solely for its
  // members that we need to create a readback buffer and command buffer).
  CopyData op;

  // Create the promise that will be fulfilled once the copy is done.
  auto promise = std::make_shared<std::promise<void>>();

  // Create a readback buffer that we can map for reading.
  {
    WGPUBufferDescriptor readbackBufferDescriptor = {
        .label = {.data = nullptr, .length = 0},
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = bufferSize,
    };
    op.readbackBuffer =
        wgpuDeviceCreateBuffer(ctx.device, &readbackBufferDescriptor);
  }

  // Create a command encoder which copies from the provided buffer to the
  // readback buffer.
  {
    WGPUCommandEncoder commandEncoder =
        wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, buffer, sourceOffset,
                                         op.readbackBuffer, 0, bufferSize);
    op.commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
    wgpuCommandEncoderRelease(commandEncoder);
    check(op.commandBuffer, "Create command buffer", __FILE__, __LINE__);
  }

  // Submit the command and release the command buffer.
  wgpuQueueSubmit(ctx.queue, 1, &op.commandBuffer);
  wgpuCommandBufferRelease(op.commandBuffer);

  // Allocate callback data
  CallbackData *cbData = new CallbackData{
      op.readbackBuffer, // The readback buffer created above.
      bufferSize,        // Size of the copy.
      data,   // Destination CPU memory.         // Offset in the GPU buffer.
      promise // Our promise to satisfy when done.
  };

  // Set up the queue work-done callback info.
  WGPUQueueWorkDoneCallbackInfo workDoneCallbackInfo = {
      .mode = WGPUCallbackMode_AllowSpontaneous,
      .callback = queueWorkDoneCallback, // Our free function callback.
      .userdata1 = cbData,               // Pass the callback data pointer.
      .userdata2 = nullptr};

  // Start the asynchronous chain by registering the work-done callback.
  wgpuQueueOnSubmittedWorkDone(ctx.queue, workDoneCallbackInfo);

  return promise->get_future();
}

/**
 * @brief Overload of the toCPU function to copy data from a GPU buffer to CPU
 * memory for an array of floats instead of a pointer to a float buffer.
 * @param[in] ctx Context instance to manage the operation
 * @param[in] tensor Tensor instance representing the GPU buffer to copy from
 * @param[out] data Array of floats to copy the data to
 *
 * @code
 * std::future<void> toCPUFuture = toCPU(ctx, tensor, data);
 * wait(ctx, toCPUFuture);
 * @endcode
 */
template <size_t N>
inline std::future<void> toCPUAsync(Context &ctx, Tensor &tensor,
                                    std::array<float, N> &data,
                                    size_t sourceOffset = 0) {
  return toCPUAsync(ctx, tensor, data.data(), sizeof(data), sourceOffset);
}

/**
 * @brief Synchronous wrapper for copying from a Tensor GPU buffer to CPU
 * memory.
 *
 * This function synchronously waits for the asynchronous copy operation to
 * complete, ensuring that the data is fully transferred from the GPU buffer to
 * the CPU memory before returning.
 *
 * @param ctx Context instance to manage the operation
 * @param tensor Tensor instance representing the GPU buffer to copy from
 * @param data Pointer to the CPU memory to copy the data to
 * @param bufferSize Size of the data buffer in bytes
 * @param instance WGPUInstance used for processing events during waiting
 *
 * @code
 * toCPU(ctx, tensor, data, bufferSize, instance);
 * @endcode
 */
inline void toCPU(Context &ctx, Tensor &tensor, void *data, size_t bufferSize,
                  size_t sourceOffset = 0) {
  auto future = toCPUAsync(ctx, tensor, data, bufferSize, sourceOffset);
  wait(ctx, future);
}

/**
 * @brief Synchronous wrapper for copying from a GPU buffer to CPU memory.
 *
 * This function synchronously waits for the asynchronous copy operation to
 * complete, ensuring that the data is fully transferred from the GPU buffer to
 * the CPU memory before returning.
 *
 * @param ctx Context instance to manage the operation
 * @param buffer WGPUBuffer instance representing the GPU buffer to copy from
 * @param data Pointer to the CPU memory to copy the data to
 * @param size Size of the data buffer in bytes
 * @param instance WGPUInstance used for processing events during waiting
 *
 * @code
 * toCPU(ctx, buffer, data, size, instance);
 * @endcode
 */
inline void toCPU(Context &ctx, WGPUBuffer buffer, void *data, size_t size,
                  size_t sourceOffset = 0) {
  auto future = toCPUAsync(ctx, buffer, data, size, sourceOffset);
  wait(ctx, future);
}

/**
 * @brief Synchronous wrapper for copying from a Tensor GPU buffer to CPU
 * memory for an array of floats instead of a pointer to a float buffer.
 *
 * This function synchronously waits for the asynchronous copy operation to
 * complete, ensuring that the data is fully transferred from the GPU buffer to
 * the CPU memory before returning.
 *
 * @param ctx Context instance to manage the operation
 * @param tensor Tensor instance representing the GPU buffer to copy from
 * @param data Array of floats to copy the data to
 * @param instance WGPUInstance used for processing events during waiting
 *
 * @code
 * toCPU(ctx, tensor, data, instance);
 * @endcode
 */
template <size_t N>
inline void toCPU(Context &ctx, Tensor &tensor, std::array<float, N> &data,
                  size_t sourceOffset = 0) {
  auto future = toCPUAsync(ctx, tensor, data, sourceOffset);
  wait(ctx, future);
}

inline void toCPU(Context &ctx, Tensor &tensor, NumType dtype, void *output,
                  size_t sourceOffset = 0) {
  size_t numElements = size(tensor.shape);
  switch (dtype) {
  // These types are directly supported.
  case kf16:
  case kf32:
  case ku32:
  case ki32:
    toCPU(ctx, tensor, output, tensor.data.size, sourceOffset);
    break;

  // kf64 to reverse bit‐packing of doubles.
  case kf64: {
    // We expect each double to have been packed into 2 uint32_t values.
    std::vector<uint32_t> tmp(numElements * 2);
    // Read the packed data (each element is 4 bytes)
    toCPU(ctx, tensor, tmp.data(), tmp.size() * sizeof(uint32_t), sourceOffset);
    double *dst = static_cast<double *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      uint32_t low = tmp[2 * i];
      uint32_t high = tmp[2 * i + 1];
      // Reassemble the 64-bit raw representation.
      uint64_t bits = (static_cast<uint64_t>(high) << 32) | low;
      // Copy the raw bits into a double.
      double d;
      std::memcpy(&d, &bits, sizeof(double));
      dst[i] = d;
    }
    break;
  }

  // For int8_t: four 8‑bit ints packed into one int32_t.
  case ki8: {
    size_t packedCount = (numElements + 3) / 4;
    std::vector<int32_t> tmp(packedCount);
    toCPU(ctx, tensor, tmp.data(), tmp.size() * sizeof(int32_t), sourceOffset);
    int8_t *dst = static_cast<int8_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 4;
      size_t shift = (i % 4) * 8;
      dst[i] = static_cast<int8_t>((tmp[idx] >> shift) & 0xFF);
    }
    break;
  }

  // For int16_t: two 16‑bit ints packed into one int32_t.
  case ki16: {
    size_t packedCount = (numElements + 1) / 2;
    std::vector<int32_t> tmp(packedCount);
    toCPU(ctx, tensor, tmp.data(), tmp.size() * sizeof(int32_t), sourceOffset);
    int16_t *dst = static_cast<int16_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 2;
      size_t shift = (i % 2) * 16;
      dst[i] = static_cast<int16_t>((tmp[idx] >> shift) & 0xFFFF);
    }
    break;
  }

  // For int64_t: each 64‑bit int was packed into two int32_t.
  case ki64: {
    std::vector<int32_t> tmp(numElements * 2);
    toCPU(ctx, tensor, tmp.data(), tmp.size() * sizeof(int32_t), sourceOffset);
    int64_t *dst = static_cast<int64_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      int32_t low = tmp[2 * i];
      int32_t high = tmp[2 * i + 1];
      dst[i] =
          (static_cast<int64_t>(high) << 32) | (static_cast<uint32_t>(low));
    }
    break;
  }

  // For uint8_t: four 8‑bit uints packed into one uint32_t.
  case ku8: {
    size_t packedCount = (numElements + 3) / 4;
    std::vector<uint32_t> tmp(packedCount);
    toCPU(ctx, tensor, tmp.data(), tmp.size() * sizeof(uint32_t), sourceOffset);
    uint8_t *dst = static_cast<uint8_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 4;
      size_t shift = (i % 4) * 8;
      dst[i] = static_cast<uint8_t>((tmp[idx] >> shift) & 0xFF);
    }
    break;
  }

  // For uint16_t: two 16‑bit uints packed into one uint32_t.
  case ku16: {
    size_t packedCount = (numElements + 1) / 2;
    std::vector<uint32_t> tmp(packedCount);
    toCPU(ctx, tensor, tmp.data(), tmp.size() * sizeof(uint32_t), sourceOffset);
    uint16_t *dst = static_cast<uint16_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 2;
      size_t shift = (i % 2) * 16;
      dst[i] = static_cast<uint16_t>((tmp[idx] >> shift) & 0xFFFF);
    }
    break;
  }

  // For uint64_t: each 64‑bit unsigned int was packed into two uint32_t.
  case ku64: {
    std::vector<uint32_t> tmp(numElements * 2);
    toCPU(ctx, tensor, tmp.data(), tmp.size() * sizeof(uint32_t), sourceOffset);
    uint64_t *dst = static_cast<uint64_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      uint32_t low = tmp[2 * i];
      uint32_t high = tmp[2 * i + 1];
      dst[i] = (static_cast<uint64_t>(high) << 32) | low;
    }
    break;
  }

  default:
    LOG(kDefLog, kError, "Unsupported dtype in toCPUUnpack");
    break;
  }
}

inline void toCPU(Context &ctx, WGPUBuffer buffer, NumType dtype, void *output,
                  size_t numElements, size_t sourceOffset = 0) {
  switch (dtype) {
  // Directly supported types.
  case kf16:
  case kf32:
  case ku32:
  case ki32: {
    size_t byteSize = sizeBytes(dtype, numElements);
    toCPU(ctx, buffer, output, byteSize, sourceOffset);
    break;
  }

  // kf64 to reverse bit‐packing of doubles.
  case kf64: {
    // We expect each double to have been packed into 2 uint32_t values.
    std::vector<uint32_t> tmp(numElements * 2);
    // Read the packed data (each element is 4 bytes)
    toCPU(ctx, buffer, tmp.data(), tmp.size() * sizeof(uint32_t), sourceOffset);
    double *dst = static_cast<double *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      uint32_t low = tmp[2 * i];
      uint32_t high = tmp[2 * i + 1];
      // Reassemble the 64-bit raw representation.
      uint64_t bits = (static_cast<uint64_t>(high) << 32) | low;
      // Copy the raw bits into a double.
      double d;
      std::memcpy(&d, &bits, sizeof(double));
      dst[i] = d;
    }
    break;
  }

  // For int8_t: four 8‑bit ints packed into one int32_t.
  case ki8: {
    size_t packedCount = (numElements + 3) / 4;
    std::vector<int32_t> tmp(packedCount);
    toCPU(ctx, buffer, tmp.data(), tmp.size() * sizeof(int32_t), sourceOffset);
    int8_t *dst = static_cast<int8_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 4;
      size_t shift = (i % 4) * 8;
      dst[i] = static_cast<int8_t>((tmp[idx] >> shift) & 0xFF);
    }
    break;
  }

  // For int16_t: two 16‑bit ints packed into one int32_t.
  case ki16: {
    size_t packedCount = (numElements + 1) / 2;
    std::vector<int32_t> tmp(packedCount);
    toCPU(ctx, buffer, tmp.data(), packedCount * sizeof(int32_t), sourceOffset);
    int16_t *dst = static_cast<int16_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 2;
      size_t shift = (i % 2) * 16;
      dst[i] = static_cast<int16_t>((tmp[idx] >> shift) & 0xFFFF);
    }
    break;
  }

  // For int64_t: each 64‑bit int is packed into two int32_t.
  case ki64: {
    std::vector<int32_t> tmp(numElements * 2);
    toCPU(ctx, buffer, tmp.data(), tmp.size() * sizeof(int32_t), sourceOffset);
    int64_t *dst = static_cast<int64_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      int32_t low = tmp[2 * i];
      int32_t high = tmp[2 * i + 1];
      dst[i] =
          (static_cast<int64_t>(high) << 32) | (static_cast<uint32_t>(low));
    }
    break;
  }

  // For uint8_t: four 8‑bit uints packed into one uint32_t.
  case ku8: {
    size_t packedCount = (numElements + 3) / 4;
    std::vector<uint32_t> tmp(packedCount);
    toCPU(ctx, buffer, tmp.data(), packedCount * sizeof(uint32_t),
          sourceOffset);
    uint8_t *dst = static_cast<uint8_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 4;
      size_t shift = (i % 4) * 8;
      dst[i] = static_cast<uint8_t>((tmp[idx] >> shift) & 0xFF);
    }
    break;
  }

  // For uint16_t: two 16‑bit uints packed into one uint32_t.
  case ku16: {
    size_t packedCount = (numElements + 1) / 2;
    std::vector<uint32_t> tmp(packedCount);
    toCPU(ctx, buffer, tmp.data(), packedCount * sizeof(uint32_t),
          sourceOffset);
    uint16_t *dst = static_cast<uint16_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      size_t idx = i / 2;
      size_t shift = (i % 2) * 16;
      dst[i] = static_cast<uint16_t>((tmp[idx] >> shift) & 0xFFFF);
    }
    break;
  }

  // For uint64_t: each 64‑bit unsigned int packed into two uint32_t.
  case ku64: {
    std::vector<uint32_t> tmp(numElements * 2);
    toCPU(ctx, buffer, tmp.data(), tmp.size() * sizeof(uint32_t), sourceOffset);
    uint64_t *dst = static_cast<uint64_t *>(output);
    for (size_t i = 0; i < numElements; ++i) {
      uint32_t low = tmp[2 * i];
      uint32_t high = tmp[2 * i + 1];
      dst[i] = (static_cast<uint64_t>(high) << 32) | low;
    }
    break;
  }

  default:
    LOG(kDefLog, kError, "Unsupported dtype in toCPU (raw buffer override)");
    break;
  }
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

// Overload for float: directly copy the float data.
inline void toGPU(Context &ctx, const float *data, WGPUBuffer buffer,
                  size_t size) {
  toGPU(ctx, static_cast<const void *>(data), buffer, size);
}

// Overload for half: directly copy the half data.
inline void toGPU(Context &ctx, const half *data, WGPUBuffer buffer,
                  size_t size) {
  toGPU(ctx, static_cast<const void *>(data), buffer, size);
}

// Overload for double: bit-pack each double into two 32‑bit unsigned integers.
inline void toGPU(Context &ctx, const double *data, WGPUBuffer buffer,
                  size_t size) {
  // Number of doubles = size / sizeof(double)
  size_t numElements = size / sizeof(double);
  std::vector<uint32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    uint64_t bits;
    std::memcpy(&bits, &data[i],
                sizeof(double)); // Reinterpret double as raw bits.
    packed[2 * i] = static_cast<uint32_t>(bits & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<uint32_t>(bits >> 32);
  }
  toGPU(ctx, packed.data(), buffer, packed.size() * sizeof(uint32_t));
}

// Overload for int8_t: pack four 8‑bit ints into one 32‑bit integer.
inline void toGPU(Context &ctx, const int8_t *data, WGPUBuffer buffer,
                  size_t size) {
  // Number of int8_t elements equals size (sizeof(int8_t)==1)
  size_t numElements = size;
  size_t packedCount = (numElements + 3) / 4;
  std::vector<int32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 4;
    size_t shift = (i % 4) * 8;
    packed[idx] |= (static_cast<uint8_t>(data[i]) << shift);
    // LOG(kDefLog, kInfo, "toGPU: %d %d %d", data[i], packed[idx], idx);
  }
  toGPU(ctx, packed.data(), buffer, packedCount * sizeof(int32_t));
}

// Overload for int16_t: pack two 16‑bit ints into one 32‑bit integer.
inline void toGPU(Context &ctx, const int16_t *data, WGPUBuffer buffer,
                  size_t size) {
  size_t numElements = size / sizeof(int16_t);
  size_t packedCount = (numElements + 1) / 2;
  std::vector<int32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 2;
    size_t shift = (i % 2) * 16;
    packed[idx] |= (static_cast<uint16_t>(data[i]) << shift);
  }
  toGPU(ctx, packed.data(), buffer, packedCount * sizeof(int32_t));
}

// Overload for int64_t: pack each 64‑bit int into two 32‑bit integers.
inline void toGPU(Context &ctx, const int64_t *data, WGPUBuffer buffer,
                  size_t size) {
  size_t numElements = size / sizeof(int64_t);
  std::vector<int32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    int64_t val = data[i];
    packed[2 * i] = static_cast<int32_t>(val & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<int32_t>((val >> 32) & 0xFFFFFFFF);
  }
  toGPU(ctx, packed.data(), buffer, packed.size() * sizeof(int32_t));
}

// Overload for uint8_t: pack four 8‑bit uints into one 32‑bit unsigned integer.
inline void toGPU(Context &ctx, const uint8_t *data, WGPUBuffer buffer,
                  size_t size) {
  size_t numElements = size; // sizeof(uint8_t)==1
  size_t packedCount = (numElements + 3) / 4;
  std::vector<uint32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 4;
    size_t shift = (i % 4) * 8;
    packed[idx] |= (static_cast<uint32_t>(data[i]) << shift);
  }
  toGPU(ctx, packed.data(), buffer, packedCount * sizeof(uint32_t));
}

// Overload for uint16_t: pack two 16‑bit uints into one 32‑bit unsigned
// integer.
inline void toGPU(Context &ctx, const uint16_t *data, WGPUBuffer buffer,
                  size_t size) {
  size_t numElements = size / sizeof(uint16_t);
  size_t packedCount = (numElements + 1) / 2;
  std::vector<uint32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 2;
    size_t shift = (i % 2) * 16;
    packed[idx] |= (static_cast<uint32_t>(data[i]) << shift);
  }
  toGPU(ctx, packed.data(), buffer, packedCount * sizeof(uint32_t));
}

// Overload for uint64_t: pack each 64‑bit uint into two 32‑bit unsigned
// integers.
inline void toGPU(Context &ctx, const uint64_t *data, WGPUBuffer buffer,
                  size_t size) {
  size_t numElements = size / sizeof(uint64_t);
  std::vector<uint32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    uint64_t val = data[i];
    packed[2 * i] = static_cast<uint32_t>(val & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<uint32_t>(val >> 32);
  }
  toGPU(ctx, packed.data(), buffer, packed.size() * sizeof(uint32_t));
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

// Overload for double: bit-pack each double into two 32‑bit unsigned integers.
inline void toGPU(Context &ctx, const double *data, Tensor &tensor) {
  size_t numElements = tensor.data.size / sizeof(double);
  std::vector<uint32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    uint64_t bits;
    std::memcpy(&bits, &data[i],
                sizeof(double)); // Reinterpret double as raw bits.
    packed[2 * i] = static_cast<uint32_t>(bits & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<uint32_t>(bits >> 32);
  }
  toGPU(ctx, packed.data(), tensor.data.buffer,
        packed.size() * sizeof(uint32_t));
}

// Overload for int8_t: pack four 8‑bit integers into one 32‑bit integer
inline void toGPU(Context &ctx, const int8_t *data, Tensor &tensor) {
  size_t numElements = size(tensor.shape);
  size_t packedCount = (numElements + 3) / 4;
  std::vector<int32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 4;
    size_t shift = (i % 4) * 8;
    // Pack as unsigned then reinterpret (shader will unpack)
    packed[idx] |= (static_cast<uint8_t>(data[i]) << shift);
  }
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
}

// Overload for int16_t: pack two 16‑bit integers into one 32‑bit integer
inline void toGPU(Context &ctx, const int16_t *data, Tensor &tensor) {
  size_t numElements = size(tensor.shape);
  size_t packedCount = (numElements + 1) / 2;
  std::vector<int32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 2;
    size_t shift = (i % 2) * 16;
    packed[idx] |= (static_cast<uint16_t>(data[i]) << shift);
  }
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
}

// Overload for int64_t: pack each 64‑bit integer into two 32‑bit integers
inline void toGPU(Context &ctx, const int64_t *data, Tensor &tensor) {
  size_t numElements = size(tensor.shape);
  std::vector<int32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    int64_t val = data[i];
    packed[2 * i] = static_cast<int32_t>(val & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<int32_t>((val >> 32) & 0xFFFFFFFF);
  }
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
}

// Overload for uint8_t: pack four 8‑bit unsigned integers into one 32‑bit
// unsigned
inline void toGPU(Context &ctx, const uint8_t *data, Tensor &tensor) {
  size_t numElements = size(tensor.shape);
  size_t packedCount = (numElements + 3) / 4;
  std::vector<uint32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 4;
    size_t shift = (i % 4) * 8;
    packed[idx] |= (static_cast<uint32_t>(data[i]) << shift);
  }
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
}

// Overload for uint16_t: pack two 16‑bit unsigned integers into one 32‑bit
// unsigned
inline void toGPU(Context &ctx, const uint16_t *data, Tensor &tensor) {
  size_t numElements = size(tensor.shape);
  size_t packedCount = (numElements + 1) / 2;
  std::vector<uint32_t> packed(packedCount, 0);
  for (size_t i = 0; i < numElements; ++i) {
    size_t idx = i / 2;
    size_t shift = (i % 2) * 16;
    packed[idx] |= (static_cast<uint32_t>(data[i]) << shift);
  }
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
}

// Overload for uint64_t: pack each 64‑bit unsigned integer into two 32‑bit
// unsigned
inline void toGPU(Context &ctx, const uint64_t *data, Tensor &tensor) {
  size_t numElements = size(tensor.shape);
  std::vector<uint32_t> packed(numElements * 2);
  for (size_t i = 0; i < numElements; ++i) {
    uint64_t val = data[i];
    packed[2 * i] = static_cast<uint32_t>(val & 0xFFFFFFFF);
    packed[2 * i + 1] = static_cast<uint32_t>(val >> 32);
  }
  wgpuQueueWriteBuffer(ctx.queue, tensor.data.buffer, 0, packed.data(),
                       tensor.data.size);
}

template <typename Params>
inline void toGPU(Context &ctx, Params &params, Kernel &op) {
  // TODO(avh): Maintain params metadata in Kernel and check for consistency.
  // If a kernel does not have parameters this will quietly overwrite
  // the last buffer in the bind group with the parameters buffer.
  if (op->numBindings > 0) {
    wgpuQueueWriteBuffer(ctx.queue, op->buffers[op->numBindings - 1], 0,
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
    wgpuComputePassEncoderSetPipeline(computePassEncoder, op->computePipeline);
    wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, op->bindGroup, 0,
                                       nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        computePassEncoder, op->totalWorkgroups[0], op->totalWorkgroups[1],
        op->totalWorkgroups[2]);
    wgpuComputePassEncoderEnd(computePassEncoder);
    wgpuComputePassEncoderRelease(computePassEncoder);
    op->commandBuffer = wgpuCommandEncoderFinish(commandEncoder, nullptr);
    wgpuCommandEncoderRelease(commandEncoder);
    op->used = false;
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
 * @brief Packages the shader compilation information along with a promise for
 * asynchronous signaling.
 *
 * This structure holds a pointer to a CompilationInfo instance that collects
 * details such as status, messages, line numbers, and positions from the shader
 * compilation. It also contains a shared pointer to a std::promise<void> which
 * is used to signal the completion of the asynchronous shader compilation
 * process.
 */
struct CompData {
  CompilationInfo *compInfo;
  std::shared_ptr<std::promise<void>> compPromise;
};

/**
 * @brief A factory function to create a kernel asynchronously on the GPU.
 * The kernel is created with the given WGSL code, input tensors,
 * output tensor, and optional parameters.
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
 * @param[in] totalWorkgroups Shape of the workgroup
 * @param[in] params Optional parameters for the kernel. If the kernel does
 * not have any parameters, use NoParam. This is cast as void* to allow for
 * arbitrary types to be passed as parameters.
 * @param[in] paramsSize Size of the parameters buffer in bytes.
 * @return Kernel instance representing the created kernel
 *
 * @code
 * std::future<Kernel> kernelFuture = createKernelAsync(ctx, code, dataBindings,
 numInputs, output, nThreads, params, paramsSize);
 * Kernel kernel = wait(ctx.instance, kernelFuture);
 * @endcode

 */
inline std::future<Kernel>
createKernelAsync(Context &ctx, const KernelCode &code,
                  const Tensor *dataBindings, size_t numTensors,
                  const size_t *viewOffsets, const Shape &totalWorkgroups,
                  const void *params = nullptr, size_t paramsSize = 0,
                  CompilationInfo *compilationInfo = nullptr,
                  const char *cacheKey = nullptr) {
  // Create a cache key by the pointer values of the data bindings and the
  // kernel code
  if (cacheKey != nullptr &&
      ctx.kernelPool.data.find(cacheKey) != ctx.kernelPool.data.end()) {
    std::promise<Kernel> ready;
    ready.set_value(ctx.kernelPool.data[cacheKey]);
    return ready.get_future();
  }

  // Create an outer promise for the new kernel.
  std::promise<Kernel> outerPromise;
  std::future<Kernel> outerFuture = outerPromise.get_future();

  assert(totalWorkgroups.rank == 3);
  WGPUDevice device = ctx.device;
  WGPUQueue queue = ctx.queue;
  Kernel op(new RawKernel());
  // paramIndex is the index into bgLayoutEntries for the parameters buffer If
  // there are no parameters for the kernel, paramsSize == 0 and paramIndex is
  // effectively undefined (== -1)
  size_t paramIndex = static_cast<size_t>(-1);
  // Note: paramIndex is undefined unless paramsSize > 0
  size_t numBindings = numTensors;
  if (paramsSize > 0) {
    numBindings++;                // parameters buffer
    paramIndex = numBindings - 1; // index of the parameters buffer within
                                  // op.buffers, op.bufferSizes and
                                  // bgLayoutEntries
  }

  op->buffers = std::make_unique<WGPUBuffer[]>(numBindings);
  op->bufferSizes = std::make_unique<size_t[]>(numBindings);
  op->numBindings = numBindings;

  // Create layout entries for input buffers
  std::vector<WGPUBindGroupLayoutEntry> bgLayoutEntries(numBindings);
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
      .entries = bgLayoutEntries.data()};
  WGPUBindGroupLayout bgLayout =
      wgpuDeviceCreateBindGroupLayout(device, &bgLayoutDesc);

  // Assign buffers from dataBindings.
  for (size_t i = 0; i < numTensors; ++i) {
    op->buffers[i] = dataBindings[i].data.buffer;
    op->bufferSizes[i] = dataBindings[i].data.size;
  }
  // Create a buffer for the Params struct
  if (paramsSize > 0) {
    WGPUBufferDescriptor paramsBufferDesc = {
        .label = {.data = nullptr, .length = 0},
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size = paramsSize,
        .mappedAtCreation = false,
    };
    op->buffers[paramIndex] = wgpuDeviceCreateBuffer(device, &paramsBufferDesc);
    op->bufferSizes[paramIndex] = paramsSize;
    wgpuQueueWriteBuffer(queue, op->buffers[paramIndex], 0, params, paramsSize);
    LOG(kDefLog, kTrace, "Params buffer written");
  } else {
    LOG(kDefLog, kTrace, "No params buffer needed");
  }

  // Build bind group entries and the bind group.
  std::vector<WGPUBindGroupEntry> bindGroupEntries(numBindings);
  for (size_t i = 0; i < numTensors; ++i) {
    bindGroupEntries[i] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(i),
        .buffer = op->buffers[i],
        .offset = viewOffsets[i],
        .size = op->bufferSizes[i],
    };
  }
  if (paramsSize > 0) {
    LOG(kDefLog, kInfo, "Create bind group entry for the params buffer");
    LOG(kDefLog, kInfo, "paramIndex: %d", paramIndex);
    bindGroupEntries[paramIndex] = WGPUBindGroupEntry{
        .binding = static_cast<uint32_t>(paramIndex),
        .buffer = op->buffers[paramIndex],
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
  op->bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);

  // Create pipeline layout.
  WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &bgLayout,
  };
  WGPUPipelineLayout pipelineLayout =
      wgpuDeviceCreatePipelineLayout(device, &pipelineLayoutDesc);

  // Prepare the WGSL source and shader module descriptor.
  WGPUShaderSourceWGSL wgslDesc = {
      .chain = {.sType = WGPUSType_ShaderSourceWGSL},
      .code = {.data = code.data.c_str(), .length = code.data.length()}};
  WGPUShaderModuleDescriptor shaderModuleDesc = {};
  shaderModuleDesc.nextInChain = &wgslDesc.chain;
  shaderModuleDesc.label = {code.label.c_str(), code.label.length()};

  // Create the shader module.
  WGPUShaderModule shaderModule =
      wgpuDeviceCreateShaderModule(device, &shaderModuleDesc);

  // If compilation info is requested, register the callback immediately.
  if (compilationInfo) {
    auto compPromise = std::make_shared<std::promise<void>>();
    std::future<void> compFuture = compPromise->get_future();
    // Allocate helper data to pass to the callback.
    auto *compData = new CompData{compilationInfo, compPromise};

    auto compilationCallback = [](WGPUCompilationInfoRequestStatus status,
                                  WGPUCompilationInfo const *info,
                                  void *userdata1, void * /*userdata2*/) {
      CompData *cd = reinterpret_cast<CompData *>(userdata1);
      if (info && cd->compInfo) {
        cd->compInfo->status = status;
        for (uint32_t i = 0; i < info->messageCount; ++i) {
          cd->compInfo->messages.push_back(
              std::string(info->messages[i].message.data,
                          info->messages[i].message.length));
          cd->compInfo->lineNums.push_back(info->messages[i].lineNum);
          cd->compInfo->linePos.push_back(info->messages[i].linePos);
        }
        cd->compInfo->finished = true;
      }
      cd->compPromise->set_value();
      delete cd;
    };

    WGPUCompilationInfoCallbackInfo compilationCallbackInfo = {};
    compilationCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    compilationCallbackInfo.callback = compilationCallback;
    compilationCallbackInfo.userdata1 = compData;
    compilationCallbackInfo.userdata2 = nullptr;

    // Register callback and then wait for the result.
    wgpuShaderModuleGetCompilationInfo(shaderModule, compilationCallbackInfo);
    wait(ctx, compFuture);
  }

  // Now create the compute pipeline using the shader module.
  WGPUComputePipelineDescriptor computePipelineDesc = {};
  computePipelineDesc.layout = pipelineLayout;
  computePipelineDesc.compute.module = shaderModule;
  computePipelineDesc.compute.entryPoint = {code.entryPoint.c_str(),
                                            code.entryPoint.length()};
  computePipelineDesc.label = {code.label.c_str(), code.label.length()};
  op->computePipeline =
      wgpuDeviceCreateComputePipeline(device, &computePipelineDesc);

  op->totalWorkgroups = {totalWorkgroups[0], totalWorkgroups[1],
                         totalWorkgroups[2]};

  resetCommandBuffer(device, op);
  if (cacheKey != nullptr)
    ctx.kernelPool.data[cacheKey] = op;

  outerPromise.set_value(op);
  return outerFuture;
}

/*
 * @brief Overload which wraps the createKernelAsync factory function to create
 * a kernel on the GPU. This overload uses takes a pointer and size for the
 * input tensors instead of a static collection and a void pointer for params
 * instead of a static type.
 *
 * @param[in] ctx Context instance to manage the kernel
 * @param[in] code WGSL code for the kernel
 * @param[in] dataBindings Pointer to a span of tensors bound to the kernel
 * @param[in] numTensors Number of tensors in the dataBindings span
 * @param[in] totalWorkgroups Number of workgroups in the x, y, z grid, must be
 * a Shape of rank == 3.
 * @param[in] params Optional parameters for the kernel. If the kernel does
 * not have any parameters, use NoParam.
 * @return Kernel instance representing the created kernel
 *
 * @code
 * std::future<Kernel> kernelFuture = createKernel(ctx, code, tensorData,
 * output,totalWorkgroups, params); Kernel kernel = wait(ctx.instance,
 * kernelFuture);
 * @endcode
 */
inline Kernel createKernel(Context &ctx, const KernelCode &code,
                           const Tensor *dataBindings, size_t numTensors,
                           const size_t *viewOffsets,
                           const Shape &totalWorkgroups,
                           const void *params = nullptr, size_t paramsSize = 0,
                           CompilationInfo *compilationInfo = nullptr,
                           const char *cacheKey = nullptr) {
  std::future<Kernel> kernelFuture = createKernelAsync(
      ctx, code, dataBindings, numTensors, viewOffsets, totalWorkgroups, params,
      paramsSize, compilationInfo, cacheKey);
  return wait(ctx, kernelFuture);
}

/**
 * @brief Overload which wraps the createKernelAsync factory function to create
 * a kernel asynchronously on the GPU. This overload uses takes a static
 * collection of input tensors instead of a pointer and a statically determined
 * ParamsType instead of casting params to a void pointer.
 *
 * @param[in] ctx Context instance to manage the kernel
 * @param[in] code WGSL code for the kernel
 * @param[in] dataBindings A Bindings of tensors whose GPU buffers are bound
 * to the kernel as inputs and outputs.
 * @param[in] totalWorkgroups Number of workgroups in the x, y, z grid, must be
 * a Shape of rank == 3.
 * @param[in] params Optional parameters for the kernel. If the kernel does
 * not have any parameters, use NoParam.
 * @return Kernel instance representing the created kernel
 *
 * @code
 * std::future<Kernel> kernelFuture = createKernel(ctx, code, tensorData,
 * output,totalWorkgroups, params); Kernel kernel = wait(ctx.instance,
 * kernelFuture);
 * @endcode
 */
template <typename ParamsType = NoParam, size_t numInputs>
std::future<Kernel>
createKernelAsync(Context &ctx, const KernelCode &code,
                  const Bindings<numInputs> &dataBindings,
                  const Shape &totalWorkgroups,
                  const ParamsType &params = ParamsType{},
                  CompilationInfo *compilationInfo = nullptr,
                  const char *cacheKey = nullptr) {
  if constexpr (!IsNoParam<ParamsType>) {
    return createKernelAsync(ctx, code, dataBindings.data.data(), numInputs,
                             dataBindings.viewOffsets.data(), totalWorkgroups,
                             reinterpret_cast<const void *>(&params),
                             sizeof(ParamsType), compilationInfo, cacheKey);
  } else {
    return createKernelAsync(ctx, code, dataBindings.data.data(), numInputs,
                             dataBindings.viewOffsets.data(), totalWorkgroups,
                             nullptr, 0, compilationInfo, cacheKey);
  }
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
 * @param[in] totalWorkgroups Number of workgroups in the x, y, z grid, must be
 * a Shape of rank == 3.
 * @param[in] params Optional parameters for the kernel. If the kernel does
 * not have any parameters, use NoParam.
 * @return Kernel instance representing the created kernel
 *
 * @code
 * Kernel kernel = createKernel(ctx, code, tensorData, output,totalWorkgroups,
 * params);
 * @endcode
 */
template <typename ParamsType = NoParam, size_t numInputs>
Kernel createKernel(Context &ctx, const KernelCode &code,
                    const Bindings<numInputs> &dataBindings,
                    const Shape &totalWorkgroups,
                    const ParamsType &params = ParamsType{},
                    CompilationInfo *compilationInfo = nullptr,
                    const char *cacheKey = nullptr) {
  if constexpr (!IsNoParam<ParamsType>) {
    return createKernel(ctx, code, dataBindings.data.data(), numInputs,
                        dataBindings.viewOffsets.data(), totalWorkgroups,
                        reinterpret_cast<const void *>(&params),
                        sizeof(ParamsType), compilationInfo, cacheKey);
  } else {
    return createKernel(ctx, code, dataBindings.data.data(), numInputs,
                        dataBindings.viewOffsets.data(), totalWorkgroups,
                        nullptr, 0, compilationInfo, cacheKey);
  }
}

/**
 * @brief Free‑standing callback for dispatchKernel’s asynchronous work‐done.
 *
 * This callback is invoked when the GPU queue signals the completion of the
 * submitted workload for a kernel dispatch. It receives the work-done status
 * and a userdata pointer, which is expected to be a heap‑allocated pointer to a
 * std::promise<void>.
 *
 * On success, the promise is fulfilled by calling set_value(). Otherwise, it is
 * set with an exception. After setting the promise state, the allocated memory
 * for the promise is freed.
 *
 * @param status The status of the work done. Expected to be
 * WGPUQueueWorkDoneStatus_Success on success.
 * @param userdata1 A heap allocated pointer to std::promise<void> which is set
 * when the work is done.
 * @param userdata2 Unused.
 */
inline void dispatchKernelCallback(WGPUQueueWorkDoneStatus status,
                                   WGPUStringView message,
                                   void *userdata1, void * /*userdata2*/) {
  // Cast the userdata pointer back to our heap‑allocated promise.
  auto *p = reinterpret_cast<std::promise<void> *>(userdata1);
  if (status == WGPUQueueWorkDoneStatus_Success) {
    p->set_value();
  } else {
    p->set_exception(std::make_exception_ptr(
        std::runtime_error("Queue work did not complete successfully.")));
  }
  delete p; // free the heap allocation
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
 * std::future<void> dispatchFuture = dispatchKernel(ctx, kernel);
 * wait(ctx.instance, dispatchFuture);
 * @endcode
 */
inline std::future<void> dispatchKernelAsync(Context &ctx, Kernel &kernel) {
  // If the kernel was used before, reset the command buffer.
  if (kernel->used) {
    resetCommandBuffer(ctx.device, kernel);
  }

  // Submit the command buffer and release it.
  wgpuQueueSubmit(ctx.queue, 1, &kernel->commandBuffer);
  wgpuCommandBufferRelease(kernel->commandBuffer);
  kernel->used = true;

  // Allocate a promise on the heap so it remains valid beyond this function’s
  // scope.
  std::promise<void> *promise = new std::promise<void>();
  std::future<void> future = promise->get_future();

  // Set up the callback info.
  WGPUQueueWorkDoneCallbackInfo workDoneCallbackInfo = {};
  workDoneCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  workDoneCallbackInfo.callback = dispatchKernelCallback;
  workDoneCallbackInfo.userdata1 = reinterpret_cast<void *>(promise);
  workDoneCallbackInfo.userdata2 = nullptr;

  wgpuQueueOnSubmittedWorkDone(ctx.queue, workDoneCallbackInfo);

  return future;
}

/**
 * @brief Synchronous wrapper for dispatchKernelAsync. This function submits
 * the kernel to the GPU queue and waits for it to finish executing.
 *
 * @param[in] ctx Context instance to manage the kernel, from which the queue
 * for the GPU is obtained
 * @param[in] kernel Kernel instance to dispatch
 *
 * @code
 * dispatchKernel(ctx, kernel);
 * @endcode
 */
inline void dispatchKernel(Context &ctx, Kernel &kernel) {
  auto future = dispatchKernelAsync(ctx, kernel);
  wait(ctx, future);
}

} // namespace gpu

#endif // GPU_H
