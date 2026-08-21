#include "gpu.hpp"

#include <emscripten/bind.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using emscripten::val;
using namespace gpu;

Shape shape(const val &value, const char *name) {
  if (value["length"].as<size_t>() != 3)
    throw std::invalid_argument(std::string(name) + " must have three dimensions");
  return {value[0].as<size_t>(), value[1].as<size_t>(), value[2].as<size_t>()};
}

NumType numType(const val &array) {
  const auto name = array["constructor"]["name"].as<std::string>();
  if (name == "Float32Array") return kf32;
  if (name == "Int32Array") return ki32;
  if (name == "Float16Array" || name == "Uint16Array") return kf16;
  throw std::invalid_argument(
      "binding data must be Float16Array, Uint16Array, Float32Array, or Int32Array");
}

std::vector<std::byte> bytes(const val &array) {
  const auto length = array["byteLength"].as<size_t>();
  std::vector<std::byte> result(length);
  auto source = val::global("Uint8Array")
                    .new_(array["buffer"], array["byteOffset"], length);
  auto destination = val(emscripten::typed_memory_view(
      length, reinterpret_cast<uint8_t *>(result.data())));
  destination.call<void>("set", source);
  return result;
}

void copy(const void *source, size_t length, const val &array) {
  auto sourceView = val(emscripten::typed_memory_view(
      length, static_cast<const uint8_t *>(source)));
  auto destination = val::global("Uint8Array")
                         .new_(array["buffer"], array["byteOffset"], length);
  destination.call<void>("set", sourceView);
}

class BrowserContext {
  Context context;
  bool running = false;

  struct RunGuard {
    bool &running;
    explicit RunGuard(bool &running) : running(running) {
      if (std::exchange(running, true))
        throw std::runtime_error("a gpu.cpp run is already in progress");
    }
    ~RunGuard() { running = false; }
  };

  template <typename T>
  void download(const Tensor &tensor, const val &array) {
    std::vector<T> values(size(tensor.shape));
    auto future = toCPU(context, tensor, values);
    wait(context, future);
    copy(values.data(), values.size() * sizeof(T), array);
  }

public:
  explicit BrowserContext(bool shaderF16) {
    ContextOptions options;
    if (shaderF16) options.requiredFeatures = {wgpu::FeatureName::ShaderF16};
    context = createContext(options);
  }

  void run(const val &spec) {
    RunGuard guard(running);
    const auto code = spec["code"].as<std::string>();
    const auto workgroupSize = shape(spec["workgroupSize"], "workgroupSize");
    const auto workgroups = shape(spec["workgroups"], "workgroups");
    const auto specs = spec["bindings"];

    std::vector<Tensor> tensors;
    std::vector<Binding> bindings;
    std::vector<val> arrays;
    std::vector<bool> downloads;
    const auto count = specs["length"].as<size_t>();
    tensors.reserve(count);
    bindings.reserve(count);
    arrays.reserve(count);
    downloads.reserve(count);

    for (size_t i = 0; i < count; ++i) {
      const auto binding = specs[i];
      const auto array = binding["data"];
      const auto type = numType(array);
      const auto data = bytes(array);
      if (data.empty() || data.size() % sizeBytes(type))
        throw std::invalid_argument("binding data has an invalid byte length");
      tensors.push_back(createTensor(context, {data.size() / sizeBytes(type)}, type));
      context.queue.WriteBuffer(tensors.back().buffer, 0, data.data(), data.size());

      const auto access = binding["access"].as<std::string>();
      if (access == "read") {
        bindings.push_back(read(tensors.back()));
        downloads.push_back(false);
      } else if (access == "readWrite") {
        bindings.push_back(readWrite(tensors.back()));
        downloads.push_back(true);
      } else {
        throw std::invalid_argument("binding access must be read or readWrite");
      }
      arrays.push_back(array);
    }

    std::vector<std::byte> parameters;
    const auto parameterValue = spec["parameters"];
    if (!parameterValue.isUndefined() && !parameterValue.isNull())
      parameters = bytes(parameterValue);

    auto kernel = createKernel(context, WGSL{code, workgroupSize}, bindings,
                               workgroups, parameters);
    auto dispatched = dispatchKernel(context, kernel);
    wait(context, dispatched);

    for (size_t i = 0; i < tensors.size(); ++i) {
      if (!downloads[i]) continue;
      switch (tensors[i].type) {
      case kf16: download<uint16_t>(tensors[i], arrays[i]); break;
      case kf32: download<float>(tensors[i], arrays[i]); break;
      case ki32: download<int32_t>(tensors[i], arrays[i]); break;
      }
    }
  }
};

BrowserContext *createBrowserContext(bool shaderF16) {
  return new BrowserContext(shaderF16);
}

} // namespace

EMSCRIPTEN_BINDINGS(gpu_cpp_web) {
  emscripten::class_<BrowserContext>("Context")
      .function("run", &BrowserContext::run, emscripten::async());
  emscripten::function("createContext", &createBrowserContext,
                       emscripten::allow_raw_pointers(), emscripten::async());
}
