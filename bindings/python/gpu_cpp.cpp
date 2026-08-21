#include "gpu.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace gpu;

namespace {

Shape shape(const std::vector<py::ssize_t> &dimensions) {
  if (dimensions.size() > Shape::kMaxRank)
    throw std::invalid_argument("tensor rank exceeds 8");
  Shape result;
  result.rank = dimensions.size();
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] < 0)
      throw std::invalid_argument("tensor dimensions cannot be negative");
    result.data[i] = dimensions[i];
  }
  return result;
}

std::vector<py::ssize_t> dimensions(const Shape &shape) {
  std::vector<py::ssize_t> result(shape.rank);
  for (size_t i = 0; i < shape.rank; ++i) result[i] = shape[i];
  return result;
}

void requireContiguous(const py::array &array) {
  if (!(array.flags() & py::array::c_style))
    throw std::invalid_argument("array must be C-contiguous");
}

NumType numType(const py::array &array) {
  if (array.dtype().is(py::dtype::of<float>())) return kf32;
  if (array.dtype().is(py::dtype::of<int32_t>())) return ki32;
  if (array.dtype().is(py::dtype("float16"))) return kf16;
  throw std::invalid_argument("array dtype must be float16, float32, or int32");
}

py::dtype dtype(NumType type) {
  switch (type) {
  case kf16: return py::dtype("float16");
  case kf32: return py::dtype::of<float>();
  case ki32: return py::dtype::of<int32_t>();
  }
  throw std::invalid_argument("unknown numeric type");
}

template <typename T>
std::span<const T> input(const py::buffer_info &buffer) {
  return {static_cast<const T *>(buffer.ptr),
          static_cast<size_t>(buffer.size)};
}

Tensor tensor(Context &context, const py::array &array) {
  requireContiguous(array);
  const auto type = numType(array);
  const auto buffer = array.request();
  const auto tensorShape = shape(buffer.shape);
  switch (type) {
  case kf16:
    return createTensor(context, tensorShape, type, input<uint16_t>(buffer));
  case kf32:
    return createTensor(context, tensorShape, type, input<float>(buffer));
  case ki32:
    return createTensor(context, tensorShape, type, input<int32_t>(buffer));
  }
  throw std::invalid_argument("unknown numeric type");
}

void upload(Context &context, const Tensor &tensor, const py::array &array) {
  requireContiguous(array);
  if (numType(array) != tensor.type)
    throw std::invalid_argument("array and tensor dtypes differ");
  const auto buffer = array.request();
  switch (tensor.type) {
  case kf16: return toGPU(context, tensor, input<uint16_t>(buffer));
  case kf32: return toGPU(context, tensor, input<float>(buffer));
  case ki32: return toGPU(context, tensor, input<int32_t>(buffer));
  }
  throw std::invalid_argument("unknown numeric type");
}

template <typename T>
void download(Context &context, const Tensor &tensor, const py::buffer_info &buffer) {
  auto future = toCPU(
      context, tensor,
      std::span<T>(static_cast<T *>(buffer.ptr), static_cast<size_t>(buffer.size)));
  py::gil_scoped_release release;
  wait(context, future);
}

py::array numpy(Context &context, const Tensor &tensor) {
  py::array result(dtype(tensor.type), dimensions(tensor.shape));
  const auto buffer = result.request();
  switch (tensor.type) {
  case kf16: download<uint16_t>(context, tensor, buffer); break;
  case kf32: download<float>(context, tensor, buffer); break;
  case ki32: download<int32_t>(context, tensor, buffer); break;
  }
  return result;
}

struct PythonFuture {
  explicit PythonFuture(gpu::Future value) : value(std::move(value)) {}
  PythonFuture(const PythonFuture &) = delete;
  PythonFuture &operator=(const PythonFuture &) = delete;
  PythonFuture(PythonFuture &&) = default;
  PythonFuture &operator=(PythonFuture &&) = default;
  gpu::Future value;
};

} // namespace

PYBIND11_MODULE(gpu_cpp, module) {
  module.doc() = "Native WebGPU compute with gpu.cpp";

  py::enum_<NumType>(module, "NumType")
      .value("f16", kf16)
      .value("f32", kf32)
      .value("i32", ki32)
      .export_values();

  py::class_<Context>(module, "Context")
      .def(py::init([](bool shaderF16, bool spirv) {
             ContextOptions options{.enableSPIRV = spirv};
             if (shaderF16)
               options.requiredFeatures = {wgpu::FeatureName::ShaderF16};
             py::gil_scoped_release release;
             return std::make_unique<Context>(createContext(options));
           }),
           py::kw_only(), py::arg("shader_f16") = false,
           py::arg("spirv") = false);

  py::class_<WGSL>(module, "WGSL")
      .def(py::init([](std::string code,
                       const std::vector<py::ssize_t> &workgroupSize,
                       NumType precision) {
             return WGSL(std::move(code), shape(workgroupSize), precision);
           }),
           py::arg("code"), py::arg("workgroup_size") =
                                std::vector<py::ssize_t>{256, 1, 1},
           py::arg("precision") = kf32)
      .def_readwrite("code", &WGSL::code)
      .def_readwrite("label", &WGSL::label)
      .def_readwrite("entry_point", &WGSL::entryPoint)
      .def_property_readonly("workgroup_size",
                             [](const WGSL &shader) {
                               return dimensions(shader.workgroupSize);
                             });

  py::class_<SPIRV>(module, "SPIRV")
      .def(py::init([](std::vector<uint32_t> words) {
             return SPIRV{std::move(words)};
           }),
           py::arg("words"))
      .def_readwrite("words", &SPIRV::code)
      .def_readwrite("label", &SPIRV::label)
      .def_readwrite("entry_point", &SPIRV::entryPoint);

  py::class_<Tensor>(module, "Tensor")
      .def_property_readonly("shape",
                             [](const Tensor &tensor) {
                               return dimensions(tensor.shape);
                             })
      .def_readonly("dtype", &Tensor::type);
  py::class_<Binding>(module, "Binding");
  py::class_<Kernel>(module, "Kernel");
  py::class_<PythonFuture>(module, "Future");

  module.def("create_tensor",
             [](Context &context,
                const std::vector<py::ssize_t> &tensorShape, NumType type) {
               return createTensor(context, shape(tensorShape), type);
             },
             py::arg("context"), py::arg("shape"), py::arg("dtype"));
  module.def("tensor", &tensor, py::arg("context"), py::arg("array"));
  module.def("read", py::overload_cast<const Tensor &>(&read),
             py::arg("tensor"));
  module.def("read_write", py::overload_cast<const Tensor &>(&readWrite),
             py::arg("tensor"));
  module.def(
      "create_kernel",
      [](Context &context, const Shader &shader,
         const std::vector<Binding> &bindings,
         const std::vector<py::ssize_t> &workgroups, py::bytes parameters) {
        const std::string bytes = parameters;
        const auto *data = reinterpret_cast<const std::byte *>(bytes.data());
        return createKernel(context, shader, std::span<const Binding>(bindings),
                            shape(workgroups), {data, bytes.size()});
      },
      py::arg("context"), py::arg("shader"), py::arg("bindings"),
      py::arg("workgroups") = std::vector<py::ssize_t>{1, 1, 1},
      py::arg("parameters") = py::bytes());
  module.def("dispatch_kernel",
             [](Context &context, const Kernel &kernel) {
               return PythonFuture(dispatchKernel(context, kernel));
             });
  module.def("wait", [](Context &context, PythonFuture &future) {
    py::gil_scoped_release release;
    wait(context, future.value);
  });
  module.def("to_gpu", &upload, py::arg("context"), py::arg("tensor"),
             py::arg("array"));
  module.def("to_numpy", &numpy, py::arg("context"), py::arg("tensor"));
}
