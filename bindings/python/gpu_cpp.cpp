#include "gpu.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
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

struct PythonFuture {
  std::shared_ptr<Context> context;
  gpu::Future value;
  py::object result;
};

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
PythonFuture download(std::shared_ptr<Context> context, const Tensor &tensor) {
  auto values = std::make_shared<std::vector<T>>(size(tensor.shape));
  auto owner = py::capsule(
      new std::shared_ptr<std::vector<T>>(values), [](void *pointer) {
        delete static_cast<std::shared_ptr<std::vector<T>> *>(pointer);
      });
  py::array result(dtype(tensor.type), dimensions(tensor.shape), values->data(),
                   owner);
  auto future = toCPU(*context, tensor, values);
  return {std::move(context), std::move(future), std::move(result)};
}

PythonFuture numpy(std::shared_ptr<Context> context, const Tensor &tensor) {
  switch (tensor.type) {
  case kf16: return download<uint16_t>(std::move(context), tensor);
  case kf32: return download<float>(std::move(context), tensor);
  case ki32: return download<int32_t>(std::move(context), tensor);
  }
  throw std::invalid_argument("unknown numeric type");
}

bool pollFuture(PythonFuture &future) {
  py::gil_scoped_release release;
  return poll(*future.context, future.value);
}

py::object waitFuture(PythonFuture &future) {
  {
    py::gil_scoped_release release;
    wait(*future.context, future.value);
  }
  return future.result;
}

} // namespace

PYBIND11_MODULE(_gpu_cpp, module) {
  module.doc() = "Native WebGPU compute with gpu.cpp";

  py::enum_<NumType>(module, "NumType")
      .value("f16", kf16)
      .value("f32", kf32)
      .value("i32", ki32)
      .export_values();

  py::class_<Context, std::shared_ptr<Context>>(module, "Context")
      .def(py::init([](bool shaderF16, bool spirv) {
             ContextOptions options{.enableSPIRV = spirv};
             if (shaderF16)
               options.requiredFeatures = {wgpu::FeatureName::ShaderF16};
             py::gil_scoped_release release;
             return std::make_shared<Context>(createContext(options));
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
  py::class_<PythonFuture>(module, "Future")
      .def("_poll", &pollFuture)
      .def("_result", [](PythonFuture &future) { return future.result; })
      .def("__await__", [](py::object self) {
        return py::module_::import("gpu_cpp._async")
            .attr("wait")(std::move(self))
            .attr("__await__")();
      });

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
             [](std::shared_ptr<Context> context, const Kernel &kernel) {
               auto future = dispatchKernel(*context, kernel);
               return PythonFuture{std::move(context), std::move(future),
                                   py::none()};
             },
             py::arg("context"), py::arg("kernel"));
  module.def("wait", &waitFuture, py::arg("future"));
  module.def("to_gpu", &upload, py::arg("context"), py::arg("tensor"),
             py::arg("array"));
  module.def("to_numpy", &numpy, py::arg("context"), py::arg("tensor"));
}
