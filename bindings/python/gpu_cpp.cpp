#include "gpu.hpp"
#include <array>
#include <cstdio>
#include <future>

using namespace gpu;

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

Shape vector_to_shape(const std::vector<int> &dims) {
  switch(dims.size()){
  case 1:
    return Shape{(unsigned long)dims[0]};
    break;
  case 2:
    return Shape{(unsigned long)dims[0],(unsigned long)dims[1]};
    break;
  case 3:
    return Shape{(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2]};
    break;
  case 4:
    return Shape{(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2],(unsigned long)dims[3]};
    break;
  case 5:
    return Shape{(unsigned long)dims[0],(unsigned long)dims[1],(unsigned long)dims[2],(unsigned long)dims[3],(unsigned long)dims[4]};
    break;
  }
  return Shape{0};
}

Context* py_createContext() {
  return new Context(createContext());
}

KernelCode* py_createKernelCode(const std::string &pData, size_t workgroupSize, int precision) {
  return new KernelCode(pData, workgroupSize, (NumType)precision);
}

Kernel py_createKernel(Context *ctx, const KernelCode *code,
                        // const Tensor *dataBindings, size_t numTensors,
                        const py::list& dataBindings_py,
                        // const size_t *viewOffsets,
                        const py::list& viewOffsets_py,
                        const std::vector<int> &totalWorkgroups){
  std::vector<Tensor> bindings;
  for (auto item : dataBindings_py) {
    bindings.push_back(item.cast<Tensor>());
  }
  std::vector<size_t> viewOffsets;
  for (auto item : viewOffsets_py) {
    viewOffsets.push_back(item.cast<size_t>());
  }
  return createKernel(*ctx, *code, bindings.data(), bindings.size(), viewOffsets.data(), vector_to_shape(totalWorkgroups));
}

Tensor* py_createTensor(Context *ctx, const std::vector<int> &dims, int dtype) {
  return new Tensor(createTensor(*ctx, vector_to_shape(dims), (NumType)dtype));
}

py::array_t<float> py_toCPU_float(Context *ctx, Tensor* tensor) {
  auto result = py::array_t<float>(tensor->data.size/sizeof(float));
  py::buffer_info buf = result.request();
  toCPU(*ctx, *tensor, static_cast<float *>(buf.ptr), tensor->data.size);
  return result;
}


void py_toGPU_float(Context *ctx, py::array_t<float> array, Tensor *tensor) {
  py::buffer_info buf = array.request();
  float *ptr = static_cast<float *>(buf.ptr);
  toGPU(*ctx, ptr, *tensor);
}

struct GpuAsync {
  std::promise<void> promise;
  std::future<void> future ;
  GpuAsync(): future(promise.get_future()){
  }
};

GpuAsync* py_dispatchKernel(Context *ctx, Kernel kernel) {
  auto async = new GpuAsync();
  dispatchKernel(*ctx, kernel, async->promise);
  return async;
}

void py_wait(Context *ctx, GpuAsync* async) {
  wait(*ctx, async->future);
}

PYBIND11_MODULE(gpu_cpp, m) {
    m.doc() = "gpu.cpp plugin";
    py::class_<Context>(m, "Context");
    py::class_<Tensor>(m, "Tensor");
    py::class_<RawKernel, std::shared_ptr<RawKernel>>(m, "Kernel");
    py::class_<KernelCode>(m, "KernelCode");
    py::class_<GpuAsync>(m, "GpuAsync");
    m.def("create_context", &py_createContext, py::return_value_policy::take_ownership);
    m.def("create_tensor", &py_createTensor, py::return_value_policy::take_ownership);
    m.def("create_kernel", &py_createKernel);
    m.def("create_kernel_code", &py_createKernelCode, py::return_value_policy::take_ownership);
    m.def("dispatch_kernel", &py_dispatchKernel, py::return_value_policy::take_ownership);
    m.def("wait", &py_wait, "Wait for GPU");
    m.def("to_cpu_float", &py_toCPU_float);
    m.def("to_gpu_float", &py_toGPU_float);
    m.attr("kf32") = (int)kf32;
}
