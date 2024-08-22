#include "evaluator.h"
#include "gpu.h"
// #include "scaffold.h"
#include "webprint.h"
#include <array>
#include <cstdio>
#include <emscripten/emscripten.h>
#include <future>
#include <memory>
#include <string>
#include <vector>

using namespace gpu;

constexpr size_t kN = 100;

template <size_t nInputs> struct HostSpec {
  const Shape wgSize;
  const Shape nWorkgroups;
  const std::string kernelCode;
  std::array<std::vector<float>, nInputs> inputs;
};

template <size_t nInputs>
void executeKernel(Context &ctx, const HostSpec<nInputs> &spec,
                   float *outputPtr, size_t outputSize) {
  std::array<Tensor, nInputs + 1> bindingsArr; // + 1 for output binding
  for (size_t inputIndex = 0; inputIndex < nInputs; ++inputIndex) {
    bindingsArr[inputIndex] =
        createTensor(ctx, Shape{spec.inputs[inputIndex].size()}, kf32,
                     spec.inputs[inputIndex].data());
  }
  Tensor output = createTensor(ctx, Shape{outputSize}, kf32);
  bindingsArr[nInputs] = output;
  Bindings bindings{bindingsArr};
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {spec.kernelCode, spec.wgSize, kf32}, bindings,
                           spec.nWorkgroups);
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputPtr, outputSize * sizeof(float));
}

extern "C" {

void generatePreamble(size_t nInputs, Shape &wgSize, Shape &nWorkgroups,
                      const char *out, size_t outSize) {
  std::string result = "";
  for (size_t i = 0; i < nInputs; ++i) {
    result += "@group(0) @binding(" + std::to_string(i) + ") var input" +
              std::to_string(i) + " : array;\n";
  }
  result += "@group(0) @binding(" + std::to_string(nInputs) +
            ") var output : array;\n";
  result += "@compute @workgroup_size(" + std::to_string(wgSize[0]) + ", " +
            std::to_string(wgSize[1]) + ", " + std::to_string(wgSize[2]) +
            ")\n";
  std::strncpy(const_cast<char *>(out), result.c_str(), outSize);
}

EMSCRIPTEN_KEEPALIVE
bool evaluate(const char *kernelCode, int puzzleIndex) {

  Context ctx = createContext({});
  return evaluate(ctx, kernelCode, puzzleIndex);
}

} // extern "C"

#ifndef STANDALONE_WASM
#include "emscripten/bind.h"
EMSCRIPTEN_BINDINGS(module) {

  emscripten::value_array<std::array<size_t, 3>>("ArrayST")
      .element(emscripten::index<0>())
      .element(emscripten::index<1>())
      .element(emscripten::index<2>());
  emscripten::register_vector<std::vector<float>>("VectorFloat");
  emscripten::register_vector<std::vector<int>>("VectorInt");

  emscripten::function("evaluate",
                       emscripten::optional_override(
                           [](const std::string &kernelCode, int puzzleIndex) {
                             return evaluate(kernelCode.c_str(), puzzleIndex);
                           }));

  emscripten::function("getTemplate",
                       emscripten::optional_override([](int puzzleIndex) {
                         return getTemplate(puzzleIndex);
                       }));
}
#endif
