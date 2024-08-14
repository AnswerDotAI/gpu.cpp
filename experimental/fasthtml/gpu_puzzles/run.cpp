#include "gpu.h"
#include "evaluator.h"
#include "scaffold.h"
#include <array>
#include <cstdio>
#include <emscripten/emscripten.h>
#include <future>
#include <memory>
#include <string>
#include <vector>

using namespace gpu;

constexpr size_t kN = 100;

EM_JS(void, js_print, (const char *str), {
  if (typeof window != 'undefined' && window.customPrint) {
    window.customPrint(UTF8ToString(str));
  } else {
    console.log("window.customPrint is not defined.");
    console.log(UTF8ToString(str));
  }
});

template <size_t nInputs>
struct HostSpec {
  const Shape wgSize;
  const Shape nWorkgroups;
  const std::string kernelCode;
  std::array<std::vector<float>, nInputs> inputs;
};

template <size_t nInputs>
void executeKernel(Context& ctx, 
                    const HostSpec<nInputs>& spec,
                   float* outputPtr, size_t outputSize) {
  std::array<Tensor, nInputs + 1> bindingsArr; // + 1 for output binding
  for (size_t inputIndex = 0; inputIndex < nInputs; ++inputIndex) {
    bindingsArr[inputIndex] = createTensor(ctx, Shape{spec.inputs[inputIndex].size()}, kf32, spec.inputs[inputIndex].data());
  }
  Tensor output = createTensor(ctx, Shape{outputSize}, kf32);
  bindingsArr[nInputs] = output;
  Bindings bindings{bindingsArr};
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {spec.kernelCode, spec.wgSize, kf32},
  bindings, spec.nWorkgroups);
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputPtr, outputSize * sizeof(float));
}

extern "C" {

void generatePreamble(size_t nInputs, Shape& wgSize, Shape& nWorkgroups, const char* out, size_t outSize) {
  std::string result = "";
  for (size_t i = 0; i < nInputs; ++i) {
    result += "@group(0) @binding(" + std::to_string(i) + ") var input" + std::to_string(i) + " : array;\n";
  }
  result += "@group(0) @binding(" + std::to_string(nInputs) + ") var output : array;\n";
  result += "@compute @workgroup_size(" + std::to_string(wgSize[0]) + ", " + std::to_string(wgSize[1]) + ", " + std::to_string(wgSize[2]) + ")\n";
  std::strncpy(const_cast<char*>(out), result.c_str(), outSize);
}


EMSCRIPTEN_KEEPALIVE
void runCheck(const char *kernelCode, const Shape &wgSize,
              const Shape &nWorkgroups) {
  Context ctx = createContext({});
  std::array<float, kN> output;
  std::vector<float> input(N);
  for (int i = 0; i < kN; ++i) {
    input[i] = static_cast<float>(i);
  }
  HostSpec<1> spec = {
    wgSize,
    nWorkgroups,
    kernelCode,
    std::array<std::vector<float>, 1> {input}
  };
  executeKernel<1>(ctx, spec, output.data(), kN); 
}

EMSCRIPTEN_KEEPALIVE
bool evaluate(const char *kernelCode, const Shape &wgSize,
              const Shape &nWorkgroups, int puzzleIndex) {
  char buffer[1024]; // for printing

  snprintf(buffer, sizeof(buffer), "Evaluating kernel with workgroup size (%zu, %zu, %zu) and nWorkgroups (%zu, %zu, %zu)",
           wgSize[0], wgSize[1], wgSize[2], nWorkgroups[0], nWorkgroups[1], nWorkgroups[2]);
  js_print(buffer);
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


  emscripten::function(
      "evaluate",
      emscripten::optional_override(
          [](const std::string &kernelCode, const std::array<size_t, 3> &wgSize,
             const std::array<size_t, 3> &nWorkgroups, int puzzleIndex) {
            return evaluate(kernelCode.c_str(),
                     Shape{static_cast<size_t>(wgSize[0]),
                           static_cast<size_t>(wgSize[1]),
                           static_cast<size_t>(wgSize[2])},
                     Shape{static_cast<size_t>(nWorkgroups[0]),
                           static_cast<size_t>(nWorkgroups[1]),
                           static_cast<size_t>(nWorkgroups[2])}, puzzleIndex);
          }));
}
#endif
