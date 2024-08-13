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

EM_JS(void, js_print, (const char *str), {
  if (typeof window != 'undefined' && window.customPrint) {
    window.customPrint(UTF8ToString(str));
  } else {
    console.log("window.customPrint is not defined.");
    console.log(UTF8ToString(str));
  }
});

constexpr size_t kN = 5000;

extern "C" {

EMSCRIPTEN_KEEPALIVE bool checkAnswer(std::array<float, kN> &outputArr) {
  return outputArr[0] == 10;
  // return false;
}

EMSCRIPTEN_KEEPALIVE
void executeKernel(Context& ctx, const char *kernelCode, const Shape &wgSize,
                   const Shape &nWorkgroups,
                   std::array<float, kN> &outputArr) {

  // TODO(avh): use puzzle dispatch from scaffold.h for host implementation
  char buffer[1024]; // for printing
  constexpr size_t N = 5000;
  std::array<float, N> inputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i);
  }
  Tensor input = createTensor(ctx, Shape{N}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kernelCode, wgSize, kf32},
                             Bindings{input, output}, nWorkgroups);
  
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 10; ++i) {
    snprintf(buffer, sizeof(buffer), "  [%d] kernel(%.1f) = %.4f", i,
             inputArr[i], outputArr[i]);
    js_print(buffer);
  }
  js_print(" ...");
  for (int i = N - 10; i < N; ++i) {
    snprintf(buffer, sizeof(buffer), "  [%d] kernel(%.1f) = %.4f", i,
             inputArr[i], outputArr[i]);
    js_print(buffer);
  }
  snprintf(buffer, sizeof(buffer), "Computed %zu values", N);
  js_print(buffer);
} // executeKernel

EMSCRIPTEN_KEEPALIVE
bool runCheck(const char *kernelCode, const Shape &wgSize,
              const Shape &nWorkgroups) {
  Context ctx = createContext({});
  std::array<float, kN> outputArr;
  executeKernel(ctx, kernelCode, wgSize, nWorkgroups, outputArr);
  Evaluator evaluator;
  return evaluator.evaluate(ctx, std::string(kernelCode), 0);
  // return checkAnswer(outputArr);
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
      "runCheck",
      emscripten::optional_override(
          [](const std::string &kernelCode, const std::array<size_t, 3> &wgSize,
             const std::array<size_t, 3> &nWorkgroups) {
            return runCheck(kernelCode.c_str(),
                     Shape{static_cast<size_t>(wgSize[0]),
                           static_cast<size_t>(wgSize[1]),
                           static_cast<size_t>(wgSize[2])},
                     Shape{static_cast<size_t>(nWorkgroups[0]),
                           static_cast<size_t>(nWorkgroups[1]),
                           static_cast<size_t>(nWorkgroups[2])});
          }));

  emscripten::function("checkAnswer", &checkAnswer);
}
#endif
