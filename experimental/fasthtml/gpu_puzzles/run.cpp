#include "gpu.h"
#include <array>
#include <cstdio>
#include <emscripten/emscripten.h>
#include <future>
#include <memory>
#include <string>

using namespace gpu;

EM_JS(void, js_print, (const char *str), {
  if (typeof window != 'undefined' && window.customPrint) {
    window.customPrint(UTF8ToString(str));
  } else {
    console.log("window.customPrint is not defined.");
    console.log(UTF8ToString(str));
  }
});

extern "C" {

EMSCRIPTEN_KEEPALIVE bool checkAnswer(std::array<float, 5000> &outputArr) {
  return true;
}

EMSCRIPTEN_KEEPALIVE
void executeKernel(const char *kernelCode, const Shape &wgSize,
                   const Shape &nWorkgroups, std::array<float, 5000> &outputArr) {
  constexpr size_t N = 5000;
  Context ctx = createContext({});
  std::array<float, N> inputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i);
  }
  Tensor input = createTensor(ctx, Shape{N}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  try {
    Kernel op = createKernel(ctx, {kernelCode, wgSize, kf32},
                             Bindings{input, output}, nWorkgroups);
    dispatchKernel(ctx, op, promise);
    wait(ctx, future);
  } catch (const std::exception &e) {
    js_print("Invalid kernel code.");
    exit(1);
  }
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  char buffer[1024];
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
  std::array<float, 5000> outputArr;
  executeKernel(kernelCode, wgSize, nWorkgroups, outputArr);
  return checkAnswer(outputArr);
}

} // extern "C"

#ifndef STANDALONE_WASM
#include "emscripten/bind.h"
EMSCRIPTEN_BINDINGS(module) {
  emscripten::function(
      "executeKernel",
      emscripten::optional_override([](const std::string &kernelCode,
                                       const std::array<size_t, 3>& wgSize,
                                       const std::array<size_t, 3>& nWorkgroups) {
        runCheck(kernelCode.c_str(),
                      Shape{wgSize[0], wgSize[1], wgSize[2]},
                      Shape{nWorkgroups[0], nWorkgroups[1], nWorkgroups[2]});
      }));
  emscripten::function("checkAnswer", &checkAnswer);
}
#endif
