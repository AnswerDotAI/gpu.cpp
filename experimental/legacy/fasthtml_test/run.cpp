#include <array>
#include <cstdio>
#include <emscripten/emscripten.h>
#include <future>
#include <memory>
#include <string>

#include "gpu.hpp"

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

EMSCRIPTEN_KEEPALIVE
void executeKernel(const char *kernelCode) {
  Context ctx = createContext({});
  static constexpr size_t N = 5000;
  std::array<float, N> inputArr, outputArr;

  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i);
  }

  Tensor input = createTensor(ctx, Shape{N}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, Shape{N}, kf32);

  std::promise<void> promise;
  std::future<void> future = promise.get_future();

  try {
    Kernel op = createKernel(ctx, {kernelCode, 256, kf32},
                             Bindings{input, output}, {cdiv(N, 256), 1, 1});

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

} // extern "C"

#ifndef STANDALONE_WASM
#include "emscripten/bind.h"
EMSCRIPTEN_BINDINGS(module) {
  emscripten::function("executeKernel", emscripten::optional_override(
                                            [](const std::string &kernelCode) {
                                              executeKernel(kernelCode.c_str());
                                            }));
}
#endif
