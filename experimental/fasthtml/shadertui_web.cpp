#include "gpu.h"
#include <array>
#include <cstdio>
#include <emscripten/emscripten.h>
#include <future>
#include <chrono>
#include <string>
#include <algorithm>

using namespace gpu;

static constexpr size_t kRows = 64;
static constexpr size_t kCols = 96;

template <size_t rows, size_t cols>
void rasterize(const std::array<float, rows * cols> &values,
               std::array<char, rows *(cols + 1)> &raster) {
  static const char intensity[] = " .`'^-+=*x17X$8#%@";
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      size_t index =
          std::min(sizeof(intensity) - 2,
                   std::max(0ul, static_cast<size_t>(values[i * cols + j] *
                                                     (sizeof(intensity) - 2))));
      raster[i * (cols + 1) + j] = intensity[index];
    }
    raster[i * (cols + 1) + cols] = '\n';
  }
}

float getCurrentTimeInMilliseconds(
    std::chrono::time_point<std::chrono::high_resolution_clock> &zeroTime) {
  std::chrono::duration<float> duration =
      std::chrono::high_resolution_clock::now() - zeroTime;
  return duration.count() * 1000.0f;  // Convert to milliseconds
}

EM_JS(void, js_print, (const char *str), {
  if (typeof window != 'undefined' && window.customPrint) {
    window.customPrint(UTF8ToString(str));
  } else {
    console.log(UTF8ToString(str));
  }
});

extern "C" {
EMSCRIPTEN_KEEPALIVE
void executeShader(const char *shaderCode) {
  Context ctx = createContext();
  
  std::array<float, kRows * kCols> screenArr;
  Tensor screen = createTensor(ctx, {kRows, kCols}, kf32, screenArr.data());

  struct Params {
    float time;
    uint32_t screenWidth;
    uint32_t screenHeight;
  } params = {0.0f, kCols, kRows};

  KernelCode shader{shaderCode, Shape{16, 16, 1}};
  Kernel renderKernel =
      createKernel(ctx, shader, Bindings{screen},
                   cdiv({kCols, kRows, 1}, shader.workgroupSize), params);

  std::array<char, kRows *(kCols + 1)> raster;

  auto start = std::chrono::high_resolution_clock::now();
  
  // Render a few frames
  for (int frame = 0; frame < 10; ++frame) {
    params.time = getCurrentTimeInMilliseconds(start);
    toGPU(ctx, params, renderKernel);
    
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    dispatchKernel(ctx, renderKernel, promise);
    wait(ctx, future);
    
    toCPU(ctx, screen, screenArr.data());
    rasterize<kRows, kCols>(screenArr, raster);
    
    // Print the rasterized frame
    js_print(raster.data());
    
    // Add a small delay between frames
    emscripten_sleep(100);
  }
}
}

#ifndef STANDALONE_WASM
#include <emscripten/bind.h>
EMSCRIPTEN_BINDINGS(module) {
  emscripten::function("executeShader", emscripten::optional_override(
                                            [](const std::string &shaderCode) {
                                              executeShader(shaderCode.c_str());
                                            }));
}
#endif
