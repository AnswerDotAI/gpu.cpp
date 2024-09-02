#include <array>
#include <cstdio>
#include <fstream>
#include <future>
#include <random>
#include <string>
#include <thread>

#include "gpu.hpp"
#include "utils/array_utils.hpp"
#include "utils/logging.hpp"

using namespace gpu;

template <size_t rows, size_t cols>
void rasterize(const std::array<float, rows * cols> &values,
               std::array<char, rows *(cols + 1)> &raster) {
  // Note: We can experiment with the rasterization characters here but fewer
  // characters looks better by imposing temporal coherence whereas more
  // characters can start to look like noise.
  // static const char intensity[] = " `.-':_,^=;><+!ngrc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";
  static const char intensity[] = " .`'^-+=*x17X$8#%@";
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      // values ranges b/w 0 and 1
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
  return duration.count();
}

void loadKernelCode(const std::string &filename, std::string &codeString) {
  codeString = "";
  FILE *file = fopen(filename.c_str(), "r");
  int nTries = 0;
  while (!file) {
    fclose(file);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    file = fopen(filename.c_str(), "r");
    if (++nTries > 5) {
      LOG(kDefLog, kError, "Failed to open file: %s", filename.c_str());
      return;
    }
  }
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), file)) {
    codeString += buffer;
  }
  fclose(file);
}

int main() {

  Context ctx = createContext();
  
  // static constexpr size_t kRows = 64;
  // static constexpr size_t kCols = 96;
  static constexpr size_t kRows = 64;
  static constexpr size_t kCols = 96;

  kDefLog.level = kError; // suppress screen logging
  LOG(kDefLog, kInfo, "Creating screen tensor");

  std::array<float, kRows * kCols> screenArr;
  // std::fill(begin(screenArr), end(screenArr), 0.0);
  auto gen = std::mt19937{std::random_device{}()};
  randint(screenArr, gen, 0, 1);
  Tensor screen = createTensor(ctx, {kRows, kCols}, kf32, screenArr.data());

  std::promise<void> promise;
  std::future<void> future = promise.get_future();

  std::string codeString;
  struct Params {
    float time;
    uint32_t screenWidth;
    uint32_t screenHeight;
  } params = {0.0, kCols, kRows};

  LOG(kDefLog, kInfo, "Loading shader code from shader.wgsl");

  loadKernelCode("shader.wgsl", codeString);
  KernelCode shader{codeString.c_str(), Shape{16, 16, 1}};
  Kernel renderKernel =
      createKernel(ctx, shader, Bindings{screen},
                   cdiv({kCols, kRows, 1}, shader.workgroupSize), params);

  LOG(kDefLog, kInfo, "Starting render loop");

  std::array<char, kRows *(kCols + 1)> raster;

  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed;
  size_t ticks = 0;
  printf("\033[2J\033[H");
  size_t framesPerLoad = 20;
  size_t frame = 0;
  while (true) {
    if (frame % framesPerLoad == 0) { 
      loadKernelCode("shader.wgsl", codeString);
      if (codeString != shader.data) {
        // TODO(avh): Use a better way to avoid write/read race conditions
        // and recover from partial write errors
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        loadKernelCode("shader.wgsl", codeString);
        shader = {codeString.c_str(), Shape{16, 16, 1}};
        renderKernel =
            createKernel(ctx, shader, Bindings{screen},
                         cdiv({kCols, kRows, 1}, shader.workgroupSize), params);
        ticks++;
        start = std::chrono::high_resolution_clock::now();
      }
      frame = 0;
    }
    params.time = getCurrentTimeInMilliseconds(start);
    toGPU(ctx, params, renderKernel);
    auto frameStart = std::chrono::high_resolution_clock::now();
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    dispatchKernel(ctx, renderKernel, promise);
    wait(ctx, future);
    resetCommandBuffer(ctx.device, renderKernel);
    toCPU(ctx, screen, screenArr);
    rasterize<kRows, kCols>(screenArr, raster);
    auto frameEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> frameElapsed = frameEnd - frameStart;
    elapsed = frameEnd - start;
    std::this_thread::sleep_for(std::chrono::milliseconds(10) - frameElapsed);
    printf("\033[H%s\nRender loop running (full screen recommended) ...\nEdit and save shader.wgsl to see changes here.\nReloaded shader.wgsl %zu times\n", raster.data(), ticks);
    fflush(stdout);
  }

  LOG(kDefLog, kInfo, "Done");
}
