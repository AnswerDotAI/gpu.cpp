#include "gpu.h"
#include <array>
#include <cstdio>
#include <fstream>
#include <future>
#include <string>
#include <thread>

#include "utils/array_utils.h"
#include "utils/logging.h"

using namespace gpu;

template <size_t rows, size_t cols>
void rasterize(const std::array<float, rows * cols> &values,
               std::array<char, rows *(cols + 1)> &raster) {
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

void loadShaderCode(const std::string &filename, std::string &codeString) {
  codeString = "";
  FILE *file = fopen(filename.c_str(), "r");
  while (!file) {
    fclose(file);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    file = fopen(filename.c_str(), "r");
  }
  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), file)) {
    codeString += buffer;
  }
  fclose(file);
}

int main() {

  Context ctx = createContext();
  static constexpr size_t kRows = 40;
  static constexpr size_t kCols = 70;

  LOG(kDefLog, kInfo, "Creating screen tensor");

  std::array<float, kRows * kCols> screenArr;
  std::fill(begin(screenArr), end(screenArr), 0.0);
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

  LOG(kDefLog, kInfo, "Creating shader and kernel");

  loadShaderCode("shader.wgsl", codeString);
  ShaderCode shader = createShader(codeString.c_str(), Shape{16, 16, 1});
  Kernel renderKernel =
      createKernel(ctx, shader, Bindings{screen},
                   cdiv({kCols, kRows, 1}, shader.workgroupSize), params);

  LOG(kDefLog, kInfo, "Starting render loop");

  std::array<char, kRows *(kCols + 1)> raster;

  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> elapsed;
  size_t ticks = 0;
  while (true) {
    if (elapsed.count() - static_cast<float>(ticks) > 1.0) {
      loadShaderCode("shader.wgsl", codeString);
      if (codeString != shader.data) {
        shader = createShader(codeString.c_str(), Shape{16, 16, 1});
        renderKernel =
            createKernel(ctx, shader, Bindings{screen},
                         cdiv({kCols, kRows, 1}, shader.workgroupSize), params);
        ticks++;
      }
    }
    params.time = getCurrentTimeInMilliseconds(start);
    wgpuQueueWriteBuffer(ctx.queue,
                         renderKernel.buffers[renderKernel.numBindings - 1], 0,
                         static_cast<void *>(&params), sizeof(params));
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
    std::this_thread::sleep_for(std::chrono::milliseconds(20) - frameElapsed);
    printf("\033[H\033[J%s\nReloaded file %zu times\n", raster.data(), ticks);
  }

  LOG(kDefLog, kInfo, "Done");
}
