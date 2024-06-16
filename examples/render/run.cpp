#include <array>
#include <chrono>
#include <cstdio>

#include "gpu.h"
#include "utils/array_utils.h"
#include "utils/logging.h"

using namespace gpu;

const char *kSDF = R"(
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    focalLength: f32,
    screenWidth: u32,
    screenHeight: u32,
    sphereRadius: f32,
    sphereCenterX: f32,
    sphereCenterY: f32,
    sphereCenterZ: f32,
    time: i32,
};

fn sdf(p: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  return length(p - c) - r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    if (GlobalInvocationID.x >= params.screenWidth 
        || GlobalInvocationID.y >= params.screenHeight) {
      return;
    }
    let id: u32 = GlobalInvocationID.y * params.screenWidth + GlobalInvocationID.x;

    let x: f32 = f32(GlobalInvocationID.x);
    let y: f32 = f32(GlobalInvocationID.y);

    // ray position, starting at the camera
    var p: vec3<f32> = vec3<f32>((x / f32(params.screenWidth)) * 4.0 - 2.0,
                                 (y / f32(params.screenHeight)) * 2.0 - 1.0,
                                 params.focalLength);

    let dir: vec3<f32> = p / length(p); // direction from focal point to pixel

    // object dynamics
    var offsetX: f32 = sin(f32(params.time) / 750) * 0.5;
    var offsetY: f32 = cos(f32(params.time) / 750) * 0.5;
    var offsetZ: f32 = sin(f32(params.time) / 1000) * 1.0;
    let c: vec3<f32> = vec3<f32>(params.sphereCenterX + offsetX,
                                 params.sphereCenterY + offsetY,
                                 params.sphereCenterZ + offsetZ);

    let dist: f32 = 0.0;
    out[id] = 0.0;

    let maxIter: u32 = 30;
    // march the ray in the direction of dir by length derived by the SDF
    for (var i: u32 = 0; i < maxIter; i++) {
      // largest step we can take w/o intersection is = SDF value at point
      let step : f32 = sdf(p, c, params.sphereRadius);
      if (abs(step) < .001) {
        return;
      }
      out[id] = 
        max(0, min(5.0, out[id] + step));
      if (out[id] == 10.0) {
        return;
      }
      p = p + dir * step;
    }

}
)";

std::uint32_t getCurrentTimeInMilliseconds() {
  auto now = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch());
  return static_cast<uint32_t>(duration.count());
}

int main(int argc, char **argv) {

  constexpr size_t NROWS = 16;
  constexpr size_t NCOLS = 64;

  std::array<float, NROWS * NCOLS> screen;

  struct Params {
    float focalLength;
    uint32_t screenWidth;
    uint32_t screenHeight;
    float sphereRadius;
    float sphereCenterX;
    float sphereCenterY;
    float sphereCenterZ;
    uint32_t time;
  } params = {/* focal length */ 1.0,
              NCOLS,
              NROWS,
              /* radius */ 1.0,
              /* x */ 0.0,
              /* y */ 0.0,
              /* z */ 2.5,
              0};

  std::fill(begin(screen), end(screen), 0.0f);

  GPUContext ctx = CreateContext();
  GPUTensor devScreen = CreateTensor(ctx, {NROWS, NCOLS}, kf32, screen.data());
  uint32_t zeroTime = getCurrentTimeInMilliseconds();

  while (true) {
    params.time = getCurrentTimeInMilliseconds() - zeroTime;
    Kernel render =
        CreateKernel(ctx, CreateShader(kSDF), {}, 0, devScreen, params);
    DispatchKernel(ctx, render);
    Wait(ctx, render.future);
    ToCPU(ctx, devScreen, screen.data(), sizeof(screen));

    static const char intensity[] = "@%#8X7x*+=-:^~'.` ";

    // clear the screen, move cursor to the top
    printf("\033[2J\033[H");

    fprintf(stdout, "%s",
            show<float, NROWS, NCOLS>(screen, "Raw values").c_str());

    // normalize values
    float min = *std::min_element(screen.begin(), screen.end());
    float max = min + params.sphereRadius; // maximize dynamic range

    for (size_t i = 0; i < screen.size(); ++i) {
      screen[i] = (screen[i] - min) / (max - min);
    }
    fprintf(stdout, "%s",
            show<float, NROWS, NCOLS>(screen, "Scaled").c_str());

    // index into intensity array
    std::array<char, screen.size()> raster;
    for (size_t i = 0; i < screen.size(); ++i) {
      size_t index =
          std::min(sizeof(intensity) - 2,
                   std::max(0ul, static_cast<size_t>(screen[i] *
                                                     (sizeof(intensity) - 2))));
      raster[i] = intensity[index];
    }

    for (size_t row = 0; row < NROWS; ++row) {
      for (size_t col = 0; col < NCOLS; ++col) {
        printf("%c", raster[row * NCOLS + col]);
      }
      printf("\n");
    }
  }
}
