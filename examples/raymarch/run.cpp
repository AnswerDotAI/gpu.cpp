#include <array>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "gpu.h"
#include "utils/array_utils.h"

using namespace gpu;

const char *kSDF = R"(
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
const FOCAL_LENGTH: f32 = 0.5;

struct Params {
    screenWidth: u32,
    screenHeight: u32,
    sphereRadius: f32,
    sphereCenterX: f32,
    sphereCenterY: f32,
    sphereCenterZ: f32,
};

fn sdf(p: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
  let l: vec3<f32> = p - c;
  return sqrt(l.x * l.x + l.y * l.y + l.z * l.z) - r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    if (GlobalInvocationID.x >= params.screenWidth 
        || GlobalInvocationID.y >= params.screenHeight) {
      return;
    }

    // Screen coordinates for this thread.
    let x: f32 = f32(GlobalInvocationID.x);
    let y: f32 = f32(GlobalInvocationID.y);

    let p: vec3<f32> = vec3<f32>((x / f32(params.screenWidth)) - 0.5,
                                 (y / f32(params.screenWidth)) - 0.5,
                                 FOCAL_LENGTH);
    let c: vec3<f32> = vec3<f32>(params.sphereCenterX, params.sphereCenterY, params.sphereCenterZ);
    let len: f32 = length(p);

    let dir: vec3<f32> = vec3<f32>(p.x / len, p.y / len, p.z / len);

    let maxIter: u32 = 5;
    let dist: f32 = 0.0;
    for (var i: u32 = 0; i < maxIter; i++) {
      let step: f32 = sdf(p, c, params.sphereRadius);
      if (step < .001) {
        return;
      }
      // out[GlobalInvocationID.y * params.screenWidth + GlobalInvocationID.x] += step;
      out[GlobalInvocationID.y * params.screenWidth + GlobalInvocationID.x] +=  1.0; // debugging
    }

}
)";

constexpr size_t NROWS = 24;
constexpr size_t NCOLS = 80;


std::int64_t getCurrentTimeInMilliseconds() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return duration.count();
}


int main(int argc, char **argv) {
  std::array<float, NROWS * NCOLS> screen;

  struct Params {
    uint32_t screenWidth;
    uint32_t screenHeight;
    float sphereRadius;
    float sphereCenterX;
    float sphereCenterY;
    float sphereCenterZ;
  } params = {NCOLS, NROWS, 0.5, 0.0, 0.0, 0.0};

  GPUContext ctx = CreateContext();
  GPUTensor devScreen = CreateTensor(ctx, {NROWS, NCOLS}, kf32, screen.data());
  Kernel render = CreateKernel(ctx, ShaderCode{kSDF, 64}, {}, 0, devScreen, params);
  DispatchKernel(ctx, render);
  Wait(ctx, render.future);
  ToCPU(ctx, devScreen, screen.data(), sizeof(screen));

  // https://stackoverflow.com/questions/30097953/ascii-art-sorting-an-array-of-ascii-characters-by-brightness-levels-c-c
  static const char intensity[] = "`.-':_,^=;><+!rc*/"
                                  "z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]"
                                  "2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@";

  fprintf(stdout, "%s", show<float, NROWS, NCOLS>(screen).c_str());
  std::array<char, NROWS *(NCOLS + 1)> raster;
  for (size_t row = 0; row < NROWS; ++row) {
    for (size_t col = 0; col < NCOLS; ++col) {
      // TODO(av): clamp distance + rasterize
      // raster[row * NCOLS + col] =
    }
  }
}
