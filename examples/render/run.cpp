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
    // TODO{(avh): explicitly encode aspect ratio
    var p: vec3<f32> = vec3<f32>((x / f32(params.screenWidth)) * 2.0 - 1.0,
                                 (y / f32(params.screenHeight)) * 2.0 - 1.0,
                                 params.focalLength);

    let dir: vec3<f32> = p / length(p); // direction from focal point to pixel

    // object dynamics - 2 spheres symmetrically moving in z
    var offsetX: f32 = cos(f32(params.time) / 666) * 0.75;
    var offsetY: f32 = sin(f32(params.time) / 666) * 0.75;
    var offsetZ: f32 = sin(f32(params.time) / 2000) * 1.5;
    let c: vec3<f32> = vec3<f32>(params.sphereCenterX + offsetX,
                                 params.sphereCenterY + offsetY,
                                 params.sphereCenterZ + offsetZ);
    let c2: vec3<f32> = vec3<f32>(params.sphereCenterX - offsetX,
                                 params.sphereCenterY - offsetY,
                                 params.sphereCenterZ + offsetZ);

    let dist: f32 = 0.0;
    out[id] = 0.0;

    let maxIter: u32 = 30;
    // march the ray in the direction of dir by length derived by the SDF
    for (var i: u32 = 0; i < maxIter; i++) {
      // largest step we can take w/o intersection is = SDF value at point
      let step : f32 = min(sdf(p, c, params.sphereRadius), sdf(p, c2, params.sphereRadius));
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

  constexpr size_t NROWS = 32;
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
              /* z */ 3.5,
              0};

  std::fill(begin(screen), end(screen), 0.0f);

  Context ctx = CreateContext();
  Tensor devScreen = CreateTensor(ctx, {NROWS, NCOLS}, kf32, screen.data());
  uint32_t zeroTime = getCurrentTimeInMilliseconds();

  ShaderCode shader = CreateShader(kSDF, Shape{16, 16, 1});
  Kernel renderKernel =
      CreateKernel(ctx, shader, {}, 0, devScreen, {NCOLS, NROWS, 1}, params);
  while (true) {
    DispatchKernel(ctx, renderKernel);
    Wait(ctx, renderKernel.future);
    ToCPU(ctx, devScreen, screen.data(), sizeof(screen));
    // Update the time field, write pparams to GPU, and create a new command
    // buffer
    params.time = getCurrentTimeInMilliseconds() - zeroTime;
    wgpuQueueWriteBuffer(ctx.queue,
                         renderKernel.buffers[renderKernel.numBuffers - 1], 0,
                         static_cast<void *>(&params), sizeof(params));
    ResetCommandBuffer(ctx.device, /*nthreads*/ {NCOLS, NROWS, 1},
                       renderKernel);

    static const char intensity[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/"
                                    "\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
    // static const char intensity[] = "@%#8$X71x*+=-:^~'.` ";

    // normalize values
    float min = 0.0;
    float max = params.sphereRadius * 3;

    for (size_t i = 0; i < screen.size(); ++i) {
      screen[i] = (screen[i] - min) / (max - min);
    }

    // index into intensity array
    std::array<char, screen.size()> raster;
    for (size_t i = 0; i < screen.size(); ++i) {
      size_t index =
          std::min(sizeof(intensity) - 2,
                   std::max(0ul, static_cast<size_t>(screen[i] *
                                                     (sizeof(intensity) - 2))));
      raster[i] = intensity[index];
    }

    // Draw the raster
    char buffer[(NROWS + 2) * (NCOLS + 2)];
    char *offset = buffer;
    sprintf(offset, "+");
    for (size_t col = 0; col < NCOLS; ++col) {
      sprintf(offset + col + 1, "-");
    }
    sprintf(buffer + NCOLS + 1, "+\n");
    offset += NCOLS + 3;
    for (size_t row = 0; row < NROWS; ++row) {
      sprintf(offset, "|");
      for (size_t col = 0; col < NCOLS; ++col) {
        sprintf(offset + col + 1, "%c", raster[row * NCOLS + col]);
      }
      sprintf(offset + NCOLS + 1, "|\n");
      offset += NCOLS + 3;
    }
    sprintf(offset, "+");
    for (size_t col = 0; col < NCOLS; ++col) {
      sprintf(offset + col + 1, "-");
    }
    sprintf(offset + NCOLS + 1, "+\n");
    printf("\033[2J\033[H");
    printf("Workgroup size: %zu %zu %zu \n", shader.workgroupSize[0],
           shader.workgroupSize[1], shader.workgroupSize[2]);
    printf("Number of Threads: %zu %zu %d \n", devScreen.shape[1],
           devScreen.shape[0], 1);
    printf("%s", buffer);
  }
}
