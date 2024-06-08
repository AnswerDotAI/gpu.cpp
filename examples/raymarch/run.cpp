#include <array>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "gpu.h"

using namespace gpu;

// test function - multiply by constant
const char *kTest = R"(
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let x = GlobalInvocationID.x;
    let y = GlobalInvocationID.y;
    if (idx < arrayLength(&input)) {
      output[idx] = x * y;
    }
  }
)";

const char* kSpere = R"(
// The buffer to store the depth map results.
@group(0) @binding(0)
var<storage, write> depthMap: array<f32>;

// Constants for the shader, like screen dimensions and sphere properties.
@group(1) @binding(0)
var<uniform> constants: Constants;

struct Constants {
    screenWidth: u32;
    screenHeight: u32;
    sphereRadius: f32;
    sphereCenterX: f32;
    sphereCenterY: f32;
    sphereCenterZ: f32;
};

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    // Screen coordinates for this thread.
    let x: f32 = f32(GlobalInvocationID.x);
    let y: f32 = f32(GlobalInvocationID.y);

    // Calculate normalized device coordinates (NDC) ranging from -1 to 1.
    let ndcX = (x / f32(constants.screenWidth)) * 2.0 - 1.0;
    let ndcY = (y / f32(constants.screenHeight)) * 2.0 - 1.0;

    // Calculate the distance from the pixel to the center of the sphere in NDC.
    let distX = ndcX - constants.sphereCenterX;
    let distY = ndcY - constants.sphereCenterY;
    let distZ = -constants.sphereCenterZ; // Assume the camera is looking along -Z axis.

    let distanceToCenter = sqrt(distX * distX + distY * distY + distZ * distZ);
    let sdfValue = distanceToCenter - constants.sphereRadius;

    // Write the depth value to the buffer. Convert SDF to depth (non-negative).
    let index = GlobalInvocationID.y * constants.screenWidth + GlobalInvocationID.x;
    if (index < constants.screenWidth * constants.screenHeight) {
        depthMap[index] = max(0.0, sdfValue);
    }
}
)";

std::string getCurrentTime() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

int main(int argc, char **argv) {
  fprintf(stdout, "Time: %s\n", getCurrentTime().c_str());
  constexpr size_t NROWS = 24;
  constexpr size_t NCOLS = 80;
  std::array<float, NROWS * NCOLS> screen;

  GPUContext ctx = CreateGPUContext();
  GPUTensor devScreen = Tensor(ctx, {NROWS, NCOLS}, kf32, screen.data());
  Kernel op = PrepareKernel(ctx, ShaderCode{kTest, 64}, {}, 0, devScreen);
}
