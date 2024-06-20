#include "gpu.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <future>

using namespace gpu; // CreateContext, CreateTensor, CreateKernel,
                     // CreateShader, DispatchKernel, Wait, ToCPU
                     // Tensor, TensorList Kernel, Context, Shape, kf32

const char *kShaderSimulation = R"(
const G: f32 = 9.81;
const dt: f32 = 0.04;
@group(0) @binding(0) var<storage, read_write> theta1: array<f32>;
@group(0) @binding(1) var<storage, read_write> theta2: array<f32>;
@group(0) @binding(2) var<storage, read_write> thetaVel1: array<f32>;
@group(0) @binding(3) var<storage, read_write> thetaVel2: array<f32>;
@group(0) @binding(4) var<storage, read_write> length: array<f32>;
@group(0) @binding(5) var<storage, read_write> pos: array<f32>;  // x1, y1 for each pendulum
//@group(0) @binding(6) var<storage, read_write> pos2: array<f32>;  // x2, y2 for each pendulum
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&theta1)) {
        return;
    }
    let l = length[idx];

    // Update angular velocities and angles for theta1
    let accel1 = -(G / l) * sin(theta1[idx]);
    thetaVel1[idx] += accel1 * dt;
    theta1[idx] += thetaVel1[idx] * dt;

    // Update angular velocities and angles for theta2
    let accel2 = -(G / l) * sin(theta2[idx]);
    thetaVel2[idx] += accel2 * dt;
    theta2[idx] += thetaVel2[idx] * dt;

    // Calculate new positions based on updated angles
    pos[4 * idx] = l * sin(theta1[idx]);   // x1
    pos[4 * idx + 1] = -l * cos(theta1[idx]);  // y1

    let l_total = 2 * l;  // Assuming the second pendulum extends from the end of the first
    pos[4 * idx + 2] = pos[4 * idx] + l * sin(theta2[idx]);   // x2
    pos[4 * idx + 3] = pos[4 * idx + 1] - l * cos(theta2[idx]);  // y2

}
)";

void render(float *pos, size_t n, float maxX, float maxY, size_t screenWidth,
            size_t screenHeight) {
  static const char reverse_intensity[] = " .`'^-+=*x17X$8#%@";
  const size_t eps = 2;
  // iterate over screen
  for (size_t i = 0; i < screenHeight; ++i) {
    for (size_t j = 0; j < screenWidth; ++j) {
      int count = 0;
      for (size_t k = 0; k < 2 * n; k += 2) {
        float nx =
            (1.0 + pos[k] / maxX) / 2.0 * static_cast<float>(screenWidth);
        // negate y since it extends from top to bottom
        float ny = (1.0 - (pos[k + 1] / maxY)) / 2.0 *
                   static_cast<float>(screenHeight);
        // printf("x: %.2f, y: %.2f\n", nx, ny);
        float length = std::sqrt((nx - j) * (nx - j) + (ny - i) * (ny - i));
        if (length < eps) {
          count++;
        }
      }
      count = std::min(count, 17);
      // printf("%d", n);
      printf("%c", reverse_intensity[count]);
    }
    printf("|\n");
  }
  for(size_t i = 0; i < screenWidth + 1; ++i) {
    printf("-");
  }
}

int main() {
  Context ctx = CreateContext();

  // N can be quite a bit larger than this on most GPUs
  static constexpr size_t N = 1000;

  // Since m1 = m2, no mass in the update equation
  std::array<float, N> theta1Arr, theta2Arr, v1Arr, v2Arr, lengthArr;

  std::fill(v1Arr.begin(), v1Arr.end(), 0.0);
  std::fill(v2Arr.begin(), v2Arr.end(), 0.0);
  for (size_t i = 0; i < N; ++i) {
    theta1Arr[i] = 3.14159 / 2 + i * 3.14159 / N;
    theta2Arr[i] = 3.14159 / 2 + i * 3.14159 / N;
    lengthArr[i] = 1.0 - i * 0.5 / N;
  }
  Tensor theta1 = CreateTensor(ctx, Shape{N}, kf32, theta1Arr.data());
  Tensor theta2 = CreateTensor(ctx, Shape{N}, kf32, theta2Arr.data());
  Tensor vel1 = CreateTensor(ctx, Shape{N}, kf32, v1Arr.data());
  Tensor vel2 = CreateTensor(ctx, Shape{N}, kf32, v2Arr.data());
  Tensor length = CreateTensor(ctx, Shape{N}, kf32, lengthArr.data());

  std::array<float, 2 * 2 * N> posArr;
  Tensor pos = CreateTensor(ctx, Shape{N * 4}, kf32);
  Shape nThreads{N, 1, 1};
  ShaderCode shader = CreateShader(kShaderSimulation, 256, kf32);
  printf("Shader code: %s\n", shader.data.c_str());
  Kernel update = CreateKernel(
      ctx, shader, TensorList{theta1, theta2, vel1, vel2, length, pos},
      nThreads);

  while (true) {
    auto start = std::chrono::high_resolution_clock::now();
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    DispatchKernel(ctx, update, promise);
    ResetCommandBuffer(ctx.device, nThreads, update);
    Wait(ctx, future);

    ToCPU(ctx, pos, posArr.data(), sizeof(pos));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // printf("x1: %.2f, y1: %.2f\nx2: %.2f, y2: %.2f\n", pos1Arr[0],
    // pos1Arr[1],pos2Arr[0], pos2Arr[1]);
    printf("\033[2J\033[1;1H");
    // render(posArr.data(), N * 2, 2.0, 2.0, 40, 40);
    render(posArr.data(), N, 2.0, 2.0, 80, 40);
    std::this_thread::sleep_for(std::chrono::milliseconds(16) - elapsed);
  }
}
