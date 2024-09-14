#include <array>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

#include "experimental/tui.h" // rasterize
#include "gpu.hpp"

using namespace gpu;

const char *kUpdateSim = R"(
const G: f32 = 9.81;
const dt: f32 = 0.03;
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

int main() {
  // N can be quite a bit larger than this on most GPUs (~ 1M on MBP M1)
  static constexpr size_t N = 1000;
  Context ctx = createContext();

  // Host-side data
  std::array<float, N> theta1Arr, theta2Arr, v1Arr, v2Arr, lengthArr;
  std::fill(v1Arr.begin(), v1Arr.end(), 0.0);
  std::fill(v2Arr.begin(), v2Arr.end(), 0.0);
  for (size_t i = 0; i < N; ++i) {
    theta1Arr[i] = 3.14159 / 2 + i * 3.14159 / 16 / N;
    theta2Arr[i] = 3.14159 / 2 + i * 3.14159 / 16 / N - 0.1;
    lengthArr[i] = 1.0 - i * 0.5 / N;
  }

  // GPU buffers
  Tensor theta1 = createTensor(ctx, Shape{N}, kf32, theta1Arr.data());
  Tensor theta2 = createTensor(ctx, Shape{N}, kf32, theta2Arr.data());
  Tensor vel1 = createTensor(ctx, Shape{N}, kf32, v1Arr.data());
  Tensor vel2 = createTensor(ctx, Shape{N}, kf32, v2Arr.data());
  Tensor length = createTensor(ctx, Shape{N}, kf32, lengthArr.data());
  std::array<float, 2 * 2 * N> posArr; // x, y outputs for each pendulum
  std::string screen(80 * 40, ' ');
  Tensor pos = createTensor(ctx, Shape{N * 4}, kf32);

  // Prepare computation
  KernelCode kernel{kUpdateSim, 256, kf32};
  printf("WGSL code: %s\n", kernel.data.c_str());
  Kernel update = createKernel(
      ctx, kernel, Bindings{theta1, theta2, vel1, vel2, length, pos},
      /* nWorkgroups */ cdiv({N, 1, 1}, kernel.workgroupSize));

  // Main simulation update loop
  printf("\033[2J\033[H");
  while (true) {
    auto start = std::chrono::high_resolution_clock::now();
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    dispatchKernel(ctx, update, promise);
    wait(ctx, future);
    toCPU(ctx, pos, posArr.data(), sizeof(posArr));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // N * 2 because there's two objects per pendulum
    rasterize(posArr.data(), N * 2, 2.0, 2.0, screen, 80, 40);
    printf("\033[1;1H" // reset cursor
           "# simulations: %lu\n%s",
           N, screen.c_str());
    resetCommandBuffer(ctx.device, update); // Prepare kernel command
                                            // buffer for nxt iteration
    std::this_thread::sleep_for(std::chrono::milliseconds(8) - elapsed);
  }
}
