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
const dt: f32 = 0.01;

// size = 2 * # of pendulums
@group(0) @binding(0) var<storage, read_write> pos1: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> vel1: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> pos2: array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> vel2: array<{{precision}}>;

// size = # of pendulums
@group(0) @binding(4) var<storage, read_write> length: array<{{precision}}>;
@group(0) @binding(5) var<storage, read_write> mass: array<{{precision}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let ic: u32 = GlobalInvocationID.x * 2; // x and y values are adjacent
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&pos1)) {
    // Double pendulum x and y values are adjacent in the arrays
    let x1: f32 = pos1[ic];
    let y1: f32 = pos1[ic + 1];
    let vx1: f32 = vel1[ic];
    let vy1: f32 = vel1[ic + 1];
    let x2: f32 = pos2[ic];
    let y2: f32 = pos2[ic + 1];
    let vx2: f32 = vel2[ic];
    let vy2: f32 = vel2[ic + 1];
    let l: f32 = length[i];
    let m: f32 = mass[i];


    }
}
)";

int main() {
  Context ctx = CreateContext();
  static constexpr size_t N = 1000;

  std::array<float, N> x1Arr, x2Arr, y1Arr, y2Arr, vx1Arr, vy1Arr, vx2Arr,
      vy2Arr, lengthArr, massArr;

  Tensor pos1 = CreateTensor(ctx, Shape{N}, kf32, x1Arr.data());
  Tensor pos2 = CreateTensor(ctx, Shape{N}, kf32, x2Arr.data());
  Tensor vel1 = CreateTensor(ctx, Shape{N}, kf32, vx1Arr.data());
  Tensor vel2 = CreateTensor(ctx, Shape{N}, kf32, vy1Arr.data());
  Tensor length = CreateTensor(ctx, Shape{N}, kf32, lengthArr.data());
  Tensor mass = CreateTensor(ctx, Shape{N}, kf32, massArr.data());

  Shape nThreads{N, 1, 1};
  Kernel update =
      CreateKernel(ctx, CreateShader(kShaderSimulation, 256, kf32),
                   TensorList{pos1, vel1, pos2, vel2, length, mass}, nThreads);
  while (true) {
    auto start = std::chrono::high_resolution_clock::now();
    ResetCommandBuffer(ctx.device, nThreads, update);
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    DispatchKernel(ctx, update, promise);
    Wait(ctx, future);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::this_thread::sleep_for(std::chrono::milliseconds(16) - elapsed);
  }
}
