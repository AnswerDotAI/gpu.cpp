#include "gpu.h"
#include <array>
#include <chrono>
#include <cstdio>

using namespace gpu; // CreateContext, CreateTensor, CreateKernel,
                     // CreateShader, DispatchKernel, Wait, ToCPU
                     // Tensor, TensorList Kernel, Context, Shape, kf32

const char *kShaderSimulation = R"(
const G: f32 = 9.81;
const dt: f32 = 0.01;
@group(0) @binding(0) var<storage, read_write> pos1: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> vel1: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> pos2: array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> vel2: array<{{precision}}>;
@group(0) @binding(4) var<storage, read_write> length: array<{{precision}}>;
@group(0) @binding(5) var<storage, read_write> mass: array<{{precision}}>;
@group(0) @binding(6) var<storage, read_write> output: array<{{precision}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&pos1)) {
    // TODO

    }
}
)";

int main() {
  printf("\nHello, gpu.cpp\n\n");
  Context ctx = CreateContext();
  static constexpr size_t N = 1000;

  std::array<float, N> x1Arr, x2Arr, y1Arr, y2Arr, vx1Arr, vy1Arr, vx2Arr, vy2Arr, lengthArr, massArr;

  Tensor pos1 = CreateTensor(ctx, Shape{N}, kf32, x1Arr.data());
  Tensor pos2 = CreateTensor(ctx, Shape{N}, kf32, x2Arr.data());
  Tensor vel1 = CreateTensor(ctx, Shape{N}, kf32, vx1Arr.data());
  Tensor vel2 = CreateTensor(ctx, Shape{N}, kf32, vy1Arr.data());
  Tensor length = CreateTensor(ctx, Shape{N}, kf32, lengthArr.data());
  Tensor mass = CreateTensor(ctx, Shape{N}, kf32, massArr.data());

  // TODO: no need to have output
  Tensor output = CreateTensor(ctx, Shape{N}, kf32);

  Shape nThreads{N, 1, 1};
  Kernel update = CreateKernel(
      ctx, CreateShader(kShaderSimulation, 256, kf32),
      TensorList{pos1, vel1, pos2, vel2,
       length, mass}, output,
      nThreads);
  while (true) {
    auto start = std::chrono::high_resolution_clock::now();
    ResetCommandBuffer(ctx.device, nThreads, update);

    DispatchKernel(ctx, update);
    Wait(ctx, update.future);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::this_thread::sleep_for(std::chrono::milliseconds(16) - elapsed);
  }

}
