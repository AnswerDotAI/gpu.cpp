#include "gpu.hpp"
#include <array>
#include <cstdio>

using namespace gpu;

static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                 * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

int main(int argc, char **argv) {
  printf("\033[2J\033[1;1H");
  printf("\nHello gpu.cpp!\n");
  printf("--------------\n\n");

  Context ctx = createContext();
  static constexpr size_t N = 10000;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i) / 10.0; // dummy input data
  }
  Tensor input = createTensor(
      ctx, Shape{N}, kf32, std::span<const float>(inputArr));
  Tensor output = createTensor(ctx, Shape{N}, kf32);
  Kernel op = createKernel(ctx, WGSL{kGelu, 256, kf32},
                           Bindings{read(input), readWrite(output)},
                           {(N + 255) / 256, 1, 1});
  auto future = dispatchKernel(ctx, op);
  wait(ctx, future);
  auto readback = toCPU(ctx, output, std::span<float>(outputArr));
  wait(ctx, readback);
  for (int i = 0; i < 12; ++i) {
    printf("  gelu(%.2f) = %.2f\n", inputArr[i], outputArr[i]);
  }
  printf("  ...\n\n");
  printf("Computed %zu values of GELU(x)\n\n", N);
}
