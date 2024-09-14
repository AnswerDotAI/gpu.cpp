#include <array>
#include <cstdio>
#include <future>

#include "gpu.hpp"
#include "numeric_types/half.hpp"

using namespace gpu; // createContext, createTensor, createKernel,
                     // createShader, dispatchKernel, wait, toCPU
                     // Tensor, Kernel, Context, Shape, kf32
                     //

static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f16 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dummy: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f16 = inp[i];
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                 * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

int main(int argc, char **argv) {
  printf("\033[2J\033[1;1H");
  printf("\nHello float16!\n");
  printf("--------------\n\n");

  Context ctx = createContext(
      {}, {},
      /*device descriptor, enabling f16 in WGSL*/
      {
          .requiredFeatureCount = 1,
          .requiredFeatures = std::array{WGPUFeatureName_ShaderF16}.data(),
      });
  static constexpr size_t N = 10000;
  std::array<half, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = half(static_cast<float>(i) / 10.0f); // dummy input data
  }
  Tensor input = createTensor(ctx, Shape{N}, kf16, inputArr.data());
  Tensor output = createTensor(ctx, Shape{N}, kf16);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kGelu, 256, kf16}, Bindings{input, output},
                           {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));

  for (int i = 0; i < 12; ++i) {
    // Cast to float32 for printing to the screen
    printf("  gelu(%.2f) = %.2f\n", static_cast<float>(inputArr[i]),
           static_cast<float>(outputArr[i]));
  }

  printf("  ...\n\n");
  printf("Computed %zu float16 values of GELU(x: float16)\n\n", N);
  return 0;

}
