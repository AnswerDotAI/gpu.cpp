#include "gpu.h"
#include <array>
#include <cstdio>

using namespace gpu; // CreateContext, CreateTensor, CreateKernel,
                     // CreateShader, DispatchKernel, Wait, ToCPU
                     // GPUTensor, Kernel, GPUContext, Shape, kf32

static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        // select is more stable than tanh for large x
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
               * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

int main(int argc, char **argv) {
  printf("\nHello, gpu.cpp\n\n");
  GPUContext ctx = CreateContext();
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i) / 2.0; // dummy input data
  }
  GPUTensor input = CreateTensor(ctx, Shape{N}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, Shape{N}, kf32);
  Kernel op = CreateKernel(ctx, CreateShader(kGelu, 256, kf32), input, output,
                           /* nthreads */ {N, 1, 1});
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 32; ++i) {
    printf("out[%d] : gelu(%.2f) = %.2f\n", i, inputArr[i], outputArr[i]);
  }
  printf("...\n\n");
  return 0;
}
