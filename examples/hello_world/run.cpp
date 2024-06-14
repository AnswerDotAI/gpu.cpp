#include "gpu.h"
#include "nn/shaders.h"
#include "utils/logging.h"
#include <array>
#include <cstdio>

using namespace gpu;

static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    // Ensure we do not access out of bounds
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + cube)));
    }
}
)";

int main(int argc, char **argv) {
  log(kDefLog, kInfo, "Hello, gpu.cpp!");
  GPUContext ctx = CreateContext();
  fprintf(stdout, "\nHello, gpu.cpp\n\n");
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr;
  std::array<float, N> outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  GPUTensor input = CreateTensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32);

  Kernel op = CreateKernel(ctx, CreateShader(kGelu, 256, kf32),
                           input, output);
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 10; ++i) {
    fprintf(stdout, "%d : %f\n", i, outputArr[i]);
  }
  fprintf(stdout, "...\n\n");
  return 0;
}
