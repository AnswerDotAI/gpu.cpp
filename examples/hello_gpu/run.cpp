#include <cstdio>
#include "gpu.h"
#include "kernels.h"

using namespace gpu;

int main(int argc, char **argv) {
  GPUContext ctx = CreateGPUContext();
  fprintf(stdout, "Hello, gpu.\n\n");

  // Create input and output tensors
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr;
  std::array<float, N> outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i);
  }
  WGPUTensor input = Tensor(ctx, std::array{N}, kf32, inputArr.data());
  WGPUTensor output = Tensor(ctx, std::array{N}, kf32, outputArr.data());

  // Run the kernel
  Op op =
      CreateOp(ctx, GeluKernel(256, "f32"), std::array{input}, output);
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));

  // Print output
  for (int i = 0; i < 10; ++i) {
    fprintf(stdout, "%d : %f\n", i, outputArr[i]);
  }
  fprintf(stdout, "...\n\nDone.\n");
  return 0;
}
