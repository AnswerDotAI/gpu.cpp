#include <array>
#include <cstdio>
#include "gpu.h"
#include "kernels.h"

using namespace gpu;

int main(int argc, char **argv) {
  GPUContext ctx = CreateGPUContext();
  fprintf(stdout, "\nHello, gpu.cpp\n\n");
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr;
  std::array<float, N> outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  GPUTensor input = Tensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor output = Tensor(ctx, {N}, kf32, outputArr.data());

  Kernel op =
      PrepareKernel(ctx, GeluShader(256, kf32), std::array{input}, output);
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 10; ++i) {
    fprintf(stdout, "%d : %f\n", i, outputArr[i]);
  }
  fprintf(stdout, "...\n\n");
  return 0;
}
