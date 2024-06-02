#include <array>
#include <future>
#include <random>

#include "array_utils.h"
#include "reference_impls.h"
#include "spdlog/spdlog.h"
#include "gpu.h"
#include "kernels.h"

using namespace gpu;

bool isclose(float *a, float *b, size_t n, float tol = 1e-4) {
  for (size_t i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

void LoggingConfig() {
  auto logger = spdlog::default_logger();
  logger->set_pattern("[%^%l%$] %v");
  logger->set_level(spdlog::level::info);
}

void TestResidual(GPUContext &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  ShaderCode shaderCode;
  WGPUTensor input1 = Tensor(ctx, std::array{N}, kf32, input1Arr.data());
  WGPUTensor input2 = Tensor(ctx, std::array{N}, kf32, input2Arr.data());
  WGPUTensor output = Tensor(ctx, std::array{N}, kf32, outputArr.data());
  shaderCode = ResidualKernel(workgroupSize, "f32");
  spdlog::info("Shader Code :\n{}", shaderCode.code);
  Op op = CreateOp(ctx, ResidualKernel(workgroupSize, "f32"),
                         std::array{input1, input2}, output);
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  spdlog::info("{}", show<float, N, 1>(outputArr, "Output"));
  spdlog::info("Done with Residual Test");
}

void TestHadamard(GPUContext &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  ShaderCode shaderCode;
  WGPUTensor input1 = Tensor(ctx, std::array{N}, kf32, input1Arr.data());
  WGPUTensor input2 = Tensor(ctx, std::array{N}, kf32, input2Arr.data());
  WGPUTensor output = Tensor(ctx, std::array{N}, kf32, outputArr.data());
  spdlog::info("Shader Code :\n{}", shaderCode.code);
  Op op = CreateOp(ctx, HadamardKernel(workgroupSize, "f32"),
                         std::array{input1, input2}, output, {});
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  spdlog::info("{}", show<float, N, 1>(outputArr, "Output"));
}

void TestMatmul(GPUContext &ctx) {
  static constexpr size_t M = 4;
  static constexpr size_t K = 5;
  static constexpr size_t N = 4;
  auto gen = std::mt19937(31415);
  std::array<float, M * K> input1Arr;
  std::array<float, K * N> input2Arr;
  std::array<float, M * N> outputArr;
  randint(input1Arr, gen, 0, 5);
  range(input2Arr);
  WGPUTensor input1 = Tensor(ctx, std::array{M, K}, kf32, input1Arr.data());
  WGPUTensor input2 = Tensor(ctx, std::array{K, N}, kf32, input2Arr.data());
  WGPUTensor output = Tensor(ctx, std::array{M, N}, kf32, outputArr.data());
  Op op = CreateOp(ctx, MatmulKernel(256, kShaderMatMul1, "f32", M, K, N),
                       std::array{input1, input2}, output);
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  spdlog::info("{}", show<float, M, K>(input1Arr, "A"));
  spdlog::info("{}", show<float, K, N>(input2Arr, "B"));
  spdlog::info("{}", show<float, M, N>(outputArr, "C"));

  std::array<float, M * N> refOutputArr;
  std::array<float, K * N> input2ArrT;
  transpose(input2Arr.data(), input2ArrT.data(), K, N);
  spdlog::info("{}", show<float, N, K>(input2ArrT, "B'"));
  matmul_forward_cpu(refOutputArr.data(), input1Arr.data(), input2ArrT.data(),
                     nullptr, 1, M, K, N);
  spdlog::info("{}", show<float, M, N>(refOutputArr, "C (reference)"));

  spdlog::info("Done with Matmul Test");
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N);
  assert(passed);
}

void TestTensorPool(GPUContext &ctx) {
  spdlog::info("Starting Tensor Pool Test");
  // Test using the tensor pool to prepare tensor buffers for kernel invocation
  TensorPool pool(&ctx);
  std::array<float, 6> inputArr = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  WGPUTensor input =
      Tensor(ctx, std::array<size_t, 2>{2, 3}, kf32, inputArr.data());
  WGPUTensor output = Tensor<2>(ctx, {2, 3}, kf32);
  // TODO(avh): LaunchKernel with buffers and/or tensors
  // Test using the tenor pool to create tensors
  for (int i = 0; i < 10; i++) {
    WGPUTensor t = Tensor<2>(ctx, {2, 3}, kf32);
  }
  // initializing a gpu buffer w/ value and then copy it back to CPU
  std::array<float, 6> initValue = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  spdlog::info("making tensors withi init");
  WGPUTensor tInit =
      Tensor(ctx, std::array<size_t, 2>{2, 3}, kf32, initValue.data());
  spdlog::info("made tensors with init");
  spdlog::info("Done with Tensor Pool Test");
  std::array<float, 6> targetValue;
  ToCPU(ctx, tInit, targetValue.data(), sizeof(initValue));
  spdlog::info("{}", show<float, 2, 3>(initValue, "initialized GPU value"));
  spdlog::info("{}", show<float, 2, 3>(targetValue, "To CPU from GPU"));
  spdlog::info("Done with Tensor Pool Test");
}

void TestGelu(GPUContext &ctx) {
  static constexpr size_t N = 100;
  std::array<float, N> inputArr;
  range(inputArr);
  std::array<float, N> outputArr;
  std::array<size_t, 1> shapeArray = {N};
  Shape geluShape = Shape{shapeArray.data(), 1};
  WGPUTensor geluIn = Tensor(ctx, geluShape, kf32, inputArr.data());
  WGPUTensor geluOut = Tensor(ctx, geluShape, kf32, outputArr.data());
  spdlog::info("Creating GELU Kernel");
  Op op =
      CreateOp(ctx, GeluKernel(256, "f32"), std::array{geluIn}, geluOut);
  spdlog::info("Launching GELU Kernel");
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, geluOut, outputArr.data(), sizeof(outputArr));
  spdlog::info("{}", show<float, N, 1>(inputArr, "GELU Input"));
  spdlog::info("{}", show<float, N, 1>(outputArr, "GELU Output"));
  std::array<float, N> refOutputArr;
  gelu_forward_cpu(refOutputArr.data(), inputArr.data(), N);
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N);
  assert(passed);
  spdlog::info("Gelu Passed? {}", passed);
  spdlog::info("Done with Gelu Test");
}

void TestLayerNorm(GPUContext &ctx) {
  struct LNParam {
    uint32_t N; // check
    uint32_t C;
  };
  constexpr size_t N = 6;
  constexpr size_t C = 3072;
  std::mt19937 gen(31415);
  std::array<float, N * C> inputArr;
  randint(inputArr, gen, 0, 3);
  // range(inputArr);
  std::array<float, N * C> outputArr;
  std::array<float, C> weightArr;
  std::array<float, C> biasArr;
  WGPUTensor input = Tensor(ctx, std::array{N, C}, kf32, inputArr.data());
  LNParam params = {N, C};
  randint(weightArr, gen, 0, 5); // populate randomly
  randint(biasArr, gen, 0, 5);
  WGPUTensor weight = Tensor(ctx, std::array{C}, kf32, weightArr.data());
  WGPUTensor bias = Tensor(ctx, std::array{C}, kf32, biasArr.data());
  WGPUTensor output = Tensor(ctx, std::array{N, C}, kf32, outputArr.data());
  Op op =
      CreateOp<LNParam, 3>(ctx, LayerNormKernel(256, "f32"),
                           std::array{input, weight, bias}, output, params);
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  spdlog::info("{}", show<float, N, C>(inputArr, "LayerNorm Input"));
  spdlog::info("{}", show<float, 1, C>(weightArr, "LayerNorm Weight"));
  spdlog::info("{}", show<float, 1, C>(biasArr, "LayerNorm Bias"));
  spdlog::info("{}", show<float, N, C>(outputArr, "LayerNorm Output"));
  std::array<float, N * C> refOutputArr;
  layernorm_forward_cpu(refOutputArr.data(), inputArr.data(), weightArr.data(),
                        biasArr.data(), N, 1, C);
  spdlog::info("{}",
               show<float, N, C>(refOutputArr,
                                 "LayerNorm Reference Implementation Output"));
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N * C);
  assert(passed);
  spdlog::info("LayerNorm passed? {}", passed);
  spdlog::info("Done with LayerNorm Test");
}

void TestSoftmax(GPUContext &ctx) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  static constexpr size_t B = 6; // batch size
  static constexpr size_t T = 8; // token index // TODO(avh): show can segfault
                                 // if dimensions are too large
  static constexpr size_t C = 3072; // input channels
  std::array<float, B * T * C> inputArr;
  std::array<float, B * T * C> outputArr;
  std::mt19937 gen(31415);
  randint(inputArr, gen, 0, 3);
  WGPUTensor input = Tensor(ctx, std::array{B, T, C}, kf32, inputArr.data());
  WGPUTensor output = Tensor(ctx, std::array{B, T, C}, kf32, outputArr.data());
  Op op = CreateOp<SoftmaxParam, 1>(ctx, SoftmaxKernel(256, "f32"),
                                         {input}, output, {B * T, C});
  LaunchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  spdlog::info("{}", show<float, B * T, C>(inputArr, "Softmax Input"));
  spdlog::info("{}", show<float, B * T, C>(outputArr, "Softmax Output"));

  std::array<float, B * T * C> refOutputArr;
  softmax_forward_cpu(refOutputArr.data(), inputArr.data(), B * T, C);
  spdlog::info("{}",
               show<float, B * T, C>(refOutputArr, "Softmax reference Output"));
  bool passed = isclose(outputArr.data(), refOutputArr.data(), B * T * C);
  assert(passed);
  spdlog::info("Softmax passed? {}", passed);
  spdlog::info("Done with Softmax Test");
}

void TestAttention(GPUContext &ctx) {
  static constexpr size_t B = 6;
  static constexpr size_t T = 32;   // token index
  static constexpr size_t C = 3072; // input channels
  static constexpr size_t QKV_DIM = 256;
  static constexpr size_t N_HEADS = 12;
  static constexpr size_t OC =
      QKV_DIM * N_HEADS * 3; // output channels, 3 for Q, K, V
  std::array<float, B * T * C> inputArr;
  std::array<float, B * OC> outputArr;
  std::array<float, C * OC> weightArr;
  // TODO: finish
}


int main(int argc, char **argv) {
  LoggingConfig();
  GPUContext ctx = CreateGPUContext(/* verbose logging */false);

  TestTensorPool(ctx);
  TestResidual(ctx);
  TestHadamard(ctx);
  TestMatmul(ctx);
  TestGelu(ctx);
  TestLayerNorm(ctx);
  TestSoftmax(ctx);

  spdlog::info("Done with all tests");
}
