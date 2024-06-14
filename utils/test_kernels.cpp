#include <array>
#include <future>
#include <memory>
#include <random>

#include "array_utils.h"
#include "gpu.h"
#include "nn/shaders.h"
#include "reference_impls.h"
#include "utils/logging.h"

using namespace gpu;

bool isclose(float *a, float *b, size_t n, float tol = 1e-4) {
  for (size_t i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

void TestResidual(GPUContext &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  GPUTensor input1 = CreateTensor(ctx, {N}, kf32, input1Arr.data());
  GPUTensor input2 = CreateTensor(ctx, {N}, kf32, input2Arr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  ShaderCode shaderCode = CreateShader(kShaderResidual, workgroupSize, kf32);
  log(kDefLog, kInfo, "Shader Code :\n%s", shaderCode.data.c_str());
  Kernel op = CreateKernel(
      ctx, CreateShader(kShaderResidual, workgroupSize, kf32),
      GPUTensors{input1, input2}, output);
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s", show<float, N, 1>(outputArr, "Output").c_str());
  log(kDefLog, kInfo, "Done with Residual Test");
}

void TestHadamard(GPUContext &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  GPUTensor input1 = CreateTensor(ctx, {N}, kf32, input1Arr.data());
  GPUTensor input2 = CreateTensor(ctx, {N}, kf32, input2Arr.data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  ShaderCode shaderCode = CreateShader(kShaderHadamard, workgroupSize, kf32);
  log(kDefLog, kInfo, "Shader Code :\n%s", shaderCode.data.c_str());
  Kernel op =
      CreateKernel(ctx, CreateShader(kShaderHadamard, workgroupSize, kf32),
                   GPUTensors{input1, input2}, output);
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  log(kDefLog, kInfo, "%s", show<float, N, 1>(outputArr, "Output").c_str());
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
  GPUTensor input1 = CreateTensor(ctx, {M, K}, kf32, input1Arr.data());
  GPUTensor input2 = CreateTensor(ctx, {K, N}, kf32, input2Arr.data());
  GPUTensor output = CreateTensor(ctx, {M, N}, kf32, outputArr.data());
  Kernel op =
      CreateKernel(ctx, MatmulShader(256, kShaderMatMul1, kf32, M, K, N),
                   GPUTensors{input1, input2}, output);
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s", show<float, M, K>(input1Arr, "A").c_str());
  log(kDefLog, kInfo, "%s", show<float, K, N>(input2Arr, "B").c_str());
  log(kDefLog, kInfo, "%s", show<float, M, N>(outputArr, "C").c_str());

  std::array<float, M * N> refOutputArr;
  std::array<float, K * N> input2ArrT;
  transpose(input2Arr.data(), input2ArrT.data(), K, N);
  log(kDefLog, kInfo, "%s", show<float, N, K>(input2ArrT, "B'").c_str());
  matmul_forward_cpu(refOutputArr.data(), input1Arr.data(), input2ArrT.data(),
                     nullptr, 1, M, K, N);
  log(kDefLog, kInfo, show<float, M, N>(refOutputArr, "C (reference)").c_str());

  log(kDefLog, kInfo, "Done with Matmul Test");
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N);
  assert(passed);
}

void TestTensorPool(GPUContext &ctx) {
  log(kDefLog, kInfo, "Starting Tensor Pool Test");
  // Test using the tensor pool to prepare tensor buffers for kernel invocation
  TensorPool pool = ctx.pool;
  std::array<float, 6> inputArr = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  GPUTensor input = CreateTensor(ctx, {2, 3}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {2, 3}, kf32);
  for (int i = 0; i < 10; i++) {
    GPUTensor t = CreateTensor(ctx, {2, 3}, kf32);
  }
  // initializing a gpu buffer w/ value and then copy it back to CPU
  std::array<float, 6> initValue = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  log(kDefLog, kInfo, "making tensors with init");
  GPUTensor tInit = CreateTensor(ctx, {2, 3}, kf32, initValue.data());
  log(kDefLog, kInfo, "Done with Tensor Pool Test");
  std::array<float, 6> targetValue;
  ToCPU(ctx, tInit, targetValue.data(), sizeof(initValue));
  log(kDefLog, kInfo, "%s",
      show<float, 2, 3>(initValue, "initialized GPU value").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, 2, 3>(targetValue, "To CPU from GPU").c_str());
  log(kDefLog, kInfo, "Done with Tensor Pool Test");
}

void TestGelu(GPUContext &ctx) {
  static constexpr size_t N = 100;
  std::array<float, N> inputArr;
  range(inputArr);
  std::array<float, N> outputArr;
  GPUTensor geluIn = CreateTensor(ctx, {N}, kf32, inputArr.data());
  GPUTensor geluOut = CreateTensor(ctx, {N}, kf32, outputArr.data());
  log(kDefLog, kInfo, "Creating GELU Shader");
  Kernel op = CreateKernel(ctx, CreateShader(kShaderGELU, 256, kf32),
                           geluIn, geluOut);
  log(kDefLog, kInfo, "Dispatching GELU Shader");
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, geluOut, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s", show<float, N, 1>(inputArr, "GELU Input").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, N, 1>(outputArr, "GELU Output").c_str());
  std::array<float, N> refOutputArr;
  gelu_forward_cpu(refOutputArr.data(), inputArr.data(), N);
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N);
  assert(passed);
  log(kDefLog, kInfo, "Gelu passed? %d", passed);
  log(kDefLog, kInfo, "Done with Gelu Test");
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
  GPUTensor input = CreateTensor(ctx, {N, C}, kf32, inputArr.data());
  LNParam params = {N, C};
  randint(weightArr, gen, 0, 5); // populate randomly
  randint(biasArr, gen, 0, 5);
  GPUTensor weight = CreateTensor(ctx, {C}, kf32, weightArr.data());
  GPUTensor bias = CreateTensor(ctx, {C}, kf32, biasArr.data());
  GPUTensor output = CreateTensor(ctx, {N, C}, kf32, outputArr.data());
  Kernel op =
      CreateKernel(ctx, CreateShader(kShaderLayerNorm1, 256, kf32),
          GPUTensors{input, weight, bias}, output, params);
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s",
      show<float, N, C>(inputArr, "LayerNorm Input").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, 1, C>(weightArr, "LayerNorm Weight").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, 1, C>(biasArr, "LayerNorm Bias").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, N, C>(outputArr, "LayerNorm Output").c_str());
  std::array<float, N * C> refOutputArr;
  layernorm_forward_cpu(refOutputArr.data(), inputArr.data(), weightArr.data(),
                        biasArr.data(), N, 1, C);
  log(kDefLog, kInfo, "%s",
      show<float, N, C>(refOutputArr,
                        "LayerNorm Reference Implementation Output")
          .c_str());
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N * C);
  assert(passed);
  log(kDefLog, kInfo, "LayerNorm passed? %d", passed);
  log(kDefLog, kInfo, "Done with LayerNorm Test");
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
  GPUTensor input = CreateTensor(ctx, {B, T, C}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {B, T, C}, kf32, outputArr.data());
  Kernel op = CreateKernel(
      ctx, CreateShader(kShaderSoftmax1, 256, kf32), input, output,
     SoftmaxParam{B * T, C});
  DispatchKernel(ctx, op);
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output").c_str());
  std::array<float, B * T * C> refOutputArr;
  softmax_forward_cpu(refOutputArr.data(), inputArr.data(), B * T, C);
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(refOutputArr, "Softmax reference Output").c_str());
  bool passed = isclose(outputArr.data(), refOutputArr.data(), B * T * C);
  assert(passed);
  log(kDefLog, kInfo, "Softmax passed? %d", passed);
  log(kDefLog, kInfo, "Done with Softmax Test");
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

void TestMultiKernel1(GPUContext &ctx) {

  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  static constexpr size_t B = 6;    // batch size
  static constexpr size_t T = 8;    // token index
  static constexpr size_t C = 3072; // input channels
  std::array<float, B * T * C> inputArr;
  std::array<float, B * T * C> outputArr;
  std::mt19937 gen(31415);
  randint(inputArr, gen, 0, 3);
  GPUTensor input = CreateTensor(ctx, {B, T, C}, kf32, inputArr.data());
  GPUTensor output = CreateTensor(ctx, {B, T, C}, kf32, outputArr.data());
  auto shader = CreateShader(kShaderSoftmax1, 256, kf32);
  constexpr size_t size = sizeof(SoftmaxParam);
  auto param = SoftmaxParam{B * T, C};
  std::array<size_t, 1> numInputs = {1};
  // First test with the degenerate case of a 1-shader multi kernel
  MultiKernelDesc desc{
      .numShaders = 1,
      .shader = &shader,
      .inputs = &input,
      .numInputs = numInputs.data(),
      .output = &output,
      .params = &param,
      .paramSizes = &size,
  };
  MultiKernel pipeline = CreateMultiKernel(ctx, desc);
  DispatchMultiKernel(ctx, pipeline);
  Wait(ctx, pipeline.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output").c_str());
  log(kDefLog, kInfo, "Done with MultiKernel Test 1");
}

void TestMultiKernel2(GPUContext &ctx) {

  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  static constexpr size_t B = 6;    // batch size
  static constexpr size_t T = 8;    // token index
  static constexpr size_t C = 3072; // input channels
  std::array<float, B * T * C> inputArr;
  std::array<float, B * T * C> outputArr;
  std::mt19937 gen(31415);
  randint(inputArr, gen, 0, 3);

  std::array<ShaderCode, 2> shaders;
  std::array<GPUTensor, 2> inputs;
  std::array<GPUTensor, 2> outputs;
  std::array<SoftmaxParam, 2> params;

  inputs[0] = CreateTensor(ctx, {B, T, C}, kf32, inputArr.data());
  outputs[0] = CreateTensor(ctx, {B, T, C}, kf32, outputArr.data());
  shaders[0] = CreateShader(kShaderSoftmax1, 256, kf32);
  params[0] = SoftmaxParam{B * T, C};

  inputs[1] = CreateTensor(ctx, {B, T, C}, kf32, inputArr.data());
  outputs[1] = CreateTensor(ctx, {B, T, C}, kf32, outputArr.data());
  shaders[1] = CreateShader(kShaderSoftmax1, 256, kf32);
  params[1] = SoftmaxParam{B * T, C};

  std::array<size_t, 2> numInputs = {1, 1};
  std::array<size_t, 2> paramSizes = {sizeof(SoftmaxParam),
                                      sizeof(SoftmaxParam)};

  // First test with the degenerate case of a 1-shader multi kernel
  MultiKernelDesc desc{
      .numShaders = 2,
      .shader = shaders.data(),
      .inputs = inputs.data(),
      .numInputs = numInputs.data(),
      .output = outputs.data(),
      .params = params.data(),
      .paramSizes = paramSizes.data(),
  };
  MultiKernel pipeline = CreateMultiKernel(ctx, desc);
  DispatchMultiKernel(ctx, pipeline);
  Wait(ctx, pipeline.future);

  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  ToCPU(ctx, outputs[0], outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output 0").c_str());
  ToCPU(ctx, outputs[1], outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output 1").c_str());
  log(kDefLog, kInfo, "Done with MultiKernel Test 2");
}

int main(int argc, char **argv) {
  GPUContext ctx = CreateContext(/* verbose logging */ false);

  TestTensorPool(ctx);
  TestResidual(ctx);
  TestHadamard(ctx);
  TestMatmul(ctx);
  TestGelu(ctx);
  TestLayerNorm(ctx);
  TestSoftmax(ctx);
  TestMultiKernel1(ctx);
  TestMultiKernel2(ctx);

  log(kDefLog, kInfo, "Done with all tests");
}
