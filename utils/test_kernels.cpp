#include <array>
#include <future>
#include <memory>
#include <random>

#include "gpu.h"
#include "utils/array_utils.h"
#include "utils/reference_impls.h"
#include "utils/logging.h"
#include "utils/shaders.h"

using namespace gpu;

bool isclose(float *a, float *b, size_t n, float tol = 1e-3) {
  for (size_t i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) > tol || std::isnan(a[i]) || std::isnan(b[i])) {
      log(kDefLog, kInfo, "Mismatch at index %d: %f != %f", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

void TestResidual(Context &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  Tensor input1 = CreateTensor(ctx, {N}, kf32, input1Arr.data());
  Tensor input2 = CreateTensor(ctx, {N}, kf32, input2Arr.data());
  Tensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  ShaderCode shaderCode = CreateShader(kShaderResidual, workgroupSize, kf32);
  log(kDefLog, kInfo, "Shader Code :\n%s", shaderCode.data.c_str());
  Kernel op =
      CreateKernel(ctx, CreateShader(kShaderResidual, workgroupSize, kf32),
                   TensorList{input1, input2, output}, /* nthreads */ {N, 1, 1});
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s", show<float, N, 1>(outputArr, "Residual Output").c_str());
  std::array<float, N> outputRef;
  residual_forward_cpu(outputRef.data(), input1Arr.data(), input2Arr.data(), N);
  log(kDefLog, kInfo, "%s", show<float, N, 1>(outputRef, "Residual Reference Output").c_str());
  assert(isclose(outputArr.data(), outputRef.data(), N));
  log(kDefLog, kInfo, "Done with Residual Test");
}

void TestHadamard(Context &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  Tensor input1 = CreateTensor(ctx, {N}, kf32, input1Arr.data());
  Tensor input2 = CreateTensor(ctx, {N}, kf32, input2Arr.data());
  Tensor output = CreateTensor(ctx, {N}, kf32, outputArr.data());
  ShaderCode shaderCode = CreateShader(kShaderHadamard, workgroupSize, kf32);
  log(kDefLog, kInfo, "Shader Code :\n%s", shaderCode.data.c_str());
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op =
      CreateKernel(ctx, CreateShader(kShaderHadamard, workgroupSize, kf32),
                   TensorList{input1, input2, output}, /* nthreads */ {N, 1, 1});
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
  log(kDefLog, kInfo, "%s", show<float, N, 1>(outputArr, "Hadamard Output").c_str());
}

void TestMatmul(Context &ctx) {
  static constexpr size_t M = 4;
  static constexpr size_t K = 5;
  static constexpr size_t N = 4;
  auto gen = std::mt19937(31415);
  std::array<float, M * K> input1Arr;
  std::array<float, K * N> input2Arr;
  std::array<float, M * N> outputArr;
  randint(input1Arr, gen, 0, 5);
  range(input2Arr);
  Tensor input1 = CreateTensor(ctx, {M, K}, kf32, input1Arr.data());
  Tensor input2 = CreateTensor(ctx, {K, N}, kf32, input2Arr.data());
  Tensor output = CreateTensor(ctx, {M, N}, kf32, outputArr.data());
  Kernel op =
      CreateKernel(ctx, MatmulShader(256, kShaderMatMul1, kf32, M, K, N),
                   TensorList{input1, input2, output}, /* nthreads */ {M * N, 1, 1});
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
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

void TestTensorPool(Context &ctx) {
  log(kDefLog, kInfo, "Starting Tensor Pool Test");
  // Test using the tensor pool to prepare tensor buffers for kernel invocation
  TensorPool pool = ctx.pool;
  std::array<float, 6> inputArr = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Tensor input = CreateTensor(ctx, {2, 3}, kf32, inputArr.data());
  Tensor output = CreateTensor(ctx, {2, 3}, kf32);
  for (int i = 0; i < 10; i++) {
    Tensor t = CreateTensor(ctx, {2, 3}, kf32);
  }
  // initializing a gpu buffer w/ value and then copy it back to CPU
  std::array<float, 6> initValue = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  log(kDefLog, kInfo, "making tensors with init");
  Tensor tInit = CreateTensor(ctx, {2, 3}, kf32, initValue.data());
  log(kDefLog, kInfo, "Done with Tensor Pool Test");
  std::array<float, 6> targetValue;
  ToCPU(ctx, tInit, targetValue.data(), sizeof(initValue));
  log(kDefLog, kInfo, "%s",
      show<float, 2, 3>(initValue, "initialized GPU value").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, 2, 3>(targetValue, "To CPU from GPU").c_str());
  log(kDefLog, kInfo, "Done with Tensor Pool Test");
}

void TestGelu(Context &ctx) {
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr;
  // range(inputArr);
  auto gen = std::mt19937(31415);
  // TODO(avh): investigate - on metal tanh seems to produce nan for values > 10
  randint(inputArr, gen, 0, 10); // for debugging
  std::array<float, N> outputArr;
  Tensor geluIn = CreateTensor(ctx, {N}, kf32, inputArr.data());
  Tensor geluOut = CreateTensor(ctx, {N}, kf32, outputArr.data());
  log(kDefLog, kInfo, "Creating GELU Shader");
  ShaderCode shader = CreateShader(kShaderGelu, 256, kf32);
  Kernel op = CreateKernel(ctx, shader, TensorList{geluIn, geluOut}, /* nthreads */ {N, 1, 1});
  log(kDefLog, kInfo, "Workgroup size: %s", ToString(shader.workgroupSize).c_str());
  log(kDefLog, kInfo, "Dispatching GELU Shader");
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
  ToCPU(ctx, geluOut, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s", show<float, N, 1>(inputArr, "GELU Input").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, N, 1>(outputArr, "GELU Output").c_str());
  std::array<float, N> refOutputArr;
  gelu_forward_cpu(refOutputArr.data(), inputArr.data(), N);
  log(kDefLog, kInfo, "%s",
      show<float, N, 1>(refOutputArr, "GELU Reference Output").c_str());
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N);
  assert(passed);
  log(kDefLog, kInfo, "Gelu passed? %d", passed);
  log(kDefLog, kInfo, "Done with Gelu Test");
}

void TestLayerNorm(Context &ctx) {
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
  Tensor input = CreateTensor(ctx, {N, C}, kf32, inputArr.data());
  LNParam params = {N, C};
  randint(weightArr, gen, 0, 5); // populate randomly
  randint(biasArr, gen, 0, 5);
  Tensor weight = CreateTensor(ctx, {C}, kf32, weightArr.data());
  Tensor bias = CreateTensor(ctx, {C}, kf32, biasArr.data());
  Tensor output = CreateTensor(ctx, {N, C}, kf32, outputArr.data());
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = CreateKernel(ctx, CreateShader(kShaderLayerNorm1, 256, kf32),
                           TensorList{input, weight, bias, output}, /* n threads */{N, 1, 1}, params);
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
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

void TestSoftmax(Context &ctx) {

  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  static constexpr size_t B = 6; // batch size
  static constexpr size_t T = 8; // token index
  static constexpr size_t C = 3072; // input channels
  std::array<float, B * T * C> inputArr;
  std::array<float, B * T * C> outputArr;
  std::mt19937 gen(31415);
  randint(inputArr, gen, 0, 3);
  Tensor input = CreateTensor(ctx, {B * T, C}, kf32, inputArr.data());
  Tensor output = CreateTensor(ctx, {B * T, C}, kf32, outputArr.data());
  log(kDefLog, kInfo, "num threads: %d", B * T);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = CreateKernel(ctx, CreateShader(kShaderSoftmax1, 256, kf32), TensorList{input,
                           output}, /* nthreads */ Shape{B * T, 1, 1}, SoftmaxParam{B * T, C});
  DispatchKernel(ctx, op, promise);
  Wait(ctx, future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output").c_str());
  std::array<float, B * T * C> refOutputArr;
  softmax_forward_cpu(refOutputArr.data(), inputArr.data(), B * T, C);
  log(kDefLog, kInfo, "%s",
      show<float, B * T, C>(refOutputArr, "Softmax reference Output").c_str());

  log(kDefLog, kInfo, "number of elements: %d", B * T * C);
  bool passed = isclose(outputArr.data(), refOutputArr.data(), B * T * C);
  assert(passed);
  log(kDefLog, kInfo, "Softmax passed? %d", passed);
  log(kDefLog, kInfo, "Done with Softmax Test");
}

void TestAttention(Context &ctx) {
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
}

int main(int argc, char **argv) {
  Context ctx = CreateContext();

  TestTensorPool(ctx);
  TestResidual(ctx);
  TestHadamard(ctx);
  TestMatmul(ctx);
  TestGelu(ctx);
  TestLayerNorm(ctx);
  TestSoftmax(ctx);

  log(kDefLog, kInfo, "Done with all tests");
}
