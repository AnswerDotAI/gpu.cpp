#include <array>
#include <future>
#include <memory>
#include <random>

#include "gpu.h"
#include "utils/array_utils.h"
#include "utils/logging.h"

#include "shaders.h"
#include "reference_impls.h"

using namespace gpu;

bool isclose(float *a, float *b, size_t n, float tol = 1e-3) {
  for (size_t i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) > tol || std::isnan(a[i]) || std::isnan(b[i])) {
      LOG(kDefLog, kInfo, "Mismatch at index %d: %f != %f", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

void testResidual(Context &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  Tensor input1 = createTensor(ctx, {N}, kf32, input1Arr.data());
  Tensor input2 = createTensor(ctx, {N}, kf32, input2Arr.data());
  Tensor output = createTensor(ctx, {N}, kf32, outputArr.data());
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  ShaderCode shaderCode = createShader(kShaderResidual, workgroupSize, kf32);
  LOG(kDefLog, kInfo, "Shader Code :\n%s", shaderCode.data.c_str());
  Kernel op =
      createKernel(ctx, createShader(kShaderResidual, workgroupSize, kf32),
                   TensorList{input1, input2, output}, /* nthreads */ {N, 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s", show<float, N, 1>(outputArr, "Residual Output").c_str());
  std::array<float, N> outputRef;
  residual_forward_cpu(outputRef.data(), input1Arr.data(), input2Arr.data(), N);
  LOG(kDefLog, kInfo, "%s", show<float, N, 1>(outputRef, "Residual Reference Output").c_str());
  assert(isclose(outputArr.data(), outputRef.data(), N));
  LOG(kDefLog, kInfo, "Done with Residual Test");
}

void testHadamard(Context &ctx) {
  constexpr size_t N = 200000;
  constexpr size_t workgroupSize = 256;
  std::array<float, N> input1Arr;
  std::array<float, N> input2Arr;
  range(input1Arr);
  range(input2Arr);
  std::array<float, N> outputArr;
  Tensor input1 = createTensor(ctx, {N}, kf32, input1Arr.data());
  Tensor input2 = createTensor(ctx, {N}, kf32, input2Arr.data());
  Tensor output = createTensor(ctx, {N}, kf32, outputArr.data());
  ShaderCode shaderCode = createShader(kShaderHadamard, workgroupSize, kf32);
  LOG(kDefLog, kInfo, "Shader Code :\n%s", shaderCode.data.c_str());
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op =
      createKernel(ctx, createShader(kShaderHadamard, workgroupSize, kf32),
                   TensorList{input1, input2, output}, /* nthreads */ {N, 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  LOG(kDefLog, kInfo, "%s", show<float, N, 1>(outputArr, "Hadamard Output").c_str());
}

void testMatmul(Context &ctx) {
  static constexpr size_t M = 4;
  static constexpr size_t K = 5;
  static constexpr size_t N = 4;
  auto gen = std::mt19937(31415);
  std::array<float, M * K> input1Arr;
  std::array<float, K * N> input2Arr;
  std::array<float, M * N> outputArr;
  randint(input1Arr, gen, 0, 5);
  range(input2Arr);
  Tensor input1 = createTensor(ctx, {M, K}, kf32, input1Arr.data());
  Tensor input2 = createTensor(ctx, {K, N}, kf32, input2Arr.data());
  Tensor output = createTensor(ctx, {M, N}, kf32, outputArr.data());
  Kernel op =
      createKernel(ctx, MatmulShader(256, kShaderMatMul1, kf32, M, K, N),
                   TensorList{input1, input2, output}, /* nthreads */ {M * N, 1, 1});
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s", show<float, M, K>(input1Arr, "A").c_str());
  LOG(kDefLog, kInfo, "%s", show<float, K, N>(input2Arr, "B").c_str());
  LOG(kDefLog, kInfo, "%s", show<float, M, N>(outputArr, "C").c_str());

  std::array<float, M * N> refOutputArr;
  std::array<float, K * N> input2ArrT;
  transpose(input2Arr.data(), input2ArrT.data(), K, N);
  LOG(kDefLog, kInfo, "%s", show<float, N, K>(input2ArrT, "B'").c_str());
  matmul_forward_cpu(refOutputArr.data(), input1Arr.data(), input2ArrT.data(),
                     nullptr, 1, M, K, N);
  LOG(kDefLog, kInfo, show<float, M, N>(refOutputArr, "C (reference)").c_str());

  LOG(kDefLog, kInfo, "Done with Matmul Test");
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N);
  assert(passed);
}

void testTensorPool(Context &ctx) {
  LOG(kDefLog, kInfo, "Starting Tensor Pool Test");
  // Test using the tensor pool to prepare tensor buffers for kernel invocation
  TensorPool pool = ctx.pool;
  std::array<float, 6> inputArr = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Tensor input = createTensor(ctx, {2, 3}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, {2, 3}, kf32);
  for (int i = 0; i < 10; i++) {
    Tensor t = createTensor(ctx, {2, 3}, kf32);
  }
  // initializing a gpu buffer w/ value and then copy it back to CPU
  std::array<float, 6> initValue = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  LOG(kDefLog, kInfo, "making tensors with init");
  Tensor tInit = createTensor(ctx, {2, 3}, kf32, initValue.data());
  LOG(kDefLog, kInfo, "Done with Tensor Pool Test");
  std::array<float, 6> targetValue;
  toCPU(ctx, tInit, targetValue.data(), sizeof(initValue));
  LOG(kDefLog, kInfo, "%s",
      show<float, 2, 3>(initValue, "initialized GPU value").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, 2, 3>(targetValue, "To CPU from GPU").c_str());
  LOG(kDefLog, kInfo, "Done with Tensor Pool Test");
}

void testGelu(Context &ctx) {
  static constexpr size_t N = 3072;
  std::array<float, N> inputArr;
  // range(inputArr);
  auto gen = std::mt19937(31415);
  // TODO(avh): investigate - on metal tanh seems to produce nan for values > 10
  randint(inputArr, gen, 0, 10); // for debugging
  std::array<float, N> outputArr;
  Tensor geluIn = createTensor(ctx, {N}, kf32, inputArr.data());
  Tensor geluOut = createTensor(ctx, {N}, kf32, outputArr.data());
  LOG(kDefLog, kInfo, "Creating GELU Shader");
  ShaderCode shader = createShader(kShaderGelu, 256, kf32);
  Kernel op = createKernel(ctx, shader, TensorList{geluIn, geluOut}, /* nthreads */ {N, 1, 1});
  LOG(kDefLog, kInfo, "Workgroup size: %s", toString(shader.workgroupSize).c_str());
  LOG(kDefLog, kInfo, "dispatching GELU Shader");
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, geluOut, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s", show<float, N, 1>(inputArr, "GELU Input").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, N, 1>(outputArr, "GELU Output").c_str());
  std::array<float, N> refOutputArr;
  gelu_forward_cpu(refOutputArr.data(), inputArr.data(), N);
  LOG(kDefLog, kInfo, "%s",
      show<float, N, 1>(refOutputArr, "GELU Reference Output").c_str());
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N);
  assert(passed);
  LOG(kDefLog, kInfo, "Gelu passed? %d", passed);
  LOG(kDefLog, kInfo, "Done with Gelu Test");
}

void testLayerNorm(Context &ctx) {
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
  Tensor input = createTensor(ctx, {N, C}, kf32, inputArr.data());
  LNParam params = {N, C};
  randint(weightArr, gen, 0, 5); // populate randomly
  randint(biasArr, gen, 0, 5);
  Tensor weight = createTensor(ctx, {C}, kf32, weightArr.data());
  Tensor bias = createTensor(ctx, {C}, kf32, biasArr.data());
  Tensor output = createTensor(ctx, {N, C}, kf32, outputArr.data());
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, createShader(kShaderLayerNorm1, 256, kf32),
                           TensorList{input, weight, bias, output}, /* n threads */{N, 1, 1}, params);
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float, N, C>(inputArr, "LayerNorm Input").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, 1, C>(weightArr, "LayerNorm Weight").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, 1, C>(biasArr, "LayerNorm Bias").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, N, C>(outputArr, "LayerNorm Output").c_str());
  std::array<float, N * C> refOutputArr;
  layernorm_forward_cpu(refOutputArr.data(), inputArr.data(), weightArr.data(),
                        biasArr.data(), N, 1, C);
  LOG(kDefLog, kInfo, "%s",
      show<float, N, C>(refOutputArr,
                        "LayerNorm Reference Implementation Output")
          .c_str());
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N * C);
  assert(passed);
  LOG(kDefLog, kInfo, "LayerNorm passed? %d", passed);
  LOG(kDefLog, kInfo, "Done with LayerNorm Test");
}

void testSoftmax(Context &ctx) {

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
  Tensor input = createTensor(ctx, {B * T, C}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, {B * T, C}, kf32, outputArr.data());
  LOG(kDefLog, kInfo, "num threads: %d", B * T);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, createShader(kShaderSoftmax1, 256, kf32), TensorList{input,
                           output}, /* nthreads */ Shape{B * T, 1, 1}, SoftmaxParam{B * T, C});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output").c_str());
  std::array<float, B * T * C> refOutputArr;
  softmax_forward_cpu(refOutputArr.data(), inputArr.data(), B * T, C);
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(refOutputArr, "Softmax reference Output").c_str());

  LOG(kDefLog, kInfo, "number of elements: %d", B * T * C);
  bool passed = isclose(outputArr.data(), refOutputArr.data(), B * T * C);
  assert(passed);
  LOG(kDefLog, kInfo, "Softmax passed? %d", passed);
  LOG(kDefLog, kInfo, "Done with Softmax Test");
}

void testAttention(Context &ctx) {
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
  Context ctx = createContext();

  testTensorPool(ctx);
  testResidual(ctx);
  testHadamard(ctx);
  testMatmul(ctx);
  testGelu(ctx);
  testLayerNorm(ctx);
  testSoftmax(ctx);

  LOG(kDefLog, kInfo, "Done with all tests");
}
