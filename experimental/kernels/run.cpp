#include <array>
#include <future>
#include <memory>
#include <random>

#include "gpu.hpp"
#include "utils/array_utils.hpp"
#include "utils/logging.hpp"

#include "llmc/reference_impls.h"
#include "kernels.h"

using namespace gpu;


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
  Kernel op = createKernel(ctx, {kShaderLayerNorm1, 256, kf32},
                           Bindings{input, weight, bias, output},
                           /* n threads */ {N, 1, 1}, params);
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
  ref::layernorm_forward_cpu(refOutputArr.data(), inputArr.data(),
                             weightArr.data(), biasArr.data(), N, 1, C);
  LOG(kDefLog, kInfo, "%s",
      show<float, N, C>(refOutputArr,
                        "LayerNorm Reference Implementation Output")
          .c_str());
  bool passed = isclose(outputArr.data(), refOutputArr.data(), N * C);
  assert(passed);
  LOG(kDefLog, kInfo, "LayerNorm passed? %d", passed);
}

void testSoftmax(Context &ctx) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
    uint32_t Cp;
  };
  static constexpr size_t B = 6;    // batch size
  static constexpr size_t T = 8;    // token index
  static constexpr size_t C = 3072; // input channels
  static constexpr size_t Cp = 3072; // input channels with padding
  std::array<float, B * T * C> inputArr;
  std::array<float, B * T * C> outputArr;
  std::mt19937 gen(31415);
  randint(inputArr, gen, 0, 3);
  Tensor input = createTensor(ctx, {B * T, C}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, {B * T, C}, kf32, outputArr.data());
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(
      ctx, {kShaderSoftmax1, 256, kf32}, Bindings{input, output},
      Shape{cdiv(B * T, 256), 1, 1}, SoftmaxParam{B * T, C, Cp});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(inputArr, "Softmax Input").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(outputArr, "Softmax Output").c_str());
  std::array<float, B * T * C> refOutputArr;
  ref::softmax_forward_cpu(refOutputArr.data(), inputArr.data(), B * T, C);
  LOG(kDefLog, kInfo, "%s",
      show<float, B * T, C>(refOutputArr, "Softmax reference Output").c_str());
  LOG(kDefLog, kInfo, "number of elements: %d", B * T * C);
  bool passed = isclose(outputArr.data(), refOutputArr.data(), B * T * C);
  assert(passed);
  LOG(kDefLog, kInfo, "Softmax passed? %d", passed);
}


int main(int argc, char **argv) {
  Context ctx = createContext();
  testLayerNorm(ctx);
  testSoftmax(ctx);

  LOG(kDefLog, kInfo, "Done with all tests");
}


