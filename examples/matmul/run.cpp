#include <array>
#include <future>
#include <random>
#include "gpu.h"
#include "utils/logging.h"
#include "array_utils.h"

using namespace gpu;

static const char *kShaderMatmul1 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>) {
    let row = global_id.y; // row and column of C 
    let col = global_id.x;
    if (row >= {{M}} || col >= {{N}}) {
        return;
    }
    var total: f32 = A[row * {{K}}] * B[col * {{K}}]; // assumes size >= 1
    for (var k = 1u; k < {{K}}; k = k + 1u) {
        // B is stored as B^T, effectively column-major
        total += A[row * {{K}} + k] * B[col * {{K}} + k];
    }
    C[row * {{N}} + col] = total;
}
)";

inline ShaderCode createMatmul(const char *shaderTemplate, const size_t M,
                               const size_t K, const size_t N,
                               const Shape &workgroupSize = {256, 1, 1},
                               NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  ReplaceAll(codeString, "{{workgroupSize}}", toString(workgroupSize));
  ReplaceAll(codeString, "{{workgroupSizeX}}",
             std::to_string(workgroupSize[0]));
  ReplaceAll(codeString, "{{workgroupSizeY}}",
             std::to_string(workgroupSize[1]));
  ReplaceAll(codeString, "{{workgroupSizeZ}}",
             std::to_string(workgroupSize[2]));
  ReplaceAll(codeString, "{{precision}}", toString(precision));
  ReplaceAll(codeString, "{{M}}", std::to_string(M));
  ReplaceAll(codeString, "{{K}}", std::to_string(K));
  ReplaceAll(codeString, "{{N}}", std::to_string(N));
  // LOG(kDefLog, kInfo, "Shader code:\n%s\n", codeString.c_str());
  return ShaderCode{codeString, workgroupSize};
}

int main() {
  LOG(kDefLog, kInfo, "Done.");
  static constexpr size_t M = 1;
  static constexpr size_t K = 10;
  static constexpr size_t N = 8;

  Context ctx = createContext();

  std::mt19937 gen(314159);

  std::array<float, M * K> inputArr;
  randint(inputArr, gen, -2, 2);
  std::array<float, N * K> weightsArr;
  randint(weightsArr, gen, -2, 2);
  LOG(kDefLog, kInfo, "%s",
      show<float>(inputArr.data(), 1, K, "Input").c_str());
  Tensor input = createTensor(ctx, Shape{M, K}, kf32, inputArr.data());

  // column-major
  Tensor weights = createTensor(ctx, Shape{N, K}, kf32, weightsArr.data());
  LOG(kDefLog, kInfo, "%s",
      show<float>(weightsArr.data(), N, K, "Weights").c_str());

  ShaderCode matmul = createMatmul(kShaderMatmul1, M, K, N);
  Tensor output = createTensor(ctx, Shape{M, N}, kf32);
  Kernel kernel = createKernel(
      ctx, matmul, TensorList{input, weights, output}, {N * M, 1, 1});
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, kernel, promise);

  std::array<float, M * N> outputArr;
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float>(outputArr.data(), M, N, "Output").c_str());

  return 0;
}
