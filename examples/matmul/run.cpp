#include <array>
#include <future>
#include <random>

#include "gpu.h"

#include "array_utils.h"
#include "llmc/reference_impls.h"
#include "utils/logging.h"

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

// Tiling with 1D global and local indexing
static const char *kShaderMatmul2 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
const tileSize: u32 = {{tileSize}}; // TODO make this static
var<workgroup> tileA: array<f32, {{tileSize}} * {{tileSize}}>;
var<workgroup> tileB: array<f32, {{tileSize}} * {{tileSize}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(local_invocation_index) local_index : u32,
    @builtin(workgroup_id) group_id : vec3<u32>) {
    let row = global_id.x / {{N}};
    let col = global_id.x % {{N}};
    let localRow = local_id.x /  tileSize;
    let localCol = local_id.x % tileSize;
    var total: f32 = 0.0;
    let ti = localRow * {{tileSize}} + localCol;
    for (var tileIndex = 0u;
          tileIndex < {{K}} / {{tileSize}};
          tileIndex = tileIndex + 1u) {
      tileA[ti] =
        A[row * {{K}} + tileIndex * {{tileSize}} + localCol];
      tileB[ti] =
        B[col * {{K}} + tileIndex * {{tileSize}} + localRow];

      workgroupBarrier();

      for (var k = 0u; k < {{tileSize}}; k = k + 1u) {
        total += tileA[localRow * {{tileSize}} + k] *
                 tileB[localCol * {{tileSize}} + k];

      }

      workgroupBarrier();

    }
    if (row >= {{M}} || col >= {{N}}) {
      return;
    }
    C[row * {{N}} + col] = total;
    // C[row * {{N}} + col] = f32(localRow);
}
)";

inline ShaderCode createMatmul(const char *shaderTemplate, const size_t M,
                               const size_t K, const size_t N,
                               const Shape &workgroupSize = {256, 1, 1},
                               NumType precision = kf32) {
  std::string codeString(shaderTemplate);

  ReplaceAll(codeString, "{{workgroupSize}}", toString(workgroupSize));
  ReplaceAll(codeString, "{{tileSize}}",
             std::to_string(static_cast<size_t>(sqrt(workgroupSize[0])))); // assumes 1D workgroup spread onto 2D tile
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
  LOG(kDefLog, kInfo, "Shader code:\n%s\n", codeString.c_str());
  return ShaderCode{codeString, workgroupSize};
}

int main() {
  LOG(kDefLog, kInfo, "Done.");
  static constexpr size_t M = 8;
  static constexpr size_t K = 8;
  static constexpr size_t N = 8;
  int version = 1;

  Context ctx = createContext();

  std::mt19937 gen(314159);

  std::array<float, M * K> inputArr;
  // randint(inputArr, gen, -2, 2);
  range(inputArr);
  std::array<float, N * K> weightsArr;
  // randint(weightsArr, gen, -2, 2);
  range(weightsArr);
  LOG(kDefLog, kInfo, "%s",
      show<float>(inputArr.data(), 1, K, "Input").c_str());
  Tensor input = createTensor(ctx, Shape{M, K}, kf32, inputArr.data());

  // column-major
  Tensor weights = createTensor(ctx, Shape{N, K}, kf32, weightsArr.data());
  LOG(kDefLog, kInfo, "%s",
      show<float>(weightsArr.data(), N, K, "Weights").c_str());

  Tensor output = createTensor(ctx, Shape{M, N}, kf32);

  Bindings<3> bindings = {input, weights, output};

  Kernel kernel;
  if (version == 1) {
    ShaderCode matmul = createMatmul(kShaderMatmul1, M, K, N, {16, 1, 1});
    kernel = createKernel(ctx, matmul, bindings,
                                  /*nthreads*/ {M, N, 1});
  } else if (version == 2) {
    ShaderCode matmul = createMatmul(kShaderMatmul2, M, K, N, {16, 1, 1});
    kernel = createKernel(ctx, matmul, bindings,
  /*nthreads*/ {M * N, 1, 1});
  }

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, kernel, promise);

  std::array<float, M * N> outputArr;
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "%s",
      show<float>(outputArr.data(), M, N, "Output").c_str());

  std::array<float, M * N> outputRefArr;
  ref::matmul_forward_cpu(outputRefArr.data(), inputArr.data(),
                          weightsArr.data(), nullptr, 1, M, K, N);
  LOG(kDefLog, kInfo, "Reference Output: %s",
      show<float>(outputRefArr.data(), M, N, "Output (Reference)").c_str());
  LOG(kDefLog, kInfo,
      isclose(outputArr.data(), outputRefArr.data(), M * N) ? "PASS" : "FAIL");
  return 0;
}
