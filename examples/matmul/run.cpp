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
    @builtin(global_invocation_id) globalID : vec3<u32>) {
    let row = globalID.x; // Use x as row makes mapping to Shape more intuitive
    let col = globalID.y;
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
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(local_invocation_index) localIdx : u32,
    @builtin(workgroup_id) groupID: vec3<u32>) {
    let localRow = localIdx /  tileSize;
    let localCol = localIdx % tileSize;
    // let localRow = localID.x;
    // let localCol = localID.y;
    let row = groupID.x * tileSize + localRow;
    let col = groupID.y * tileSize + localCol;
    var total: f32 = 0.0;
    for (var tile = 0u; tile < ({{K}} + {{tileSize}} - 1) / {{tileSize}}; tile = tile + 1u) {
    let aRow = groupID.x * {{tileSize}} + localRow;
    let aCol = tile * tileSize + localCol;
      let aIndex =  aRow * {{K}} + aCol;
      let bRow = groupID.y * {{tileSize}} + localCol;
      let bCol = tile * tileSize + localRow;
      let bIndex = bRow * {{K}} + bCol;

      // We can eliminate the f32(...) masking scalar
      // *if* it's known apriori that tileSize is evenly
      // divisible into M, K, and N dimensions
      tileA[localRow * {{tileSize}} + localCol] =
        A[aIndex] * f32(aRow < {{M}} && aCol < {{K}});
      tileB[localCol * {{tileSize}} + localRow] =
        B[bIndex] * f32(bRow < {{N}} && bCol < {{K}});

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
    // C[row * {{N}} + col] = f32(localIdx);
}
)";

inline ShaderCode createMatmul(const char *shaderTemplate, const size_t M,
                               const size_t K, const size_t N,
                               const Shape &workgroupSize = {256, 1, 1},
                               NumType precision = kf32) {
  std::string codeString(shaderTemplate);

  ReplaceAll(codeString, "{{workgroupSize}}", toString(workgroupSize));
  ReplaceAll(
      codeString, "{{tileSize}}",
      std::to_string(static_cast<size_t>(
          sqrt(workgroupSize[0])))); // assumes 1D workgroup spread onto 2D tile
  ReplaceAll(codeString, "{{precision}}", toString(precision));
  ReplaceAll(codeString, "{{M}}", std::to_string(M));
  ReplaceAll(codeString, "{{K}}", std::to_string(K));
  ReplaceAll(codeString, "{{N}}", std::to_string(N));
  LOG(kDefLog, kInfo, "Shader code:\n%s\n", codeString.c_str());
  return ShaderCode{codeString, workgroupSize};
}

int main() {
  LOG(kDefLog, kInfo, "Done.");
  static constexpr size_t M = 16;
  static constexpr size_t K = 3072;
  static constexpr size_t N = 3072;
  int version = 2;

  Context ctx = createContext();

  std::mt19937 gen(314159);

  std::unique_ptr<float[]> inputPtr = std::make_unique<float[]>(M * K);
  randint(inputPtr.get(), M * K, gen, -2, 2);
  std::unique_ptr<float[]> weightsPtr = std::make_unique<float[]>(N * K);
  randint(weightsPtr.get(), N * K, gen, -2, 2);
  LOG(kDefLog, kInfo, "%s", show<float>(inputPtr.get(), 1, K, "Input").c_str());
  Tensor input = createTensor(ctx, Shape{M, K}, kf32, inputPtr.get());

  // column-major
  Tensor weights = createTensor(ctx, Shape{N, K}, kf32, weightsPtr.get());
  LOG(kDefLog, kInfo, "%s",
      show<float>(weightsPtr.get(), N, K, "Weights").c_str());

  Tensor output = createTensor(ctx, Shape{M, N}, kf32);

  Kernel kernel;
  if (version == 1) {
    Shape wgSize = {2, 2, 1};
    LOG(kDefLog, kInfo, "wgSize: %s", toString(wgSize).c_str());
    ShaderCode matmul =
        createMatmul(kShaderMatmul1, M, K, N, /*wgsize*/ wgSize);
    kernel = createKernel(ctx, matmul, Bindings{input, weights, output},
                          /*nWorkgroups*/ cdiv({M, N, 1}, wgSize));
  } else if (version == 2) {
    static constexpr size_t tileSize = 16;
    ShaderCode matmul = createMatmul(kShaderMatmul2, M, K, N,
                                     /*wgSize*/ {tileSize * tileSize, 1, 1});
    kernel =
        createKernel(ctx, matmul, Bindings{input, weights, output},
                     /* nWorkgroups*/ cdiv({M, N, 1}, {tileSize, tileSize, 1}));
  }

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, kernel, promise);
  std::unique_ptr<float[]> outputPtr = std::make_unique<float[]>(M * N);
  toCPU(ctx, output, outputPtr.get(), M * N * sizeof(float));
  LOG(kDefLog, kInfo, "%s",
      show<float>(outputPtr.get(), M, N, "Output").c_str());

  std::unique_ptr<float[]> outputRefPtr = std::make_unique<float[]>(M * N);
  ref::matmul_forward_cpu(outputRefPtr.get(), inputPtr.get(), weightsPtr.get(),
                          nullptr, 1, M, K, N);
  LOG(kDefLog, kInfo, "Reference Output: %s",
      show<float>(outputRefPtr.get(), M, N, "Output (Reference)").c_str());
  LOG(kDefLog, kInfo,
      isclose(outputPtr.get(), outputRefPtr.get(), M * N) ? "PASS" : "FAIL");
  return 0;
}
