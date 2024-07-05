#include <array>
#include <chrono>
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
var<workgroup> tileA: array<f32, {{tileSize}} * {{tileSize}}>;
var<workgroup> tileB: array<f32, {{tileSize}} * {{tileSize}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_index) localIdx : u32,
  @builtin(workgroup_id) groupID: vec3<u32>) {
    let localRow = localIdx /  {{tileSize}};
    let localCol = localIdx % {{tileSize}};
    let row = groupID.x * {{tileSize}} + localRow;
    let col = groupID.y * {{tileSize}} + localCol;
    let aRow = groupID.x * {{tileSize}} + localRow;
    let bRow = groupID.y * {{tileSize}} + localCol;
    var total: f32 = 0.0;
    for (var tile = 0u;
         tile < ({{K}} + {{tileSize}} - 1) / {{tileSize}};
         tile = tile + 1u) {
      let aCol = tile * {{tileSize}} + localCol;
      let bCol = tile * {{tileSize}} + localRow;
      // We can skip masking here *iff* tileSize is evenly
      // divisible into M, K, and N dimensions
      tileA[localRow * {{tileSize}} + localCol] =
        A[aRow * {{K}} + aCol];
        // A[aRow * {{K}} + aCol] * f32(aRow < {{M}} && aCol < {{K}}); // masked version
      tileB[localCol * {{tileSize}} + localRow] =
        B[bRow * {{K}} + bCol];
        // B[bRow * {{K}} + bCol] * f32(bRow < {{N}} && bCol < {{K}}); // masked version
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
  // Configuration
  static constexpr size_t M = 2048;
  static constexpr size_t K = 4096;
  static constexpr size_t N = 2 * 4096;
  int version = 2; // 1 == naive implementation
                   // 2 == tiled

  // Initialize Data (host side)
  std::unique_ptr<float[]> inputPtr = std::make_unique<float[]>(M * K);
  std::unique_ptr<float[]> weightsPtr = std::make_unique<float[]>(N * K);
  std::mt19937 gen(314159);
  randn(inputPtr.get(), M * K, gen);
  randn(weightsPtr.get(), N * K, gen);
  LOG(kDefLog, kInfo, "%s", show<float>(inputPtr.get(), M, K, "Input").c_str());
  LOG(kDefLog, kInfo, "Allocating GPU buffer and copying data");
  LOG(kDefLog, kInfo, "%s",
      show<float>(weightsPtr.get(), N, K, "Weights").c_str());

  // Allocate GPU buffers and copy data
  Context ctx = createContext();
  Tensor input = createTensor(ctx, Shape{M, K}, kf32, inputPtr.get());
  Tensor weights =
      createTensor(ctx, Shape{N, K}, kf32, weightsPtr.get()); // column-major
  Tensor output = createTensor(ctx, Shape{M, N}, kf32);

  // Initialize Kernel and bind GPU buffers
  LOG(kDefLog, kInfo, "Creating Kernel");
  Kernel kernel;
  if (version == 1) {
    Shape wgSize = {16, 16, 1};
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

  // Dispatch kernel execution
  LOG(kDefLog, kInfo, "Dispatching + waiting");

  // pre-allocate promises and futures for async dispatch
  // TODO(avh): implement a pooling mechanism for promises/futures
  constexpr size_t nIter = 10;
  std::array<std::promise<void>, nIter> promises;
  std::array<std::future<void>, nIter> futures;
  for (int i = 0; i < nIter; i++) {
    futures[i] = promises[i].get_future();
  }
  
  // Dispatch kernel nIter times
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nIter; i++) {
    dispatchKernel(ctx, kernel, promises[i]);
    wait(ctx, futures[i]);
    resetCommandBuffer(ctx.device, kernel);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Report performance
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  float gigaflops =
      2 * M * N * K / // factor of 2 for multiplication & accumulation
      (static_cast<float>(duration.count()) / 1000.0) / 1000000000.0 * static_cast<float>(nIter);
  LOG(kDefLog, kInfo,
      "Execution Time: (M = %d, K = %d, N = %d) x %d iterations :  %.1f milliseconds / dispatch ~ %.2f "
      "GFLOPS/s",
      M, K, N, nIter, duration.count() / static_cast<float>(nIter), gigaflops);
  std::unique_ptr<float[]> outputPtr = std::make_unique<float[]>(M * N);
  LOG(kDefLog, kInfo, "Copying result to CPU");
  toCPU(ctx, output, outputPtr.get(), M * N * sizeof(float));
  LOG(kDefLog, kInfo, "%s",
      show<float>(outputPtr.get(), M, N, "Output").c_str());

  // CPU reference implementation can be pretty slow for (M > 256) x 3072 x 3072
  {
    /*
    LOG(kDefLog, kInfo, "Computing CPU reference implementation");
    std::unique_ptr<float[]> outputRefPtr = std::make_unique<float[]>(M * N);
    ref::matmul_forward_cpu(outputRefPtr.get(), inputPtr.get(),
                            weightsPtr.get(), nullptr, 1, M, K, N);
    LOG(kDefLog, kInfo, "Reference Output: %s",
        show<float>(outputRefPtr.get(), M, N, "Output (Reference)").c_str());
    LOG(kDefLog, kInfo,
        isclose(outputPtr.get(), outputRefPtr.get(), M * N) ? "PASS" : "FAIL");
    */
  }
  LOG(kDefLog, kInfo, "Done.");
  return 0;
}
