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

// Shared memory cache-blocking
static const char *kShaderMatmul2 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
var<workgroup> As: array<f32, {{tileSize}} * {{tileSize}}>;
var<workgroup> Bs: array<f32, {{tileSize}} * {{tileSize}}>;
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
      As[localRow * {{tileSize}} + localCol] =
        A[aRow * {{K}} + aCol];
        // A[aRow * {{K}} + aCol] * f32(aRow < {{M}} && aCol < {{K}}); // masked version
      Bs[localCol * {{tileSize}} + localRow] =
        B[bRow * {{K}} + bCol];
        // B[bRow * {{K}} + bCol] * f32(bRow < {{N}} && bCol < {{K}}); // masked version
      workgroupBarrier();
      for (var k = 0u; k < {{tileSize}}; k = k + 1u) {
        total += As[localRow * {{tileSize}} + k] *
                 Bs[localCol * {{tileSize}} + k];
      }
      workgroupBarrier();
    }
    if (row >= {{M}} || col >= {{N}}) {
      return;
    }
    C[row * {{N}} + col] = total;
}
)";

/* 1D block-tiling
 * This is a more advanced version of the tile-based approach
 * that uses 1D workgroups to map to 2D tiles.
 *
 * - A block tile in C is of size BM x BN
 * - Each workgroup computes a BM x BN block of C
 * - The BM rows of a block tile in As are split into TM x TK
 *   tiles, where TM is the number of rows in a workgroup
 *
 * In other words a single thread computing a single value will iterate over
 * block tiles of BM x BN, within which it iterates over tiles TM x BK in As
 * and BK x BN in Bs.
 *
 * There are three nested loops in the kernel:
 * - The outer loop over block tiles which increments
 *   from 0..K by increments of BK
 *
 *   In this outer loop we load BM x BK tiles shared by
 *   the threads in the workgroup.
 *
 * - The second loop which iterates from 0..BK aggregating the partial dot
 *   product contribution of a single tile
 *
 *  - The innermost loop iterates from 0..TM. Each thread in the workgroup
 *  computes a different row of the block tile in C.
 *
 */
static const char *kShaderMatmul3 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

var<workgroup> As: array<f32, {{BM}} * {{BK}}>;
var<workgroup> Bs: array<f32, {{BK}} * {{BN}}>;

@compute @workgroup_size({{BN * (BM / TM)}})
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) groupId: vec3<u32>,
  @builtin(global_invocation_id) globalId: vec3<u32>
) {
    let tileCol = groupId.x;
    let tileRow = groupId.y;
    let threadCol = localId.x % {{BN}};
    let threadRow = localId.x / {{BN}};
    
    let innerColA = localId.x % {{BK}};
    let innerRowA = localId.x / {{BK}};
    let innerColB = localId.x % {{BN}};
    let innerRowB = localId.x / {{BN}};

    var aptr = (tileRow * {{BM}}) * {{K}};
    var bptr = tileCol * {{BN}};
    let cptr = (tileRow * {{BM}}) * {{N}} + tileCol * {{BN}};

    var threadResults: array<f32, {{TM}}>;
    for (var i = 0u; i < {{TM}}; i = i + 1u) {
        threadResults[i] = 0.0;
    }

    for (var tileIdx = 0u; tileIdx < {{K}}; tileIdx = tileIdx + {{BK}}) {
        As[innerRowA * {{BK}} + innerColA] = A[aptr + {{K}} * innerRowA + innerColA];
        Bs[innerRowB * {{BN}} + innerColB] = B[bptr + {{N}} * innerRowB + innerColB];
        
        workgroupBarrier();

        aptr = aptr + {{BK}};
        bptr = bptr + {{BK}} * {{N}};

        for (var k = 0u; k < {{BK}}; k = k + 1u) {
            let tmp = Bs[k * {{BN}} + threadCol];
            for (var resIdx = 0u; resIdx < {{TM}}; resIdx = resIdx + 1u) {
                threadResults[resIdx] = threadResults[resIdx] + As[(threadRow * {{TM}} + resIdx) * {{BK}} + k] * tmp;
            }
        }

        workgroupBarrier();
    }

    for (var resIdx = 0u; resIdx < {{TM}}; resIdx = resIdx + 1u) {
        C[cptr + {{N}} * (threadRow * {{TM}} + resIdx) + threadCol] = threadResults[resIdx];
    }
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

inline ShaderCode createMatmul3(const char *shaderTemplate, const size_t M,
                               const size_t K, const size_t N,
                               const size_t BM, const size_t BK, const size_t BN,
                               const size_t TM,
                               const Shape &workgroupSize = {256, 1, 1},
                               NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  ReplaceAll(codeString, "{{workgroupSize}}", toString(workgroupSize));
  ReplaceAll(codeString, "{{precision}}", toString(precision));
  ReplaceAll(codeString, "{{M}}", std::to_string(M));
  ReplaceAll(codeString, "{{K}}", std::to_string(K));
  ReplaceAll(codeString, "{{N}}", std::to_string(N));
  ReplaceAll(codeString, "{{BM}}", std::to_string(BM));
  ReplaceAll(codeString, "{{BK}}", std::to_string(BK));
  ReplaceAll(codeString, "{{BN}}", std::to_string(BN));
  ReplaceAll(codeString, "{{TM}}", std::to_string(TM));
  LOG(kDefLog, kInfo, "Shader code:\n%s\n", codeString.c_str());
  return ShaderCode{codeString, workgroupSize};
}

int main() {
  // Configuration
  static constexpr size_t M = 4096;
  static constexpr size_t K = 4096;
  static constexpr size_t N = 2 * 4096;
  // static constexpr size_t M = 8;
  // static constexpr size_t K = 16;
  // static constexpr size_t N = 8;
  int version = 2; // 1 == naive
                   // 2 == tile-based
                   // 3 == 1D blocktiling

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
  } else if (version == 3) {
    static constexpr size_t BM = 64;
    static constexpr size_t BK = 64;
    static constexpr size_t BN = 8;
    static constexpr size_t TM = 4;
    ShaderCode matmul = createMatmul3(kShaderMatmul3, M, K, N, BM, BK, BN, TM,
                                      /*wgSize*/ {256, 1, 1});
    kernel = createKernel(ctx, matmul, Bindings{input, weights, output},
                          /*nWorkgroups*/ cdiv({M, N, 1}, {BM, BN, 1}));
  }

  // Dispatch kernel execution
  LOG(kDefLog, kInfo, "Dispatching + waiting");

  // pre-allocate promises and futures for async dispatch
  // TODO(avh): implement a pooling mechanism for promises/futures in gpu.h
  constexpr size_t nIter = 4;
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
  float gigaflops = 2 * M * N *
                    K / // factor of 2 for multiplication & accumulation
                    (static_cast<float>(duration.count()) / 1000.0) /
                    1000000000.0 * static_cast<float>(nIter);
  LOG(kDefLog, kInfo,
      "Execution Time: (M = %d, K = %d, N = %d) x %d iterations :  %.1f "
      "milliseconds / dispatch ~ %.2f "
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
