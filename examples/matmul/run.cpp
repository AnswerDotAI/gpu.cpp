#include <array>
#include <chrono>
#include <future>
#include <random>
#include <cstdlib>

#include "gpu.h" // createContext, createTensor, createKernel, dispatchKernel,
                 // wait, resetCommandBuffer, toCPU

#include "llmc/reference_impls.h" // for CPU reference implementation
#include "utils/array_utils.h"    // show, isclose, randn, randint
#include "utils/logging.h"        // LOG

using namespace gpu;

static const char *kShaderMatmul1 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> C: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>) {
    let row = globalID.x; // Use x as row makes mapping to Shape more intuitive
    let col = globalID.y;
    if (row >= {{M}} || col >= {{N}}) {
        return;
    }
    var total: {{precision}} = A[row * {{K}}] * B[col * {{K}}]; // assumes size >= 1
    for (var k = 1u; k < {{K}}; k = k + 1u) {
        // B is stored as B^T, effectively column-major
        total += A[row * {{K}} + k] * B[col * {{K}} + k];
    }
    C[row * {{N}} + col] = total;
}
)";

inline ShaderCode createMatmul1(const char *shaderTemplate, const size_t M,
                                const size_t K, const size_t N,
                                const Shape &workgroupSize = {256, 1, 1},
                                NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)},
                          {"{{M}}", toString(M)},
                          {"{{K}}", toString(K)},
                          {"{{N}}", toString(N)}});
  return ShaderCode{codeString, workgroupSize};
}

// Shared memory cache-blocking
static const char *kShaderMatmul2 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> C: array<{{precision}}>;
var<workgroup> As: array<{{precision}}, {{tileSize}} * {{tileSize}}>;
var<workgroup> Bs: array<{{precision}}, {{tileSize}} * {{tileSize}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_index) localIdx : u32,
  @builtin(workgroup_id) groupID: vec3<u32>) {
    let loadRow = localIdx /  {{tileSize}};
    let loadCol = localIdx % {{tileSize}};
    let row = groupID.x * {{tileSize}} + loadRow;
    let col = groupID.y * {{tileSize}} + loadCol;
    let aRow = groupID.x * {{tileSize}} + loadRow;
    let bRow = groupID.y * {{tileSize}} + loadCol;
    var total: {{precision}} = 0.0;
    for (var tile = 0u;
         tile < ({{K}} + {{tileSize}} - 1) / {{tileSize}};
         tile = tile + 1u) {
      let aCol = tile * {{tileSize}} + loadCol;
      let bCol = tile * {{tileSize}} + loadRow;
      // We can skip masking here *iff* tileSize is evenly
      // divisible into M, K, and N dimensions
      As[loadRow * {{tileSize}} + loadCol] =
        A[aRow * {{K}} + aCol];
        // A[aRow * {{K}} + aCol] * {{precision}}(aRow < {{M}} && aCol < {{K}}); // masked version
      Bs[loadCol * {{tileSize}} + loadRow] =
        B[bRow * {{K}} + bCol];
        // B[bRow * {{K}} + bCol] * {{precision}}(bRow < {{N}} && bCol < {{K}}); // masked version
      workgroupBarrier();
      for (var k = 0u; k < {{tileSize}}; k = k + 1u) {
        total += As[loadRow * {{tileSize}} + k] *
                 Bs[loadCol * {{tileSize}} + k];
      }
      workgroupBarrier();
    }
    if (row >= {{M}} || col >= {{N}}) {
      return;
    }
    C[row * {{N}} + col] = total;
}
)";

inline ShaderCode createMatmul2(const char *shaderTemplate, const size_t M,
                                const size_t K, const size_t N,
                                const Shape &workgroupSize = {256, 1, 1},
                                NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString,
             {{"{{workgroupSize}}", toString(workgroupSize)},
              {"{{precision}}", toString(precision)},
              {"{{M}}", toString(M)},
              {"{{K}}", toString(K)},
              {"{{N}}", toString(N)},
              {"{{tileSize}}",
               toString(static_cast<size_t>(sqrt(workgroupSize[0])))}});
  return ShaderCode{codeString, workgroupSize};
}

/* 1D block-tiling
 *
 * - A block tile in C is of size BM x BN
 * - Each workgroup computes a BM x BN block of C
 * - The BM rows of a block tile in As are split into TM x TK
 *   tiles
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
@group(0) @binding(0) var<storage, read_write> a: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> b: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> c: array<{{precision}}>;
var<workgroup> tileA: array<{{precision}}, {{BM}} * {{BK}}>;
var<workgroup> tileB: array<{{precision}}, {{BN}} * {{BK}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(local_invocation_index) localIdx : u32,
    @builtin(workgroup_id) groupid : vec3<u32>) {

    var threadResults: array<{{precision}}, {{TM}}>;

    let cRow: u32 = groupid.x;
    let cCol: u32 = groupid.y;

    // position of the first c element computed by the thread
    let threadRow: u32 = localID.x / {{BN}} * {{TM}};
    let threadCol: u32 = localID.x % {{BN}};

    // value of a to cache in as
    // value of b to cache in bs (b is stored as b^t)
    // Both tiles are of width BK
    let loadColA = localID.x % {{BK}};
    let loadRowA = localID.x / {{BK}};
    let loadColB = loadColA;
    let loadRowB = loadRowA;

    // aPtr and bPtr are the starting positions of the tiles in a and b,
    // incremented in the bkidx loop. 
    // cPtr is the starting position of the tile in c which is fixed.

    var aPtr = cRow * {{BM}} * {{K}};
    var bPtr = (cCol * {{BN}}) * {{K}};
    let cPtr = cRow * {{BM}} * {{N}} + cCol * {{BN}};

    for (var bkidx = 0; bkidx < {{K}}; bkidx += {{BK}}) {

      // Load tile
      tileA[loadRowA * {{BK}} + loadColA] = a[aPtr + loadRowA * {{K}} + loadColA];
      tileB[loadRowB * {{BK}} + loadColB] = b[bPtr + loadRowB * {{K}} + loadColB];
      aPtr += {{BK}};
      bPtr += {{BK}};
      workgroupBarrier();

      // Compute tile
      for (var dotIdx: u32 = 0; dotIdx < {{BK}}; dotIdx = dotIdx + 1) {
        let tmp = tileB[threadCol * {{BK}} + dotIdx];
        for (var residx: u32 = 0; residx < {{TM}}; residx = residx + 1) {

          threadResults[residx] += tileA[(threadRow + residx) * {{BK}} + dotIdx] * tmp;
        }
      }
      workgroupBarrier();
    }

    for (var residx: u32 = 0; residx < {{TM}}; residx = residx + 1) {
      c[cPtr + (threadRow + residx) * {{N}} + threadCol] = threadResults[residx];
    }
    
}
)";

inline ShaderCode createMatmul3(const char *shaderTemplate, const size_t M,
                                const size_t K, const size_t N, const size_t BM,
                                const size_t BK, const size_t BN,
                                const size_t TM,
                                const Shape &workgroupSize = {256, 1, 1},
                                NumType precision = kf32) {
  assert(BM % TM == 0);
  assert(K % BK == 0);
  assert(M % BM == 0);
  assert(N % BN == 0);
  // # threads = tile A size == tile B size == # threads for computing C
  assert(/* tile A size */ BM * BK == /* tile B size */ BK * BN);
  assert(/* tile A size */ BM * BK == /* # of threads for C */ BM * BN / TM);
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)},
                          {"{{M}}", toString(M)},
                          {"{{K}}", toString(K)},
                          {"{{N}}", toString(N)},
                          {"{{BM}}", toString(BM)},
                          {"{{BK}}", toString(BK)},
                          {"{{BN}}", toString(BN)},
                          {"{{TM}}", toString(TM)}});
  return ShaderCode{codeString, workgroupSize};
}

/**
 * @brief No-Op shader with matmul bindings for performance testing
 */
static const char *kShaderNoOp = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> C: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>) {
}
)";

/* 2D block-tiling
 *
 */
static const char *kShaderMatmul4 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> b: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> c: array<{{precision}}>;
var<workgroup> tileA: array<{{precision}}, {{BM}} * {{BK}}>;
var<workgroup> tileB: array<{{precision}}, {{BN}} * {{BK}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>) {

    var threadResults: array<{{precision}}, {{TM}} * {{TN}}>;
    var localM: array<{{precision}}, {{TM}}>;
    var localN: array<{{precision}}, {{TN}}>;

    let cRow: u32 = groupid.x;
    let cCol: u32 = groupid.y;
    let numThread: u32 = ({{BM}} * {{BN}}) / ({{TM}} * {{TN}});

    // position of the first c element computed by the thread
    let threadRow: u32 = (localID.x / ({{BN}} / {{TN}})) * {{TM}};
    let threadCol: u32 = (localID.x % ({{BN}} / {{TN}})) * {{TN}};

    let numIterA: u32 = {{BM}} * {{BK}} / ({{BM}} * {{BN}} / ({{TM}} * {{TN}}));
    let numIterB: u32 = {{BK}} * {{BN}} / ({{BM}} * {{BN}} / ({{TM}} * {{TN}}));

    // aPtr and bPtr are the starting positions of the tiles in a and b,
    // incremented in the bkidx loop. 
    // cPtr is the starting position of the tile in c which is fixed.

    var aPtr = cRow * {{BM}} * {{K}};
    var bPtr = cCol * {{BN}} * {{K}};
    let cPtr = cRow * {{BM}} * {{N}} + cCol * {{BN}};

    for (var bkidx = 0; bkidx < {{K}}; bkidx += {{BK}}) {

      // Load tile
      // Load BM x BK by numThread(BM * BN / (TM * TN))
      // The number of iteration == BM * BK / (BM * BN / (TM * TN))
      for (var i: u32 = 0; i < numIterA; i++) {
        let loadColA: u32 = (localID.x + i * numThread) % {{BK}};
        let loadRowA: u32 = (localID.x + i * numThread) / {{BK}};
        tileA[loadRowA * {{BK}} + loadColA] = a[aPtr + loadRowA * {{K}} + loadColA];
      }
      // Load BK x BN by numThread(BM * BN / (TM * TN))
      // The number of iteration == BK * BN / (BM * BN / (TM * TN))
      for (var i: u32 = 0; i < numIterB; i++) {
        let loadColB: u32 = (localID.x + i * numThread) % {{BK}};
        let loadRowB: u32 = (localID.x + i * numThread) / {{BK}};
        tileB[loadRowB * {{BK}} + loadColB] = b[bPtr + loadRowB * {{K}} + loadColB];
      }

      aPtr += {{BK}};
      bPtr += {{BK}};

      workgroupBarrier();
      // Compute tile
      for (var dotIdx: u32 = 0; dotIdx < {{BK}}; dotIdx = dotIdx + 1) {
        for (var i: u32 = 0; i < {{TM}}; i++) {
          localM[i] = tileA[(threadRow + i) * {{BK}} + dotIdx];
        }
        for (var i: u32 = 0; i < {{TN}}; i++) {
          localN[i] = tileB[(threadCol + i) * {{BK}} + dotIdx];
        }
        for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
          for (var resIdxN: u32 = 0; resIdxN < {{TN}}; resIdxN++) {
            threadResults[resIdxM * {{TN}} + resIdxN] += localM[resIdxM] * localN[resIdxN];
          }
        }
      }
      workgroupBarrier();
    }

    for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
      for (var resIdxN: u32 = 0; resIdxN < {{TN}}; resIdxN++) {
        c[cPtr + (threadRow + resIdxM) * {{N}} + threadCol + resIdxN] = threadResults[resIdxM * {{TN}} + resIdxN];
      }
    }
}
)";

inline ShaderCode createMatmul4(const char *shaderTemplate, const size_t M,
                                const size_t K, const size_t N, const size_t BM,
                                const size_t BK, const size_t BN,
                                const size_t TM, const size_t TN,
                                const Shape &workgroupSize = {256, 1, 1},
                                NumType precision = kf32) {
  assert(BM % TM == 0);
  assert(BN % TN == 0);
  assert(K % BK == 0);
  assert(M % BM == 0);
  assert(N % BN == 0);
  // # threads = tile A size == tile B size == # threads for computing C
  //assert(/* tile A size */ BM * BK == /* tile B size */ BK * BN);
  //assert(/* tile A size */ BM * BK == /* # of threads for C */ BM * BN / (TM * TN));
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)},
                          {"{{M}}", toString(M)},
                          {"{{K}}", toString(K)},
                          {"{{N}}", toString(N)},
                          {"{{BM}}", toString(BM)},
                          {"{{BK}}", toString(BK)},
                          {"{{BN}}", toString(BN)},
                          {"{{TM}}", toString(TM)},
                          {"{{TN}}", toString(TN)}});
  return ShaderCode{codeString, workgroupSize};
}

inline ShaderCode createNoOp(const char *shaderTemplate,
                             const Shape &workgroupSize = {256, 1, 1},
                             NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)}});
  return ShaderCode{codeString, workgroupSize};
}

void initData(size_t M, size_t K, size_t N, std::unique_ptr<float[]> &inputPtr,
              std::unique_ptr<float[]> &weightsPtr) {
  std::mt19937 gen(314159);
  randn(inputPtr.get(), M * K, gen);
  randn(weightsPtr.get(), N * K, gen);
  // randint(inputPtr.get(), M * K, gen, 1, 2);
  // randint(weightsPtr.get(), N * K, gen, 1, 2);
  LOG(kDefLog, kInfo, "%s", show<float>(inputPtr.get(), M, K, "Input").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<float>(weightsPtr.get(), N, K, "Weights").c_str());
}

void checkCPU(size_t M, size_t K, size_t N, std::unique_ptr<float[]> &inputPtr,
              std::unique_ptr<float[]> &weightsPtr,
              std::unique_ptr<float[]> &outputPtr) {
  LOG(kDefLog, kInfo, "Computing CPU reference implementation");
  std::unique_ptr<float[]> outputRefPtr = std::make_unique<float[]>(M * N);
  ref::matmul_forward_cpu(outputRefPtr.get(), inputPtr.get(), weightsPtr.get(),
                          nullptr, 1, M, K, N);
  LOG(kDefLog, kInfo, "Reference Output: %s",
      show<float>(outputRefPtr.get(), M, N, "Output (Reference)").c_str());
  LOG(kDefLog, kInfo,
      isclose(outputPtr.get(), outputRefPtr.get(), M * N) ? "CPU Check: PASS"
                                                          : "CPU Check: FAIL");
}

Kernel selectMatmul(Context &ctx, int version,
                    const Bindings</* input, weights, output */ 3> &bindings,
                    size_t M, size_t K, size_t N) {
  Kernel kernel;
  if (version == 1) {
    Shape wgSize = {16, 16, 1};
    LOG(kDefLog, kInfo, "wgSize: %s", toString(wgSize).c_str());
    ShaderCode matmul =
        createMatmul1(kShaderMatmul1, M, K, N, /*wgsize*/ wgSize);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ cdiv({M, N, 1}, wgSize));
  } else if (version == 2) {
    static constexpr size_t tileSize = 16;
    ShaderCode matmul = createMatmul2(kShaderMatmul2, M, K, N,
                                      /*wgSize*/ {tileSize * tileSize, 1, 1});
    kernel =
        createKernel(ctx, matmul, bindings,
                     /* nWorkgroups*/ cdiv({M, N, 1}, {tileSize, tileSize, 1}));
  } else if (version == 3) {
    static constexpr size_t BM = 64;
    static constexpr size_t BK = 4;
    static constexpr size_t BN = BM;
    static constexpr size_t TM =
        BN / BK; //  BM * BN / TM == BM * BK, therefore TM == BN / BK
    Shape wgSize = {BM * BN / TM, 1,
                    1}; // BM * BN values per workgroup, TM values per thread
    Shape nWorkgroups = {cdiv(M, BM), cdiv(N, BN), 1};
    LOG(kDefLog, kInfo, "M: %d, K: %d, N: %d", M, K, N);
    LOG(kDefLog, kInfo, "BM: %d, BK: %d, BN: %d, TM: %d", BM, BK, BN, TM);
    LOG(kDefLog, kInfo, "wgSize: ( %s )", toString(wgSize).c_str());
    LOG(kDefLog, kInfo, "nWorkgroups: ( %s )", toString(nWorkgroups).c_str());
    ShaderCode matmul = createMatmul3(kShaderMatmul3, M, K, N, BM, BK, BN, TM,
                                      /*wgSize*/ wgSize);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  } else if (version == 4) {
    static constexpr size_t BM = 64;
    static constexpr size_t BK = 16;
    static constexpr size_t BN = 64;
    static constexpr size_t TM = BM / BK;
    static constexpr size_t TN = BN / BK;
    Shape wgSize = {(BM / TM) * (BN / TN), 1, 1}; // This is the same as BK * BK.
    Shape nWorkgroups = {cdiv(M, BM), cdiv(N, BN), 1};
    LOG(kDefLog, kInfo, "M: %d, K: %d, N: %d", M, K, N);
    LOG(kDefLog, kInfo, "BM: %d, BK: %d, BN: %d, TM: %d, TN: %d", BM, BK, BN, TM, TN);
    LOG(kDefLog, kInfo, "wgSize: ( %s )", toString(wgSize).c_str());
    LOG(kDefLog, kInfo, "nWorkgroups: ( %s )", toString(nWorkgroups).c_str());
    ShaderCode matmul = createMatmul4(kShaderMatmul4, M, K, N, BM, BK, BN, TM, TN,
                                      /*wgSize*/ wgSize);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  } else if (version == 5) {
    Shape wgSize = {256, 1, 1};
    Shape nWorkgroups = cdiv({M, N, 1}, {16, 16, 1});
    ShaderCode matmul = createNoOp(kShaderNoOp, /*wgsize*/ wgSize);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  }
  return kernel;
}

void runTest(int version, size_t M, size_t K, size_t N,
             std::unique_ptr<float[]> &inputPtr,
             std::unique_ptr<float[]> &weightsPtr,
             std::unique_ptr<float[]> &outputPtr) {

  // Allocate GPU buffers and copy data
  Context ctx = createContext();
  Tensor input = createTensor(ctx, Shape{M, K}, kf32, inputPtr.get());
  Tensor weights =
      createTensor(ctx, Shape{N, K}, kf32, weightsPtr.get()); // column-major
  Tensor output = createTensor(ctx, Shape{M, N}, kf32);

  constexpr size_t nIter = 5;

  // Initialize Kernel and bind GPU buffers
  LOG(kDefLog, kInfo, "Creating Kernel");
  Kernel kernel = selectMatmul(ctx, version, {input, weights, output}, M, K, N);

  // Dispatch kernel execution
  LOG(kDefLog, kInfo, "Dispatching Kernel version %d, %d iterations ...", version,
      nIter);

  // pre-allocate promises and futures for async dispatch
  // TODO(avh): implement a pooling mechanism for promises/futures in gpu.h
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
      std::chrono::duration_cast<std::chrono::seconds>(end - start);
  float gflops = 2 * M * N *
                 K / // factor of 2 for multiplication & accumulation
                 (static_cast<float>(duration.count())) /
                 1000000000.0 * static_cast<float>(nIter);
  LOG(kDefLog, kInfo,
      "Execution Time: (M = %d, K = %d, N = %d) x %d iterations :  %.1f "
      "milliseconds / dispatch ~ %.2f "
      "GFLOPS",
      M, K, N, nIter, duration.count() / static_cast<float>(nIter), gflops);
  LOG(kDefLog, kInfo, "Copying result to CPU");
  toCPU(ctx, output, outputPtr.get(), M * N * sizeof(float));
  LOG(kDefLog, kInfo, "%s",
      show<float>(outputPtr.get(), M, N, "Output").c_str());
}

int main() {
  char* version_str = getenv("MATMUL_VERSION");
  int version = version_str == NULL ? 3 : atoi(version_str);
    // 1 == naive matmul
    // 2 == tiling
    // 3 == 1D blocktiling
    // 4 == 2D blocktiling
    // 5 == No-Op

  size_t M, K, N;  // Matrix dimensions
  static constexpr int kTestSize = 2;
  if constexpr (kTestSize == 0) {
    // Tiny test
    M = 32;
    K = 32;
    N = 32;
  } else if constexpr (kTestSize == 1) {
    // Small test
    M = 256;
    K = 128;
    N = 512;
  } else {
    // Large test
    M = 4096;
    K = 4096;
    N = 2 * 4096;
  }

  std::unique_ptr<float[]> inputPtr = std::make_unique<float[]>(M * K);
  std::unique_ptr<float[]> weightsPtr = std::make_unique<float[]>(N * K);
  std::unique_ptr<float[]> outputPtr = std::make_unique<float[]>(M * N);

  initData(M, K, N, inputPtr, weightsPtr);
  runTest(version, M, K, N, inputPtr, weightsPtr, outputPtr);

  if constexpr (kTestSize <= 1) {
    // Check result with CPU reference implementation for tiny/small tests
    checkCPU(M, K, N, inputPtr, weightsPtr, outputPtr);
  }

  LOG(kDefLog, kInfo, "Done.");
  return 0;
}
