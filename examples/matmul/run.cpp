#include <array>
#include <chrono>
#include <future>
#include <random>
#include <cstdlib>

#include "gpu.hpp" // createContext, createTensor, createKernel, dispatchKernel,
                   // wait, resetCommandBuffer, toCPU

#include "llmc/reference_impls.h"   // for CPU reference implementation
#include "utils/array_utils.hpp"    // show, isclose, randn, randint
#include "utils/logging.hpp"        // LOG
#include "experimental/wgsl.h"      // loopUnrolling
#include "numeric_types/half.hpp"

#ifdef METAL_PROFILER
#include "experimental/profiler/metal.hpp"
#endif

using namespace gpu;

const std::string versionToStr(int version);

void matmulf16_forward_cpu(half* out,
			   const half* inp, const half* weight, const half* bias,
			   int B, int T, int C, int OC) {
  // OC is short for "output channels"
  // inp is (B,T,C), weight is (OC, C)
  // out will be (B,T,OC)
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      half* out_bt = out + b * T * OC + t * OC;
      const half* inp_bt = inp + b * T * C + t * C;
      for (int o = 0; o < OC; o++) {
	float val = (bias != NULL) ? halfToFloat(bias[o]) : 0.0f;
	const half* wrow = weight + o*C;
	for (int i = 0; i < C; i++) {
	  val += halfToFloat(inp_bt[i]) * halfToFloat(wrow[i]);
	}
	out_bt[o] = val;
      }
    }
  }
}

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

inline KernelCode createMatmul1(const char *shaderTemplate, const size_t M,
                                const size_t K, const size_t N,
                                const Shape &workgroupSize = {256, 1, 1},
                                NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)},
                          {"{{M}}", toString(M)},
                          {"{{K}}", toString(K)},
                          {"{{N}}", toString(N)}});
  return {codeString, workgroupSize, precision};
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

inline KernelCode createMatmul2(const char *shaderTemplate, const size_t M,
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
  return {codeString, workgroupSize, precision};
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
        for (var residx: u32 = 0; residx < {{TM}}; residx++) {
          threadResults[residx] += tileA[(threadRow + residx) * {{BK}} + dotIdx] * tmp;
        }
      }
      workgroupBarrier();
    }

    for (var residx: u32 = 0; residx < {{TM}}; residx++) {
      c[cPtr + (threadRow + residx) * {{N}} + threadCol] = threadResults[residx];
    }
    
}
)";

inline KernelCode createMatmul3(const char *shaderTemplate, const size_t M,
                                const size_t K, const size_t N, const size_t BM,
                                const size_t BK, const size_t BN,
                                const size_t TM,
                                const Shape &workgroupSize = {256, 1, 1},
                                NumType precision = kf32,
                                bool unrolling = false) {
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
  if (unrolling) {
    std::string unrolledCode = loopUnrolling(codeString);
    // LOG(kDefLog, kInfo, "Unrolled code:\n%s", unrolledCode.c_str());
    return {unrolledCode, workgroupSize, precision};
  } else {
    return {codeString, workgroupSize, precision};
  }
}

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
      for (var idx: u32 = 0; idx < {{NUM_TILEA}}; idx++) {
        tileA[localID.x + idx * numThread] = a[aPtr + ((localID.x + idx * numThread) / {{BK}}) * {{K}} + (localID.x + idx * numThread) % {{BK}}];
      }
      // Load BK x BN by numThread(BM * BN / (TM * TN))
      // The number of iteration == BK * BN / (BM * BN / (TM * TN))
      for (var idx: u32 = 0; idx < {{NUM_TILEB}}; idx++) {
        tileB[localID.x + idx * numThread] = b[bPtr + ((localID.x + idx * numThread) / {{BK}}) * {{K}} + ((localID.x + idx * numThread) % {{BK}})];
      }

      aPtr += {{BK}};
      bPtr += {{BK}};

      workgroupBarrier();
      // Compute tile
      for (var dotIdx: u32 = 0; dotIdx < {{BK}}; dotIdx = dotIdx + 1) {
        for (var idx: u32 = 0; idx < {{TM}}; idx++) {
          localM[idx] = tileA[(threadRow + idx) * {{BK}} + dotIdx];
        }
        for (var idx: u32 = 0; idx < {{TN}}; idx++) {
          localN[idx] = tileB[(threadCol + idx) * {{BK}} + dotIdx];
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

inline KernelCode createMatmul4(const char *shaderTemplate, const size_t M,
                                const size_t K, const size_t N, const size_t BM,
                                const size_t BK, const size_t BN,
                                const size_t TM, const size_t TN,
                                const Shape &workgroupSize = {256, 1, 1},
                                NumType precision = kf32,
                                bool unrolling = false) {
  assert(BM % TM == 0);
  assert(BN % TN == 0);
  assert(K % BK == 0);
  assert(M % BM == 0);
  assert(N % BN == 0);
  // # threads = tile A size == tile B size == # threads for computing C
  int num_threads = BM * BN / (TM * TN);
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
                          {"{{TN}}", toString(TN)},
                          {"{{NUM_TILEA}}", toString(BM * BK / num_threads)},
                          {"{{NUM_TILEB}}", toString(BN * BK / num_threads)}
                          });
  if (unrolling) {
    std::string unrolledCode = loopUnrolling(codeString);
    // LOG(kDefLog, kInfo, "Unrolled code:\n%s", unrolledCode.c_str());
    return {unrolledCode, workgroupSize, precision};
  } else {
    return {codeString, workgroupSize, precision};
  }
}

/* 2D block-tiling with vectorization
 *
 */
static const char *kShaderMatmulWithVectorization = R"(
@group(0) @binding(0) var<storage, read_write> a: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> b: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4<{{precision}}>>;
var<workgroup> tileA: array<{{precision}}, {{BM}} * {{BK}}>;
var<workgroup> tileB: array<{{precision}}, {{BN}} * {{BK}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>) {

    var threadResults: array<vec4<{{precision}}>, {{TM}} * {{TN4}}>;
    var localM: array<{{precision}}, {{TM}}>;
    var localN: array<vec4<{{precision}}>, {{TN4}}>;

    let cRow: u32 = groupid.x;
    let cCol: u32 = groupid.y;
    let numThread: u32 = ({{BM}} * {{BN}}) / ({{TM}} * {{TN}});

    // position of the first c element computed by the thread
    let threadRow: u32 = (localID.x / ({{BN}} / {{TN}})) * {{TM}};
    let threadCol: u32 = (localID.x % ({{BN}} / {{TN}})) * {{TN}};

    // aPtr and bPtr are the starting positions of the tiles in a and b,
    // incremented in the bkidx loop. 
    // cPtr is the starting position of the tile in c which is fixed.

    var aPtr = cRow * {{BM}} * {{K}};
    var bPtr = cCol * {{BN}} * {{K}};
    let cPtr = cRow * {{BM}} * {{N4}} + cCol * {{BN4}};

    for (var bkidx = 0; bkidx < {{K}}; bkidx += {{BK}}) {

      // Load tile
      // Load BM x BK by numThread(BM * BN / (TM * TN))
      // The number of iteration == BM * BK / (BM * BN / (TM * TN))
      for (var idx: u32 = 0; idx < {{NUM_TILEA}}; idx++) {
        tileA[localID.x + idx * numThread] = a[aPtr + ((localID.x + idx * numThread) / {{BK}}) * {{K}} + (localID.x + idx * numThread) % {{BK}}];
      }
      // Load BK x BN by numThread(BM * BN / (TM * TN))
      // The number of iteration == BK * BN / (BM * BN / (TM * TN))
      for (var idx: u32 = 0; idx < {{NUM_TILEB}}; idx++) {
        tileB[localID.x + idx * numThread] = b[bPtr + ((localID.x + idx * numThread) / {{BK}}) * {{K}} + ((localID.x + idx * numThread) % {{BK}})];
      }

      aPtr += {{BK}};
      bPtr += {{BK}};

      workgroupBarrier();
      // Compute tile
      for (var dotIdx: u32 = 0; dotIdx < {{BK}}; dotIdx = dotIdx + 1) {
        for (var idx: u32 = 0; idx < {{TM}}; idx++) {
          localM[idx] = tileA[(threadRow + idx) * {{BK}} + dotIdx];
        }
        for (var idx: u32 = 0; idx < {{TN4}}; idx++) {
          localN[idx] = vec4<{{precision}}>(tileB[(threadCol + idx*4    ) * {{BK}} + dotIdx],
                                            tileB[(threadCol + idx*4 + 1) * {{BK}} + dotIdx],
                                            tileB[(threadCol + idx*4 + 2) * {{BK}} + dotIdx],
                                            tileB[(threadCol + idx*4 + 3) * {{BK}} + dotIdx]);
        }
        for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
          for (var resIdxN: u32 = 0; resIdxN < {{TN4}}; resIdxN++) {
            threadResults[resIdxM * {{TN4}} + resIdxN] += localM[resIdxM] * localN[resIdxN];
          }
        }
      }
      workgroupBarrier();
    }

    for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
      for (var resIdxN: u32 = 0; resIdxN < {{TN4}}; resIdxN++) {
        c[cPtr + (threadRow + resIdxM) * {{N4}} + (threadCol/4) + resIdxN] = threadResults[resIdxM * {{TN4}} + resIdxN];
      }
    }
}
)";

inline KernelCode createMatmulWithVectorization(const char *shaderTemplate, const size_t M,
                                                const size_t K, const size_t N, const size_t BM,
                                                const size_t BK, const size_t BN,
                                                const size_t TM, const size_t TN,
                                                const Shape &workgroupSize = {256, 1, 1},
                                                NumType precision = kf32,
                                                bool unrolling = false) {
  assert(BM % TM == 0);
  assert(BN % TN == 0);
  assert(K % BK == 0);
  assert(M % BM == 0);
  assert(N % BN == 0);
  // # threads = tile A size == tile B size == # threads for computing C
  int num_threads = BM * BN / (TM * TN);
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
                          {"{{TN}}", toString(TN)},
                          {"{{NUM_TILEA}}", toString(BM * BK / num_threads)},
                          {"{{NUM_TILEB}}", toString(BN * BK / num_threads)},
                          {"{{TN4}}", toString(TN / 4)},
                          {"{{N4}}", toString(N / 4)},
                          {"{{BN4}}", toString(BN / 4)},
                          });
  if (unrolling) {
    std::string unrolledCode = loopUnrolling(codeString);
    // LOG(kDefLog, kInfo, "Unrolled code:\n%s", unrolledCode.c_str());
    return {unrolledCode, workgroupSize, precision};
  } else {
    return {codeString, workgroupSize, precision};
  }
}

/* 2D block-tiling with transpose
 *
 */
static const char *kShaderMatmulWithTranspose = R"(
@group(0) @binding(0) var<storage, read_write> a: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> b: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> c: array<vec4<{{precision}}>>;
var<workgroup> tileA: array<{{precision}}, {{BM}} * {{BK}}>;
var<workgroup> tileB: array<{{precision}}, {{BK}} * {{BN}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>) {

    var threadResults: array<vec4<{{precision}}>, {{TM}} * {{TN4}}>;
    var localM: array<{{precision}}, {{TM}}>;
    var localN: array<vec4<{{precision}}>, {{TN4}}>;

    let cRow: u32 = groupid.x;
    let cCol: u32 = groupid.y;
    let numThread: u32 = ({{BM}} * {{BN}}) / ({{TM}} * {{TN}});

    // position of the first c element computed by the thread
    let threadRow: u32 = (localID.x / ({{BN}} / {{TN}})) * {{TM}};
    let threadCol: u32 = (localID.x % ({{BN}} / {{TN}})) * {{TN}};

    // aPtr and bPtr are the starting positions of the tiles in a and b,
    // incremented in the bkidx loop. 
    // cPtr is the starting position of the tile in c which is fixed.

    var aPtr: u32 = cRow * {{BM}} * {{K}};
    var bPtr: u32 = cCol * {{BN}};
    let cPtr: u32 = cRow * {{BM}} * {{N4}} + cCol * {{BN4}};

    for (var bkidx = 0; bkidx < {{K}}; bkidx += {{BK}}) {

      // Load tile
      // Load BM x BK by numThread(BM * BN / (TM * TN))
      // The number of iteration == BM * BK / (BM * BN / (TM * TN))
      for (var idx: u32 = 0; idx < {{NUM_TILEA}}; idx++) {
        tileA[localID.x + idx * numThread] = a[aPtr + ((localID.x + idx * numThread) / {{BK}}) * {{K}} + (localID.x + idx * numThread) % {{BK}}];
      }
      // Load BK x BN by numThread(BM * BN / (TM * TN))
      // The number of iteration == BK * BN / (BM * BN / (TM * TN))
      for (var idx: u32 = 0; idx < {{NUM_TILEB}}; idx++) {
        tileB[localID.x + idx * numThread] = b[bPtr + ((localID.x + idx * numThread) / {{BN}}) * {{N}} + ((localID.x + idx * numThread) % {{BN}})];
      }

      aPtr += {{BK}};
      bPtr += {{BK}} * {{N}};

      workgroupBarrier();
      // Compute tile
      for (var dotIdx: u32 = 0; dotIdx < {{BK}}; dotIdx = dotIdx + 1) {
        for (var idx: u32 = 0; idx < {{TM}}; idx++) {
          localM[idx] = tileA[(threadRow + idx) * {{BK}} + dotIdx];
        }
        for (var idx: u32 = 0; idx < {{TN4}}; idx++) {
          localN[idx] = vec4<{{precision}}>(tileB[(threadCol + idx*4    ) + dotIdx * {{BN}}],
                                            tileB[(threadCol + idx*4 + 1) + dotIdx * {{BN}}],
                                            tileB[(threadCol + idx*4 + 2) + dotIdx * {{BN}}],
                                            tileB[(threadCol + idx*4 + 3) + dotIdx * {{BN}}]);
        }
        for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
          for (var resIdxN: u32 = 0; resIdxN < {{TN4}}; resIdxN++) {
            threadResults[resIdxM * {{TN4}} + resIdxN] += localM[resIdxM] * localN[resIdxN];
          }
        }
      }
      workgroupBarrier();
    }

    for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
      for (var resIdxN: u32 = 0; resIdxN < {{TN4}}; resIdxN++) {
        c[cPtr + (threadRow + resIdxM) * {{N4}} + (threadCol/4) + resIdxN] = threadResults[resIdxM * {{TN4}} + resIdxN];
      }
    }
}
)";

inline KernelCode createMatmulWithTranspose(const char *shaderTemplate, const size_t M,
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
  int num_threads = BM * BN / (TM * TN);
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
                          {"{{TN}}", toString(TN)},
                          {"{{NUM_TILEA}}", toString(BM * BK / num_threads)},
                          {"{{NUM_TILEB}}", toString(BN * BK / num_threads)},
                          {"{{TN4}}", toString(TN / 4)},
                          {"{{N4}}", toString(N / 4)},
                          {"{{BN4}}", toString(BN / 4)},
                          });
  std::string unrolledCode = loopUnrolling(codeString);
  // LOG(kDefLog, kInfo, "Unrolled code:\n%s", unrolledCode.c_str());
  return {unrolledCode, workgroupSize, precision};
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

inline KernelCode createNoOp(const char *shaderTemplate,
                             const Shape &workgroupSize = {256, 1, 1},
                             NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)}});
  return {codeString, workgroupSize, precision};
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

void initData(size_t M, size_t K, size_t N, std::unique_ptr<half[]> &inputPtr,
              std::unique_ptr<half[]> &weightsPtr) {
  std::mt19937 gen(314159);
  randn(inputPtr.get(), M * K, gen);
  randn(weightsPtr.get(), N * K, gen);
  // randint(inputPtr.get(), M * K, gen, 1, 2);
  // randint(weightsPtr.get(), N * K, gen, 1, 2);
  LOG(kDefLog, kInfo, "%s", show<half>(inputPtr.get(), M, K, "Input").c_str());
  LOG(kDefLog, kInfo, "%s",
      show<half>(weightsPtr.get(), N, K, "Weights").c_str());
}

template<class precision=float>
void checkCPU(size_t M, size_t K, size_t N, std::unique_ptr<precision[]> &inputPtr,
              std::unique_ptr<precision[]> &weightsPtr,
              std::unique_ptr<precision[]> &outputPtr) {
  LOG(kDefLog, kInfo, "Computing CPU reference implementation");
  std::unique_ptr<precision[]> outputRefPtr = std::make_unique<precision[]>(M * N);
  if constexpr (std::is_same<precision, float>::value) {
    ref::matmul_forward_cpu(outputRefPtr.get(), inputPtr.get(), weightsPtr.get(),
			    nullptr, 1, M, K, N);
  } else if constexpr (std::is_same<precision, half>::value) {
    matmulf16_forward_cpu(outputRefPtr.get(), inputPtr.get(), weightsPtr.get(),
			  nullptr, 1, M, K, N);
  }
  LOG(kDefLog, kInfo, "Reference Output: %s",
      show<precision>(outputRefPtr.get(), M, N, "Output (Reference)").c_str());
  LOG(kDefLog, kInfo,
      isclose(outputPtr.get(), outputRefPtr.get(), M * N) ? "CPU Check: PASS"
                                                          : "CPU Check: FAIL");
}

Kernel selectMatmul(Context &ctx, int version,
                    const Bindings</* input, weights, output */ 3> &bindings,
                    size_t M, size_t K, size_t N, NumType numtype) {
  Kernel kernel;
  if (version == 1) {
    Shape wgSize = {256, 1, 1};
    Shape nWorkgroups = cdiv({M, N, 1}, {16, 16, 1});
    KernelCode matmul = createNoOp(kShaderNoOp, /*wgsize*/ wgSize);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  } else if (version == 2) {
    Shape wgSize = {16, 16, 1};
    LOG(kDefLog, kInfo, "wgSize: %s", toString(wgSize).c_str());
    KernelCode matmul =
      createMatmul1(kShaderMatmul1, M, K, N, /*wgsize*/ wgSize, numtype);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ cdiv({M, N, 1}, wgSize));
  } else if (version == 3) {
    static constexpr size_t tileSize = 16;
    KernelCode matmul = createMatmul2(kShaderMatmul2, M, K, N,
                                      /*wgSize*/ {tileSize * tileSize, 1, 1}, numtype);
    kernel =
        createKernel(ctx, matmul, bindings,
                     /* nWorkgroups*/ cdiv({M, N, 1}, {tileSize, tileSize, 1}));
  } else if (version == 4 || version == 6) {
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
    KernelCode matmul = createMatmul3(kShaderMatmul3, M, K, N, BM, BK, BN, TM,
                                      /*wgSize*/ wgSize,
				      numtype,
				      /*Loop unrolling*/ version == 6 ? true: false);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  } else if (version == 5 || version == 7) {
    static constexpr size_t BM = 64;
    static constexpr size_t BK = 8;
    static constexpr size_t BN = 64;
    static constexpr size_t TM = BM / BK;
    static constexpr size_t TN = BN / BK;
    Shape wgSize = {(BM / TM) * (BN / TN), 1, 1}; // This is the same as BK * BK.
    Shape nWorkgroups = {cdiv(M, BM), cdiv(N, BN), 1};
    LOG(kDefLog, kInfo, "M: %d, K: %d, N: %d", M, K, N);
    LOG(kDefLog, kInfo, "BM: %d, BK: %d, BN: %d, TM: %d, TN: %d", BM, BK, BN, TM, TN);
    LOG(kDefLog, kInfo, "wgSize: ( %s )", toString(wgSize).c_str());
    LOG(kDefLog, kInfo, "nWorkgroups: ( %s )", toString(nWorkgroups).c_str());
    KernelCode matmul = createMatmul4(kShaderMatmul4, M, K, N, BM, BK, BN, TM, TN,
                                      /*wgSize*/ wgSize,
				      numtype,
				      /*Loop unrolling*/ version == 7 ? true: false);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  } else if (version == 8 || version == 10) {
    static constexpr size_t BM = 64;
    static constexpr size_t BK = 8;
    static constexpr size_t BN = 64;
    static constexpr size_t TM = BM / BK;
    static constexpr size_t TN = BN / BK;
    Shape wgSize = {(BM / TM) * (BN / TN), 1, 1}; // This is the same as BK * BK.
    Shape nWorkgroups = {cdiv(M, BM), cdiv(N, BN), 1};
    LOG(kDefLog, kInfo, "M: %d, K: %d, N: %d", M, K, N);
    LOG(kDefLog, kInfo, "BM: %d, BK: %d, BN: %d, TM: %d, TN: %d", BM, BK, BN, TM, TN);
    LOG(kDefLog, kInfo, "wgSize: ( %s )", toString(wgSize).c_str());
    LOG(kDefLog, kInfo, "nWorkgroups: ( %s )", toString(nWorkgroups).c_str());
    KernelCode matmul = createMatmulWithVectorization(kShaderMatmulWithVectorization, M, K, N, BM, BK, BN, TM, TN,
                                                      /*wgSize*/ wgSize,
						      numtype,
						      /*Loop unrolling*/ true);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  } else if (version == 9 || version == 11) {
    static constexpr size_t BM = 64;
    static constexpr size_t BK = 8;
    static constexpr size_t BN = 64;
    static constexpr size_t TM = BM / BK;
    static constexpr size_t TN = BN / BK;
    Shape wgSize = {(BM / TM) * (BN / TN), 1, 1}; // This is the same as BK * BK.
    Shape nWorkgroups = {cdiv(M, BM), cdiv(N, BN), 1};
    LOG(kDefLog, kInfo, "M: %d, K: %d, N: %d", M, K, N);
    LOG(kDefLog, kInfo, "BM: %d, BK: %d, BN: %d, TM: %d, TN: %d", BM, BK, BN, TM, TN);
    LOG(kDefLog, kInfo, "wgSize: ( %s )", toString(wgSize).c_str());
    LOG(kDefLog, kInfo, "nWorkgroups: ( %s )", toString(nWorkgroups).c_str());
    KernelCode matmul = createMatmulWithTranspose(kShaderMatmulWithTranspose, M, K, N, BM, BK, BN, TM, TN,
						  /*wgSize*/ wgSize,
						  numtype);
    kernel = createKernel(ctx, matmul, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  }
  return kernel;
}

template<class precision=float>
void runTest(int version, size_t M, size_t K, size_t N,
             std::unique_ptr<precision[]> &inputPtr,
             std::unique_ptr<precision[]> &weightsPtr,
             std::unique_ptr<precision[]> &outputPtr,
	     NumType numtype) {
  if constexpr (std::is_same<precision, float>::value) {
    assert(numtype == kf32);
  } else if constexpr (std::is_same<precision, half>::value) {
    assert(numtype == kf16);
  }

  // Allocate GPU buffers and copy data
  WGPUDeviceDescriptor devDescriptor = {};
  devDescriptor.requiredFeatureCount = 1;
  devDescriptor.requiredFeatures = std::array{WGPUFeatureName_ShaderF16}.data();

  Context ctx;
  if (numtype == kf16) {
    ctx = createContext(
        {}, {},
        /*device descriptor, enabling f16 in WGSL*/
        {
          .requiredFeatureCount = 1,
          .requiredFeatures = std::array{WGPUFeatureName_ShaderF16}.data()
        });
    if (ctx.adapterStatus != WGPURequestAdapterStatus_Success) {
      LOG(kDefLog, kError, "Failed to create adapter with f16 support, try running an f32 test instead (`export MATMUL_VERSION=9).");
      exit(1);
    }
    if (ctx.deviceStatus != WGPURequestDeviceStatus_Success) {
      LOG(kDefLog, kError, "Failed to create device with f16 support, try running an f32 test instead. (`export MATMUL_VERSION=9)");
      exit(1);
    }
  }

  if (numtype == kf32) {
    ctx = createContext({}, {}, {});
    if (ctx.adapterStatus != WGPURequestAdapterStatus_Success ||
        ctx.deviceStatus != WGPURequestDeviceStatus_Success) {
      LOG(kDefLog, kError, "Failed to create adapter or device");
      // stop execution
      exit(1);
    } else {
      LOG(kDefLog, kInfo, "Successfully created adapter and device");
    }
  } 

  Tensor input = createTensor(ctx, Shape{M, K}, numtype, inputPtr.get());
  Tensor weights = createTensor(ctx, Shape{N, K}, numtype, weightsPtr.get()); // column-major

#ifdef METAL_PROFILER
  constexpr size_t nIter = 1;
#else
  constexpr size_t nIter = 30;
#endif

  // Initialize Kernel and bind GPU buffers
  // pre-allocate for async dispatch
  std::array<std::promise<void>, nIter> promises;
  std::array<std::future<void>, nIter> futures;
  std::array<Kernel, nIter> kernels;
  std::array<Tensor, nIter> outputs;
  for (int i = 0; i < nIter; i++) {
    futures[i] = promises[i].get_future();
    outputs[i] = createTensor(ctx, Shape{M, N}, numtype);
    kernels[i] = selectMatmul(ctx, version, {input, weights, outputs[i]}, M, K, N, numtype);
  }

  LOG(kDefLog, kInfo, "Dispatching Kernel version %d: %s, %d iterations ...",
      version, versionToStr(version).c_str(), nIter);

  // Dispatch kernel nIter times
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < nIter; i++) {
    dispatchKernel(ctx, kernels[i], promises[i]);
  }
  for (int i = 0; i < nIter; i++) {
    wait(ctx, futures[i]);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Report performance.
  // Use microsecond for more accurate time measurement
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  float gflops = 2 * M * N *
                 K / // factor of 2 for multiplication & accumulation
                 (static_cast<double>(duration.count()) / 1000000.0) /
                 1000000000.0 * static_cast<float>(nIter);

  LOG(kDefLog, kInfo, "Copying result to CPU");
  toCPU(ctx, outputs[0], outputPtr.get(), M * N * sizeof(precision));
  LOG(kDefLog, kInfo, "%s",
      show<precision>(outputPtr.get(), M, N, "Output[0]").c_str());

  LOG(kDefLog, kInfo, "\n\n===================================================================="
      "============\nExecution Time: (M = %d, K = %d, N = %d) x %d iterations "
      ":\n%.1f "
      "milliseconds / dispatch ~ %.2f "
      "GFLOPS\n================================================================"
      "================\n\n",
      M, K, N, nIter, duration.count() / static_cast<double>(nIter) / 1000.0 /* us -> ms */, gflops);
}

template<class precision=float>
void runTestWithCheck(int version, size_t M, size_t K, size_t N,
                      bool transposedInput, int kTestSize, NumType numtype) {
    std::unique_ptr<precision[]> inputPtr = std::make_unique<precision[]>(M * K);
    std::unique_ptr<precision[]> weightsPtr = std::make_unique<precision[]>(N * K);
    std::unique_ptr<precision[]> outputPtr = std::make_unique<precision[]>(M * N);

    initData(M, K, N, inputPtr, weightsPtr);
    if (transposedInput) {
      std::unique_ptr<precision[]> transposedWeightPtr = std::make_unique<precision[]>(K * N);
      transpose(weightsPtr.get(), transposedWeightPtr.get(), N, K);
      runTest(version, M, K, N, inputPtr, transposedWeightPtr, outputPtr, numtype);
    } else {
      runTest(version, M, K, N, inputPtr, weightsPtr, outputPtr, numtype);
    }

    if (kTestSize <= 1) {
      // Check result with CPU reference implementation for tiny/small tests
      checkCPU(M, K, N, inputPtr, weightsPtr, outputPtr);
    }
}

const std::string versionToStr(int version){
  switch (version) {
  case 1: return  "f32: No-Op";
  case 2: return  "f32: naive matmul";
  case 3: return  "f32: tiling";
  case 4: return  "f32: 1D blocktiling";
  case 5: return  "f32: 2D blocktiling";
  case 6: return  "f32: 1D blocktiling with loop unrolling";
  case 7: return  "f32: 2D blocktiling with loop unrolling";
  case 8: return  "f32: 2D blocktiling with loop unrolling and vectorization";
  case 9: return  "f32: 2D blocktiling with loop unrolling, vectorization and transpose";
  case 10: return "f16: 2D blocktiling with loop unrolling and vectorization";
  case 11: return "f16: 2D blocktiling with loop unrolling, vectorization and transpose";
  default: return "Not specified";
  }
}

int main() {
  char* version_str = getenv("MATMUL_VERSION");
  int version = version_str == NULL ? 10 : atoi(version_str);
    // 1 == f32: No-Op
    // 2 == f32: naive matmul
    // 3 == f32: tiling
    // 4 == f32: 1D blocktiling
    // 5 == f32: 2D blocktiling
    // 6 == f32: 1D blocktiling with loop unrolling
    // 7 == f32: 2D blocktiling with loop unrolling
    // 8 == f32: 2D blocktiling with loop unrolling and vectorization
    // 9 == f32: 2D blocktiling with loop unrolling, vectorization and transpose
    // 10 == f16: 2D blocktiling with loop unrolling and vectorization (default)
    // 11 == f16: 2D blocktiling with loop unrolling, vectorization and transpose
  bool enableF16 = version == 10 || version ==11;
  bool transposedInput = version == 9 || version == 11;
  NumType numtype = enableF16 ? kf16 : kf32;

  size_t M, K, N;  // Matrix dimensions
  char* kTestSize_str = getenv("MATMUL_SIZE");
  int kTestSize = kTestSize_str == NULL ? 2 : atoi(kTestSize_str);
  if (kTestSize == 0) {
    // Tiny test
    M = 32;
    K = 32;
    N = 32;
  } else if (kTestSize == 1) {
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

#ifdef METAL_PROFILER
  startCapture();
#endif
  if (enableF16) {
    runTestWithCheck<half>(version, M, K, N, transposedInput, kTestSize, numtype);
  } else {
    runTestWithCheck<float>(version, M, K, N, transposedInput, kTestSize, numtype);
  }
#ifdef METAL_PROFILER
  stopCapture();
#endif

  LOG(kDefLog, kInfo, "Done.");
  return 0;
}
