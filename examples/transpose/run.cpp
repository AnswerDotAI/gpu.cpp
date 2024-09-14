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

using namespace gpu;

// This implements the tranpose kernels in https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc .

static const char *kShaderTranspose1 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>) {
    let bRow: u32 = globalID.x;
    let bCol: u32 = globalID.y;
    B[bRow * {{M}} + bCol] = A[bCol * {{N}} + bRow];
}
)";

inline KernelCode createTranspose1(const char *shaderTemplate,
				   const size_t M, const size_t N,
                                   const Shape &workgroupSize = {256, 1, 1},
                                   NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)},
                          {"{{M}}", toString(M)},
                          {"{{N}}", toString(N)}});
  return {codeString, workgroupSize};
}

// Shared memory cache-blocking
static const char *kShaderTranspose2 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
var<workgroup> tile: array<{{precision}}, {{BN}} * {{BM}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) localID : vec3<u32>,
  @builtin(workgroup_id) groupID: vec3<u32>) {
    let bRow: u32 = groupID.x * {{BN}};
    let bCol: u32 = groupID.y * {{BM}};

    let aPtr = bCol * {{N}} + bRow;
    let bPtr = bRow * {{M}} + bCol;
    let numThread: u32 = ({{BM}} * {{BN}}) / ({{TM}} * {{TN}});

    for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
      for (var resIdxN: u32 = 0; resIdxN < {{TN}}; resIdxN++) {
        let idx: u32 = localID.x + numThread * (resIdxN + {{TN}} * resIdxM);
        let loadRow: u32 = idx / {{BN}};
        let loadCol: u32 = idx % {{BN}};
        tile[loadCol * {{BN}} + loadRow] = A[aPtr + loadRow * {{N}} + loadCol];
      }
    }

    workgroupBarrier();

    for (var resIdxN: u32 = 0; resIdxN < {{TN}}; resIdxN++) {
      for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
        let idx: u32 = localID.x + numThread * (resIdxM + {{TM}} * resIdxN);
        let loadRow: u32 = idx / {{BM}};
        let loadCol: u32 = idx % {{BM}};
        B[bPtr + loadRow * {{M}} + loadCol] = tile[loadRow * {{BM}} + loadCol];
      }
    }
}
)";

inline KernelCode createTranspose2(const char *shaderTemplate,
				   const size_t M, const size_t N,
				   const size_t BM, const size_t BN,
                                   const size_t TM, const size_t TN,
                                   const Shape &workgroupSize = {256, 1, 1},
                                   NumType precision = kf32) {
  assert(BM % TM == 0);
  assert(BN % TN == 0);
  assert(M % BM == 0);
  assert(N % BN == 0);
  std::string codeString(shaderTemplate);
  replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
                          {"{{precision}}", toString(precision)},
                          {"{{M}}", toString(M)},
                          {"{{N}}", toString(N)},
                          {"{{BM}}", toString(BM)},
                          {"{{BN}}", toString(BN)},
                          {"{{TM}}", toString(TM)},
                          {"{{TN}}", toString(TN)}
                          });
  std::string unrolledCode = codeString ;// loopUnrolling(codeString);
  return {unrolledCode, workgroupSize};
}

void initData(size_t M, size_t N, std::unique_ptr<float[]> &inputPtr) {
  std::mt19937 gen(314159);
  randn(inputPtr.get(), M * N, gen);
  LOG(kDefLog, kInfo, "%s", show<float>(inputPtr.get(), M, N, "Input").c_str());
}

Kernel selectTranspose(Context &ctx, int version,
                       const Bindings</* input, output */ 2> &bindings,
                       size_t M, size_t N) {
  Kernel kernel;
  if (version == 1) {
    Shape wgSize = {16, 16, 1};
    LOG(kDefLog, kInfo, "wgSize: %s", toString(wgSize).c_str());
    KernelCode transpose =
        createTranspose1(kShaderTranspose1, M, N, /*wgsize*/ wgSize); // The shape of input == M x N
    kernel = createKernel(ctx, transpose, bindings,
                          /*nWorkgroups*/ cdiv({N, M, 1}, wgSize)); // The shape of output == N x M
  } else if (version == 2) {
    static constexpr size_t BM = 64;
    static constexpr size_t BK = 16;
    static constexpr size_t BN = 64;
    static constexpr size_t TM = BM / BK;
    static constexpr size_t TN = BN / BK;
    Shape wgSize = {(BM / TM) * (BN / TN), 1, 1}; // This is the same as BK * BK.
    Shape nWorkgroups = {cdiv(N, BN), cdiv(M, BM), 1};
    LOG(kDefLog, kInfo, "M: %d, N: %d", M, N);
    LOG(kDefLog, kInfo, "BM: %d, BK: %d, BN: %d, TM: %d, TN: %d", BM, BK, BN, TM, TN);
    LOG(kDefLog, kInfo, "wgSize: ( %s )", toString(wgSize).c_str());
    LOG(kDefLog, kInfo, "nWorkgroups: ( %s )", toString(nWorkgroups).c_str());
    KernelCode transpose = createTranspose2(kShaderTranspose2, M, N, BM, BN, TM, TN,
					    /*wgSize*/ wgSize,
					    kf32);
    kernel = createKernel(ctx, transpose, bindings,
                          /*nWorkgroups*/ nWorkgroups);
  } else if (version == 0) {
    LOG(kDefLog, kInfo, "Skip Creating Kernel", M, N);
  }
  return kernel;
}

void runTest(int version, size_t M, size_t N,
             std::unique_ptr<float[]> &inputPtr,
             std::unique_ptr<float[]> &outputPtr) {
  bool isCPU = version == 0;
  
  // Allocate GPU buffers and copy data
  Context ctx = createContext();
  Tensor input = createTensor(ctx, Shape{M, N}, kf32, inputPtr.get());
  Tensor output = createTensor(ctx, Shape{N, M}, kf32);

  constexpr size_t nIter = 50;

  // Initialize Kernel and bind GPU buffers
  LOG(kDefLog, kInfo, "Creating Kernel");
  Kernel kernel = selectTranspose(ctx, version, {input, output}, M, N);

  // Dispatch kernel execution
  LOG(kDefLog, kInfo, "Dispatching Kernel version %d, %d iterations ...",
      version, nIter);

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
    if (!isCPU) {
      dispatchKernel(ctx, kernel, promises[i]);
      wait(ctx, futures[i]);
      resetCommandBuffer(ctx.device, kernel);
    } else {
      transpose(inputPtr.get(), outputPtr.get(), M, N);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Report performance.
  // Use microsecond for more accurate time measurement
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  float gbps = sizeof(float) * M * N /
               (static_cast<double>(duration.count()) / 1000000.0) /
               1000000000.0 * static_cast<float>(nIter);

  LOG(kDefLog, kInfo, "Copying result to CPU");
  if (!isCPU) {
    toCPU(ctx, output, outputPtr.get(), M * N * sizeof(float));
  }
  LOG(kDefLog, kInfo, "%s",
      show<float>(outputPtr.get(), N, M, "Output").c_str());

  LOG(kDefLog, kInfo, "\n\n===================================================================="
      "============\nExecution Time: (M = %d, N = %d) x %d iterations "
      ":\n%.3f "
      "milliseconds / dispatch ~ %.2f "
      "GB/s\n================================================================"
      "================\n\n",
      M, N, nIter, duration.count() / static_cast<double>(nIter) / 1000.0 /* us -> ms */, gbps);
}

int main() {
  char* version_str = getenv("TEST_VERSION");
  int version = version_str == NULL ? 2 : atoi(version_str);
    // 0 == cpu
    // 1 == naive transpose
    // 2 == tiling with shared memory

  size_t M, N;  // Matrix dimensions
  static constexpr int kTestSize = 2;
  if constexpr (kTestSize == 0) {
    // Tiny test
    M = 16;
    N = 32;
  } else if constexpr (kTestSize == 1) {
    // Small test
    M = 256;
    N = 512;
  } else {
    // Large test
    M = 4096;
    N = 2 * 4096;
  }

  std::unique_ptr<float[]> inputPtr = std::make_unique<float[]>(M * N);
  std::unique_ptr<float[]> outputPtr = std::make_unique<float[]>(N * M);

  initData(M, N, inputPtr);
  runTest(version, M, N, inputPtr, outputPtr);

  LOG(kDefLog, kInfo, "Done.");
  return 0;
}
