#include "gpu.h"
#include <array>
#include <cstdio>
#include <future>

#include "kernels.h"
#include "kernels_c.h"

using namespace gpu; // createContext, createTensor, createKernel,
                     // createShader, dispatchKernel, wait, toCPU
                     // Tensor, Kernel, Context, Shape, kf32

void gelu_forward(float* out, float* inp, int n) {
  unsigned long N = static_cast<unsigned long>(n);
  setLogLevel(kError);
  Context ctx = createContext();
  Tensor input = createTensor(ctx, Shape{N}, kf32, inp);
  Tensor output = createTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kShaderGelu, 256, kf32},
                           Bindings{input, output},
                           /* nWorkgroups */ {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, out, N * sizeof(float));
}

void softmax_forward(float* probs, float* logits, int b, int t, int v, int vp) {
  struct SoftmaxParam {
    uint32_t N;
    uint32_t C;
  };
  uint32_t B = static_cast<uint32_t>(b);
  uint32_t T = static_cast<uint32_t>(t);
  uint32_t C = static_cast<uint32_t>(v);
  Context ctx = createContext();
  Tensor input = createTensor(ctx, {B * T, C}, kf32, logits);
  Tensor output = createTensor(ctx, {B * T, C}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(
      ctx, {kShaderSoftmax1, 256, kf32}, Bindings{input, output},
      Shape{cdiv(B * T, 256), 1, 1}, SoftmaxParam{B * T, C});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, probs, sizeof(float)*B*T*C);
}

// static const char *kShaderMatmul = R"(
// @group(0) @binding(0) var<storage, read_write> a: array<{{precision}}>;
// @group(0) @binding(1) var<storage, read_write> b: array<{{precision}}>;
// @group(0) @binding(2) var<storage, read_write> c: array<vec4<{{precision}}>>;
// var<workgroup> tileA: array<{{precision}}, {{BM}} * {{BK}}>;
// var<workgroup> tileB: array<{{precision}}, {{BN}} * {{BK}}>;

// @compute @workgroup_size({{workgroupSize}})
// fn main(
//     @builtin(global_invocation_id) globalID : vec3<u32>,
//     @builtin(local_invocation_id) localID : vec3<u32>,
//     @builtin(workgroup_id) groupid : vec3<u32>) {

//     var threadResults: array<vec4<{{precision}}>, {{TM}} * {{TN4}}>;
//     var localM: array<{{precision}}, {{TM}}>;
//     var localN: array<vec4<{{precision}}>, {{TN4}}>;

//     let cRow: u32 = groupid.x;
//     let cCol: u32 = groupid.y;
//     let numThread: u32 = ({{BM}} * {{BN}}) / ({{TM}} * {{TN}});

//     // position of the first c element computed by the thread
//     let threadRow: u32 = (localID.x / ({{BN}} / {{TN}})) * {{TM}};
//     let threadCol: u32 = (localID.x % ({{BN}} / {{TN}})) * {{TN}};

//     // aPtr and bPtr are the starting positions of the tiles in a and b,
//     // incremented in the bkidx loop. 
//     // cPtr is the starting position of the tile in c which is fixed.

//     var aPtr = cRow * {{BM}} * {{K}};
//     var bPtr = cCol * {{BN}} * {{K}};
//     let cPtr = cRow * {{BM}} * {{N4}} + cCol * {{BN4}};

//     for (var bkidx = 0; bkidx < {{K}}; bkidx += {{BK}}) {

//       // Load tile
//       // Load BM x BK by numThread(BM * BN / (TM * TN))
//       // The number of iteration == BM * BK / (BM * BN / (TM * TN))
//       for (var idx: u32 = 0; idx < {{NUM_TILEA}}; idx++) {
//         tileA[localID.x + idx * numThread] = a[aPtr + ((localID.x + idx * numThread) / {{BK}}) * {{K}} + (localID.x + idx * numThread) % {{BK}}];
//       }
//       // Load BK x BN by numThread(BM * BN / (TM * TN))
//       // The number of iteration == BK * BN / (BM * BN / (TM * TN))
//       for (var idx: u32 = 0; idx < {{NUM_TILEB}}; idx++) {
//         tileB[localID.x + idx * numThread] = b[bPtr + ((localID.x + idx * numThread) / {{BK}}) * {{K}} + ((localID.x + idx * numThread) % {{BK}})];
//       }

//       aPtr += {{BK}};
//       bPtr += {{BK}};

//       workgroupBarrier();
//       // Compute tile
//       for (var dotIdx: u32 = 0; dotIdx < {{BK}}; dotIdx = dotIdx + 1) {
//         for (var idx: u32 = 0; idx < {{TM}}; idx++) {
//           localM[idx] = tileA[(threadRow + idx) * {{BK}} + dotIdx];
//         }
//         for (var idx: u32 = 0; idx < {{TN4}}; idx++) {
//           localN[idx] = vec4<{{precision}}>(tileB[(threadCol + idx*4    ) * {{BK}} + dotIdx],
//                                             tileB[(threadCol + idx*4 + 1) * {{BK}} + dotIdx],
//                                             tileB[(threadCol + idx*4 + 2) * {{BK}} + dotIdx],
//                                             tileB[(threadCol + idx*4 + 3) * {{BK}} + dotIdx]);
//         }
//         for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
//           for (var resIdxN: u32 = 0; resIdxN < {{TN4}}; resIdxN++) {
//             threadResults[resIdxM * {{TN4}} + resIdxN] += localM[resIdxM] * localN[resIdxN];
//           }
//         }
//       }
//       workgroupBarrier();
//     }

//     for (var resIdxM: u32 = 0; resIdxM < {{TM}}; resIdxM++) {
//       for (var resIdxN: u32 = 0; resIdxN < {{TN4}}; resIdxN++) {
//         c[cPtr + (threadRow + resIdxM) * {{N4}} + (threadCol/4) + resIdxN] = threadResults[resIdxM * {{TN4}} + resIdxN];
//       }
//     }
// }
// )";

// inline KernelCode createMatmul(const char *shaderTemplate, const size_t M,
// 			       const size_t K, const size_t N, const size_t BM,
// 			       const size_t BK, const size_t BN,
// 			       const size_t TM, const size_t TN,
// 			       const Shape &workgroupSize = {256, 1, 1},
// 			       NumType precision = kf32) {
//   assert(BM % TM == 0);
//   assert(BN % TN == 0);
//   assert(K % BK == 0);
//   assert(M % BM == 0);
//   assert(N % BN == 0);
//   // # threads = tile A size == tile B size == # threads for computing C
//   int num_threads = BM * BN / (TM * TN);
//   std::string codeString(shaderTemplate);
//   replaceAll(codeString, {{"{{workgroupSize}}", toString(workgroupSize)},
//                           {"{{precision}}", toString(precision)},
//                           {"{{M}}", toString(M)},
//                           {"{{K}}", toString(K)},
//                           {"{{N}}", toString(N)},
//                           {"{{BM}}", toString(BM)},
//                           {"{{BK}}", toString(BK)},
//                           {"{{BN}}", toString(BN)},
//                           {"{{TM}}", toString(TM)},
//                           {"{{TN}}", toString(TN)},
//                           {"{{NUM_TILEA}}", toString(BM * BK / num_threads)},
//                           {"{{NUM_TILEB}}", toString(BN * BK / num_threads)},
//                           {"{{TN4}}", toString(TN / 4)},
//                           {"{{N4}}", toString(N / 4)},
//                           {"{{BN4}}", toString(BN / 4)},
//                           });
//   std::string unrolledCode = loopUnrolling(codeString);
//   return {unrolledCode, workgroupSize, precision};
// }

// void matmul_forward_gpu(float* out,
// 			const float* inp, const float* weight, const float* bias,
// 			int B, int T, int C, int OC) {
//   static constexpr size_t BM = 128;
//   static constexpr size_t BK = 16;
//   static constexpr size_t BN = 64;
//   static constexpr size_t TM = 4;
//   static constexpr size_t TN = 8;
//   static constexpr Shape wgSize = {(BM / TM) * (BN / TN), 1, 1}; // This is the same as BK * BK.
//   static constexpr Shape nWorkgroups = {cdiv(M, BM), cdiv(N, BN), 1};
//   static constexpr KernelCode matmul = createMatmul(kShaderMatmul, M, K, N, BM, BK, BN, TM, TN,
// 						    /*wgSize*/ wgSize,
// 						    kf32);
//   setLogLevel(kError);
//   Context ctx = createContext();

//   Tensor input = createTensor(ctx, Shape{M, K}, kf32, inp);
//   Tensor weights = createTensor(ctx, Shape{N, K}, kf32, weight); // column-major
//   Tensor output = createTensor(ctx, Shape{M, N}, kf32); // column-major
  
//   std::promise<void> promise;
//   std::future<void> future = promise.get_future();
//   Kernel op = createKernel(ctx, matmul, bindings, nWorkgroups);

//   dispatchKernel(ctx, op, promise);
//   wait(ctx, future);
//   toCPU(ctx, output, out, N * sizeof(float));

// }
