#include "gpu.hpp"
#include "utils/array_utils.hpp"
#include "utils/logging.hpp"
#include <array>

#include "llmc/reference_impls.h"

using namespace gpu;

static const char *kShaderGelu = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        // select is more stable for larger values of x
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                  * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

static const char *kMLPGate = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> gate: array<{{precision}}>;
@group(0) @binding(0) var<storage, read_write> gated: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&a)) {
        let x: f32 = gate[i];
        out[i] = gated[i] * select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                    * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

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

static const char *kShaderMatmul2 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
var<workgroup> tileA: array<f32, {{workgroupSizeX}} * {{workgroupSizeY}}>;
var<workgroup> tileB: array<f32, {{workgroupSizeX}} * {{workgroupSizeY}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(local_invocation_index) local_index : u32,
    @builtin(workgroup_id) group_id : vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let localRow = local_id.y;
    let localCol = local_id.x;
    if (row >= {{M}} || col >= {{N}}) {
        return;
    }
    var total: f32 = 0;
    for (var tileIndex = 0u; tileIndex < {{K}} / {{workgroupSizeX}}; tileIndex = tileIndex + 1u) {
      // TODO
    }
    C[row * {{N}} + col] = total;
}
)";

struct Transformer {
  Tensor qkv;         // modelDim * 3 * qkvDim
  Tensor rmsNormPre;  // modelDim
  Tensor rmsNormPost; // modelDim
  Tensor out;         // 3 * qkvDim * modelDim
  Tensor mlp1;        // modelDim * (2 * hidden_width * modelDim)
  Tensor mlp2;        // modelDim * (2 * hidden_width * modelDim)
};

struct Activations {
  Tensor qkv; // batchSize * 3 * nHeads * qkvDim
  Tensor qk;
  Tensor att;
};

struct KVCache {
  Tensor keyCache;
  Tensor valueCache;
};

void createTransformer(Context &ctx, size_t modelDim, size_t qkvDim,
                       size_t nHeads, size_t batchSize, size_t seqLen,
                       size_t hiddenWidth, Transformer &transformer,
                       Activations &activations, KVCache &kvCache) {
  std::mt19937 gen(314159);
  transformer = {
      .qkv = createTensor(ctx, Shape{3 * nHeads * qkvDim, modelDim},
                          kf32), // column-major
      .rmsNormPre = createTensor(ctx, Shape{modelDim}, kf32),
      .rmsNormPost = createTensor(ctx, Shape{modelDim}, kf32),
      .out = createTensor(ctx, Shape{3 * qkvDim, modelDim}, kf32),
      .mlp1 = createTensor(ctx, Shape{modelDim, 2 * hiddenWidth}, kf32),
      .mlp2 = createTensor(ctx, Shape{modelDim, 2 * hiddenWidth}, kf32),
  };

  // Initialize values
  std::unique_ptr<float[]> qkvInit(new float[modelDim * 3 * nHeads * qkvDim]);
  // randint(qkvInit.get(), size(transformer.qkv.shape), gen, -2, 2);
  range(qkvInit.get(), size(transformer.qkv.shape), 0.0);
  LOG(kDefLog, kInfo, "%s",
      show<float>(qkvInit.get(), transformer.qkv.shape[0],
                  transformer.qkv.shape[1], "QKV Weights")
          .c_str());
  toGPU(ctx, qkvInit.get(), transformer.qkv);

  activations = {
      .qkv = createTensor(ctx, Shape{batchSize * 3 * nHeads * qkvDim}, kf32),
      .qk = createTensor(ctx, Shape{batchSize * nHeads}, kf32),
      .att = createTensor(ctx, Shape{batchSize * nHeads}, kf32)};

  kvCache = {
      .keyCache = createTensor(ctx, Shape{seqLen, qkvDim}, kf32),
      .valueCache = createTensor(ctx, Shape{seqLen, qkvDim}, kf32),
  };
  std::unique_ptr<float[]> keyCacheInit(new float[seqLen * qkvDim]);
  std::unique_ptr<float[]> valueCacheInit(new float[seqLen * qkvDim]);
  range(keyCacheInit.get(), size(kvCache.keyCache.shape), 0.0);
  range(valueCacheInit.get(), size(kvCache.valueCache.shape), 0.0);
  toGPU(ctx, keyCacheInit.get(), kvCache.keyCache);
  toGPU(ctx, valueCacheInit.get(), kvCache.valueCache);
}

inline KernelCode createMatmul(const char *shaderTemplate, const size_t M,
                               const size_t K, const size_t N,
                               const Shape &workgroupSize = {256, 1, 1},
                               NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString, "{{workgroupSize}}", toString(workgroupSize));
  replaceAll(codeString, "{{workgroupSizeX}}",
             std::to_string(workgroupSize[0]));
  replaceAll(codeString, "{{workgroupSizeY}}",
             std::to_string(workgroupSize[1]));
  replaceAll(codeString, "{{workgroupSizeZ}}",
             std::to_string(workgroupSize[2]));
  replaceAll(codeString, "{{precision}}", toString(precision));
  replaceAll(codeString, "{{M}}", std::to_string(M));
  replaceAll(codeString, "{{K}}", std::to_string(K));
  replaceAll(codeString, "{{N}}", std::to_string(N));
  // LOG(kDefLog, kInfo, "Shader code:\n%s\n", codeString.c_str());
  return KernelCode{codeString, workgroupSize};
}

int main() {
  printf("\033[2J\033[1;1H");
  Context ctx = createContext();
  static constexpr size_t seqLen = 24;
  static constexpr size_t batchSize = 1;
  static constexpr size_t modelDim = 4; // 3072;
  static constexpr size_t hiddenWidth = modelDim * 2;
  static constexpr size_t qkvDim = 3; // 256;
  static constexpr size_t nHeads = 8;
  std::mt19937 gen(314);

  Transformer transformer;
  Activations activations;
  KVCache kvcache;
  LOG(kDefLog, kInfo, "Initializing transformer, allocating GPU buffers ...\n");
  createTransformer(ctx, modelDim, qkvDim, nHeads, batchSize, seqLen,
                    hiddenWidth, transformer, activations, kvcache);

  std::array<float, modelDim> inputArr;
  randint(inputArr, gen, -2, 2);
  LOG(kDefLog, kInfo, "%s",
      show<float>(inputArr.data(), 1, modelDim, "Input").c_str());
  Tensor input = createTensor(ctx, Shape{modelDim}, kf32, inputArr.data());

  /* QKV Projection */

  LOG(kDefLog, kInfo, "QKV Projection");
  {
    KernelCode matmul = createMatmul(kShaderMatmul1, /*M*/ batchSize,
                                     /*K*/ modelDim, /*N*/ 3 * qkvDim);
    Kernel qkv = createKernel(
        ctx, matmul, Bindings{input, transformer.qkv, activations.qkv},
        /*nthreads*/ {modelDim, 1, 1});
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    dispatchKernel(ctx, qkv, promise);
    wait(ctx, future);
    std::array<float, 3 * qkvDim> outputArr;
    toCPU(ctx, activations.qkv, outputArr.data(), sizeof(outputArr));
    LOG(kDefLog, kInfo, "Output: %s",
        show<float>(outputArr.data(), 1, 3 * qkvDim, "QKV Output").c_str());
    std::array<float, 3 * qkvDim> outputRefArr;
    std::array<float, modelDim * 3 * qkvDim> weightsArr;
    toCPU(ctx, transformer.qkv, weightsArr.data(), sizeof(weightsArr));
    ref::matmul_forward_cpu(
        outputRefArr.data(), inputArr.data(), weightsArr.data(), nullptr,
        /* batch */ 1, /* T */ 1, /* C */ modelDim, /* OC */ 3 * qkvDim);
    LOG(kDefLog, kInfo, "Reference Output: %s",
        show<float>(outputRefArr.data(), 1, 3 * qkvDim,
                    "QKV Output (Reference)")
            .c_str());
    LOG(kDefLog, kInfo,
        isclose(outputArr.data(), outputRefArr.data(), 3 * qkvDim) ? "PASS"
                                                                   : "FAIL");
  }

  /* QK Dot Products */

  LOG(kDefLog, kInfo, "QK Dot Product");
  {
    KernelCode dot = createMatmul(kShaderMatmul1, /*M*/ batchSize * nHeads,
                                  /*K*/ qkvDim, /*M*/ 1);

    /*
    // TODO(avh): need to pass in activation views that don't overlap here
    Kernel qk = createKernel(
        ctx, dot, Bindings{activations.qkv, activations.qkv, activations.qk},
        {batchSize * nHeads, 1, 1});
    */
    // TODO(avh): check nThreads
  }

  LOG(kDefLog, kInfo, "Done");
}
