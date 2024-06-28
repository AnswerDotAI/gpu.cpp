#include "gpu.h"
#include "utils/array_utils.h"
#include "utils/logging.h"
#include <array>

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

static const char *kShaderMatmul0 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(workgroupSizeX, workgroupSizeY, 1)
fn matmul(
    @builtin(global_invocation_id) global_id : vec3<u32>) {
    // row and column of C
    let row = global_id.y;
    let col = global_id.x;
    for (var k = 0u; k < {{K}}; k = k + 1u) {
        // B is stored as B^T, effectively column-major
        C[row * {{N}} + col] += A[row * {{K}} + k] * B[k + col * {{N}}];
    }
}
");

static const char *kShaderMatMul = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
var<workgroup> tileA: array<f32, workgroupSizeY * workgroupSizeX>;
var<workgroup> tileB: array<f32, workgroupSizeY * workgroupSizeX>;
@compute @workgroup_size(workgroupSizeX, workgroupSizeY, 1)
fn matmul(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= {{M}} || col >= {{N}}) {
        return;
    }
    var result: f32 = 0.0;
    for (var i = 0u; i < {{K}}; i = i + workgroupSizeX) {
        // Load tiles into shared memory
        tileA[local_id.y][local_id.x] = A[row][i + local_id.x];
        tileB[local_id.y][local_id.x] = B[i + local_id.y][col];
        // Synchronize to make sure the tile is loaded
        workgroupBarrier();
        // Perform partial dot product for the current tile
        for (var k = 0u; k < workgroupSizeX; k = k + 1u) {
            result = result + tileA[local_id.y][k] * tileB[k][local_id.x];
        }
        // Synchronize before loading the next tile
        workgroupBarrier();
    }
    C[row][col] = result;
}
)";

struct Transformer {
  Tensor qkv;         // modelDim * 3 * qkvDim
  Tensor rmsNormPre;  // modelDim
  Tensor rmsNormPost; // modelDim
  Tensor out;         // 3 * qkvDim * modelDim

  Tensor mlp1; // modelDim * (2 * hidden_width * modelDim)
  Tensor mlp2; // modelDim * (2 * hidden_width * modelDim)
};

struct Activations {
  Tensor normPre; // modelDim * seqLen
  Tensor attOut;
};

struct KVCache {
  Tensor key_cache;
  Tensor value_cache;
};

void initTransformer(Context &ctx, size_t modelDim, size_t qkvDim,
                     size_t batchSize, size_t seqLen, size_t hiddenWidth,
                     Transformer &transformer, Activations &activations,
                     KVCache &kvcache) {
  std::mt19937 gen(314159);
  transformer = {
      .qkv = createTensor(ctx, Shape{modelDim, 3 * qkvDim}, kf32),
      .rmsNormPre = createTensor(ctx, Shape{modelDim}, kf32),
      .rmsNormPost = createTensor(ctx, Shape{modelDim}, kf32),
      .out = createTensor(ctx, Shape{3 * qkvDim, modelDim}, kf32),
      .mlp1 = createTensor(ctx, Shape{modelDim, 2 * hiddenWidth}, kf32),
      .mlp2 = createTensor(ctx, Shape{modelDim, 2 * hiddenWidth}, kf32),
  };

  // Initialize values
  std::unique_ptr<float[]> qkvInit(new float[modelDim * 3 * qkvDim]);
  randn(qkvInit.get(), size(transformer.qkv.shape), gen);
  printf("%s", show<float>(qkvInit.get(), transformer.qkv.shape[0], transformer.qkv.shape[1], "QKV Weights").c_str());
  toGPU(ctx, qkvInit.get(), transformer.qkv);

  activations = {
      // TODO
  };
  kvcache = {
      .key_cache = createTensor(ctx, Shape{seqLen, qkvDim}, kf32),
      .value_cache = createTensor(ctx, Shape{seqLen, qkvDim}, kf32),
  };
}

inline ShaderCode createMatmul(const char *shaderTemplate,
                        const size_t M, const size_t K, const size_t N,
                        const Shape &workgroupSize = {256, 1, 1},
                        NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  ReplaceAll(codeString, "{{workgroupSize}}", toString(workgroupSize));
  ReplaceAll(codeString, "{{precision}}", toString(precision));
  ReplaceAll(codeString, "{{M}}", std::to_string(M));
  ReplaceAll(codeString, "{{K}}", std::to_string(K));
  ReplaceAll(codeString, "{{N}}", std::to_string(N));
  return ShaderCode{codeString, workgroupSize};
}

int main() {
  printf("\033[2J\033[1;1H");
  Context ctx = createContext();
  // static constexpr N = 3072;
  static constexpr size_t N = 128;
  static constexpr size_t seqLen = 24;
  static constexpr size_t batchSize = 1;
  static constexpr size_t modelDim = 3072;
  static constexpr size_t hiddenWidth = modelDim * 2;
  static constexpr size_t qkvDim = 256;
  std::mt19937 gen(314);

  Transformer transformer;
  Activations activations;
  KVCache kvcache;
  printf("Initializing transformer, allocating GPU buffers ...\n");
  initTransformer(ctx, modelDim, qkvDim, batchSize, seqLen, hiddenWidth,
                  transformer, activations, kvcache);

  std::array<float, modelDim> inputArr;
  std::array<float, modelDim * 3 * qkvDim> weightsArr;
  randn(inputArr, gen);
  randn(weightsArr, gen);
  Tensor input = createTensor(ctx, Shape{modelDim}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, Shape{3 * qkvDim}, kf32);

  ShaderCode matmul = createMatmul(kShaderMatmul0, modelDim, 3 * qkvDim, modelDim);





  printf("Done\n");
}
