#include "gpu.h"
#include "utils/array_utils.h"
#include "utils/logging.h"
#include <array>

#include "reference_impls.h"

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
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>) {
    let row = global_id.y; // row and column of C 
    let col = global_id.x;
    if (row >= {{M}} || col >= {{N}}) {
        return;
    }
    // [row * {{N}} + col] = 0;
    // C[row * {{N}} + col] = f32(row * {{N}} + col);
    // C[row * {{N}} + col] = f32(row * {{N}});
    // C[row * {{N}} + col] = f32(col);
    var total: f32 = 0; // A[row * {{K}}] * B[col * {{N}}];
    for (var k = 0u; k < {{K}}; k = k + 1u) {
        // B is stored as B^T, effectively column-major
        total += A[row * {{K}} + k] * B[col * {{N}} + k];
    }
    C[row * {{N}} + col] = total;
    // C[row * {{N}} + col] = A[row * {{K}} + col];
    // C[row * {{N}} + col] = 0;
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

void createTransformer(Context &ctx, size_t modelDim, size_t qkvDim,
                     size_t batchSize, size_t seqLen, size_t hiddenWidth,
                     Transformer &transformer, Activations &activations,
                     KVCache &kvcache) {
  std::mt19937 gen(314159);
  transformer = {
      .qkv = createTensor(ctx, Shape{3 * qkvDim, modelDim}, kf32), // column-major
      .rmsNormPre = createTensor(ctx, Shape{modelDim}, kf32),
      .rmsNormPost = createTensor(ctx, Shape{modelDim}, kf32),
      .out = createTensor(ctx, Shape{3 * qkvDim, modelDim}, kf32),
      .mlp1 = createTensor(ctx, Shape{modelDim, 2 * hiddenWidth}, kf32),
      .mlp2 = createTensor(ctx, Shape{modelDim, 2 * hiddenWidth}, kf32),
  };

  // Initialize values
  std::unique_ptr<float[]> qkvInit(new float[modelDim * 3 * qkvDim]);
  // randint(qkvInit.get(), size(transformer.qkv.shape), gen, -2, 2);
  range(qkvInit.get(), size(transformer.qkv.shape), 0.0);
  LOG(kDefLog, kInfo, "%s",
      show<float>(qkvInit.get(), transformer.qkv.shape[0],
                  transformer.qkv.shape[1], "QKV Weights")
          .c_str());
  toGPU(ctx, qkvInit.get(), transformer.qkv);

  activations = {
      // TODO
  };
  kvcache = {
      .key_cache = createTensor(ctx, Shape{seqLen, qkvDim}, kf32),
      .value_cache = createTensor(ctx, Shape{seqLen, qkvDim}, kf32),
  };
}

inline ShaderCode createMatmul(const char *shaderTemplate, const size_t M,
                               const size_t K, const size_t N,
                               const Shape &workgroupSize = {256, 1, 1},
                               NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  ReplaceAll(codeString, "{{workgroupSize}}", toString(workgroupSize));
  ReplaceAll(codeString, "{{precision}}", toString(precision));
  ReplaceAll(codeString, "{{M}}", std::to_string(M));
  ReplaceAll(codeString, "{{K}}", std::to_string(K));
  ReplaceAll(codeString, "{{N}}", std::to_string(N));
  LOG(kDefLog, kInfo, "Shader code:\n%s\n", codeString.c_str());
  return ShaderCode{codeString, workgroupSize};
}

int main() {
  printf("\033[2J\033[1;1H");
  Context ctx = createContext();
  static constexpr size_t seqLen = 24;
  static constexpr size_t batchSize = 1;
  static constexpr size_t modelDim = 3; // 3072;
  static constexpr size_t hiddenWidth = modelDim * 2;
  static constexpr size_t qkvDim = 1; //256;
  std::mt19937 gen(314);

  Transformer transformer;
  Activations activations;
  KVCache kvcache;
  LOG(kDefLog, kInfo, "Initializing transformer, allocating GPU buffers ...\n");
  createTransformer(ctx, modelDim, qkvDim, batchSize, seqLen, hiddenWidth,
                  transformer, activations, kvcache);

  std::array<float, modelDim> inputArr;
  std::array<float, modelDim * 3 * qkvDim> weightsArr;
  randint(inputArr, gen, -2, 2);
  LOG(kDefLog, kInfo, "%s",
      show<float>(inputArr.data(), 1, modelDim, "Input").c_str());
  Tensor input = createTensor(ctx, Shape{modelDim}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, Shape{3 * qkvDim}, kf32);

  ShaderCode matmul = createMatmul(kShaderMatmul0, 1, modelDim, 3 * qkvDim);
  Kernel qkv =
      createKernel(ctx, matmul, TensorList{input, transformer.qkv, output},
                   /*nthreads*/ {modelDim, 1, 1});
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, qkv, promise);
  wait(ctx, future);
  std::array<float, 3 * qkvDim> outputArr;
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  LOG(kDefLog, kInfo, "Output: %s",
      show<float>(outputArr.data(), 1, 3 * qkvDim, "QKV Output").c_str());

  std::array<float, 3 * qkvDim> outputRefArr;
  ref::matmul_forward_cpu(
      outputRefArr.data(), inputArr.data(), weightsArr.data(), NULL,
      /* batch */ 1, /* T */ 1, /* C */ modelDim, /* OC */ 3 * qkvDim);
  LOG(kDefLog, kInfo, "Reference Output: %s",
      show<float>(outputRefArr.data(), 1, 3 * qkvDim, "QKV Output (Reference)")
          .c_str());

  LOG(kDefLog, kInfo, isclose(outputArr.data(), outputRefArr.data(), 3 * qkvDim) ? "PASS" : "FAIL");

  LOG(kDefLog, kInfo, "Done");
}
