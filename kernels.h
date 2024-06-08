#ifndef KERNELS_H
#define KERNELS_H

#include "gpu.h"

namespace gpu {

std::string ReplaceAll(std::string str, const std::string &from,
                       const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return str;
}

std::string PrepareShader(const char *shaderRaw, NumType precision,
                          size_t workgroupSize) {
  std::string shader(shaderRaw);
  const char *precisionStr = ToString(precision);
  shader =
      ReplaceAll(shader, "{{workgroupSize}}", std::to_string(workgroupSize));
  shader = ReplaceAll(shader, "{{precision}}", precisionStr);
  return shader;
}

// test function - multiply by constant
const char *kShaderCMul = R"(
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    if (idx < arrayLength(&input)) {
      output[idx] = input[idx] * 2.0;
    }
  }
)";

// approximate gelu
const char *kShaderGELU = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    // Ensure we do not access out of bounds
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR * (x + cube)));
    }
}
)";

// TODO(avh): 3D workgroup specs
ShaderCode GeluShader(size_t workgroupSize = 32 * 32,
                      const NumType precision = kf32) {
  return ShaderCode{PrepareShader(kShaderGELU, precision, workgroupSize),
                    workgroupSize};
}

const char *kShaderHadamard = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> C: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    if (idx < arrayLength(&A)) {
      C[idx] = A[idx] * B[idx];
    }
}
)";

ShaderCode HadamardShader(size_t workgroupSize = 32 * 32,
                          const NumType precision = kf32) {
  return ShaderCode{PrepareShader(kShaderHadamard, precision, workgroupSize),
                    workgroupSize};
}

const char *kShaderResidual = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> C: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    if (idx < arrayLength(&A)) {
      C[idx] = A[idx] + B[idx];
    }
}
)";

ShaderCode ResidualShader(size_t workgroupSize = 32 * 32,
                          const NumType precision = kf32) {
  return ShaderCode{PrepareShader(kShaderResidual, precision, workgroupSize),
                    workgroupSize};
}

/* LayerNorm
 * v1:
 * - No caching mean/std for backwards
 * - No parallel reduction
 * - Simple 1 thread for each 1..N
 */
// TODO(avh): Allow larger virtual 1D workgroups by making use of y / z
// dimensions and calculating the threadID accordingly.
const char *kShaderLayerNorm1 = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> weight: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> bias: array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    N: u32,
    C: u32,
};

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>,
        @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
        @builtin(workgroup_id) WorkgroupID: vec3<u32>) {
    let idx: u32 = GlobalInvocationID.x;

    if (idx >= params.N) { return; }

    let C: u32 = params.C;

    // Calculate mean
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < C; i = i + 1) {
        sum += inp[idx * C + i];
    }
    let mean_val: f32 = sum / f32(C);

    // Calculate rstd
    sum = 0.0;
    for (var i: u32 = 0; i < C; i = i + 1) {
        let diff: f32 = inp[idx * C + i] - mean_val;
        sum += diff * diff;
    }
    let rstd_val: f32 = 1.0 / sqrt(sum / f32(C) + 1e-5);

    for (var i: u32 = 0; i < C; i = i + 1) {
        let n: f32 = rstd_val * (inp[idx * C + i] - mean_val);
        out[idx * C + i] = n * weight[i] + bias[i];
    }
}
)";

ShaderCode LayerNormShader(size_t workgroupSize = 256,
                           const NumType precision = kf32) {
  return ShaderCode{PrepareShader(kShaderLayerNorm1, precision, workgroupSize),
                    workgroupSize};
}

// matrix multiplication (naive implementation)
const char *kShaderMatMul1 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> B: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> C: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x / {{N}};
    let j: u32 = GlobalInvocationID.x % {{N}};
    if (i < {{M}} && j < {{N}}) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < {{K}}; k = k + 1) {
            sum = sum + A[i * {{K}} + k] * B[k * {{N}} + j];
        }
        C[i * {{N}} + j] = sum;
    }
}
)";

const char *kShaderMatMul2 = R"(
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

// Shared memory for tiling - TODO(avh): fix
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

    if (row >= M || col >= N) {
        return;
    }

    var result: f32 = 0.0;

    for (var i = 0u; i < K; i = i + workgroupSizeX) {
        // Load tiles into shared memory
        /*
        tileA[local_id.y][local_id.x] = A[row][i + local_id.x];
        tileB[local_id.y][local_id.x] = B[i + local_id.y][col];
        */

        // Synchronize to make sure the tile is loaded
        workgroupBarrier();

        /*
        // Perform partial dot product for the current tile
        for (var k = 0u; k < workgroupSizeX; k = k + 1u) {
            result = result + tileA[local_id.y][k] * tileB[k][local_id.x];
        }
        */

        // Synchronize before loading the next tile
        workgroupBarrier();
    }

    // Write the result to the output matrix
    // C[row][col] = result;
}
)";

/* Generates ShaderCode instance for all matmul kernels - pass in
 * the template code via `shaderRaw` */
ShaderCode MatmulShader(size_t workgroupSize, const char *shaderRaw,
                        NumType precision, size_t M, size_t K, size_t N) {
  std::string shader = PrepareShader(shaderRaw, precision, workgroupSize);
  shader = ReplaceAll(shader, "{{M}}", std::to_string(M));
  shader = ReplaceAll(shader, "{{K}}", std::to_string(K));
  shader = ReplaceAll(shader, "{{N}}", std::to_string(N));
  return ShaderCode{shader, workgroupSize};
}

/* Softmax
 * v1:
 * - equivalent to naive softmax with one thread per row
 */
const char *kShaderSoftmax1 = R"(
@group(0) @binding(0) var<storage, read_write> inp : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out : array<{{precision}}>;
@group(0) @binding(2) var<uniform> params : Params;

struct Params {
    N: u32,
    C: u32,
};

const NEG_INFINITY: f32 = -3.0e38; // WGSL has problem representing -3.4028235e+38

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let N : u32 = params.N;
    let C : u32 = params.C;
    let i : u32 = global_id.x;
    if (i < N) {
        let inp_row_start : u32 = i * C;
        var maxval : f32 = NEG_INFINITY;
        // Find the maximum value in the row
        for (var j : u32 = 0u; j < C; j++) {
            let val : f32 = inp[inp_row_start + j];
            if (val > maxval) {
                maxval = val;
            }
        }
        var sum : f32 = 0.0;
        // Compute the exponentials and sum them
        for (var j : u32 = 0u; j < C; j++) {
            let exp_val : f32 = exp(inp[inp_row_start + j] - maxval);
            out[inp_row_start + j] = exp_val;
            sum += exp_val;
        }
        // Normalize the row to get probabilities
        let norm : f32 = 1.0f / sum;
        for (var j : u32 = 0u; j < C; j++) {
            out[inp_row_start + j] *= norm;
        }
    }
}
)";

ShaderCode SoftmaxShader(size_t workgroupSize = 32,
                         const NumType precision = kf32) {
  return ShaderCode{PrepareShader(kShaderSoftmax1, precision, workgroupSize),
                    workgroupSize};
}

} // namespace gpu

#endif // KERNELS_H
