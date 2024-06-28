#ifndef KERNELS_H
#define KERNELS_H

#include "gpu.h"

namespace gpu {


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

static const char *kShaderHadamard = R"(
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

static const char *kShaderResidual = R"(
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

/* LayerNorm
 * v1:
 * - No caching mean/std for backwards
 * - No parallel reduction
 * - Simple 1 thread for each 1..N
 */
// TODO(avh): Allow larger virtual 1D workgroups by making use of y / z
// dimensions and calculating the threadID accordingly.
static const char *kShaderLayerNorm1 = R"(
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

// matrix multiplication (naive implementation)
static const char *kShaderMatMul1 = R"(
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

static const char *kShaderMatMul2 = R"(
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

/* Generates ShaderCode instance for all matmul kernels - pass in
 * the template code via `shaderRaw`.
 *
 * This is intended to be run ahead of time, so is not performance critical.
 * */
ShaderCode MatmulShader(size_t workgroupSize, const char *shaderRaw,
                        NumType precision, size_t M, size_t K, size_t N) {
  ShaderCode shader = CreateShader(shaderRaw, workgroupSize, precision);
  ReplaceAll(shader.data, "{{M}}", std::to_string(M));
  ReplaceAll(shader.data, "{{K}}", std::to_string(K));
  ReplaceAll(shader.data, "{{N}}", std::to_string(N));
  return shader;
}

/* Softmax
 * v1:
 * - equivalent to naive softmax with one thread per row
 */
static const char *kShaderSoftmax1 = R"(
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
            out[inp_row_start + j] /= sum;
        }
    }
}
)";

} // namespace gpu

#endif // KERNELS_H
