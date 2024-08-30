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
        let x: {{precision}} = inp[i];
        // select is more stable for larger values of x
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                  * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";

static const char *kShaderGeluBackward = R"(
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dout: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> dinp: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: {{precision}} = inp[i];
        let cube: {{precision}} = 0.044715f * x * x * x;
        let tanh_arg: {{precision}} = GELU_SCALING_FACTOR * (x + cube);
        let tanh_out: {{precision}} = tanh(tanh_arg);
        let cosh_out: {{precision}} = cosh(tanh_arg);
        let sech_out: {{precision}} = 1.0f / (cosh_out * cosh_out);
        let local_grad: {{precision}} = select(0.5f * (1.0f + tanh_out), 1, x > 10.0) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
)";

static const char *kTanh = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        out[i] = tan(x);
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

static const char *kShaderResidualBackward = R"(
@group(0) @binding(0) var<storage, read_write> dout: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dinp1: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> dinp2: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&dout)) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
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
