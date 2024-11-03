#ifndef KERNELS_H
#define KERNELS_H

#include "gpu.hpp"

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

static const char *kShaderTanh = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        out[i] = tanh(x);
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
    Cp: u32,
};
const NEG_INFINITY: f32 = -3.0e38; // WGSL has problem representing -3.4028235e+38
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let N : u32 = params.N;
    let C : u32 = params.C;
    let Cp : u32 = params.Cp;
    let i : u32 = global_id.x;
    if (i < N) {
        let inp_row_start : u32 = i * Cp;
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
        for (var j : u32 = C; j < Cp; j++) {
            out[inp_row_start + j] = 0;
        }
    }
}
)";

// Encoder
static const char *kShaderEncoder = R"(
@group(0) @binding(0) var<storage, read_write> inp : array<i32>;
@group(0) @binding(1) var<storage, read_write> wte : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> wpe : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> out : array<{{precision}}>;
@group(0) @binding(4) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    C: u32,
};
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;
    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;
    if (b < B && t < T) {
        let ix : u32 = u32(inp[b * T + t]);
        let out_bt : u32 = b * T * C + t * C;
        for (var i : u32 = 0u; i < C; i++) {
            out[out_bt + i] = wte[ix * C + i] + wpe[t * C + i];
        }
    }
}
)";

static const char *kShaderEncoderBackward = R"(
@group(0) @binding(0) var<storage, read_write> dwte : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dwpe : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> dout : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> inp : array<i32>;
@group(0) @binding(4) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    C: u32,
};
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;
    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;
    if (b < B && t < T) {
        let ix : u32 = u32(inp[b * T + t]);
        let dout_bt : u32 = b * T * C + t * C;
        for (var i : u32 = 0u; i < C; i++) {
            let d : {{precision}} = dout[dout_bt + i];
            atomicAdd(&dwte[ix * C + i], d);
            atomicAdd(&dwpe[t * C + i], d);
        }
    }
}
)";


// Matmul
static const char *kShaderMatmul = R"(
@group(0) @binding(0) var<storage, read_write> inp : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> weight : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> bias : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> out : array<{{precision}}>;
@group(0) @binding(4) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    C: u32,
    OC: u32,
};
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;
    let OC : u32 = params.OC;
    // N == B*T == global_id.x
    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;
    if (arrayLength(&bias) == 1) {
      if (b < B && t < T) {
          let bt : u32 = global_id.x;
          for (var o : u32 = 0u; o < OC; o++) {
              var val : {{precision}} = 0;
              for (var i : u32 = 0u; i < C; i++) {
                  val += inp[bt * C + i] * weight[o * C + i];
              }
              out[bt * OC + o] = val;
          }
      }
    } else {
      if (b < B && t < T) {
          let bt : u32 = global_id.x;
          for (var o : u32 = 0u; o < OC; o++) {
              var val : {{precision}} = bias[o];
              for (var i : u32 = 0u; i < C; i++) {
                  val += inp[bt * C + i] * weight[o * C + i];
              }
              out[bt * OC + o] = val;
          }
      }
    }
}

)";


static const char *kShaderMatmul2DTiling = R"(
@group(0) @binding(0) var<storage, read_write> inp : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> weight : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> bias : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> out : array<{{precision}}>;
@group(0) @binding(4) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    C: u32,
    OC: u32,
};
var<workgroup> tileInp: array<{{precision}}, {{BT}} * {{BC}}>;
var<workgroup> tileWeight: array<{{precision}}, {{BOC}} * {{BC}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;
    let OC : u32 = params.OC;

    var localT: array<{{precision}}, {{TT}}>;
    var localOC: array<{{precision}}, {{TOC}}>;

    let outB: u32 = groupid.x;
    let outT: u32 = groupid.y;
    let outOC: u32 = groupid.z;
    let numThread: u32 = ({{BT}} * {{BOC}}) / ({{TT}} * {{TOC}});

    // position of the first c element computed by the thread
    let threadRow: u32 = (localID.x / ({{BOC}} / {{TOC}})) * {{TT}};
    let threadCol: u32 = (localID.x % ({{BOC}} / {{TOC}})) * {{TOC}};

    // inpPtr and weightPtr are the starting positions of the tiles in a and b,
    // incremented in the bkidx loop. 
    // outPtr is the starting position of the tile in c which is fixed.

    var inpPtr = (outB * T + outT * {{BT}}) * C; // BTC 
    var weightPtr = outOC * {{BOC}} * C; //OCC
    var threadResults: array<{{precision}}, {{TT}} * {{TOC}}>;
    let outPtr = (outB * T + outT * {{BT}}) * OC + outOC * {{BOC}}; //BTOC
    let biasPtr = outOC * {{BOC}};

    for (var bkidx: u32 = 0; bkidx < C; bkidx += {{BC}}) {
      // Load BC x BOC by numThread(BT * BOC / (TT * TOC))
      // The number of iteration == BC * BOC / (BT * BOC / (TT * TOC))
      for (var idx: u32 = 0; idx < {{NUM_TILEW}}; idx++) {
        tileWeight[localID.x + idx * numThread] = weight[weightPtr + ((localID.x + idx * numThread) / {{BC}}) * C + ((localID.x + idx * numThread) % {{BC}})];
      }
      weightPtr += {{BC}};
    
      // Load tile
      // Load BT x BC by numThread(BT * BOC / (TT * TOC))
      // The number of iteration == BT * BC / (BT * BOC / (TT * TOC))
      for (var idx: u32 = 0; idx < {{NUM_TILEI}}; idx++) {
        tileInp[localID.x + idx * numThread] = inp[inpPtr + ((localID.x + idx * numThread) / {{BC}}) * C + (localID.x + idx * numThread) % {{BC}}];
      }
      inpPtr += {{BC}};
    
      workgroupBarrier();
      // Compute tile
      for (var dotIdx: u32 = 0; dotIdx < {{BC}}; dotIdx = dotIdx + 1) {
        for (var idx: u32 = 0; idx < {{TT}}; idx++) {
          localT[idx] = tileInp[(threadRow + idx) * {{BC}} + dotIdx];
        }
        for (var idx: u32 = 0; idx < {{TOC}}; idx++) {
          localOC[idx] = tileWeight[(threadCol + idx) * {{BC}} + dotIdx];
        }
        for (var resIdxT: u32 = 0; resIdxT < {{TT}}; resIdxT++) {
          for (var resIdxOC: u32 = 0; resIdxOC < {{TOC}}; resIdxOC++) {
            threadResults[resIdxT * {{TOC}} + resIdxOC] += localT[resIdxT] * localOC[resIdxOC];
          }
        }
      }
      workgroupBarrier();
    }
    
    if (arrayLength(&bias) == 1) {
      for (var resIdxT: u32 = 0; resIdxT < {{TT}}; resIdxT++) {
        for (var resIdxOC: u32 = 0; resIdxOC < {{TOC}}; resIdxOC++) {
          out[outPtr + (threadRow + resIdxT) * OC + threadCol + resIdxOC] = threadResults[resIdxT * {{TOC}} + resIdxOC];
        }
      }
    } else {
      for (var resIdxT: u32 = 0; resIdxT < {{TT}}; resIdxT++) {
        for (var resIdxOC: u32 = 0; resIdxOC < {{TOC}}; resIdxOC++) {
          out[outPtr + (threadRow + resIdxT) * OC + threadCol + resIdxOC] = threadResults[resIdxT * {{TOC}} + resIdxOC] + bias[biasPtr + threadCol + resIdxOC];
        }
      }
    }
}
)";

static const char *kShaderMatmulBackward = R"(
@group(0) @binding(0) var<storage, read_write> dinp : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dweight : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> dbias : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> dout : array<{{precision}}>;
@group(0) @binding(4) var<storage, read_write> inp : array<{{precision}}>;
@group(0) @binding(5) var<storage, read_write> weight : array<{{precision}}>;
@group(0) @binding(6) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    C: u32,
    OC: u32,
};
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;
    let OC : u32 = params.OC;
    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;
    if (b < B && t < T) {
        let bt : u32 = b * T + t;
        for (var o : u32 = 0u; o < OC; o++) {
            let d : {{precision}} = dout[bt * OC + o];
            atomicAdd(&dbias[o], d);
            for (var i : u32 = 0u; i < C; i++) {
                atomicAdd(&dinp[bt * C + i], weight[o * C + i] * d);
                atomicAdd(&dweight[o * C + i], inp[bt * C + i] * d);
            }
        }
    }
}
)";

// Attention
static const char *kShaderAttention = R"(
@group(0) @binding(0) var<storage, read_write> inp : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> preatt : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> att : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> out : array<{{precision}}>;
@group(0) @binding(4) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    C: u32,
    NH: u32,
};
const NEG_INFINITY: f32 = -3.0e38; // WGSL has problem representing -3.4028235e+38
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;
    let NH : u32 = params.NH;
    let C3 : u32 = C * 3u;
    let hs : u32 = C / NH;
    let scale : {{precision}} = 1.0 / sqrt({{precision}}(hs));

    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;

    if (b < B && t < T) {
        for (var h : u32 = 0u; h < NH; h++) {
            let query_t : u32 = b * T * C3 + t * C3 + h * hs;
            let preatt_bth : u32 = b * NH * T * T + h * T * T + t * T;
            let att_bth : u32 = b * NH * T * T + h * T * T + t * T;

            // pass 1: calculate query dot key and maxval
            var maxval : {{precision}} = NEG_INFINITY;
            for (var t2 : u32 = 0u; t2 <= t; t2++) {
                let key_t2 : u32 = b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                // (query_t) dot (key_t2)
                var val : {{precision}} = 0.0;
                for (var i : u32 = 0u; i < hs; i++) {
                    val += inp[query_t + i] * inp[key_t2 + i];
                }
                val *= scale;
                if (val > maxval) {
                    maxval = val;
                }

                preatt[preatt_bth + t2] = val;
            }

            // pass 2: calculate the exp and keep track of sum
            // maxval is being calculated and subtracted only for numerical stability
            var expsum : {{precision}} = 0.0;
            for (var t2 : u32 = 0u; t2 <= t; t2++) {
                let expv : {{precision}} = exp(preatt[preatt_bth + t2] - maxval);
                expsum += expv;
                att[att_bth + t2] = expv;
            }
            let expsum_inv : {{precision}} = select(0.0, 1.0 / expsum, expsum != 0.0);

            // pass 3: normalize to get the softmax
            for (var t2 : u32 = 0u; t2 < T; t2++) {
                if (t2 <= t) {
                    att[att_bth + t2] *= expsum_inv;
                } else {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att[att_bth + t2] = 0.0;
                }
            }

            // pass 4: accumulate weighted values into the output of attention
            let out_bth : u32 = b * T * C + t * C + h * hs;
            for (var i : u32 = 0u; i < hs; i++) { out[out_bth + i] = 0.0; }
            for (var t2 : u32 = 0u; t2 <= t; t2++) {
                let value_t2 : u32 = b * T * C3 + t2 * C3 + h * hs + C * 2u; // +C*2 because it's value
                let att_btht2 : {{precision}} = att[att_bth + t2];
                for (var i : u32 = 0u; i < hs; i++) {
                    out[out_bth + i] += att_btht2 * inp[value_t2 + i];
                }
            }
        }
    }
}
)";

static const char *kShaderAttentionBackward = R"(
@group(0) @binding(0) var<storage, read_write> dinp : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dpreatt : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> datt : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> dout : array<{{precision}}>;
@group(0) @binding(4) var<storage, read_write> inp : array<{{precision}}>;
@group(0) @binding(5) var<storage, read_write> att : array<{{precision}}>;
@group(0) @binding(6) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    C: u32,
    NH: u32,
};
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;
    let NH : u32 = params.NH;
    let C3 : u32 = C * 3u;
    let hs : u32 = C / NH;
    let scale : {{precision}} = 1.0 / sqrt({{precision}}(hs));

    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;

    if (b < B && t < T) {
        for (var h : u32 = 0u; h < NH; h++) {
            let att_bth : u32 = b * NH * T * T + h * T * T + t * T;
            let datt_bth : u32 = b * NH * T * T + h * T * T + t * T;
            let dpreatt_bth : u32 = b * NH * T * T + h * T * T + t * T;
            let dquery_t : u32 = b * T * C3 + t * C3 + h * hs;
            let query_t : u32 = b * T * C3 + t * C3 + h * hs;

            // backward pass 4, through the value accumulation
            let dout_bth : u32 = b * T * C + t * C + h * hs;
            for (var t2 : u32 = 0u; t2 <= t; t2++) {
                let value_t2 : u32 = b * T * C3 + t2 * C3 + h * hs + C * 2u; // +C*2 because it's value
                let dvalue_t2 : u32 = b * T * C3 + t2 * C3 + h * hs + C * 2u; // +C*2 because it's value
                for (var i : u32 = 0u; i < hs; i++) {
                    // in the forward pass this was:
                    // out_bth[i] += att_bth[t2] * value_t2[i];
                    // so now we have:
                    atomicAdd(&datt[datt_bth + t2], inp[value_t2 + i] * dout[dout_bth + i]);
                    atomicAdd(&dinp[dvalue_t2 + i], att[att_bth + t2] * dout[dout_bth + i]);
                }
            }

            // backward pass 2 & 3, the softmax
            // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
            for (var t2 : u32 = 0u; t2 <= t; t2++) {
                for (var t3 : u32 = 0u; t3 <= t; t3++) {
                    let indicator : {{precision}} = select(0.0, 1.0, t2 == t3);
                    let local_derivative : {{precision}} = att[att_bth + t2] * (indicator - att[att_bth + t3]);
                    atomicAdd(&dpreatt[dpreatt_bth + t3], local_derivative * datt[datt_bth + t2]);
                }
            }

            // backward pass 1, the query @ key matmul
            for (var t2 : u32 = 0u; t2 <= t; t2++) {
                let key_t2 : u32 = b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                let dkey_t2 : u32 = b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                for (var i : u32 = 0u; i < hs; i++) {
                    // in the forward pass this was:
                    // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                    // so now we have:
                    atomicAdd(&dinp[dquery_t + i], inp[key_t2 + i] * dpreatt[dpreatt_bth + t2] * scale);
                    atomicAdd(&dinp[dkey_t2 + i], inp[query_t + i] * dpreatt[dpreatt_bth + t2] * scale);
                }
            }
        }
    }
}
)";

// LayerNorm
static const char *kShaderLayerNorm = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> weight: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> bias: array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(4) var<storage, read_write> mean: array<{{precision}}>;
@group(0) @binding(5) var<storage, read_write> rstd: array<{{precision}}>;
@group(0) @binding(6) var<uniform> params: Params;

struct Params {
    B: u32,
    T: u32,
    C: u32,
};

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx: u32 = GlobalInvocationID.x;

    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;

    if (idx >= B * T) { return; }

    let b : u32 = idx / T;
    let t : u32 = idx % T;

    // Calculate mean
    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < C; i = i + 1) {
        sum += inp[b * T * C + t * C + i];
    }
    let mean_val: f32 = sum / f32(C);
    mean[b * T + t] = mean_val;

    // Calculate rstd
    sum = 0.0;
    for (var i: u32 = 0; i < C; i = i + 1) {
        let diff: f32 = inp[b * T * C + t * C + i] - mean_val;
        sum += diff * diff;
    }
    let rstd_val: f32 = 1.0 / sqrt(sum / f32(C) + 1e-5);
    rstd[b * T + t] = rstd_val;

    for (var i: u32 = 0; i < C; i = i + 1) {
        let n: f32 = rstd_val * (inp[b * T * C + t * C + i] - mean_val);
        out[b * T * C + t * C + i] = n * weight[i] + bias[i];
    }
}
)";

static const char *kShaderLayerNormBackward = R"(
@group(0) @binding(0) var<storage, read_write> dinp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dweight: array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> dbias: array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> dout: array<{{precision}}>;
@group(0) @binding(4) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(5) var<storage, read_write> weight: array<{{precision}}>;
@group(0) @binding(6) var<storage, read_write> mean: array<{{precision}}>;
@group(0) @binding(7) var<storage, read_write> rstd: array<{{precision}}>;
@group(0) @binding(8) var<uniform> params: Params;

struct Params {
    B: u32,
    T: u32,
    C: u32,
};

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx: u32 = GlobalInvocationID.x;

    let B : u32 = params.B;
    let T : u32 = params.T;
    let C : u32 = params.C;

    if (idx >= B * T) { return; }

    let b : u32 = idx / T;
    let t : u32 = idx % T;

    // first: two reduce operations
    var dnorm_mean: f32 = 0.0f;
    var dnorm_norm_mean: f32 = 0.0f;
    for (var i: u32 = 0; i < C; i = i + 1) {
        let norm_bti: f32 = (inp[b * T * C + t * C + i] - mean[b * T + t]) * rstd[b * T + t];
        let dnorm_i: f32 = weight[i] * dout[b * T * C + t * C + i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / f32(C);
    dnorm_norm_mean = dnorm_norm_mean / f32(C);

    // now iterate again and accumulate all the gradients
    for (var i: u32 = 0; i < C; i = i + 1) {
        let norm_bti: f32 = (inp[b * T * C + t * C + i] - mean[b * T + t]) * rstd[b * T + t];
        let dnorm_i: f32 = weight[i] * dout[b * T * C + t * C + i];
        // gradient contribution to bias
        atomicAdd(&dbias[i], dout[b * T * C + t * C + i]);
        // gradient contribution to weight
        atomicAdd(&dweight[i], norm_bti * dout[b * T * C + t * C + i]);
        // gradient contribution to input
        var dval: f32 = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd[b * T + t]; // final scale
        atomicAdd(&dinp[b * T * C + t * C + i], dval);
    }
}
)";

static const char *kShaderCrossEntropyForward = R"(
@group(0) @binding(0) var<storage, read_write> losses : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> probs : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> targets : array<i32>;
@group(0) @binding(3) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    Vp: u32,
};
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let Vp : u32 = params.Vp;
    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;
    if (b < B && t < T) {
        let probs_bt : u32 = b * T * Vp + t * Vp;
        let ix : u32 = u32(targets[b * T + t]);
        losses[b * T + t] = -log(probs[probs_bt + ix]);
    }
}
)";

static const char *kShaderCrossEntropySoftmaxBackward = R"(
@group(0) @binding(0) var<storage, read_write> dlogits : array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dlosses : array<{{precision}}>;
@group(0) @binding(2) var<storage, read_write> probs : array<{{precision}}>;
@group(0) @binding(3) var<storage, read_write> targets : array<i32>;
@group(0) @binding(4) var<uniform> params : Params;
struct Params {
    B: u32,
    T: u32,
    V: u32,
    Vp: u32,
};
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let B : u32 = params.B;
    let T : u32 = params.T;
    let V : u32 = params.V;
    let Vp : u32 = params.Vp;
    let b : u32 = global_id.x / T;
    let t : u32 = global_id.x % T;
    if (b < B && t < T) {
        let dlogits_bt : u32 = b * T * Vp + t * Vp;
        let probs_bt : u32 = b * T * Vp + t * Vp;
        let dloss : {{precision}} = dlosses[b * T + t];
        let ix : u32 = u32(targets[b * T + t]);
        for (var i : u32 = 0u; i < V; i++) {
            let p : {{precision}} = probs[probs_bt + i];
            let indicator : {{precision}} = select(0.0, 1.0, i == ix);
            dlogits[dlogits_bt + i] += (p - indicator) * dloss;
        }
    }
}
)";

static const char *kSum = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
var<workgroup> buffer: array<{{precision}}, 1024>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>,
    @builtin(num_workgroups) numGroups : vec3<u32>) {
    let blockSize3d: vec3<u32> = vec3({{workgroupSize}});
    let blockSize: u32 = blockSize3d.x;
    let threadId: u32 = localID.x;
    let blockId: u32 = groupid.x + groupid.y * numGroups.x;
    let blockStart = blockId * blockSize * 2 + threadId;

    buffer[threadId] = inp[blockStart] + inp[blockStart + blockSize];
    workgroupBarrier();
    var stride: u32 = blockSize / 2;
 
    if (blockSize >= 1024 && threadId < 512) {
        buffer[threadId] += buffer[threadId + 512];
    }
    workgroupBarrier();
   
    if (blockSize >= 512 && threadId < 256) {
        buffer[threadId] += buffer[threadId + 256];
    }
    workgroupBarrier();

    if (blockSize >= 256 && threadId < 128) {
        buffer[threadId] += buffer[threadId + 128];
    }
    workgroupBarrier();

    if (threadId < 64) {
        buffer[threadId] += buffer[threadId + 64];
    }
    workgroupBarrier();

    if (threadId < 32) {
        buffer[threadId] += buffer[threadId + 32];
    }
    workgroupBarrier();

    if (threadId < 16) {
        buffer[threadId] += buffer[threadId + 16];
    }
    workgroupBarrier();

    if (threadId < 8) {
        buffer[threadId] += buffer[threadId + 8];
    }
    workgroupBarrier();

    if (threadId < 4) {
        buffer[threadId] += buffer[threadId + 4];
    }
    workgroupBarrier();

    if (threadId < 2) {
        buffer[threadId] += buffer[threadId + 2];
    }
    workgroupBarrier();

    if (threadId == 0) {
        buffer[0] += buffer[1];
        out[blockId] = buffer[0];
    }
}
)";
  
} // namespace gpu

#endif // KERNELS_H
