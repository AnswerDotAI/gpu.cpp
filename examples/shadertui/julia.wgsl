@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

const MAX_ITERATIONS: u32 = 100;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = (vec2<f32>(globalID.xy) - 0.5 * resolution) / min(resolution.x, resolution.y);
    
    // Animate the Julia set parameters
    let t = params.time * 0.3;
    let c = 0.7885 * vec2<f32>(cos(t), sin(t));
    
    var z = uv * 3.0;
    var i: u32 = 0u;
    
    for (; i < MAX_ITERATIONS; i = i + 1u) {
        z = complex_mul(z, z) + c;
        if (dot(z, z) > 4.0) {
            break;
        }
    }
    
    let smooth_i = f32(i) + 1.0 - log2(log2(dot(z, z)));
    let color = 0.5 + 0.5 * cos(3.0 + smooth_i * 0.15 + vec3<f32>(0.0, 0.6, 1.0));
    
    let idx = globalID.y * params.screenwidth + globalID.x;
    out[idx] = (color.r + color.g + color.b) / 3.0;
}
