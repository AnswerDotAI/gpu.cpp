@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

const MAX_ITERATIONS: u32 = 1000u;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    var uv = (vec2<f32>(globalID.xy) - 0.5 * resolution) / min(resolution.x, resolution.y);
    
    // Zoom parameters
    let t = params.time * 0.3;
    let zoom = pow(2.0, t);
    let center = vec2<f32>(-0.745, 0.186);
    
    // Apply zoom
    uv = uv / zoom + center;
    
    var z = vec2<f32>(0.0, 0.0);
    let c = uv;
    var i: u32 = 0u;
    
    for (; i < MAX_ITERATIONS; i = i + 1u) {
        z = complex_mul(z, z) + c;
        if (dot(z, z) > 4.0) {
            break;
        }
    }
    
    // Smooth coloring
    let smooth_i = f32(i) + 1.0 - log2(max(1.0, log2(length(z))));
    
    // Color mapping
    let hue = smooth_i / 50.0;
    let saturation = 0.6;
    let value = 1.0;
    
    // HSV to RGB conversion
    let K = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(hue) + K) * 6.0 - 3.0);
    let color = value * mix(vec3<f32>(1.0), clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)), saturation);
    
    let idx = globalID.y * params.screenwidth + globalID.x;
    // out[idx] = pow(max(((color.r + color.g + color.b) * 2.0), 0.0) - .8, 4.0) / 400.0 ;
    out[idx] = pow(max(((color.r + color.g + color.b) * 2.0), 0.0) - .8, 4.0) / 400.0 - .2;
}
