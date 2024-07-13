@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2<f32>(0.0, 0.0)), 
                   hash(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(hash(i + vec2<f32>(0.0, 1.0)), 
                   hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = vec2<f32>(f32(globalID.x) / resolution.x, f32(globalID.y) / resolution.y);
    let t = params.time * 0.2;

    let color1 = vec3<f32>(0.5, 0.8, 0.9);
    let color2 = vec3<f32>(0.9, 0.4, 0.3);

    let n = noise(uv * 3.0 + t) * 0.5 + 
            noise(uv * 6.0 - t * 0.5) * 0.25 + 
            noise(uv * 12.0 + t * 0.25) * 0.125;

    let color = mix(color1, color2, n);
    
    let idx = globalID.y * params.screenwidth + globalID.x;
    out[idx] = log((color.r + color.g + color.b) / 1.6) * exp(0.1 + color.r + color.g + color.b) / 4.0 - .2;
    // out[idx] = log((color.r + color.g + color.b) / 1.6);
    // out[idx] = (color.r + color.g + color.b) / 3.0;
}
