@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    time: f32,
    screenwidth: u32,
    screenheight: u32,
};

const PI: f32 = 3.14159265359;

fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 2.0;
    for (var i = 0; i < 6; i = i + 1) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = vec2<f32>(f32(globalID.x) / resolution.x, f32(globalID.y) / resolution.y);
    let idx = globalID.y * params.screenwidth + globalID.x;
    
    let time = params.time * 0.8;
    
    // Create base shape of aurora
    var p = vec2<f32>(uv.x * 2.0, uv.y);
    var f = fbm(p * 3.0 + vec2<f32>(time * 0.5, time * 0.1));
    f = f * f * f + 0.5 * f * f + 0.1 * f;
    
    // Create movement
    f += 0.1 * sin(p.x * 4.0 + time) * sin(p.y * 10.0 + time);
    
    // Shape the aurora
    f *= smoothstep(0.0, 0.5, uv.y);
    f *= smoothstep(1.0, 0.7, uv.y);
    
    // Add color variation (in grayscale)
    f += 0.1 * sin(time + uv.y * PI * 2.0);
    
    // Clamp and adjust intensity
    f = clamp(f, 0.0, 1.0);
    f *= 0.7;  // Reduce overall intensity
    
    out[idx] = f;
}
