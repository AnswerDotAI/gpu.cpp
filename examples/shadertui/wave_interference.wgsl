@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    time: f32,
    screenwidth: u32,
    screenheight: u32,
};

const NUM_SOURCES: u32 = 4;
const PI: f32 = 3.14159265359;

fn wave(dist: f32, frequency: f32, speed: f32, time: f32) -> f32 {
    return sin(dist * frequency + time * speed);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = (vec2<f32>(globalID.xy) - 0.5 * resolution) / min(resolution.x, resolution.y);
    let idx = globalID.y * params.screenwidth + globalID.x;
    
    var sources: array<vec2<f32>, NUM_SOURCES>;
    sources[0] = vec2<f32>(sin(params.time * 0.5) * 0.4, cos(params.time * 0.3) * 0.4);
    sources[1] = vec2<f32>(sin(params.time * 0.4) * 0.4, sin(params.time * 0.5) * 0.4);
    sources[2] = vec2<f32>(cos(params.time * 0.3) * 0.4, sin(params.time * 0.4) * 0.4);
    sources[3] = vec2<f32>(0.0, 0.0);  // Static source at center

    var total_wave: f32 = 0.0;

    for (var i: u32 = 0u; i < NUM_SOURCES; i = i + 1u) {
        let dist = distance(uv, sources[i]);
        let frequency = 10.0 + f32(i) * 5.0;  // Different frequency for each source
        let speed = 2.0 + f32(i);  // Different speed for each source
        total_wave += wave(dist, frequency, speed, params.time);
    }

    // Normalize and shift to 0-1 range
    total_wave = (total_wave / f32(NUM_SOURCES) + 1.0) * 0.5;
    
    // Add some contrast
    total_wave = pow(total_wave, 1.5);

    out[idx] = total_wave;
}
