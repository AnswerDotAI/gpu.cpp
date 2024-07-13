@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    time: f32,
    screenwidth: u32,
    screenheight: u32,
};

const NUM_METABALLS: u32 = 5;
const THRESHOLD: f32 = 1.0;

struct Metaball {
    position: vec2<f32>,
    radius: f32,
}

fn create_metaball(id: u32, time: f32) -> Metaball {
    let angle = f32(id) * 0.8 + time * (0.2 + f32(id) * 0.05);
    let distance = 0.25 + sin(f32(id) * 1.3 + time * 0.4) * 0.1;
    let position = vec2<f32>(
        cos(angle) * distance + 0.5,
        sin(angle) * distance + 0.5
    );
    let radius = 0.1 + sin(f32(id) * 2.0 + time * 0.8) * 0.05;
    return Metaball(position, radius);
}

fn metaball_field(uv: vec2<f32>, metaball: Metaball) -> f32 {
    let d = distance(uv, metaball.position);
    return metaball.radius / d;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = vec2<f32>(f32(globalID.x) / resolution.x, f32(globalID.y) / resolution.y);
    let idx = globalID.y * params.screenwidth + globalID.x;
    
    let time = params.time * 1.0;
    
    var sum: f32 = 0.0;
    
    for (var i: u32 = 0u; i < NUM_METABALLS; i = i + 1u) {
        let metaball = create_metaball(i, time);
        sum += metaball_field(uv, metaball);
    }
    
    // Create the metaball effect
    let metaball = smoothstep(THRESHOLD - 0.05, THRESHOLD + 0.05, sum);
    
    // Add some interior detail
    let detail = sin(sum * 20.0) * 0.1 + 0.9;
    
    let color = pow(metaball * detail, 4.0) / 1.2 - .15;
    
    out[idx] = color;
}
