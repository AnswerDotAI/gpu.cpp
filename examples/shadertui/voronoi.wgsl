@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    time: f32,
    screenwidth: u32,
    screenheight: u32,
};

const NUM_CELLS: u32 = 10u;

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn voronoi(p: vec2<f32>, time: f32) -> f32 {
    var min_dist = 1.0;
    var cell_color = 0.0;

    for (var i = 0u; i < NUM_CELLS; i = i + 1u) {
        let fi = f32(i);
        let angle = hash(fi) * 6.28318 + time * (0.2 + hash(fi + 1.0) * 0.7);
        let radius = 0.3 + 0.2 * hash(fi + 2.0);
        let cell_pos = vec2<f32>(
            cos(angle) * radius + 0.5,
            sin(angle) * radius + 0.5
        );

        let dist = distance(p, cell_pos);
        if (dist < min_dist) {
            min_dist = dist;
            cell_color = hash(fi + 3.0);
        }
    }

    return cell_color;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = vec2<f32>(globalID.xy) / resolution;
    let idx = globalID.y * params.screenwidth + globalID.x;
    
    let v = voronoi(uv, params.time * 0.12);
    
    // Enhance contrast and ensure full dynamic range
    let value = smoothstep(0.1, 0.9, v) / 2.0;
    
    out[idx] = value;
}
