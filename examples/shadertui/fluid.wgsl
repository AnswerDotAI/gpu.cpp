@group(0) @binding(0) var<storage, read_write> fluid: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    time: f32,
    screenwidth: u32,
    screenheight: u32,
};

const DIFFUSION: f32 = 0.0001;
const DT: f32 = 0.016;

fn idx(x: i32, y: i32) -> u32 {
    let width = i32(params.screenwidth);
    let height = i32(params.screenheight);
    return u32((y + height) % height * width + (x + width) % width);
}

fn diffuse(x: i32, y: i32) -> f32 {
    let x0 = idx(x - 1, y);
    let x1 = idx(x + 1, y);
    let y0 = idx(x, y - 1);
    let y1 = idx(x, y + 1);
    let c = idx(x, y);

    let a = DIFFUSION * DT * f32(params.screenwidth * params.screenheight);
    return (fluid[c] + a * (fluid[x0] + fluid[x1] + fluid[y0] + fluid[y1])) / (1.0 + 4.0 * a);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let x = i32(globalID.x);
    let y = i32(globalID.y);
    let i = idx(x, y);

    // Add some "dye" to the center
    if (x == i32(params.screenwidth) / 2 && y == i32(params.screenheight) / 2) {
        fluid[i] = 1.0;
    }

    // Diffuse
    let diffused = diffuse(x, y);

    // Simple upward flow
    let above = idx(x, (y + 1) % i32(params.screenheight));
    fluid[i] = mix(diffused, fluid[above], 0.1);
}
