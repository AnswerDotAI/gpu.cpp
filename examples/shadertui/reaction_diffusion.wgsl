@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    time: f32,
    screenwidth: u32,
    screenheight: u32,
};

const DELTA_TIME: f32 = 0.05;
const DIFFUSION_RATE_A: f32 = 1.0;
const DIFFUSION_RATE_B: f32 = 0.5;
const FEED_RATE: f32 = 0.055;
const KILL_RATE: f32 = 0.062;

fn get_state(x: i32, y: i32, component: i32) -> f32 {
    let width = i32(params.screenwidth);
    let height = i32(params.screenheight);
    let wrapped_x = (x + width) % width;
    let wrapped_y = (y + height) % height;
    return state[(wrapped_y * width + wrapped_x) * 2 + component];
}

fn compute_laplacian(x: i32, y: i32, component: i32) -> f32 {
    let center = get_state(x, y, component);
    let left = get_state(x - 1, y, component);
    let right = get_state(x + 1, y, component);
    let top = get_state(x, y - 1, component);
    let bottom = get_state(x, y + 1, component);
    
    return (left + right + top + bottom - 4.0 * center) * 0.25;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let x = i32(globalID.x);
    let y = i32(globalID.y);
    let idx = (y * i32(params.screenwidth) + x) * 2;
    
    let a = get_state(x, y, 0);
    let b = get_state(x, y, 1);
    let laplacian_a = compute_laplacian(x, y, 0);
    let laplacian_b = compute_laplacian(x, y, 1);
    
    let a_next = a + (DIFFUSION_RATE_A * laplacian_a - a * b * b + FEED_RATE * (1.0 - a)) * DELTA_TIME;
    let b_next = b + (DIFFUSION_RATE_B * laplacian_b + a * b * b - (KILL_RATE + FEED_RATE) * b) * DELTA_TIME;
    
    state[idx] = a_next;
    state[idx + 1] = b_next;
}
