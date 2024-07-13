@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

fn get_cell(x: u32, y: u32) -> u32 {
    let idx = y * params.screenwidth + x;
    return u32(out[idx] > 0.5);
}

fn count_neighbors(x: u32, y: u32) -> u32 {
    let width = params.screenwidth;
    let height = params.screenheight;
    
    let left = (x + width - 1u) % width;
    let right = (x + 1u) % width;
    let up = (y + height - 1u) % height;
    let down = (y + 1u) % height;
    
    return get_cell(left, up) +
           get_cell(x, up) +
           get_cell(right, up) +
           get_cell(left, y) +
           get_cell(right, y) +
           get_cell(left, down) +
           get_cell(x, down) +
           get_cell(right, down);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let x = globalID.x;
    let y = globalID.y;
    
    if (x >= params.screenwidth || y >= params.screenheight) {
        return;
    }
    
    let idx = y * params.screenwidth + x;
    let current_state = get_cell(x, y);
    let neighbors = count_neighbors(x, y);
    
    var next_state: u32;
    if (current_state == 1u) {
        // Cell is alive
        if (neighbors == 2u || neighbors == 3u) {
            next_state = 1u; // Cell survives
        } else {
            next_state = 0u; // Cell dies
        }
    } else {
        // Cell is dead
        if (neighbors == 3u) {
            next_state = 1u; // Cell becomes alive
        } else {
            next_state = 0u; // Cell stays dead
        }
    }
    
    out[idx] = f32(next_state);
}
