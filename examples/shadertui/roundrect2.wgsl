@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

fn sdf_rounded_rectangle(p: vec2<f32>, c: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
    let d = abs(p - c) - size + vec2<f32>(radius);
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

fn rotate2d(angle: f32) -> mat2x2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return mat2x2<f32>(c, -s, s, c);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
  let xy: vec2<f32> =
    vec2<f32>(f32(globalID.x) / f32(params.screenwidth),
              f32(globalID.y) / f32(params.screenheight));
  let t: f32 = params.time / 1.0;
  let idx = globalID.y * params.screenwidth + globalID.x;
  
  let center = vec2<f32>(0.5, 0.5 + 0.3 * sin(3.0 * t));
  let size = vec2<f32>(0.2, 0.1);  // Width and height of the rectangle
  let corner_radius = 0.05;  // Radius of the rounded corners
  
  // Rotation
  let rotation_speed = 2.0;  // Adjust this to change rotation speed
  let rotation_angle = t * rotation_speed;
  let rotation_matrix = rotate2d(rotation_angle);
  
  // Apply rotation to the point
  let rotated_point = (rotation_matrix * (xy - center)) + center;
  
  out[idx] = 0.3 - min(5.0 * abs(sdf_rounded_rectangle(rotated_point, center, size, corner_radius)), 0.3);
}
