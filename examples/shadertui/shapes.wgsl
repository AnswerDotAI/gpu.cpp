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

fn sdf_circle(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
    return length(p - c) - r;
}

fn sdf_triangle(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
    let k = sqrt(3.0);
    let q = p - c;
    let p_mod = vec2<f32>(abs(q.x) - r, q.y + r / k);
    let p_final = select(
        p_mod,
        vec2<f32>(p_mod.x - k * p_mod.y, -k * p_mod.x - p_mod.y) / 2.0,
        p_mod.x + k * p_mod.y > 0.0
    );
    let x_clamped = clamp(p_final.x, -2.0 * r, 0.0);
    return -length(vec2<f32>(p_final.x - x_clamped, p_final.y)) * sign(p_final.y);
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
  
  let orbit_center = vec2<f32>(0.5, 0.5);
  let orbit_radius = 0.3;  // Increased orbit radius
  let orbit_speed = 0.5;
  let rotation_speed = 2.0;

  // Calculate positions of the three shapes
  let angle1 = t * orbit_speed;
  let angle2 = angle1 + 2.0 * 3.14159 / 3.0;
  let angle3 = angle2 + 2.0 * 3.14159 / 3.0;
  let center1 = orbit_center + orbit_radius * vec2<f32>(cos(angle1), sin(angle1));
  let center2 = orbit_center + orbit_radius * vec2<f32>(cos(angle2), sin(angle2));
  let center3 = orbit_center + orbit_radius * vec2<f32>(cos(angle3), sin(angle3));

  // Increased sizes for all shapes
  let rect_size = vec2<f32>(0.12, 0.06);  // Larger rectangle
  let rect_corner_radius = 0.015;  // Slightly larger corner radius
  let circle_radius = 0.08;  // Larger circle
  let triangle_radius = 0.09;  // Larger triangle

  // Rotation
  let rotation_angle = t * rotation_speed;
  let rotation_matrix = rotate2d(rotation_angle);

  // Apply rotation to the points
  let rotated_point1 = (rotation_matrix * (xy - center1)) + center1;
  let rotated_point2 = (rotation_matrix * (xy - center2)) + center2;
  let rotated_point3 = (rotation_matrix * (xy - center3)) + center3;

  // Calculate SDFs for each shape
  let sdf1 = sdf_rounded_rectangle(rotated_point1, center1, rect_size, rect_corner_radius);
  let sdf2 = sdf_circle(rotated_point2, center2, circle_radius);
  let sdf3 = sdf_triangle(rotated_point3, center3, triangle_radius);

  // Combine the SDFs
  let combined_sdf = min(sdf1, min(sdf2, sdf3));

  out[idx] = 0.3 - min(5.0 * abs(combined_sdf), 0.3);
}
