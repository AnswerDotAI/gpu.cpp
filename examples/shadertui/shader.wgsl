@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

fn sdf(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
  return length(p - c) - r;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
  let xy: vec2<f32> =
    vec2<f32>(f32(globalID.x) / f32(params.screenwidth),
              f32(globalID.y) / f32(params.screenheight));
  let t: f32 = params.time / 1.0;
  let idx = globalID.y * params.screenwidth + globalID.x;
  let center = vec2<f32>(0.5, 0.5 + 0.3 * sin(3.0 * t));
  let center2 = vec2<f32>(0.5 + 0.2 * cos(3.0 * t), 0.5);
  // out[idx] += 0.4 - min(5 * abs(sdf(xy, center, 0.2)), 0.4) + 0.5 * cos(xy.y + t) + 0.5 * sin(xy.x);
  out[idx] = 0.3 - min(5 * abs(sdf(xy, center, 0.2)), 0.3);
  out[idx] += 0.3 - min(5 * abs(sdf(xy, center2, 0.2)), 0.3);
  out[idx] += 0.4 * sin(xy.y +t);
}
