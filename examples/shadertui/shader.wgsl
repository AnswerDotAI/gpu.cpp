@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

fn sdf(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
  // Signed distance function (SDF) for a circle
  // See https://iquilezles.org/articles/distfunctions2d/
  return length(p - c) - r;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {

  // Normalize xy coordinates
  let xy: vec2<f32> =
    vec2<f32>(f32(globalID.x) / f32(params.screenwidth),
              f32(globalID.y) / f32(params.screenheight));

  // 1-D index into the output GPU buffer
  let idx = globalID.y * params.screenwidth + globalID.x;

  // Draw a circle, oscillating with time
  let position = vec2<f32>(0.5, 0.5 + 0.1 * sin(3.0 * params.time));
  out[idx] = 0.3 - min(5 * abs(sdf(xy, position, 0.2)), 0.3);

}
