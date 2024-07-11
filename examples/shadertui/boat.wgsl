// A boat rocking on water at night, by Claude 3.5 Sonnet

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
  time: f32,
  screenwidth: u32,
  screenheight: u32,
};

fn sdf_box(p: vec2<f32>, center: vec2<f32>, size: vec2<f32>) -> f32 {
    let d = abs(p - center) - size;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdf_triangle(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let v0 = p - p0;
    let v1 = p - p1;
    let v2 = p - p2;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d = min(min(vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                    vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                    vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
    return -sqrt(d.x) * sign(d.y);
}

fn sdf_boat(p: vec2<f32>, center: vec2<f32>, size: f32) -> f32 {
    let hull = sdf_box(p, center + vec2<f32>(0.0, -0.15) * size, vec2<f32>(0.4, 0.1) * size);
    let sail1 = sdf_triangle(p,
                             center + vec2<f32>(-0.1, -0.05) * size,
                             center + vec2<f32>(0.1, -0.05) * size,
                             center + vec2<f32>(0.0, 0.4) * size);
    let sail2 = sdf_triangle(p,
                             center + vec2<f32>(0.1, -0.05) * size,
                             center + vec2<f32>(0.3, -0.05) * size,
                             center + vec2<f32>(0.2, 0.3) * size);
    return min(min(hull, sail1), sail2);
}

fn wave(x: f32, time: f32) -> f32 {
    return 0.05 * sin(x * 3.0 + time) + 
           0.03 * sin(x * 5.0 - time * 0.8) + 
           0.02 * sin(x * 8.0 + time * 1.5);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = (vec2<f32>(globalID.xy) + 0.5) / resolution;
    let aspect = resolution.x / resolution.y;
    let p = vec2<f32>(uv.x * aspect, uv.y);
    
    let t = params.time;
    
    // Water
    let water_level = 0.5;
    let water_base = 0.6;
    let water_distance = p.y - water_level - wave(p.x, t);
    
    // Boat
    let boat_size = 0.6;
    let boat_center = vec2<f32>(0.5 * aspect, water_level + 0.1 + 0.05 * sin(t * 0.5));
    let boat_rotation = 0.1 * sin(t * 0.4);
    let rotated_p = vec2<f32>(
        cos(boat_rotation) * (p.x - boat_center.x) - sin(boat_rotation) * (p.y - boat_center.y),
        sin(boat_rotation) * (p.x - boat_center.x) + cos(boat_rotation) * (p.y - boat_center.y)
    ) + boat_center;
    let boat_distance = sdf_boat(rotated_p, boat_center, boat_size);
    
    // Coloring
    var intensity = 0.05; // Dark night sky
    
    if (water_distance < 0.0) {
        intensity = water_base - 0.2 * (1.0 + wave(p.x, t));
    }
    
    if (boat_distance < 0.0) {
        intensity = 0.2; // Dark boat silhouette
    }
    
    // Add stars
    if (intensity == 0.05 && fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453) > 0.998) {
        intensity = 1.0;
    }
    
    // Moon
    let moon_center = vec2<f32>(0.8 * aspect, 0.8);
    let moon_distance = length(p - moon_center) - 0.05;
    if (moon_distance < 0.0) {
        intensity = 0.9;
    }
    
    let idx = (params.screenheight - globalID.y) * params.screenwidth + globalID.x;
    out[idx] = intensity;
}
