@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct Params {
    time: f32,
    screenwidth: u32,
    screenheight: u32,
};

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    life: f32,
}

const NUM_PARTICLES: u32 = 1000u;
const PARTICLE_LIFE: f32 = 9.0;
const EMISSION_RATE: f32 = 300.0;

fn rand(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

fn initialize_particle(id: f32, time: f32) -> Particle {
    let random1 = rand(id * 0.01 + time * 0.1);
    let random2 = rand(id * 0.02 + time * 0.1);
    let angle = random1 * 2.0 * 3.14159;
    let speed = 0.05 + random2 * 0.05;
    
    return Particle(
        vec2<f32>(0.5, 0.5),
        vec2<f32>(cos(angle) * speed, sin(angle) * speed),
        PARTICLE_LIFE
    );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalID : vec3<u32>) {
    let resolution = vec2<f32>(f32(params.screenwidth), f32(params.screenheight));
    let uv = vec2<f32>(f32(globalID.x) / resolution.x, f32(globalID.y) / resolution.y);
    let idx = globalID.y * params.screenwidth + globalID.x;
    
    var color: f32 = 0.0;
    
    for (var i: f32 = 0.0; i < f32(NUM_PARTICLES); i += 1.0) {
        let spawn_time = i / EMISSION_RATE;
        let particle_age = fract((params.time - spawn_time) / PARTICLE_LIFE) * PARTICLE_LIFE;
        
        if (particle_age < PARTICLE_LIFE) {
            var particle = initialize_particle(i, spawn_time);
            particle.position += particle.velocity * particle_age;
            
            let distance = length(uv - particle.position);
            if (distance < 0.005) {
                color += 0.5 * (1.0 - particle_age / PARTICLE_LIFE);
            }
        }
    }
    
    out[idx] = min(color, 1.0);
}
