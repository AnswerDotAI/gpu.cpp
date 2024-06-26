#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

#include "gpu.h"
#include "utils/array_utils.h"                       

using namespace gpu;

const char *kShaderUpdateFluid = R"(
const DT: f32 = 1.0;
const GRID_SIZE: u32 = 80u;
const TAU: f32 = 0.6;
const OMEGA: f32 = 1.0 / (TAU / DT);

@group(0) @binding(0) var<storage, read_write> f_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> density: array<f32>;

const WEIGHTS: array<f32, 9> = array<f32, 9>(
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

const CX: array<i32, 9> = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const CY: array<i32, 9> = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);

fn idx(x: u32, y: u32, d: u32) -> u32 {
    return (y % GRID_SIZE) * GRID_SIZE * 9u + (x % GRID_SIZE) * 9u + d;
}

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    var rho: f32 = 0.0;
    var ux: f32 = 0.0;
    var uy: f32 = 0.0;

    // Compute macroscopic variables
    for (var i: u32 = 0u; i < 9u; i++) {
        let fi = f_in[idx(x, y, i)];
        rho += fi;
        ux += fi * f32(CX[i]);
        uy += fi * f32(CY[i]);
    }
    ux /= rho;
    uy /= rho;

    // Add velocity push from left and right
    if (x == 0u) { ux = 0.0001; }
    if (x == GRID_SIZE - 1u) { ux = -0.0001; }
    // add noise to ux
    ux *= 1 + 0.04 * f32(y % 20);


    // Collision step
    for (var i: u32 = 0u; i < 9u; i++) {
        let ci_u = f32(CX[i]) * ux + f32(CY[i]) * uy;
        let feq = rho * WEIGHTS[i] * (1.0 + 3.0 * ci_u + 4.5 * ci_u * ci_u - 1.5 * (ux * ux + uy * uy));
        f_out[idx(x, y, i)] = f_in[idx(x, y, i)] - OMEGA * (f_in[idx(x, y, i)] - feq);
    }

    // Update density
    density[y * GRID_SIZE + x] = rho;
}
)";

const char *kShaderStreamFluid = R"(
const GRID_SIZE: u32 = 80u;

@group(0) @binding(0) var<storage, read_write> f_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> f_out: array<f32>;

const CX: array<i32, 9> = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const CY: array<i32, 9> = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);

fn idx(x: u32, y: u32, d: u32) -> u32 {
    return (y % GRID_SIZE) * GRID_SIZE * 9u + (x % GRID_SIZE) * 9u + d;
}

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    // Stream step
    for (var i: u32 = 0u; i < 9u; i++) {
        let nx: u32 = u32((i32(x) + i32(GRID_SIZE) - CX[i]) % i32(GRID_SIZE));
        let ny: u32 = u32((i32(y) + i32(GRID_SIZE) - CY[i]) % i32(GRID_SIZE));
        f_out[idx(x, y, i)] = f_in[idx(nx, ny, i)];
    }
}
)";


int main() {
    static constexpr size_t GRID_SIZE = 80;
    Context ctx = CreateContext();

    // Initialize distributions
    std::array<float, GRID_SIZE * GRID_SIZE * 9> fArr;
    for (size_t i = 0; i < fArr.size(); ++i) {
        fArr[i] = 1.0f;
    }

    // GPU buffers
    Tensor f1 = CreateTensor(ctx, Shape{GRID_SIZE * GRID_SIZE * 9}, kf32, fArr.data());
    Tensor f2 = CreateTensor(ctx, Shape{GRID_SIZE * GRID_SIZE * 9}, kf32);
    Tensor density = CreateTensor(ctx, Shape{GRID_SIZE * GRID_SIZE}, kf32);

    std::array<float, GRID_SIZE * GRID_SIZE> densityArr;
    std::string screen(GRID_SIZE * (GRID_SIZE + 1), ' '); // + 1 for newline
    Shape nThreads{GRID_SIZE, GRID_SIZE, 1};

    // Prepare computations
    ShaderCode updateShader = CreateShader(kShaderUpdateFluid, 16, kf32);
    ShaderCode streamShader = CreateShader(kShaderStreamFluid, 16, kf32);
    Kernel update = CreateKernel(ctx, updateShader, TensorList{f1, f2, density}, nThreads);
    Kernel stream = CreateKernel(ctx, streamShader, TensorList{f2, f1}, nThreads);

    // Main simulation update loop
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            DispatchKernel(ctx, update, promise);
            Wait(ctx, future);
        }
        
        {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            DispatchKernel(ctx, stream, promise);
            Wait(ctx, future);
        }
        
        ToCPU(ctx, density, densityArr.data(), sizeof(densityArr));
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Adjust density values for better visualization
        for (auto& d : densityArr) {
            d = (d - 1.0f) * 5.0f + 0.5f;  // Amplify differences and shift
        }
        static const char intensity[] = " .`'^-+=*x17X$8#%@";
        // display to screen
        for (size_t i = 0; i < GRID_SIZE; ++i) {
          for (size_t j = 0; j < GRID_SIZE; ++j) {
            double densityNorm = std::max(0.0, std::min(1.0, (densityArr[i * GRID_SIZE + j] - 5.0) / 500.0));
            int index = densityNorm * 17;
            screen[i * (GRID_SIZE + 1) + j] = intensity[index];
          }
          screen[i * (GRID_SIZE + 1) + GRID_SIZE] = '\n';
        }
        // clear screen
        printf("\033[2J\033[1;1H");
        printf("%s", screen.c_str());


          ResetCommandBuffer(ctx.device, nThreads, update);
          ResetCommandBuffer(ctx.device, nThreads, stream);
          std::this_thread::sleep_for(std::chrono::milliseconds(70) - elapsed);
    }
}
