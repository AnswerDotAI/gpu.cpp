/* 
 * WIP implementation of Sasha Rush's GPU puzzles https://github.com/srush/GPU-Puzzles
 */

#include <array>
#include <cstdio>
#include "gpu.h"
#include "utils/array_utils.h"

using namespace gpu;

static constexpr size_t N = 3072;

template <size_t N>
std::array<float, N> makeData() {
  std::array<float, N> inputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  return inputArr;
}

template <size_t N>
void showResult(GPUContext& ctx, Kernel& op,  GPUTensor& output) {
  DispatchKernel(ctx, op);
  std::array<float, N> outputArr;
  Wait(ctx, op.future);
  ToCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  fprintf(stdout, "%s", show<float, N, 1>(outputArr, "output").c_str());
}

// Puzzle 1 : Map
// Implement a "kernel" (GPU function) that adds 10 to each position of vector
// a and stores it in vector out. You have 1 thread per position.
const char *kPuzzle1_Map= R"(
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    if (idx < arrayLength(&input)) {
      output[idx] = input[idx] + 10;
    }
  }
)";

void puzzle1(GPUContext& ctx) {
  fprintf(stdout, "\n\nPuzzle 1\n\n");
  GPUTensor input = CreateTensor(ctx, {N}, kf32, makeData<N>().data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32);
  Kernel op =
      CreateKernel(ctx, ShaderCode{kPuzzle1_Map, 256}, input, output);
  showResult<N>(ctx, op, output);
}

// Puzzle 2 : Zip
// Implement a kernel that adds together each position of a and b and stores it
// in out. You have 1 thread per position.
const char *kPuzzle2_Map= R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    if (idx < arrayLength(&a)) {
      output[idx] = a[idx] + b[idx];
    }
  }
)";

void puzzle2(GPUContext& ctx) {
  fprintf(stdout, "\n\nPuzzle 2\n\n");
  GPUTensor a = CreateTensor(ctx, {N}, kf32, makeData<N>().data());
  GPUTensor b = CreateTensor(ctx, {N}, kf32, makeData<N>().data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32);
  Kernel op =
      CreateKernel(ctx, ShaderCode{kPuzzle2_Map, 256}, GPUTensors{a, b}, output);
  showResult<N>(ctx, op, output);
}


// Puzzle 3 : Guards
// Implement a kernel that adds 10 to each position of a and stores it in out.
// You have more threads than positions.
const char *kPuzzle3_Map= R"(
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(4)
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    // increment by workgroup size 
    for (var i = GlobalInvocationID.x; i < arrayLength(&input); i = i + 4) {
      output[i] = input[i] + 10;
    }
  }
)";
void puzzle3(GPUContext& ctx) {
  fprintf(stdout, "\n\nPuzzle 3\n\n");
  GPUTensor input = CreateTensor(ctx, {N}, kf32, makeData<N>().data());
  GPUTensor output = CreateTensor(ctx, {N}, kf32);
  Kernel op =
      CreateKernel(ctx, ShaderCode{kPuzzle3_Map, 4}, input, output);
  showResult<N>(ctx, op, output);
}

// Puzzle 4 : Map 2D
// Implement a kernel that adds 10 to each position of a and stores it in out.
// Input a is 2D and square. You have more threads than positions.
// TODO

// Puzzle 5 : Broadcast
// Implement a kernel that adds a and b and stores it in out. Inputs a and b
// are vectors. You have more threads than positions.
// TODO

// ...

int main(int argc, char **argv) {
  GPUContext ctx = CreateGPUContext();
  puzzle1(ctx);
  puzzle2(ctx);
  puzzle3(ctx);
  return 0;
}
