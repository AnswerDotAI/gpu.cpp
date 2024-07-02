/*
 * WIP implementation of Sasha Rush's GPU puzzles
 * https://github.com/srush/GPU-Puzzles
 */

#include "gpu.h"
#include "utils/array_utils.h"
#include <array>
#include <cstdio>
#include <future>

using namespace gpu;

static constexpr size_t N = 3072;

template <size_t N> std::array<float, N> makeData() {
  std::array<float, N> inputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  return inputArr;
}

template <size_t N, size_t R = N, size_t C = 1> void showResult(Context &ctx, Kernel &op, Tensor &output) {

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  dispatchKernel(ctx, op, promise);
  std::array<float, R * C> outputArr;
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  printf("%s", show<float, R, C>(outputArr, "output").c_str());
}

// Puzzle 1 : Map
// Implement a "kernel" (GPU function) that adds 10 to each position of vector
// a and stores it in vector out. You have 1 thread per position.
const char *kPuzzle1 = R"(
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    if (idx < arrayLength(&input)) {
      output[idx] = input[idx] + 10;
    }
  }
)";

void puzzle1(Context &ctx) {
  printf("\n\nPuzzle 1\n\n");
  Tensor input = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  Kernel op = createKernel(ctx, createShader(kPuzzle1, 256), Bindings{input, output},
                           /*nthreads*/ {N, 1, 1});
  showResult<N>(ctx, op, output);
}

// Puzzle 2 : Zip
// Implement a kernel that adds together each position of a and b and stores it
// in out. You have 1 thread per position.
const char *kPuzzle2 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    if (idx < arrayLength(&a)) {
      output[idx] = a[idx] + b[idx];
    }
  }
)";

void puzzle2(Context &ctx) {
  printf("\n\nPuzzle 2\n\n");
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor b = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  Kernel op = createKernel(ctx, createShader(kPuzzle2, 256), Bindings{a, b, output},
                           {N, 1, 1});
  showResult<N>(ctx, op, output);
}

// Puzzle 3 : Guards
// Implement a kernel that adds 10 to each position of a and stores it in out.
// You have more threads than positions.
const char *kPuzzle3 = R"(
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    // increment by workgroup size 
    for (var i = GlobalInvocationID.x; i < arrayLength(&input); i = i + 4) {
      output[i] = input[i] + 10;
    }
  }
)";
void puzzle3(Context &ctx) {
  printf("\n\nPuzzle 3\n\n");
  Tensor input = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  Kernel op =
      createKernel(ctx, createShader(kPuzzle3, 4), Bindings{input, output}, {N, 1, 1});
  showResult<N>(ctx, op, output);
}

// Puzzle 4 : Map 2D
// Implement a kernel that adds 10 to each position of a and stores it in out.
// Input a is 2D and square. You have more threads than positions.
const char *kPuzzle4 = R"(
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
struct Params {
  size: u32, // input is size x size
};
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    let idx = GlobalInvocationID.x + GlobalInvocationID.y * params.size;
    if (idx < arrayLength(&input)) {
      output[idx] = input[idx] + 10;
    }
  }
)";
void puzzle4(Context &ctx) {
  printf("\n\nPuzzle 4\n\n");
  static constexpr size_t N = 9;
  Tensor input = createTensor(ctx, {N, N}, kf32, makeData<N * N>().data());
  Tensor output = createTensor(ctx, {N, N}, kf32);
  struct Params {
    uint32_t size = N;
  };
  Kernel op =
      createKernel(ctx, createShader(kPuzzle4, /*workgroup size*/ {N, N, 1}),
                   Bindings{input, output}, {N, N, 1}, Params{N});
  showResult<N, N, N>(ctx, op, output);
}

// Puzzle 5 : Broadcast
// Implement a kernel that adds a and b and stores it in out. Inputs a and b
// are vectors. You have more threads than positions.
const char *kPuzzle5_Broadcast = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}}) (
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    // TODO
  }
)";

// TODO
// ...

int main(int argc, char **argv) {
  Context ctx = createContext();
  puzzle1(ctx);
  puzzle2(ctx);
  puzzle3(ctx);
  puzzle4(ctx);
  return 0;
}
