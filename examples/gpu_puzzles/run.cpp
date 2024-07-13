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

static constexpr size_t N = 10;

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
                           /*nWorkgroups */ {1, 1, 1});
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
                           {1, 1, 1});
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
      createKernel(ctx, createShader(kPuzzle3, 4), Bindings{input, output}, {N / 4, 1, 1});
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
                   Bindings{input, output}, /* nWorkgroups */ {1, 1, 1}, Params{N});
  showResult<N, N, N>(ctx, op, output);
}

// Puzzle 5 : Broadcast
// Implement a kernel that adds a and b and stores it in out. Inputs a and b
// are vectors. You have more threads than positions.
const char *kPuzzle5 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
struct Params {
  size: u32, // input is size x size
};
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    let local_i = GlobalInvocationID.x;
    let local_j = GlobalInvocationID.y;

    if (local_i < params.size && local_j < params.size) {
      output[local_i + local_j * params.size] = a[local_i] + b[local_j];
    }
  }
)";
void puzzle5(Context &ctx) {
  printf("\n\nPuzzle 5\n\n");
  static constexpr size_t N = 9;
  Tensor a = createTensor(ctx, {N, 1}, kf32, makeData<N>().data());
  Tensor b = createTensor(ctx, {1, N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N, N}, kf32);
  struct Params {
    uint32_t size = N;
  };

  Kernel op =
      createKernel(ctx, createShader(kPuzzle5, /*workgroup size*/ {N, N, 1}),
                   Bindings{a, b, output}, {1, 1, 1}, Params{N});
  showResult<N, N, N>(ctx, op, output);
}

// Puzzle 6 : Blocks
// Implement a kernel that adds 10 to each position of a and stores it in out.
// You have fewer threads per block than the size of a.

const char *kPuzzle6 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
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

    if (idx < params.size) {
      output[idx] = a[idx] + 10;
    }
  }
)";
void puzzle6(Context &ctx) {
  printf("\n\nPuzzle 6\n\n");
  static constexpr size_t N = 9;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  struct Params {
    uint32_t size = N;
  };

  Kernel op =
      createKernel(ctx, createShader(kPuzzle6, {4, 1, 1}),
                   Bindings{a, output}, {3, 1, 1}, Params{N});
  showResult<N>(ctx, op, output);
}

// Puzzle 7 : Blocks 2D
// Implement the same kernel in 2D. 
// You have fewer threads per block than the size of a in both directions.

const char *kPuzzle7 = R"(
@group(0)@binding(0) var<storage, read_write> a: array<f32>;
@group(0)@binding(1) var<storage, read_write> output : array<f32>;
@group(0)@binding(2) var<uniform> params: Params;
struct Params {
  size: u32, // input is size x size
};
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    let idx = GlobalInvocationID.x + GlobalInvocationID.y * params.size;

    if (idx < arrayLength(&a)) {
      output[idx] = a[idx] + 10;
    }
  }
)";
void puzzle7(Context &ctx) {
  printf("\n\nPuzzle 7\n\n");
  static constexpr size_t N = 5;
  Tensor a = createTensor(ctx, {N, N}, kf32, makeData<N * N>().data());
  Tensor output = createTensor(ctx, {N, N}, kf32);
  struct Params {
    uint32_t size = N;
  };

  Kernel op =
      createKernel(ctx, createShader(kPuzzle7, {3, 3, 1}),
                   Bindings{a, output}, {2, 2, 1}, Params{N});
  showResult<N, N, N>(ctx, op, output);
}

// Puzzle 8 : Shared
// Implement a kernel that adds 10 to each position of a and stores it in out.
// You have fewer threads per block than the size of a.

// (This example does not really need shared memory or syncthreads, but it is a demo.)

const char *kPuzzle8 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
struct Params {
  size: u32, 
};
var<workgroup> sharedData: array<f32, 256>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    let idx = GlobalInvocationID.x + GlobalInvocationID.y * params.size;
    sharedData[idx] = a[idx];

    workgroupBarrier();

    if (idx < params.size) {
      output[idx] = sharedData[idx] + 10;
    }
  }
)";
void puzzle8(Context &ctx) {
  printf("\n\nPuzzle 8\n\n");
  static constexpr size_t N = 8;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  struct Params {
    uint32_t size = N;
  };

  Kernel op =
      createKernel(ctx, createShader(kPuzzle8, {4, 1, 1}),
                   Bindings{a, output}, {2, 1, 1}, Params{N});
  showResult<N>(ctx, op, output);
}

// Puzzle 9 : Pooling
// Implement a kernel that sums together the last 3 position of a and stores it in out.
// You have 1 thread per position. You only need 1 global read and 1 global write per thread.

const char *kPuzzle9 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
struct Params {
  size: u32, // input is size x size
};
var<workgroup> sharedData: array<f32, 256>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x + GlobalInvocationID.y * params.size;
    let local_idx = GlobalInvocationID.x;

    if (idx < arrayLength(&a)) {
      sharedData[local_idx] = a[idx];
    }

    workgroupBarrier();

    if (idx == 0) {
      output[idx] = sharedData[idx];
    }
    else if (idx == 1) {
      output[idx] = sharedData[idx] + sharedData[idx - 1];
    }
    else {
      output[idx] = sharedData[idx] + sharedData[idx - 1] + sharedData[idx - 2];
    }
  }
)";
void puzzle9(Context &ctx) {
  printf("\n\nPuzzle 9\n\n");
  static constexpr size_t N = 8;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  struct Params {
    uint32_t size = N;
  };

  Kernel op =
      createKernel(ctx, createShader(kPuzzle9, {N, 1, 1}),
                   Bindings{a, output}, {1, 1, 1}, Params{N});
  showResult<N>(ctx, op, output);
}

// Puzzle 10 : Dot Product
// Implement a kernel that computes the dot-product of a and b and stores it in out.
// You have 1 thread per position. You only need 2 global reads and 1 global write per thread.

const char *kPuzzle10 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
struct Params {
  size: u32, // input is size x size
};
var<workgroup> sharedData: array<f32, 256>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x + GlobalInvocationID.y * params.size;
    let local_idx = GlobalInvocationID.x;

    if (idx < arrayLength(&a)) {
      sharedData[local_idx] = a[idx] * b[idx];
    }

    workgroupBarrier();

    if (local_idx == 0) {
      var sum = 0.0;
      for (var i: u32 = 0u; i < arrayLength(&a); i = i + 1u) {
        sum = sum + sharedData[i];
      }
      output[idx] = sum;
    }
  }
)";
void puzzle10(Context &ctx) {
  printf("\n\nPuzzle 10\n\n");
  static constexpr size_t N = 8;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor b = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {1}, kf32);
  struct Params {
    uint32_t size = N;
  };

  Kernel op =
      createKernel(ctx, createShader(kPuzzle10, {N, 1, 1}),
                   Bindings{a, b, output}, {1, 1, 1}, Params{N});
  showResult<1>(ctx, op, output);
}

// Puzzle 11 : Convolution
// Implement a kernel that computes a 1D convolution between a and b and stores it in out.
// You need to handle the general case. You only need 2 global reads and 1 global write per thread.

const char *kPuzzle11 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
struct Params {
  size: u32, // input is size x size
};
var<workgroup> shared_a: array<f32, 256>;
var<workgroup> shared_b: array<f32, 256>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x + GlobalInvocationID.y * params.size;
    let local_idx = GlobalInvocationID.x;

    // TBD
  }
)";

// TODO
// ...

int main(int argc, char **argv) {
  Context ctx = createContext();
  // puzzle1(ctx);
  // puzzle2(ctx);
  // puzzle3(ctx);
  // puzzle4(ctx);
  // puzzle5(ctx);
  // puzzle6(ctx);
  // puzzle7(ctx);
  // puzzle8(ctx);
  // puzzle9(ctx);
  puzzle10(ctx);
  return 0;
}
