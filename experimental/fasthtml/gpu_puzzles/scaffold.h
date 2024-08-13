#include "gpu.h"
#include "utils/array_utils.h"
#include <array>
#include <cstdio>
#include <future>

using namespace gpu;

static constexpr size_t N = 10;

// TODO: use these on a per-puzzle basis instead of the run.cpp host code.

template <size_t N> std::array<float, N> makeData(bool transpose = false) {
  std::array<float, N> inputArr;

  for (size_t i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i); // dummy input data
  }
  if (transpose) {
    static size_t n_sqrt = static_cast<size_t>(std::sqrt(N));
    printf("Transposing %zu x %zu array\n", n_sqrt, n_sqrt);
    std::array<float, N> transposedArr;
    for (size_t row = 0; row < n_sqrt; ++row) {
      for (size_t col = 0; col < n_sqrt; ++col) {
        transposedArr[col * n_sqrt + row] = inputArr[row * n_sqrt + col];
      }
    }
    inputArr = transposedArr;
  }
  return inputArr;
}

template <size_t N, size_t R = N, size_t C = 1>
void showResult(Context &ctx, Kernel &op, Tensor &output) {
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
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>) {
    let local_idx = LocalInvocationID.x;
      output[local_idx] = a[local_idx] + 10;
  }
)";

void puzzle1(Context &ctx, const char *code) {
  printf("\n\nPuzzle 1\n\n");
  static constexpr size_t N = 4;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  Kernel op = createKernel(ctx, {code, N}, Bindings{a, output},
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
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>) {
    let local_idx = LocalInvocationID.x;
    output[local_idx] = a[local_idx] + b[local_idx];
  }
)";

void puzzle2(Context &ctx, const char *code) {
  printf("\n\nPuzzle 2\n\n");
  static constexpr size_t N = 4;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor b = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  Kernel op = createKernel(ctx, {code, N}, Bindings{a, b, output}, {1, 1, 1});
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
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>) {
    let local_idx = LocalInvocationID.x;
    if (local_idx < arrayLength(&input)) {
      output[local_idx] = input[local_idx] + 10;
    }
  }
)";
void puzzle3(Context &ctx, const char *code) {
  printf("\n\nPuzzle 3\n\n");
  static constexpr size_t N = 8;
  static constexpr size_t EXTRAS = 3;
  Tensor input = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  Kernel op =
      createKernel(ctx, {code, N + EXTRAS}, Bindings{input, output}, {1, 1, 1});
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
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  ) {
    let local_i = LocalInvocationID.x;
    let local_j = LocalInvocationID.y;

    if (local_i < params.size && local_j < params.size) {
      let idx = local_i + local_j * params.size;
      output[idx] = input[idx] + 10;
    }
  }
)";
void puzzle4(Context &ctx, const char *code) {
  printf("\n\nPuzzle 4\n\n");
  static constexpr size_t Tx = 3; // Threads along the x-axis
  static constexpr size_t Ty = 3; // Threads along the y-axis
  static constexpr size_t N = 2;
  Tensor input = createTensor(ctx, {N, N}, kf32, makeData<N * N>().data());
  Tensor output = createTensor(ctx, {N, N}, kf32);
  struct Params {
    uint32_t size = N;
  };
  Kernel op = createKernel(ctx, {code, /*workgroup size*/ {Tx, Ty, 1}},
                           Bindings{input, output}, /* nWorkgroups */ {1, 1, 1},
                           Params{N});
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
void puzzle5(Context &ctx, const char *code) {
  printf("\n\nPuzzle 5\n\n");
  static constexpr size_t N = 2;
  static constexpr size_t Tx = 3;
  Tensor a = createTensor(ctx, {N, 1}, kf32, makeData<N>().data());
  Tensor b = createTensor(ctx, {1, N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N, N}, kf32);
  struct Params {
    uint32_t size = N;
  };

  Kernel op = createKernel(ctx, {code, /*workgroup size*/ {Tx, Tx, 1}},
                           Bindings{a, b, output}, {1, 1, 1}, Params{N});
  showResult<N, N, N>(ctx, op, output);
}

// Puzzle 6 : Blocks
// Implement a kernel that adds 10 to each position of a and stores it in out.
// You have fewer threads per block than the size of a but multiple workgroups.

const char *kPuzzle6 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
  ) {
    let idx = GlobalInvocationID.x;

    if (idx < arrayLength(&a)) {
      output[idx] = a[idx] + 10;
    }
  }
)";
void puzzle6(Context &ctx, const char *code) {
  printf("\n\nPuzzle 6\n\n");
  static constexpr size_t N = 11;
  static constexpr size_t Tx = 4; // Threads along the x-axis
  static constexpr size_t Wx = 3; // Workgroups along the x-axis
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);

  Kernel op =
      createKernel(ctx, {code, {Tx, 1, 1}}, Bindings{a, output}, {Wx, 1, 1});
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
    let idx_i = GlobalInvocationID.x;
    let idx_j = GlobalInvocationID.y;

    if (idx_i < params.size && idx_j < params.size) {
      let idx = idx_i + idx_j * params.size;
      output[idx] = a[idx] + 10;
    }
  }
)";
void puzzle7(Context &ctx, const char *code) {
  printf("\n\nPuzzle 7\n\n");
  static constexpr size_t N = 5;
  static constexpr size_t Tx = 3;
  static constexpr size_t Wx = 2;
  Tensor a = createTensor(ctx, {N, N}, kf32, makeData<N * N>().data());
  Tensor output = createTensor(ctx, {N, N}, kf32);
  struct Params {
    uint32_t size;
  };

  Kernel op = createKernel(ctx, {code, {Tx, Tx, 1}}, Bindings{a, output},
                           {Wx, Wx, 1}, Params{N});
  showResult<N, N, N>(ctx, op, output);
}

// Puzzle 8 : Shared
// Implement a kernel that adds 10 to each position of a and stores it in out.
// You have fewer threads per block than the size of a.

// (This example does not really need shared memory or syncthreads, but it is a
// demo.)

const char *kPuzzle8 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
var<workgroup> sharedData: array<f32, 256>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>,
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>
  ) {
    let idx = GlobalInvocationID.x;
    let local_idx = LocalInvocationID.x;

    if (idx < arrayLength(&a)) {
      sharedData[local_idx] = a[idx];
    }

    workgroupBarrier();
   
    if (idx < arrayLength(&a)) {
      output[idx] = sharedData[local_idx] + 10;
    }
  }
)";
void puzzle8(Context &ctx, const char *code) {
  printf("\n\nPuzzle 8\n\n");
  static constexpr size_t N = 8;
  static constexpr size_t Tx = 4;
  static constexpr size_t Wx = 2;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  Kernel op =
      createKernel(ctx, {code, {Tx, 1, 1}}, Bindings{a, output}, {Wx, 1, 1});
  showResult<N>(ctx, op, output);
}

// Puzzle 9 : Pooling
// Implement a kernel that sums together the last 3 position of a and stores it
// in out. You have 1 thread per position. You only need 1 global read and 1
// global write per thread.

inline KernelCode createCustomSharedMemory(
    const char *shaderTemplate, const size_t shared_memory_size,
    const Shape &workgroupSize = {256, 1, 1}, NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString,
             {{"{{workgroupSize}}", toString(workgroupSize)},
              {"{{precision}}", toString(precision)},
              {"{{sharedMemorySize}}", toString(shared_memory_size)}});
  return {codeString, workgroupSize};
}

const char *kPuzzle9 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
var<workgroup> sharedData: array<f32, {{sharedMemorySize}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let local_idx = LocalInvocationID.x;

    if (idx < arrayLength(&a)) {
      sharedData[local_idx] = a[idx];
    }

    workgroupBarrier();

    if (idx == 0) {
      output[idx] = sharedData[local_idx];
    }
    else if (idx == 1) {
      output[idx] = sharedData[local_idx] + sharedData[local_idx - 1];
    }
    else {
      output[idx] = sharedData[local_idx] + sharedData[local_idx - 1] + sharedData[local_idx - 2];
    }
  }
)";
void puzzle9(Context &ctx, const char *codeTemplate) {
  printf("\n\nPuzzle 9\n\n");
  static constexpr size_t N = 8;
  static constexpr size_t Tx = 8;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {N}, kf32);

  KernelCode code = createCustomSharedMemory(codeTemplate, Tx, {Tx, 1, 1});

  Kernel op = createKernel(ctx, code, Bindings{a, output}, {1, 1, 1});

  showResult<N>(ctx, op, output);
}

// Puzzle 10 : Dot Product
// Implement a kernel that computes the dot-product of a and b and stores it in
// out. You have 1 thread per position. You only need 2 global reads and 1
// global write per thread.

const char *kPuzzle10 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
var<workgroup> sharedData: array<f32, {{sharedMemorySize}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let local_idx = LocalInvocationID.x;

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
void puzzle10(Context &ctx, const char *codeTemplate) {
  printf("\n\nPuzzle 10\n\n");
  static constexpr size_t N = 8;
  static constexpr size_t Tx = 8;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor b = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {1}, kf32);

  KernelCode code = createCustomSharedMemory(codeTemplate, Tx, {Tx, 1, 1});

  Kernel op = createKernel(ctx, code, Bindings{a, b, output}, {1, 1, 1});
  showResult<1>(ctx, op, output);
}

// Puzzle 11 : Convolution
// Implement a kernel that computes a 1D convolution between a and b and stores
// it in out. You need to handle the general case. You only need 2 global reads
// and 1 global write per thread.

const char *kPuzzle11 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
struct Params {
  TPB: u32,
};
var<workgroup> shared_a: array<f32, {{sharedMemorySize}}>;
var<workgroup> shared_b: array<f32, {{sharedMemorySize}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let local_idx = LocalInvocationID.x;

    if (idx < arrayLength(&a)) {
      shared_a[local_idx] = a[idx];   
    }

    if (local_idx < arrayLength(&b)) {
      shared_b[local_idx] = b[local_idx];   
    } 
    else {
      let local_idx2 = local_idx - arrayLength(&b);
      let idx2 = idx - arrayLength(&b);
      if ((idx2 + params.TPB < arrayLength(&a)) && (local_idx2 < arrayLength(&b))) {
        shared_a[local_idx2 + params.TPB] = a[idx2 + params.TPB];
      }
    }

    workgroupBarrier();

    var acc = 0.0;

    for (var i: u32 = 0u; i < arrayLength(&b); i = i + 1u) {
      if (idx + i < arrayLength(&a)) {
        acc = acc + shared_a[local_idx + i] * shared_b[i];
      }
    }

    if (idx < arrayLength(&a)) {
      output[idx] = acc;
    }
  }
)";
void puzzle11(Context &ctx, const char *codeTemplate) {
  printf("\n\nPuzzle 11\n\n");
  static constexpr size_t N = 15;
  static constexpr size_t CONV = 4;
  static constexpr size_t Tx = 8;
  static constexpr size_t Wx = 2;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor b = createTensor(ctx, {CONV}, kf32, makeData<CONV>().data());
  Tensor output = createTensor(ctx, {N}, kf32);
  struct Params {
    uint32_t TPB = Tx;
  };

  KernelCode code =
      createCustomSharedMemory(codeTemplate, Tx + CONV, {Tx, 1, 1});

  Kernel op =
      createKernel(ctx, code, Bindings{a, b, output}, {Wx, 1, 1}, Params{Tx});
  showResult<N>(ctx, op, output);
}

// Puzzle 12 : Prefix Sum
// Implement a kernel that computes a sum over a and stores it in out.
// We will do this using the parallel prefix sum algorithm in shared memory.
// That is, each step of the algorithm should sum together half the remaining
// numbers.

const char *kPuzzle12 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
var<workgroup> cache: array<f32, {{sharedMemorySize}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let idx = GlobalInvocationID.x;
    let local_idx = LocalInvocationID.x;

    if (idx < arrayLength(&a)) {
      cache[local_idx] = a[idx];
    }
    else {
      cache[local_idx] = 0.0;
    }

    workgroupBarrier();

    if (local_idx % 2 == 0) {
      cache[local_idx] = cache[local_idx] + cache[local_idx + 1];
    }
    workgroupBarrier();
    if (local_idx % 4 == 0) {
      cache[local_idx] = cache[local_idx] + cache[local_idx + 2];
    }
    workgroupBarrier();
    if (local_idx % 8 == 0) {
      cache[local_idx] = cache[local_idx] + cache[local_idx + 4];
    }
    workgroupBarrier();
    if (local_idx == 0) {
      output[idx] = cache[local_idx];
    }
  }
)";
void puzzle12(Context &ctx, const char *codeTemplate) {
  printf("\n\nPuzzle 12\n\n");
  static constexpr size_t N = 15;
  static constexpr size_t Tx = 8;
  Tensor a = createTensor(ctx, {N}, kf32, makeData<N>().data());
  Tensor output = createTensor(ctx, {2}, kf32);

  KernelCode code = createCustomSharedMemory(codeTemplate, Tx, {Tx, 1, 1});

  Kernel op = createKernel(ctx, code, Bindings{a, output}, {2, 1, 1});
  showResult<2>(ctx, op, output);
}

// Puzzle 13 : Axis Sum
// Implement a kernel that computes a sum over each column of a and stores it in
// out.

const char *kPuzzle13 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
  TPB: u32,
  size: u32, // input is BATCH x size
};

var<workgroup> cache: array<f32, {{sharedMemorySize}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i = GlobalInvocationID.x;
    let local_i = LocalInvocationID.x;
    let batch = GlobalInvocationID.y;

    if (i < params.size) {
        // Copy and sync
        cache[local_i] = a[batch * params.size + i];
        
    }

    workgroupBarrier();


    // Sum over each col
    if (i < params.size) {
      for (var k: u32 = 0u; k < 3u; k = k + 1u) {
          let p = 1u << k;
          if (local_i % (p * 2u) == 0u && i + p < params.size) {
              cache[local_i] = cache[local_i] + cache[local_i + p];
          }
      }
    }

    workgroupBarrier();

    // Each block corresponds to a different output position
    if (local_i == 0u) {
        output[batch] = cache[0];
    }
}
)";
void puzzle13(Context &ctx, const char *codeTemplate) {
  printf("\n\nPuzzle 13\n\n");
  static constexpr size_t N = 6;
  static constexpr size_t TPB = 8;
  static constexpr size_t BATCH = 4;
  Tensor a = createTensor(ctx, {BATCH, N}, kf32, makeData<N * BATCH>().data());
  Tensor output = createTensor(ctx, {BATCH}, kf32);
  struct Params {
    uint32_t TPB = TPB;
    uint32_t size = N;
  };

  KernelCode code = createCustomSharedMemory(codeTemplate, TPB, {TPB, 1, 1});

  Kernel op = createKernel(ctx, code, Bindings{a, output}, {1, BATCH, 1},
                           Params{TPB, N});
  showResult<BATCH>(ctx, op, output);
}

// Puzzle 14 : Matrix Multiply!!
// Implement a kernel that computes the matrix product of a and b and stores it
// in out. Tip: The most efficient algorithm here will copy a block into shared
// memory before computing each of the individual row-column dot products. This
// is easy to do if the matrix fits in shared memory. Do that case first. Then
// update your code to compute a partial dot-product and iteratively move the
// part you copied into shared memory. You should be able to do the hard case in
// 6 global reads.
const char *kPuzzle14 = R"(
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
  TPB: u32,
  size: u32,
};

var<workgroup> a_shared: array<f32, {{sharedMemorySize}}>;
var<workgroup> b_shared: array<f32, {{sharedMemorySize}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i = GlobalInvocationID.x;
    let j = GlobalInvocationID.y;
    let local_i = LocalInvocationID.x;
    let local_j = LocalInvocationID.y;

    var acc: f32 = 0.0;

    for (var k: u32 = 0u; k < params.size; k = k + params.TPB) {
      if (i < params.size && k + local_i < params.size) {
        a_shared[local_i + local_j * params.TPB] = a[j * params.size + (k + local_i)];
      }
      if (j < params.size && k + local_j < params.size) {
        b_shared[local_i + local_j * params.TPB] = b[i + (k + local_j) * params.size];
      }

      workgroupBarrier();

      let local_k_max = min(params.TPB, params.size - k);
      for (var local_k: u32 = 0u; local_k < local_k_max; local_k = local_k + 1u) {
          acc += a_shared[local_j * params.TPB + local_k] * b_shared[local_k * params.TPB + local_i];
      }
    }

    // Copy to out
    if (i < params.size && j < params.size) {
        output[i + j * params.size] = acc;
    }
    
}
)";
void puzzle14(Context &ctx, const char *codeTemplate) {
  printf("\n\nPuzzle 14\n\n");
  static constexpr size_t N = 8;
  static constexpr size_t TPB = 10;
  Tensor a = createTensor(ctx, {N, N}, kf32, makeData<N * N>().data());
  Tensor b = createTensor(ctx, {N, N}, kf32, makeData<N * N>(true).data());
  Tensor output = createTensor(ctx, {N, N}, kf32);
  struct Params {
    uint32_t TPB = TPB;
    uint32_t size = N;
  };

  KernelCode code =
      createCustomSharedMemory(codeTemplate, TPB * TPB, {TPB, TPB, 1});

  Kernel op = createKernel(ctx, code, Bindings{a, b, output}, {3, 3, 1},
                           Params{TPB, N});
  showResult<N, N, N>(ctx, op, output);
}

void puzzleDispatch(Context &ctx, const char *code, int puzzleIndex) {
  switch (puzzleIndex) {
  case 1:
    puzzle1(ctx, code);
    break;
  case 2:
    puzzle2(ctx, code);
    break;
  case 3:
    puzzle3(ctx, code);
    break;
  case 4:
    puzzle4(ctx, code);
    break;
  case 5:
    puzzle5(ctx, code);
    break;
  case 6:
    puzzle6(ctx, code);
    break;
  case 7:
    puzzle7(ctx, code);
    break;
  case 8:
    puzzle8(ctx, code);
    break;
  case 9:
    puzzle9(ctx, code);
    break;
  case 10:
    puzzle10(ctx, code);
    break;
  case 11:
    puzzle11(ctx, code);
    break;
  case 12:
    puzzle12(ctx, code);
    break;
  case 13:
    puzzle13(ctx, code);
    break;
  case 14:
    puzzle14(ctx, code);
    break;
  default:
    printf("Invalid puzzle index\n");
  }
}
