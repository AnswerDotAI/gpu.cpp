#include "gpu.h"
#include <array>
#include <cstdio>
#include <future>

using namespace gpu; // createContext, createTensor, createKernel,
                     // createShader, dispatchKernel, wait, toCPU
                     // Tensor, Kernel, Context, Shape, kf32

static const char *kTan = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f32 = inp[i];
        out[i] = tan(x);
    }
}
)";

int main(int argc, char **argv) {
  printf("\033[2J\033[1;1H");
  printf("\nHello gpu.cpp!\n");
  printf("--------------\n\n");

  Context ctx = createContext();
  static constexpr size_t N = 100000;
  std::array<float, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = static_cast<float>(i) / 10.0; // dummy input data
  }
  Tensor input = createTensor(ctx, Shape{N}, kf32, inputArr.data());
  Tensor output = createTensor(ctx, Shape{N}, kf32);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kTan, 256, kf32},
                           Bindings{input, output},
                           /* nWorkgroups */ {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 1000; ++i) {
    printf("  tan(%.2f) = %.10f\n", inputArr[i], outputArr[i]);
  }
  printf("  ...\n\n");
  printf("Computed %zu values of tan(x)\n\n", N);
  return 0;
}