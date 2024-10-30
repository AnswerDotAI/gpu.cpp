#include "gpu.hpp"
#include <array>
#include <cstdio>
#include <future>
#include <random>
#include <algorithm>
#include "utils/array_utils.hpp"    // show, isclose, randn, randint
#include "kernels.h"

using namespace gpu;

#define LIMITS { \
    .nextInChain = nullptr, \
    .limits = { \
      .maxTextureDimension1D=8192, \
      .maxTextureDimension2D=8192, \
      .maxTextureDimension3D=2048, \
      .maxTextureArrayLayers=256, \
      .maxBindGroups=4, \
      .maxBindGroupsPlusVertexBuffers=24, \
      .maxBindingsPerBindGroup=1000, \
      .maxDynamicUniformBuffersPerPipelineLayout=8, \
      .maxDynamicStorageBuffersPerPipelineLayout=4, \
      .maxSampledTexturesPerShaderStage=16, \
      .maxSamplersPerShaderStage=16, \
      .maxStorageBuffersPerShaderStage=8, \
      .maxStorageTexturesPerShaderStage=4, \
      .maxUniformBuffersPerShaderStage=12, \
      .maxUniformBufferBindingSize=65536, \
      .maxStorageBufferBindingSize=1073741824, \
      .minUniformBufferOffsetAlignment=256, \
      .minStorageBufferOffsetAlignment=256, \
      .maxVertexBuffers=8, \
      .maxBufferSize=0x80000000, \
      .maxVertexAttributes=16, \
      .maxVertexBufferArrayStride=2048, \
      .maxInterStageShaderComponents=64, \
      .maxInterStageShaderVariables=16, \
      .maxColorAttachments=8, \
      .maxColorAttachmentBytesPerSample=32, \
      .maxComputeWorkgroupStorageSize=16384, \
      .maxComputeInvocationsPerWorkgroup=1024, \
      .maxComputeWorkgroupSizeX=1024, \
      .maxComputeWorkgroupSizeY=1024, \
      .maxComputeWorkgroupSizeZ=64, \
      .maxComputeWorkgroupsPerDimension=65535 \
    } \
  }


struct DurationTime {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::chrono::microseconds duration;
  std::string src;
  bool verbose;
  int num;
  
  inline DurationTime(const std::string& src, bool verbose = true, int num = 1) {
    this->src = src;
    this->verbose = verbose;
    this->num = num;
    start = std::chrono::high_resolution_clock::now();
  }

  inline ~DurationTime() {
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (this->verbose) {
      printf("Duration(%s): %.1f microseconds\n", src.c_str(), static_cast<double>(duration.count()) / static_cast<double>(num));
    }
  }
};

static const char *kSumVersion1 = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
var<workgroup> buffer: array<{{precision}}, 1024>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>,
    @builtin(num_workgroups) numGroups : vec3<u32>) {
    let blockSize3d: vec3<u32> = vec3({{workgroupSize}});
    let blockSize: u32 = blockSize3d.x;
    let threadId: u32 = localID.x;
    let blockId: u32 = groupid.x + groupid.y * numGroups.x;
    let blockStart = blockId * blockSize * 2 + threadId;

    buffer[threadId] = inp[blockStart] + inp[blockStart + blockSize];
    workgroupBarrier();
 
    for (var stride: u32 = blockSize / 2; stride > 0; stride /= 2) {
        if (threadId < stride) {
            buffer[threadId] += buffer[threadId + stride];
        }
        workgroupBarrier();
    }
 
    if (threadId == 0) {
        out[blockId] = buffer[0];
    }
}
)";

static const char *kSumVersion2 = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
var<workgroup> buffer: array<{{precision}}, 1024>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>,
    @builtin(num_workgroups) numGroups : vec3<u32>) {
    let blockSize3d: vec3<u32> = vec3({{workgroupSize}});
    let blockSize: u32 = blockSize3d.x;
    let threadId: u32 = localID.x;
    let blockId: u32 = groupid.x + groupid.y * numGroups.x;
    let n: u32 = arrayLength(&inp);
    let blockStart = blockId * blockSize * 2 + threadId;

    buffer[threadId] = inp[blockStart] + inp[blockStart + blockSize];
    workgroupBarrier();
    var stride: u32 = blockSize / 2;
 
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();
   
    stride /= 2; // 1/4
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/8
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/16
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/32
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/64
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/128
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/256
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/512
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
    workgroupBarrier();

    stride /= 2; // 1/1024
    if (threadId < stride) {
        buffer[threadId] += buffer[threadId + stride];
    }
 
    if (threadId == 0) {
        out[blockId] = buffer[0];
    }
}
)";

static const char *kSum2d = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(2) var<uniform> params : Params;
struct Params {
    N: u32,
    C: u32,
};
var<workgroup> buffer: array<{{precision}}, 1024>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) globalID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>,
    @builtin(num_workgroups) numGroups : vec3<u32>) {
    let blockSize3d: vec3<u32> = vec3({{workgroupSize}});
    let blockSize: u32 = blockSize3d.x;
    let threadId: u32 = localID.x;
    let blockId: u32 = groupid.x + groupid.y * numGroups.x;
    let blockStart = blockId * blockSize * 2 + threadId;

    buffer[threadId] = inp[blockStart] + inp[blockStart + blockSize];
    workgroupBarrier();
    var stride: u32 = blockSize / 2;
 
    if (blockSize >= 1024 && threadId < 512) {
        buffer[threadId] += buffer[threadId + 512];
    }
    workgroupBarrier();
   
    if (blockSize >= 512 && threadId < 256) {
        buffer[threadId] += buffer[threadId + 256];
    }
    workgroupBarrier();

    if (blockSize >= 256 && threadId < 128) {
        buffer[threadId] += buffer[threadId + 128];
    }
    workgroupBarrier();

    if (threadId < 64) {
        buffer[threadId] += buffer[threadId + 64];
    }
    workgroupBarrier();

    if (threadId < 32) {
        buffer[threadId] += buffer[threadId + 32];
    }
    workgroupBarrier();

    if (threadId < 16) {
        buffer[threadId] += buffer[threadId + 16];
    }
    workgroupBarrier();

    if (threadId < 8) {
        buffer[threadId] += buffer[threadId + 8];
    }
    workgroupBarrier();

    if (threadId < 4) {
        buffer[threadId] += buffer[threadId + 4];
    }
    workgroupBarrier();

    if (threadId < 2) {
        buffer[threadId] += buffer[threadId + 2];
    }
    workgroupBarrier();

    if (threadId == 0) {
        buffer[0] += buffer[1];
        out[blockId] = buffer[0];
    }
}
)";

float sum_cpu(const float* data, size_t size) {
  float result = 0;
  for (size_t i = 0; i < size; ++i) {
    result += data[i];
  }
  return result;
}

Kernel createSumKernel(Context& ctx, Tensor& input, Tensor& output, size_t size) {
  uint32_t num_threads = 1024;
  uint32_t num_blocks =  ((size + num_threads -1) / num_threads);
  uint32_t size_x = 32768u < num_blocks ? 32768u : num_blocks;
  uint32_t size_y = size_x == 32768u ? num_blocks / 32768u : 1;
  size_x /= 2;
  size_x = size_x < 1 ? 1 : size_x;
  // print size_x, size_y
  // printf("size_x: %u, size_y: %u, num_blocks: %u\n", size_x, size_y, num_blocks);
  return createKernel(ctx, {kSum, num_threads, kf32}, Bindings{input, output}, {size_x, size_y, 1});
}

float sum_gpu(Context& ctx, const float* data, const float* buffer, size_t size) {
  WGPURequiredLimits requiredLimits = LIMITS;
  uint32_t num_threads = 1024;
  int nSum = round(log2(size) / log2(num_threads));
  int input_size = size;
  unsigned long output_size = size;
  std::vector<Tensor> outputs;
  std::vector<Kernel> ops;
  outputs.push_back(createTensor(ctx, Shape{std::max(size, static_cast<unsigned long>(1024*2))}, kf32));
  for(int i=size,j=0;i>0;i/=num_threads,j++){
    output_size = (output_size + num_threads - 1) / num_threads;
    outputs.push_back(createTensor(ctx, Shape{std::max(output_size, static_cast<unsigned long>(1024*2))}, kf32));
    ops.push_back(createSumKernel(ctx, outputs[j], outputs[j+1], input_size));
    // printf("size: %d\n", input_size);
    input_size = output_size;
  }
  toGPU(ctx, data, outputs[0], size * sizeof(float));


  {
    for(int i=size,j=0;i>0;i/=num_threads,j++){
      std::promise<void> promise;
      std::future<void> future = promise.get_future();
      dispatchKernel(ctx, ops[j], promise);
      wait(ctx, future);
      resetCommandBuffer(ctx.device, ops[j]);
    }
  }
  
  {
    int nIter = 100;
    DurationTime dt("GPU", true, nIter);
    for (int t = 0; t < nIter; t++){
      for(int i=size,j=0;i>0;i/=num_threads,j++){
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        dispatchKernel(ctx, ops[j], promise);
        wait(ctx, future);
        resetCommandBuffer(ctx.device, ops[j]);
      }
    }
  }

  float r = 0;
  toCPU(ctx, outputs[outputs.size()-1], (void*)buffer, 4 * sizeof(float));

  return buffer[0];
}

// float sum_gpu2d(Context& ctx, const float* data, const float* buffer, size_t size_x, size_t size_y) {
//   WGPURequiredLimits requiredLimits = LIMITS;
//   Tensor input = createTensor(ctx, Shape{size}, kf32, data);
//   Tensor output = createTensor(ctx, Shape{size}, kf32);
//   uint32_t num_threads = 1024;
//   uint32_t num_blocks =  ((size_x + num_threads -1) / num_threads);
//   printf("size: %u, size_x: %u, size_y: %u\n", size, size_x, size_y);
//   Kernel op = createKernel(ctx, {kSum, num_threads, kf32}, Bindings{input, output}, {size_x, size_y, 1});
// 
//   {
//     for (int i = 0; i < 100; ++i){
//       DurationTime dt("GPU");
//       std::promise<void> promise;
//       std::future<void> future = promise.get_future();
//       dispatchKernel(ctx, op, promise);
//       wait(ctx, future);
//       resetCommandBuffer(ctx.device, op);
//     }
//   }
// 
//   float r = 0;
//   toCPU(ctx, output, (void*)buffer, num_blocks * sizeof(float));
// 
//   for (int i = 0; i < num_blocks; i++){
//     r+=buffer[i];
//   }
//   return r;
// }

int main(int argc, char **argv) {
  static constexpr size_t M = 4096*2;
  static constexpr size_t N = 4096*2;
  static constexpr size_t BUF_SIZE = 16;
  std::unique_ptr<float[]> inputArr = std::make_unique<float[]>(M * N);
  std::unique_ptr<float[]> buffer = std::make_unique<float[]>(BUF_SIZE);
  std::mt19937 gen(314159);
  printf("Initializing %zu values\n", M*N);
  randn(inputArr.get(), M*N, gen);
  // for(int i=0;i<M*N;i++) {
  //   inputArr[i] = 1;
  // }
  for(int i=0;i<BUF_SIZE;i++) {
    buffer[i] = 0;
  }
  float cpu_result;
  float gpu_result;
  WGPURequiredLimits requiredLimits = LIMITS;
  gpu::Context ctx = gpu::createContext({}, {}, {
      .requiredLimits = &requiredLimits
  });
  Tensor input = createTensor(ctx, Shape{M*N}, kf32, inputArr.get());

  printf("Start testing sum(x) on %zu values\n", M*N);
  cpu_result = sum_cpu(inputArr.get(), M*N);
  {
    int nIter = 100;
    DurationTime dt("CPU", true, nIter);
    for (int i = 0; i < nIter; ++i){
      cpu_result = sum_cpu(inputArr.get(), M*N);
    }
  }
  printf("sum_cpu: %.6f\n", cpu_result);

  printf("Start testing sum(x) on %zu values\n", M*N);
  {
    gpu_result = sum_gpu(ctx, inputArr.get(), buffer.get(), M*N);
    printf("sum_gpu: %.6f\n", gpu_result);
  }
  // Compare cpu_result with gpu_result
  float diff = fabs(cpu_result - gpu_result);
  if (diff >= 1e-0f) {
    printf("Error: diff = %.6f\n", diff);
  } else {
    printf("Success: diff = %.6f\n", diff);
  }
  
  printf("Computed %zu values of kSum(x)\n\n", M*N);
  return 0;
}
