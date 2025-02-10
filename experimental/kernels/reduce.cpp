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
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(workgroup_id) groupid : vec3<u32>,
    @builtin(num_workgroups) numGroups : vec3<u32>) {
    let N : u32 = params.N;
    let C : u32 = params.C;
    let blockSize3d: vec3<u32> = vec3({{workgroupSize}});
    let blockSize: u32 = blockSize3d.x;
    let threadId: u32 = localID.x;
    let blockId: u32 = groupid.x + groupid.y * numGroups.x;

    for (var i: u32 = 0; i<C ; i++) {
        let blockStart = blockId * blockSize * 2 + threadId;
        if(blockStart >= N) {
        } else if(blockStart + blockSize >= N) {
            buffer[threadId] = inp[blockStart * C + i];
        } else {
            buffer[threadId] = inp[blockStart * C + i] + inp[(blockStart + blockSize) * C + i];
        }
        workgroupBarrier();
        
        for (var stride: u32 = blockSize / 2; stride > 0; stride /= 2) {
            if (threadId < stride) {
                 buffer[threadId] += buffer[threadId + stride];
            }
            workgroupBarrier();
        }
        
        if (threadId == 0) {
            out[blockId * C + i] = buffer[0];
        }
        workgroupBarrier();
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

void sum_cpu_2d(const float* data, float* out, size_t size0, size_t size1) {
  float result = 0;
  for (size_t j = 0; j < size1; ++j) {
    out[j] = 0;
  }
  for (size_t i = 0; i < size0; ++i) {
    for (size_t j = 0; j < size1; ++j) {
      out[j] += data[(i * size1) + j];
    }
  }
}

Kernel createSumKernel(Context& ctx, Tensor& input, Tensor& output, size_t size, uint32_t num_threads = 1024) {
  uint32_t num_blocks =  ((size + num_threads -1) / num_threads);
  uint32_t size_x = 32768u < num_blocks ? 32768u : num_blocks;
  uint32_t size_y = size_x == 32768u ? num_blocks / 32768u : 1;
  size_x /= 2;
  size_x = size_x < 1 ? 1 : size_x;
  // print size_x, size_y
  printf("size_x: %u, size_y: %u, num_blocks: %u\n", size_x, size_y, num_blocks);
  return createKernel(ctx, {kSum, num_threads, kf32}, Bindings{input, output}, {size_x, size_y, 1});
}

Kernel createSumKernel2d(Context& ctx, Tensor& input, Tensor& output, size_t size0, size_t size1, uint32_t num_threads = 1024) {
  struct Params {
    uint32_t N;
    uint32_t C;
  };
  uint32_t num_blocks =  ((size0 + num_threads -1) / num_threads);
  uint32_t size_x = num_blocks;
  uint32_t size_y = size1;
  size_x /= 2;
  size_x = size_x < 1 ? 1 : size_x;
  printf("size_x: %u, size_y: %u, num_blocks: %u\n", size_x, size_y, num_blocks);
  return createKernel(ctx,
                      {kSum2d, num_threads, kf32},
                      Bindings{input, output},
                      {size_x, size_y, 1},
                      Params{
                        static_cast<uint32_t>(size0),
                        static_cast<uint32_t>(size1),
                      });
}

struct SumKernel {
  std::vector<Tensor> outputs;
  std::vector<Kernel> ops;
  SumKernel(Context& ctx, size_t size, uint32_t num_threads = 1024) {
    int input_size = size;
    unsigned long output_size = size;
    outputs.push_back(createTensor(ctx, Shape{std::max(size, static_cast<unsigned long>(num_threads*2))}, kf32));
    for(int j=0;output_size>1;j++){
      output_size = (output_size + (num_threads * 2) - 1) / (num_threads * 2);
      outputs.push_back(createTensor(ctx, Shape{std::max(output_size, static_cast<unsigned long>(num_threads*2))}, kf32));
      ops.push_back(createSumKernel(ctx, outputs[j], outputs[j+1], input_size, num_threads));
      input_size = output_size;
    }
  }
  void dispatchKernel(Context& ctx) {
    for(int i=0;i<ops.size();i++){
      std::promise<void> promise;
      std::future<void> future = promise.get_future();
      gpu::dispatchKernel(ctx, ops[i], promise);
      wait(ctx, future);
      resetCommandBuffer(ctx.device, ops[i]);
    }
  }
  void toGPU(Context& ctx, const float* data, size_t size) {
    gpu::toGPU(ctx, data, outputs[0], size);
  }
  void toCPU(Context& ctx, float* data, size_t size) {
    gpu::toCPU(ctx, outputs[outputs.size()-1], data, size);
  }
};

struct SumKernel2d {
  std::vector<Tensor> outputs;
  std::vector<Kernel> ops;
  bool debug;
  SumKernel2d(Context& ctx, size_t size0, size_t size1, uint32_t num_threads = 1024) {
    debug = false;
    int input_size = size0;
    unsigned long output_size = size0;
    outputs.push_back(createTensor(ctx, Shape{std::max(size0, static_cast<unsigned long>(num_threads*2)),size1}, kf32));
    for(int j=0;output_size>1;j++){
      output_size = (output_size + (num_threads * 2) - 1) / (num_threads * 2);
      if (debug)
        printf("size0: %zu, num_threads: %d, output_size: %lu\n", size0, num_threads, output_size);
      outputs.push_back(createTensor(ctx, Shape{std::max(output_size, static_cast<unsigned long>(num_threads*2)), size1}, kf32));
      ops.push_back(createSumKernel2d(ctx, outputs[j], outputs[j+1], input_size, size1, num_threads));
      input_size = output_size;
    }
    if (debug)
      printf("ops.size(): %zu\n", ops.size());
  }
  void dispatchKernel(Context& ctx) {
    for(int i=0;i<ops.size();i++){
      std::promise<void> promise;
      std::future<void> future = promise.get_future();
      gpu::dispatchKernel(ctx, ops[i], promise);
      wait(ctx, future);
      resetCommandBuffer(ctx.device, ops[i]);
    }
    if (debug) {
      std::unique_ptr<float[]> buffer = std::make_unique<float[]>(8);
      for(int i=0;i<outputs.size();i++){
        gpu::toCPU(ctx, outputs[i], buffer.get(), 8*sizeof(float));
        printf("outputs[%d]: ", i);
        for (int j = 0; j < 8; j++) {
          printf("%.6f ", buffer[j]);
        }
        printf("\n");
      }
    }
  }
  void toGPU(Context& ctx, const float* data, size_t size) {
    gpu::toGPU(ctx, data, outputs[0], size);
  }
  void toCPU(Context& ctx, float* data, size_t size) {
    gpu::toCPU(ctx, outputs[outputs.size()-1], data, size);
  }
};

float sum_gpu(Context& ctx, const float* data, float* buffer, size_t size) {
  WGPURequiredLimits requiredLimits = LIMITS;
  SumKernel sumKernel(ctx, size);
  sumKernel.toGPU(ctx, data, size * sizeof(float));
  sumKernel.dispatchKernel(ctx);
  
  {
    int nIter = 100;
    DurationTime dt("GPU", true, nIter);
    for (int t = 0; t < nIter; t++){
      sumKernel.dispatchKernel(ctx);
    }
  }

  float r = 0;
  sumKernel.toCPU(ctx, buffer, 4 * sizeof(float));

  return buffer[0];
}

void sum_gpu_2d(Context& ctx, const float* data, float* out, size_t size0, size_t size1) {
  WGPURequiredLimits requiredLimits = LIMITS;
  SumKernel2d sumKernel(ctx, size0, size1);
  sumKernel.toGPU(ctx, data, size0 * size1 * sizeof(float));
  sumKernel.dispatchKernel(ctx);
  
  {
    int nIter = 3;
    DurationTime dt("GPU", true, nIter);
    for (int t = 0; t < nIter; t++){
      sumKernel.dispatchKernel(ctx);
    }
  }

  sumKernel.toCPU(ctx, out, size1 * sizeof(float));
}

int main_1d(int argc, char **argv) {
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

int main_2d(int argc, char **argv) {
  static constexpr size_t M = 4096;
  static constexpr size_t N = 4096;
  std::unique_ptr<float[]> inputArr = std::make_unique<float[]>(M * N);
  std::unique_ptr<float[]> outputCpuArr = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> outputGpuArr = std::make_unique<float[]>(N);
  std::mt19937 gen(314159);
  printf("Initializing %zu values\n", M*N);
  randn(inputArr.get(), M*N, gen);
  for(int i=0;i<M*N;i++) {
    inputArr[i] = 1 + i%2;
  }
  for(int i=0;i<N;i++) {
    outputCpuArr[i] = 0;
    outputGpuArr[i] = 0;
  }
  WGPURequiredLimits requiredLimits = LIMITS;
  gpu::Context ctx = gpu::createContext({}, {}, {
      .requiredLimits = &requiredLimits
  });

  printf("Start testing sum2d(x) on %zu values\n", M*N);
  sum_cpu_2d(inputArr.get(), outputCpuArr.get(), M, N);
  {
    int nIter = 100;
    DurationTime dt("CPU", true, nIter);
    for (int i = 0; i < nIter; ++i){
      sum_cpu_2d(inputArr.get(), outputCpuArr.get(), M, N);
    }
  }

  printf("Start testing sum2d(x) on %zu values\n", M*N);
  {
    sum_gpu_2d(ctx, inputArr.get(), outputGpuArr.get(), M, N);
  }

  // Compare cpu_result with gpu_result
  float diff = 0;
  for(int i=0;i<N;i++) {
    diff += fabs(outputCpuArr[i] - outputGpuArr[i]);
  }
  
  if (diff >= 1e-0f) {
    printf("Error: diff = %.6f\n", diff);
  } else {
    printf("Success: diff = %.6f\n", diff);
  }
  
  return 0;
}

int main(int argc, char **argv) {
  printf("================================\n");
  printf("Start testing reduce-1d\n");
  main_1d(argc,argv);
  printf("================================\n");
  printf("Start testing reduce-2d\n");
  main_2d(argc,argv);
  return 0;
}
