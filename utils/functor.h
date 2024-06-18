#include <array>
#include <future>
#include <memory>
#include <random>

#include "gpu.h"

namespace gpu {

template <typename F>
std::string fmap1Shader(const F& func){
  std::string kShader = R"(
    @group(0) @binding(0) var<storage, read_write> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output : array<f32>;
    @compute @workgroup_size({{workgroupSize}})
    fn main(
      @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
        let idx = GlobalInvocationID.x;
        if (idx < arrayLength(&input)) {
          output[idx] = )" + func( "input[idx]" ) + R"(;
        }
      }
  )";
  return kShader;
}

#define fmap1(arg0,body) fmap1Shader([](const std::string& arg0) {return body;})

template <typename F>
std::string fmap2Shader(const F& func){
  std::string kShader = R"(
    @group(0) @binding(0) var<storage, read_write> input0: array<f32>;
    @group(0) @binding(1) var<storage, read_write> input1: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output : array<f32>;
    @compute @workgroup_size({{workgroupSize}})
    fn main(
      @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
        let idx = GlobalInvocationID.x;
        if (idx < arrayLength(&input0) && idx < arrayLength(&input1)) {
          output[idx] = )" + func( "input0[idx]", "input1[idx]" ) + R"(;
        }
      }
  )";
  return kShader;
}

#define fmap2(arg0,arg1,body) fmap2Shader([](const std::string& arg0, const std::string& arg1) {return body;})

template <typename F>
std::string foreach1Shader(const F& func){
  std::string kShader = R"(
    @group(0) @binding(0) var<storage, read_write> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output : array<f32>;
    @compute @workgroup_size({{workgroupSize}})
    fn main(
      @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
        let idx = GlobalInvocationID.x;
        if (idx < arrayLength(&input)) {
           )" + func( "input[idx]", "output[idx]" ) + R"(;
        }
      }
  )";
  return kShader;
}

#define foreach1(input0, output0, body) foreach1Shader([](const std::string& input0, const std::string& output0) {return body;})

void callKernel(GPUContext &ctx, const std::string& kShader, GPUTensor& in, GPUTensor& out, size_t N){
    Kernel op = CreateKernel(ctx, CreateShader(kShader, {N,1,1}, kf32),
			     in, out);
    DispatchKernel(ctx, op);
    Wait(ctx, op.future);
}

void callKernel(GPUContext &ctx, const std::string& kShader, GPUTensor& in0, GPUTensor& in1, GPUTensor& out, size_t N){
    Kernel op = CreateKernel(ctx, CreateShader(kShader, {N,1,1}, kf32),
			     GPUTensors{in0, in1}, out);
    DispatchKernel(ctx, op);
    Wait(ctx, op.future);
}

}
