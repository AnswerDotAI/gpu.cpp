#include "gpu.hpp"
#include "numeric_types/half.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace gpu;

static constexpr auto twicePlusOne = R"(
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&input)) {
    output[id.x] = input[id.x] * 2 + 1;
  }
}
)";

static constexpr auto doubleF16 = R"(
@group(0) @binding(0) var<storage, read> input: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> output: array<{{precision}}>;

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&input)) {
    output[id.x] = input[id.x] * 2.0;
  }
}
)";

static std::vector<uint32_t> readSPIRV(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("could not open assembled SPIR-V test");
  const auto bytes = file.tellg();
  if (bytes <= 0 || bytes % sizeof(uint32_t))
    throw std::runtime_error("assembled SPIR-V test has an invalid size");
  std::vector<uint32_t> code(bytes / sizeof(uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(code.data()), bytes);
  return code;
}

template <typename T>
static void expect(const std::vector<T> &actual,
                   const std::vector<T> &expected, const char *story) {
  if (actual != expected) throw std::runtime_error(std::string(story) + " failed");
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3)
    throw std::runtime_error("expected assembled SPIR-V path");

  // Ordinary WGSL covers upload, explicit binding access, dispatch, and readback.
  auto context = createContext();
  std::vector<int32_t> input{1, 2, 3, 4};
  std::vector<int32_t> output(input.size());
  auto gpuInput = createTensor(context, {input.size()}, ki32, input);
  auto gpuOutput = createTensor(context, {output.size()}, ki32);
  auto wgsl = createKernel(context, WGSL{twicePlusOne, 4},
                           Bindings{read(gpuInput), readWrite(gpuOutput)});
  auto dispatched = dispatchKernel(context, wgsl);
  wait(context, dispatched);
  auto downloaded = toCPU(context, gpuOutput, output);
  wait(context, downloaded);
  expect(output, std::vector<int32_t>{3, 5, 7, 9}, "WGSL compute");

  // Native IEEE binary16 values round-trip through an f16 WGSL pipeline.
  auto f16Context = createContext(
      {.requiredFeatures = {wgpu::FeatureName::ShaderF16}});
  std::vector<half> f16Input{half(0.5f), half(-2.0f), half(3.25f),
                             half(10.0f)};
  std::vector<half> f16Output(f16Input.size());
  auto gpuF16Input = createTensor(f16Context, {f16Input.size()}, kf16, f16Input);
  auto gpuF16Output = createTensor(f16Context, {f16Output.size()}, kf16);
  auto f16Kernel =
      createKernel(f16Context, WGSL{doubleF16, 4, kf16},
                   Bindings{read(gpuF16Input), readWrite(gpuF16Output)});
  auto f16Dispatched = dispatchKernel(f16Context, f16Kernel);
  wait(f16Context, f16Dispatched);
  auto f16Downloaded = toCPU(f16Context, gpuF16Output, f16Output);
  wait(f16Context, f16Downloaded);
  expect(f16Output,
         std::vector<half>{half(1.0f), half(-4.0f), half(6.5f), half(20.0f)},
         "f16 compute");

  // The same runtime accepts SPIR-V conforming to gpu.cpp's WebGPU profile.
  auto spirvContext = createContext({.enableSPIRV = true});
  std::vector<int32_t> answer(1);
  auto gpuAnswer = createTensor(spirvContext, {1}, ki32);
  auto spirv = createKernel(spirvContext, SPIRV{readSPIRV(argv[1])},
                            Bindings{readWrite(gpuAnswer)});
  auto spirvDispatched = dispatchKernel(spirvContext, spirv);
  wait(spirvContext, spirvDispatched);
  auto answerDownloaded = toCPU(spirvContext, gpuAnswer, answer);
  wait(spirvContext, answerDownloaded);
  expect(answer, std::vector<int32_t>{42}, "SPIR-V compute");

  if (argc == 3) {
    std::vector<int32_t> externalAnswer(1);
    auto externalOutput = createTensor(spirvContext, {1}, ki32);
    auto external = createKernel(
        spirvContext, SPIRV{readSPIRV(argv[2]), "external SPIR-V", "main"},
        Bindings{readWrite(externalOutput)});
    auto externalDispatched = dispatchKernel(spirvContext, external);
    wait(spirvContext, externalDispatched);
    auto externalDownloaded =
        toCPU(spirvContext, externalOutput, externalAnswer);
    wait(spirvContext, externalDownloaded);
    expect(externalAnswer, std::vector<int32_t>{42}, "external SPIR-V compute");
  }

  std::cout << "WGSL, f16, and SPIR-V compute stories passed\n";
}
