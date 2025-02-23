#include "gpu.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <vector>

using namespace gpu;
using namespace std::chrono;


// Forward declarations:
void testToCPUWithTensor();
void testToCPUWithBuffer();
void testToCPUWithTensorSourceOffset();
void testToCPUWithBufferSourceOffset();
void stressTestToCPU();

int main() {
  LOG(kDefLog, kInfo, "Running GPU integration tests...");
  testToCPUWithTensor();
  testToCPUWithBuffer();
  testToCPUWithTensorSourceOffset();
  testToCPUWithBufferSourceOffset();
  stressTestToCPU();
  LOG(kDefLog, kInfo, "All tests passed.");
  return 0;
}


// A simple WGSL copy kernel that copies input to output.
static const char *kCopyKernel = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dummy: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < arrayLength(&inp)) {
    out[i] = inp[i];
  }
}
)";


// Test using the overload that takes a Tensor.
void testToCPUWithTensor() {
  LOG(kDefLog, kInfo, "Running testToCPUWithTensor...");

// Create a real GPU context.
#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<float, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<float>(i);
    outputData[i] = 0.0f;
  }

  // Create input and output tensors.
  Tensor inputTensor = createTensor(ctx, Shape{N}, kf32, inputData.data());
  Tensor outputTensor = createTensor(ctx, Shape{N}, kf32);

  // Create and dispatch the copy kernel.
  Kernel copyKernel =
      createKernel(ctx, {kCopyKernel, 256, kf32},
                   Bindings{inputTensor, outputTensor}, {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, copyKernel);

  // Synchronously copy GPU output to CPU using the tensor overload.
  toCPU(ctx, outputTensor, outputData.data(), sizeof(outputData));

  // Verify the output matches the input.
  for (size_t i = 0; i < N; ++i) {
    LOG(kDefLog, kInfo, "inputData[%zu] = %f", i, inputData[i]);
    LOG(kDefLog, kInfo, "outputData[%zu] = %f", i, outputData[i]);
    assert(outputData[i] == inputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithTensor passed.");
}

// Test using the overload that takes a raw GPU buffer.
// We reuse the Tensor's underlying buffer for this test.
void testToCPUWithBuffer() {
  LOG(kDefLog, kInfo, "Running testToCPUWithBuffer...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<float, N> data, outputData;
  for (size_t i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i * 2);
    outputData[i] = 0.0f;
  }

  // Create a tensor to allocate a GPU buffer and initialize it.
  Tensor tensor = createTensor(ctx, Shape{N}, kf32, data.data());

  // Now extract the raw GPU buffer from the tensor.
  WGPUBuffer gpuBuffer = tensor.data.buffer;

  // Use the WGPUBuffer overload. This call returns a future.
  auto future =
      toCPUAsync(ctx, gpuBuffer, outputData.data(), sizeof(outputData), 0);
  wait(ctx, future);

  // Verify that the CPU output matches the original data.
  for (size_t i = 0; i < N; ++i) {
    LOG(kDefLog, kInfo, "outputData[%zu] = %f", i, outputData[i]);
    assert(outputData[i] == data[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithBuffer passed.");
}

void testToCPUWithTensorSourceOffset() {
  LOG(kDefLog, kInfo, "Running testToCPUWithTensorSourceOffset...");
#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t numElements = 25;
  constexpr size_t sourceOffsetElements = 5; // Skip first 5 elements
  constexpr size_t copyCount = 10;           // Number of floats to copy
  size_t copySize = copyCount * sizeof(float);

  // Create an input array with known data.
  std::array<float, numElements> inputData{};
  for (size_t i = 0; i < numElements; ++i) {
    inputData[i] = static_cast<float>(i + 50); // Arbitrary values
  }
  // Create a tensor from the full data.
  Tensor tensor = createTensor(ctx, Shape{numElements}, kf32, inputData.data());

  // Allocate a destination CPU buffer exactly as large as the data we want to
  // copy.
  std::vector<float> cpuOutput(copyCount, -1.0f);

  // Set sourceOffset to skip the first few float elements
  size_t sourceOffsetBytes = sourceOffsetElements * sizeof(float);
  // Call the tensor overload with sourceOffset and destOffset = 0.
  auto future =
      toCPUAsync(ctx, tensor, cpuOutput.data(), copySize, sourceOffsetBytes);
  wait(ctx, future);

  // Verify the copied data matches the expected subset.
  for (size_t i = 0; i < copyCount; ++i) {
    float expected = inputData[sourceOffsetElements + i];
    float actual = cpuOutput[i];
    LOG(kDefLog, kInfo, "cpuOutput[%zu] = %f", i, actual);
    LOG(kDefLog, kInfo, "expected[%zu] = %f", i, expected);
    assert(expected == actual);
  }
  LOG(kDefLog, kInfo, "testToCPUWithTensorSourceOffset passed.");
}

void testToCPUWithBufferSourceOffset() {
  LOG(kDefLog, kInfo, "Running testToCPUWithBufferSourceOffset...");
#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t numElements = 30;
  constexpr size_t sourceOffsetElements = 7; // Skip first 7 elements
  constexpr size_t copyCount = 12;           // Number of floats to copy
  size_t copySize = copyCount * sizeof(float);

  // Create an input array with arbitrary data.
  std::array<float, numElements> inputData{};
  for (size_t i = 0; i < numElements; ++i) {
    inputData[i] = static_cast<float>(i + 100);
  }
  // Create a tensor to initialize a GPU buffer.
  Tensor tensor = createTensor(ctx, Shape{numElements}, kf32, inputData.data());
  // Extract the raw GPU buffer from the tensor.
  WGPUBuffer buffer = tensor.data.buffer;

  // Allocate a destination CPU buffer exactly as large as needed.
  std::vector<float> cpuOutput(copyCount, -2.0f);
  size_t sourceOffsetBytes = sourceOffsetElements * sizeof(float);

  // Call the buffer overload with sourceOffset and destOffset = 0.
  auto future =
      toCPUAsync(ctx, buffer, cpuOutput.data(), copySize, sourceOffsetBytes);
  wait(ctx, future);

  // Verify that the copied data matches the expected subset.
  for (size_t i = 0; i < copyCount; ++i) {
    float expected = inputData[sourceOffsetElements + i];
    float actual = cpuOutput[i];
    LOG(kDefLog, kInfo, "cpuOutput[%zu] = %f", i, actual);
    LOG(kDefLog, kInfo, "expected[%zu] = %f", i, expected);
    assert(expected == actual);
  }
  LOG(kDefLog, kInfo, "testToCPUWithBufferSourceOffset passed.");
}

void stressTestToCPU() {
  LOG(kDefLog, kInfo, "Running stressTestToCPU for 2 seconds...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  // Create a persistent tensor with some test data.
  std::vector<float> inputData(N, 0.0f);
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<float>(i);
  }
  Tensor tensor = createTensor(ctx, Shape{N}, kf32, inputData.data());

  // Prepare to run for one second.
  auto startTime = high_resolution_clock::now();
  size_t opCount = 0;
  while (high_resolution_clock::now() - startTime < seconds(2)) {
    // Allocate an output buffer (using a shared_ptr so it stays valid until the future completes)
    auto outputData = std::make_shared<std::vector<float>>(N, 0.0f);
    // Use the tensor overload; we’re copying the entire tensor (destOffset = 0)
    LOG(kDefLog, kInfo, "Copying %zu bytes from GPU to CPU...", N * sizeof(float));
    // log count
    LOG(kDefLog, kInfo, "opCount = %zu", opCount);
    auto fut = toCPUAsync(ctx, tensor, outputData->data(), N * sizeof(float), 0);
    wait(ctx, fut);
    ++opCount;
  }
  
  auto endTime = high_resolution_clock::now();
  auto totalMs = duration_cast<milliseconds>(endTime - startTime).count();
  double throughput = (opCount / (totalMs / 1000.0));

  LOG(kDefLog, kInfo, "Stress test completed:\n"
            "  %zu GPU to CPU operations in %lld ms\n"
            "  Throughput: %.2f ops/sec", opCount, totalMs, throughput);
}
