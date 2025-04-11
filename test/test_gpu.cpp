#include "gpu.hpp"
#include "numeric_types/half.hpp"
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
void testToCPUWithHalf();
void testToCPUWithFloat();
void testToCPUWithDouble();
void testToCPUWithint8();
void testToCPUWithint16();
void testToCPUWithint();
void testToCPUWithint64();
void testToCPUWithUint8();
void testToCPUWithUint16();
void testToCPUWithUint32();
void testToCPUWithUint64();
void testNumTypeSizes();
void testToCPUUnpack();

int main() {
  LOG(kDefLog, kInfo, "Running GPU integration tests...");
  testToCPUUnpack();
  testToCPUWithTensor();
  testToCPUWithBuffer();
  testToCPUWithTensorSourceOffset();
  testToCPUWithBufferSourceOffset();
  testToCPUWithHalf();
  testToCPUWithFloat();
  testToCPUWithDouble();
  testToCPUWithint8();
  testToCPUWithint16();
  testToCPUWithint();
  testToCPUWithint64();
  testToCPUWithUint8();
  testToCPUWithUint16();
  testToCPUWithUint32();
  testToCPUWithUint64();
  testNumTypeSizes();
  stressTestToCPU();
  testHalf();
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

void testToCPUUnpack() {
  LOG(kDefLog, kInfo, "Running testToCPUUnpack...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  // Test for double (kf64 -> packed as kf32)
  {
    constexpr size_t N = 1024;
    std::vector<double> inputData(N), outputData(N);
    for (size_t i = 0; i < N; ++i) {
      inputData[i] = static_cast<double>(i) * 3.14;
    }
    Tensor tensor = createTensor(ctx, Shape{N}, kf64, inputData.data());
    toCPU(ctx, tensor, kf64, outputData.data(), 0);
    for (size_t i = 0; i < N; ++i) {
      // Allow for a very small epsilon error due to float conversion.
      assert(fabs(inputData[i] - outputData[i]) < 1e-4);
    }
    LOG(kDefLog, kInfo, "toCPUUnpack for double passed.");
  }

  // Test for int8_t (ki8 -> packed as ki32)
  {
    constexpr size_t N = 1024;
    std::vector<int8_t> inputData(N), outputData(N);
    for (size_t i = 0; i < N; ++i) {
      inputData[i] = static_cast<int8_t>((i % 256) - 128);
    }
    Tensor tensor = createTensor(ctx, Shape{N}, ki8, inputData.data());
    toCPU(ctx, tensor, ki8, outputData.data(), 0);
    for (size_t i = 0; i < N; ++i) {
      assert(inputData[i] == outputData[i]);
    }
    LOG(kDefLog, kInfo, "toCPUUnpack for int8_t passed.");
  }

  // Test for int16_t (ki16 -> packed as ki32)
  {
    constexpr size_t N = 1024;
    std::vector<int16_t> inputData(N), outputData(N);
    for (size_t i = 0; i < N; ++i) {
      inputData[i] = static_cast<int16_t>((i % 65536) - 32768);
    }
    Tensor tensor = createTensor(ctx, Shape{N}, ki16, inputData.data());
    toCPU(ctx, tensor, ki16, outputData.data(), 0);
    for (size_t i = 0; i < N; ++i) {
      assert(inputData[i] == outputData[i]);
    }
    LOG(kDefLog, kInfo, "toCPUUnpack for int16_t passed.");
  }

  // Test for int64_t (ki64 -> packed as two ki32s)
  {
    constexpr size_t N = 1024;
    std::vector<int64_t> inputData(N), outputData(N);
    for (size_t i = 0; i < N; ++i) {
      inputData[i] = static_cast<int64_t>(i) - 512;
    }
    Tensor tensor = createTensor(ctx, Shape{N}, ki64, inputData.data());
    toCPU(ctx, tensor, ki64, outputData.data(), 0);
    for (size_t i = 0; i < N; ++i) {
      assert(inputData[i] == outputData[i]);
    }
    LOG(kDefLog, kInfo, "toCPUUnpack for int64_t passed.");
  }

  // Test for uint8_t (ku8 -> packed as ku32)
  {
    constexpr size_t N = 1024;
    std::vector<uint8_t> inputData(N), outputData(N);
    for (size_t i = 0; i < N; ++i) {
      inputData[i] = static_cast<uint8_t>(i % 256);
    }
    Tensor tensor = createTensor(ctx, Shape{N}, ku8, inputData.data());
    toCPU(ctx, tensor, ku8, outputData.data(), 0);
    for (size_t i = 0; i < N; ++i) {
      assert(inputData[i] == outputData[i]);
    }
    LOG(kDefLog, kInfo, "toCPUUnpack for uint8_t passed.");
  }

  // Test for uint16_t (ku16 -> packed as ku32)
  {
    constexpr size_t N = 1024;
    std::vector<uint16_t> inputData(N), outputData(N);
    for (size_t i = 0; i < N; ++i) {
      inputData[i] = static_cast<uint16_t>(i % 65536);
    }
    Tensor tensor = createTensor(ctx, Shape{N}, ku16, inputData.data());
    toCPU(ctx, tensor, ku16, outputData.data(), 0);
    for (size_t i = 0; i < N; ++i) {
      assert(inputData[i] == outputData[i]);
    }
    LOG(kDefLog, kInfo, "toCPUUnpack for uint16_t passed.");
  }

  // Test for uint64_t (ku64 -> packed as two ku32s)
  {
    constexpr size_t N = 1024;
    std::vector<uint64_t> inputData(N), outputData(N);
    for (size_t i = 0; i < N; ++i) {
      inputData[i] = static_cast<uint64_t>(i) * 123456789ULL;
    }
    Tensor tensor = createTensor(ctx, Shape{N}, ku64, inputData.data());
    toCPU(ctx, tensor, ku64, outputData.data(), 0);
    for (size_t i = 0; i < N; ++i) {
      assert(inputData[i] == outputData[i]);
    }
    LOG(kDefLog, kInfo, "toCPUUnpack for uint64_t passed.");
  }

  LOG(kDefLog, kInfo, "All toCPUUnpack tests passed.");
}

void testNumTypeSizes() {
  LOG(kDefLog, kInfo, "Running testNumTypeSizes...");

  assert(sizeBytes(kf16) == 2);
  assert(sizeBytes(kf32) == 4);
  assert(sizeBytes(ki8) == sizeof(uint8_t));   // typically 1
  assert(sizeBytes(ki16) == sizeof(uint16_t)); // typically 2
  assert(sizeBytes(ki32) == sizeof(int32_t));  // typically 4
  assert(sizeBytes(ku8) == sizeof(uint8_t));   // typically 1
  assert(sizeBytes(ku16) == sizeof(uint16_t)); // typically 2
  assert(sizeBytes(ku32) == sizeof(uint32_t)); // typically 4

  LOG(kDefLog, kInfo, "testNumTypeSizes passed.");
}

// Test using half-precision (16-bit float) data.
void testToCPUWithHalf() {
  LOG(kDefLog, kInfo, "Running testToCPUWithHalf...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<half, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    // Construct half from float.
    inputData[i] = half(static_cast<float>(i));
  }

  Tensor inputTensor = createTensor(ctx, Shape{N}, kf16, inputData.data());

  // Copy GPU output to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy (using float conversion for approximate equality).
  for (size_t i = 0; i < N; ++i) {
    float inVal = static_cast<float>(inputData[i]);
    float outVal = static_cast<float>(outputData[i]);
    // Use a small epsilon to compare half values.
    assert(fabs(inVal - outVal) <= 0.01f);
  }
  LOG(kDefLog, kInfo, "testToCPUWithHalf passed.");
}

// Test using float (32-bit) data.
void testToCPUWithFloat() {
  LOG(kDefLog, kInfo, "Running testToCPUWithFloat...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<float, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<float>(i * 1.5f);
    outputData[i] = 0.0f;
  }

  Tensor inputTensor = createTensor(ctx, Shape{N}, kf32, inputData.data());

  // Copy GPU output to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy.
  for (size_t i = 0; i < N; ++i) {
    assert(inputData[i] == outputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithFloat passed.");
}

// Test using double (64-bit floating point) data.
void testToCPUWithDouble() {
  LOG(kDefLog, kInfo, "Running testToCPUWithDouble...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<double, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<double>(i) * 2.5;
    outputData[i] = 0.0;
  }

  Tensor inputTensor = createTensor(ctx, Shape{N}, kf64, inputData.data());

  // Copy GPU output to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy.
  for (size_t i = 0; i < N; ++i) {
    assert(inputData[i] == outputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithDouble passed.");
}

void testToCPUWithint8() {
  LOG(kDefLog, kInfo, "Running testToCPUWithint8...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<int8_t, N> inputData, outputData;
  // Use a range that includes negative values.
  for (size_t i = 0; i < N; ++i) {
    // Values between -128 and 127.
    inputData[i] = static_cast<int8_t>((i % 256) - 128);
    outputData[i] = 0;
  }

  // Create a tensor for int8_t.
  Tensor inputTensor = createTensor(ctx, Shape{N}, ki8, inputData.data());

  // Synchronously copy the GPU tensor data to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy.
  for (size_t i = 0; i < N; ++i) {
    // LOG(kDefLog, kInfo, "inputData[%zu] = %d", i, inputData[i]);
    // LOG(kDefLog, kInfo, "outputData[%zu] = %d", i, outputData[i]);
    assert(outputData[i] == inputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithint8 passed.");
}

// Test using int16_t data.
void testToCPUWithint16() {
  LOG(kDefLog, kInfo, "Running testToCPUWithint16...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<int16_t, N> inputData, outputData;
  // Use a range that includes negative values.
  for (size_t i = 0; i < N; ++i) {
    // Values between -32768 and 32767.
    inputData[i] = static_cast<int16_t>((i % 65536) - 32768);
    outputData[i] = 0;
  }

  // Create a tensor for int16_t.
  Tensor inputTensor = createTensor(ctx, Shape{N}, ki16, inputData.data());

  // Synchronously copy the GPU tensor data to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy.
  for (size_t i = 0; i < N; ++i) {
    // LOG(kDefLog, kInfo, "inputData[%zu] = %d", i, inputData[i]);
    // LOG(kDefLog, kInfo, "outputData[%zu] = %d", i, outputData[i]);
    assert(outputData[i] == inputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithint16 passed.");
}

// Test using int (int32_t) data.
void testToCPUWithint() {
  LOG(kDefLog, kInfo, "Running testToCPUWithint...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<int32_t, N> inputData, outputData;
  // Fill with sample data.
  for (size_t i = 0; i < N; ++i) {
    inputData[i] =
        static_cast<int32_t>(i - 512); // Negative and positive values.
    outputData[i] = 0;
  }

  // Create a tensor for int32_t.
  Tensor inputTensor = createTensor(ctx, Shape{N}, ki32, inputData.data());

  // Synchronously copy the GPU tensor data to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy.
  for (size_t i = 0; i < N; ++i) {
    // LOG(kDefLog, kInfo, "inputData[%zu] = %d", i, inputData[i]);
    // LOG(kDefLog, kInfo, "outputData[%zu] = %d", i, outputData[i]);
    assert(outputData[i] == inputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithint passed.");
}

// Test using int64_t (64-bit signed integer) data.
void testToCPUWithint64() {
  LOG(kDefLog, kInfo, "Running testToCPUWithint64...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<int64_t, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] =
        static_cast<int64_t>(i) - 512; // Some negative and positive values.
    outputData[i] = 0;
  }

  Tensor inputTensor = createTensor(ctx, Shape{N}, ki64, inputData.data());

  // Copy GPU output to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy.
  for (size_t i = 0; i < N; ++i) {
    assert(inputData[i] == outputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithint64 passed.");
}

void testToCPUWithUint8() {
  LOG(kDefLog, kInfo, "Running testToCPUWithUint8...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<uint8_t, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<uint8_t>(i % 256);
    outputData[i] = 0;
  }

  Tensor inputTensor = createTensor(
      ctx, Shape{N}, ku8, reinterpret_cast<const uint8_t *>(inputData.data()));

  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Verify the output matches the input.
  for (size_t i = 0; i < N; ++i) {
    // LOG(kDefLog, kInfo, "inputData[%zu] = %u", i, inputData[i]);
    // LOG(kDefLog, kInfo, "outputData[%zu] = %u", i, outputData[i]);
    assert(outputData[i] == inputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithUint8 passed.");
}

void testToCPUWithUint16() {
  LOG(kDefLog, kInfo, "Running testToCPUWithUint16...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<uint16_t, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<uint16_t>(i % 65536);
    outputData[i] = 0;
  }

  Tensor inputTensor =
      createTensor(ctx, Shape{N}, ku16,
                   reinterpret_cast<const uint16_t *>(inputData.data()));

  // Synchronously copy GPU output to CPU using the tensor overload.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Verify the output matches the input.
  for (size_t i = 0; i < N; ++i) {
    // LOG(kDefLog, kInfo, "inputData[%zu] = %u", i, inputData[i]);
    // LOG(kDefLog, kInfo, "outputData[%zu] = %u", i, outputData[i]);
    assert(outputData[i] == inputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithUint16 passed.");
}

void testToCPUWithUint32() {
  LOG(kDefLog, kInfo, "Running testToCPUWithUint32...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<uint32_t, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<uint32_t>(i);
    outputData[i] = 0;
  }

  Tensor inputTensor =
      createTensor(ctx, Shape{N}, ku32,
                   reinterpret_cast<const uint32_t *>(inputData.data()));

  // Synchronously copy GPU output to CPU using the tensor overload.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Verify the output matches the input.
  for (size_t i = 0; i < N; ++i) {
    // LOG(kDefLog, kInfo, "inputData[%zu] = %u", i, inputData[i]);
    // LOG(kDefLog, kInfo, "outputData[%zu] = %u", i, outputData[i]);
    assert(outputData[i] == inputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithUint32 passed.");
}

// Test using uint64_t (64-bit unsigned integer) data.
void testToCPUWithUint64() {
  LOG(kDefLog, kInfo, "Running testToCPUWithUint64...");

#ifdef USE_DAWN_API
  Context ctx = createContextByGpuIdx(0);
#else
  Context ctx = createContext();
#endif

  constexpr size_t N = 1024;
  std::array<uint64_t, N> inputData, outputData;
  for (size_t i = 0; i < N; ++i) {
    inputData[i] = static_cast<uint64_t>(i);
    outputData[i] = 0;
  }

  // Assuming a new NumType 'ku64' for 64-bit unsigned integers.
  Tensor inputTensor = createTensor(ctx, Shape{N}, ku64, inputData.data());

  // Copy GPU output to CPU.
  toCPU(ctx, inputTensor, outputData.data(), sizeof(outputData));

  // Validate the copy.
  for (size_t i = 0; i < N; ++i) {
    assert(inputData[i] == outputData[i]);
  }
  LOG(kDefLog, kInfo, "testToCPUWithUint64 passed.");
}

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
    // LOG(kDefLog, kInfo, "inputData[%zu] = %f", i, inputData[i]);
    // LOG(kDefLog, kInfo, "outputData[%zu] = %f", i, outputData[i]);
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
    // LOG(kDefLog, kInfo, "outputData[%zu] = %f", i, outputData[i]);
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
    // LOG(kDefLog, kInfo, "cpuOutput[%zu] = %f", i, actual);
    // LOG(kDefLog, kInfo, "expected[%zu] = %f", i, expected);
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
    // LOG(kDefLog, kInfo, "cpuOutput[%zu] = %f", i, actual);
    // LOG(kDefLog, kInfo, "expected[%zu] = %f", i, expected);
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
    // Allocate an output buffer (using a shared_ptr so it stays valid until the
    // future completes)
    auto outputData = std::make_shared<std::vector<float>>(N, 0.0f);
    // Use the tensor overload; we’re copying the entire tensor (destOffset = 0)
    // log count
    auto fut =
        toCPUAsync(ctx, tensor, outputData->data(), N * sizeof(float), 0);
    wait(ctx, fut);
    ++opCount;
  }

  auto endTime = high_resolution_clock::now();
  auto totalMs = duration_cast<milliseconds>(endTime - startTime).count();
  double throughput = (opCount / (totalMs / 1000.0));

  LOG(kDefLog, kInfo,
      "Stress test completed:\n"
      "  %zu GPU to CPU operations in %lld ms\n"
      "  Throughput: %.2f ops/sec",
      opCount, totalMs, throughput);
}
