#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "gpu.h"
#include "utils/array_utils.h"
#include "webprint.h"
#include <array>
#include <cmath>
#include <cstddef> // for size_t
#include <cstdio>
#include <future>
#include <numeric>
#include <vector>

using namespace gpu;

constexpr size_t kNumTestCases = 14;

struct TestCase {
  std::vector<float> input;
  int nInputs;
  std::vector<float> expectedOutput;
  Shape workgroupSize = {1, 1, 1};
  Shape gridSize = {1, 1, 1};
  Shape sharedMemorySize = {1, 1, 1};
};

typedef std::array<std::vector<TestCase>, kNumTestCases> TestCases;

std::vector<float> getOutput(Context &ctx, Kernel &op, Tensor &output, size_t R,
                             size_t C = 1) {
  std::promise<void> promise;
  std::future<void> future = promise.get_future();

  dispatchKernel(ctx, op, promise);

  std::vector<float> outputArr(R * C);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), outputArr.size() * sizeof(float));

  return outputArr;
}

inline KernelCode createCustomSharedMemory(
    const std::string &shaderTemplate, const size_t shared_memory_size,
    const Shape &workgroupSize = {256, 1, 1}, NumType precision = kf32) {
  std::string codeString(shaderTemplate);
  replaceAll(codeString,
             {{"{{workgroupSize}}", toString(workgroupSize)},
              {"{{precision}}", toString(precision)},
              {"{{sharedMemorySize}}", toString(shared_memory_size)}});
  return {codeString, workgroupSize};
}

void map_spec(const float *a, size_t length, float *result) {
  for (size_t i = 0; i < length; ++i) {
    result[i] = a[i] + 10;
  }
}

// C++ version of zip_spec
void zip_spec(const float *a, size_t length, float *result) {
  // Get first half of a and assign to b
  float *b = new float[length / 2];

  for (size_t i = 0; i < length / 2; ++i) {
    b[i] = a[i + length / 2];
  }

  for (size_t i = 0; i < length / 2; ++i) {
    result[i] = a[i] + b[i];
  }
}

// C++ version of broadcast_spec
void broadcast_spec(const float *a, size_t length, float *result) {
  size_t N = length / 2;

  // Assuming the input length is N + N, where the first N is 'a' and the second
  // N is 'b'
  const float *b = a + N;

  // Compute the result as the outer sum of a and b
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result[i * N + j] = a[i] + b[j];
    }
  }
}

// C++ version of pool_spec
void pool_spec(const float *a, size_t length, float *result) {
  for (size_t i = 0; i < length; ++i) {
    float sum = 0;
    for (size_t j = (i < 2 ? 0 : i - 2); j <= i; ++j) {
      sum += a[j];
    }
    result[i] = sum;
  }
}

// C++ version of dot_spec
void dot_spec(const float *a, size_t length, float *result) {
  size_t N = length / 2;

  // Assume the input length is N + N, where the first N is 'a' and the second N
  // is 'b'
  const float *b = a + N;

  *result = 0.0f;

  // Compute the dot product
  for (size_t i = 0; i < N; ++i) {
    *result += a[i] * b[i];
  }
}

// C++ version of conv_spec
void conv_spec(const float *a, size_t a_length, float *result) {
  const float *b = a + (a_length - 4);
  const float *a_end = a - 4;
  a_length -= 4;
  for (size_t i = 0; i < a_length; ++i) {
    result[i] = 0.0f;
    for (size_t j = 0; j < 4; ++j) {
      if (i + j < a_length) {
        result[i] += a[i + j] * b[j];
      }
    }
  }
}

// C++ version of sum_spec
void sum_spec(const float *a, size_t length, size_t TPB, float *result) {
  size_t out_size = (length + TPB - 1) / TPB;
  for (size_t j = 0, i = 0; i < length; i += TPB, ++j) {
    result[j] = 0.0f;
    for (size_t k = i; k < i + TPB && k < length; ++k) {
      result[j] += a[k];
    }
  }
}

void axis_sum_spec(const float *a, size_t rows, size_t cols, size_t TPB,
                   float *result) {
  // Calculate the number of chunks (columns) in the result
  size_t out_cols = (cols + TPB - 1) / TPB;

  // Initialize the result array to zeros
  std::memset(result, 0, rows * out_cols * sizeof(float));

  // Loop through the input array in chunks of size TPB
  for (size_t j = 0; j < out_cols; ++j) {
    size_t start_idx = j * TPB;
    size_t end_idx = (start_idx + TPB < cols) ? start_idx + TPB : cols;

    for (size_t i = 0; i < rows; ++i) {
      for (size_t k = start_idx; k < end_idx; ++k) {
        result[i * out_cols + j] += a[i * cols + k];
      }
    }
  }
}

// C++ version of matmul_spec
void matmul_spec(const float *a, size_t N, float *result) {
  // Assume the input length is 2(N^2), where the first N^2 is 'a' and the
  // second N^2 is 'b'
  const float *b = a + N * N;

  // Compute the matrix multiplication
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      result[i * N + j] = 0.0f;
      for (size_t k = 0; k < N; ++k) {
        result[i * N + j] += a[i * N + k] * b[k * N + j];
      }
    }
  }
}

// Function to run Puzzle 1
std::vector<float> runPuzzle1(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = testCase.input.size();

  Tensor a = createTensor(ctx, {N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {N}, kf32);

  KernelCode code = {kernelString, N};
  Kernel op = createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

// Function to run Puzzle 2
std::vector<float> runPuzzle2(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = testCase.input.size() / 2;

  std::vector<float> aVec(testCase.input.begin(), testCase.input.begin() + N);
  std::vector<float> bVec(testCase.input.begin() + N, testCase.input.end());

  Tensor a = createTensor(ctx, {N}, kf32, aVec.data());
  Tensor b = createTensor(ctx, {N}, kf32, bVec.data());
  Tensor output = createTensor(ctx, {N}, kf32);

  KernelCode code = {kernelString, N};
  Kernel op = createKernel(ctx, code, Bindings{a, b, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

// Function to run Puzzle 3
std::vector<float> runPuzzle3(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = testCase.input.size();

  Tensor a = createTensor(ctx, {N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {N}, kf32);

  KernelCode code = {kernelString, testCase.workgroupSize};
  Kernel op = createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

// Function to run Puzzle 4
std::vector<float> runPuzzle4(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = std::sqrt(testCase.input.size());

  Tensor input = createTensor(ctx, {N, N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {N, N}, kf32);

  struct Params {
    uint32_t size;
    Params(uint32_t n) : size(n) {}
  };

  KernelCode code = {kernelString, testCase.workgroupSize};
  Kernel op = createKernel(ctx, code, Bindings{input, output},
                           testCase.gridSize, Params(N), &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N * N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

// Function to run Puzzle 5

std::vector<float> runPuzzle5(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = testCase.input.size() / 2;

  std::vector<float> aVec(testCase.input.begin(), testCase.input.begin() + N);
  std::vector<float> bVec(testCase.input.begin() + N, testCase.input.end());

  Tensor a = createTensor(ctx, {N, 1}, kf32, aVec.data());
  Tensor b = createTensor(ctx, {1, N}, kf32, bVec.data());
  Tensor output = createTensor(ctx, {N, N}, kf32);

  struct Params {
    uint32_t size;
    Params(uint32_t n) : size(n) {}
  };

  KernelCode code = {kernelString, testCase.workgroupSize};
  Kernel op = createKernel(ctx, code, Bindings{a, b, output}, testCase.gridSize,
                           Params{static_cast<uint32_t>(N)}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N * N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

// Function to run Puzzle 6

std::vector<float> runPuzzle6(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = testCase.input.size();

  Tensor a = createTensor(ctx, {N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {N}, kf32);

  KernelCode code = {kernelString, testCase.workgroupSize};
  Kernel op = createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle7(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = std::sqrt(testCase.input.size());

  Tensor a = createTensor(ctx, {N, N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {N, N}, kf32);

  struct Params {
    uint32_t size;
    Params(uint32_t n) : size(n) {}
  };

  KernelCode code = {kernelString, testCase.workgroupSize};
  Kernel op = createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                           Params(N), &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N * N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle8(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {
  size_t N = testCase.input.size();

  Tensor a = createTensor(ctx, {N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {N}, kf32);
  KernelCode code = {kernelString, testCase.workgroupSize};
  Kernel op = createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle9(Context &ctx, TestCase &testCase,
                              const std::string &kernelString,
                              CompilationInfo &compilationInfo) {

  size_t N = testCase.input.size();

  Tensor a = createTensor(ctx, {N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {N}, kf32);

  size_t shared_memory = testCase.sharedMemorySize[0] *
                         testCase.sharedMemorySize[1] *
                         testCase.sharedMemorySize[2];

  KernelCode code = createCustomSharedMemory(kernelString, shared_memory,
                                             testCase.workgroupSize);
  Kernel op = createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle10(Context &ctx, TestCase &testCase,
                               const std::string &kernelString,
                               CompilationInfo &compilationInfo) {

  size_t N = testCase.input.size() / 2;

  std::vector<float> aVec(testCase.input.begin(), testCase.input.begin() + N);
  std::vector<float> bVec(testCase.input.begin() + N, testCase.input.end());

  Tensor a = createTensor(ctx, {N}, kf32, aVec.data());
  Tensor b = createTensor(ctx, {N}, kf32, bVec.data());
  Tensor output = createTensor(ctx, {N}, kf32);

  size_t shared_memory = testCase.sharedMemorySize[0] *
                         testCase.sharedMemorySize[1] *
                         testCase.sharedMemorySize[2];

  KernelCode code = createCustomSharedMemory(kernelString, shared_memory,
                                             testCase.workgroupSize);
  Kernel op = createKernel(ctx, code, Bindings{a, b, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, 1);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle11(Context &ctx, TestCase &testCase,
                               const std::string &kernelString,
                               CompilationInfo &compilationInfo) {

  size_t N = testCase.input.size() - 4;

  std::vector<float> aVec(testCase.input.begin(), testCase.input.end() - 4);
  std::vector<float> bVec(testCase.input.end() - 4, testCase.input.end());

  Tensor a = createTensor(ctx, {N}, kf32, aVec.data());
  Tensor b = createTensor(ctx, {4}, kf32, bVec.data());
  Tensor output = createTensor(ctx, {N}, kf32);

  struct Params {
    uint32_t TPB;
    Params(uint32_t tpb) : TPB(tpb) {}
  };

  size_t sharedMemory = testCase.sharedMemorySize[0] *
                        testCase.sharedMemorySize[1] *
                        testCase.sharedMemorySize[2];

  KernelCode code = createCustomSharedMemory(kernelString, sharedMemory,
                                             testCase.workgroupSize);
  Kernel op =
      createKernel(ctx, code, Bindings{a, b, output}, testCase.gridSize,
                   Params{static_cast<uint32_t>(testCase.workgroupSize[0])},
                   &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle12(Context &ctx, TestCase &testCase,
                               const std::string &kernelString,
                               CompilationInfo &compilationInfo) {

  size_t N = testCase.input.size();

  Tensor a = createTensor(ctx, {N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {2}, kf32);

  size_t sharedMemory = testCase.sharedMemorySize[0] *
                        testCase.sharedMemorySize[1] *
                        testCase.sharedMemorySize[2];

  KernelCode code = createCustomSharedMemory(kernelString, sharedMemory,
                                             testCase.workgroupSize);
  Kernel op = createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                           {}, &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, 2);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle13(Context &ctx, TestCase &testCase,
                               const std::string &kernelString,
                               CompilationInfo &compilationInfo) {

  size_t N = testCase.input.size() / 4;

  Tensor a = createTensor(ctx, {4, N}, kf32, testCase.input.data());
  Tensor output = createTensor(ctx, {4}, kf32);

  struct Params {
    uint32_t TPB;
    uint32_t size;
    Params(uint32_t tpb, uint32_t n) : TPB(tpb), size(n) {}
  };

  size_t sharedMemory = testCase.sharedMemorySize[0] *
                        testCase.sharedMemorySize[1] *
                        testCase.sharedMemorySize[2];

  KernelCode code = createCustomSharedMemory(kernelString, sharedMemory,
                                             testCase.workgroupSize);
  Kernel op =
      createKernel(ctx, code, Bindings{a, output}, testCase.gridSize,
                   Params{static_cast<uint32_t>(testCase.workgroupSize[0]),
                          static_cast<uint32_t>(N)},
                   &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, 4);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

std::vector<float> runPuzzle14(Context &ctx, TestCase &testCase,
                               const std::string &kernelString,
                               CompilationInfo &compilationInfo) {

  size_t N = std::sqrt(testCase.input.size() / 2);

  std::vector<float> aVec(testCase.input.begin(),
                          testCase.input.begin() + N * N);
  std::vector<float> bVec(testCase.input.begin() + N * N, testCase.input.end());

  Tensor a = createTensor(ctx, {N, N}, kf32, aVec.data());
  Tensor b = createTensor(ctx, {N, N}, kf32, bVec.data());
  Tensor output = createTensor(ctx, {N, N}, kf32);

  struct Params {
    uint32_t TPB;
    uint32_t size;
    Params(uint32_t tpb, uint32_t n) : TPB(tpb), size(n) {}
  };

  size_t sharedMemory = testCase.sharedMemorySize[0] *
                        testCase.sharedMemorySize[1] *
                        testCase.sharedMemorySize[2];

  KernelCode code = createCustomSharedMemory(kernelString, sharedMemory,
                                             testCase.workgroupSize);
  Kernel op =
      createKernel(ctx, code, Bindings{a, b, output}, testCase.gridSize,
                   Params{static_cast<uint32_t>(testCase.workgroupSize[0]),
                          static_cast<uint32_t>(N)},
                   &compilationInfo);

  std::vector outputArr = getOutput(ctx, op, output, N * N);

  testCase.workgroupSize = code.workgroupSize;
  testCase.gridSize = op.nWorkgroups;

  return outputArr;
}

// Function to initialize the test cases
TestCases createTestCases() {
  TestCases testCases;

  // Initialize test cases for Puzzle 1
  std::vector<std::vector<float>> inputs = {{0, 1, 2, 3, 4, 5},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  std::vector<std::vector<float>> outputs = inputs;

  for (size_t i = 0; i < inputs.size(); ++i) {
    map_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[0] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()}} // Test case 2
  };

  // Initialize test cases for Puzzle 2
  inputs = {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  outputs = {{0, 0}, {0, 0, 0, 0, 0}};

  for (size_t i = 0; i < inputs.size(); ++i) {
    zip_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[1] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {4, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {10, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 3
  inputs = {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  outputs = {std::vector<float>(6, 0), std::vector<float>(10, 0)};

  for (size_t i = 0; i < inputs.size(); ++i) {
    map_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[2] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {10, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {12, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 4
  inputs = {{0, 1, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8}};

  outputs = {std::vector<float>(inputs[0].begin(), inputs[0].end()),
             std::vector<float>(inputs[1].begin(), inputs[1].end())};

  for (size_t i = 0; i < inputs.size(); ++i) {
    map_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[3] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {3, 3, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {4, 4, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 5
  inputs = {{0, 1, 2, 3, 0, 1, 2, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8}};

  outputs = {std::vector<float>(16, 0), std::vector<float>(81, 0)};

  for (size_t i = 0; i < inputs.size(); ++i) {
    broadcast_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[4] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {5, 5, 1}} // Test case 1
  };

  // Initialize test cases for Puzzle 6
  inputs = {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  outputs = {std::vector<float>(inputs[0].begin(), inputs[0].end()),
             std::vector<float>(inputs[1].begin(), inputs[1].end())};

  for (size_t i = 0; i < inputs.size(); ++i) {
    map_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[5] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {2, 1, 1},
       .gridSize = {3, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {4, 1, 1},
       .gridSize = {3, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 7
  inputs = {{0, 1, 2, 3, 4, 5, 6, 7, 8},
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};

  outputs = {std::vector<float>(inputs[0].begin(), inputs[0].end()),
             std::vector<float>(inputs[1].begin(), inputs[1].end())};

  for (size_t i = 0; i < inputs.size(); ++i) {
    map_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[6] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {2, 2, 1},
       .gridSize = {2, 2, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {2, 2, 1},
       .gridSize = {3, 3, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 8
  inputs = {{0, 1, 2, 3, 4, 5, 6, 7},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

  outputs = {std::vector<float>(inputs[0].begin(), inputs[0].end()),
             std::vector<float>(inputs[1].begin(), inputs[1].end())};

  for (size_t i = 0; i < inputs.size(); ++i) {
    map_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[7] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {4, 1, 1},
       .gridSize = {2, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {2, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 9
  inputs = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  outputs = {std::vector<float>(inputs[0].begin(), inputs[0].end()),
             std::vector<float>(inputs[1].begin(), inputs[1].end())};

  for (size_t i = 0; i < inputs.size(); ++i) {
    pool_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[8] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {1, 1, 1},
       .sharedMemorySize = {8, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {10, 1, 1},
       .gridSize = {1, 1, 1},
       .sharedMemorySize = {10, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 10
  inputs = {{0, 1, 2, 3, 0, 1, 2, 3}, {0, 1, 2, 3, 4, 0, 1, 2, 3, 4}};

  outputs = {std::vector<float>(1, 0), std::vector<float>(1, 0)};

  for (size_t i = 0; i < inputs.size(); ++i) {
    dot_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[9] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {4, 1, 1},
       .gridSize = {1, 1, 1},
       .sharedMemorySize = {4, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {5, 1, 1},
       .gridSize = {1, 1, 1},
       .sharedMemorySize = {5, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 11
  inputs = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3},
            {0,  1,  2,  3,  4,  5,  6,  7, 8, 9, 10,
             11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3}};

  outputs = {std::vector<float>(15, 0), std::vector<float>(18, 0)};

  for (size_t i = 0; i < inputs.size(); ++i) {
    conv_spec(inputs[i].data(), inputs[i].size(), outputs[i].data());
  }

  testCases[10] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {2, 1, 1},
       .sharedMemorySize = {12, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {3, 1, 1},
       .sharedMemorySize = {12, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 12
  inputs = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};

  outputs = {std::vector<float>(2, 0), std::vector<float>(2, 0)};

  for (size_t i = 0; i < inputs.size(); ++i) {
    sum_spec(inputs[i].data(), inputs[i].size(), 8, outputs[i].data());
  }

  testCases[11] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {1, 1, 1},
       .sharedMemorySize = {8, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {2, 1, 1},
       .sharedMemorySize = {8, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 13
  inputs = {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};

  outputs = {std::vector<float>(4, 0), std::vector<float>(4, 0)};

  for (size_t i = 0; i < inputs.size(); ++i) {
    axis_sum_spec(inputs[i].data(), 4, inputs[i].size() / 4, 8,
                  outputs[i].data());
  }

  testCases[12] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {1, 4, 1},
       .sharedMemorySize = {8, 1, 1}}, // Test case 1
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 1,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {8, 1, 1},
       .gridSize = {1, 4, 1},
       .sharedMemorySize = {8, 1, 1}} // Test case 2
  };

  // Initialize test cases for Puzzle 14
  inputs = {{0, 1, 2, 3, 0, 2, 1, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             0,  8,  16, 24, 32, 40, 48, 56, 1,  9,  17, 25, 33, 41, 49, 57,
             2,  10, 18, 26, 34, 42, 50, 58, 3,  11, 19, 27, 35, 43, 51, 59,
             4,  12, 20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53, 61,
             6,  14, 22, 30, 38, 46, 54, 62, 7,  15, 23, 31, 39, 47, 55, 63}};

  outputs = {std::vector<float>(4, 0), std::vector<float>(9, 0),
             std::vector<float>(16, 0), std::vector<float>(64, 0)};

  for (size_t i = 0; i < inputs.size(); ++i) {
    matmul_spec(inputs[i].data(), sqrt(inputs[i].size() / 2),
                outputs[i].data());
  }

  testCases[13] = {
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {3, 3, 1},
       .gridSize = {1, 1, 1},
       .sharedMemorySize = {3, 3, 1}},
      {.input = {inputs[0].begin(), inputs[0].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[0].begin(), outputs[0].end()},
       .workgroupSize = {1, 1, 1},
       .gridSize = {2, 2, 1},
       .sharedMemorySize = {3, 3, 1}},
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {4, 4, 1},
       .gridSize = {1, 1, 1},
       .sharedMemorySize = {4, 4, 1}},
      {.input = {inputs[1].begin(), inputs[1].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[1].begin(), outputs[1].end()},
       .workgroupSize = {2, 2, 1},
       .gridSize = {2, 2, 1},
       .sharedMemorySize = {2, 2, 1}},
      {.input = {inputs[2].begin(), inputs[2].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[2].begin(), outputs[2].end()},
       .workgroupSize = {2, 2, 1},
       .gridSize = {2, 2, 1},
       .sharedMemorySize = {2, 2, 1}},
      {.input = {inputs[3].begin(), inputs[3].end()},
       .nInputs = 2,
       .expectedOutput = {outputs[3].begin(), outputs[3].end()},
       .workgroupSize = {2, 2, 1},
       .gridSize = {4, 4, 1},
       .sharedMemorySize = {2, 2, 1}},
  };

  return testCases;
}

std::string getTemplate(int puzzleIndex) {
  const auto allTestCases = createTestCases();
  int nInputs = allTestCases[puzzleIndex][0].nInputs;
  std::string result = "";
  for (size_t i = 0; i < nInputs; ++i) {
    result += "@group(0) @binding(" + std::to_string(i) +
              ") var<storage, read_write> in" + std::to_string(i) +
              " : array<f32>;\n";
  }
  result += "@group(0) @binding(" + std::to_string(nInputs) +
            ") var<storage, read_write> out : array<f32>;\n";

  result += "const wgs = vec3({{workgroupSize}});\n\n";
  result += "@compute @workgroup_size({{workgroupSize}})\n";
  result +=
      R"(fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,) {
    let i: u32 = gid.x;
    out[i] = in0[i];
}
)";
  return result;
}

// Function to check if the output matches the expected output
bool checkOutput(const std::vector<float> &output,
                 const std::vector<float> &expectedOutput) {
  if (output.size() != expectedOutput.size())
    return false;
  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i] != expectedOutput[i])
      return false;
  }
  return true;
}

// Function to evaluate the test cases
bool evaluate(Context &ctx, const std::string &kernelCode, int puzzleIndex) {
  if (puzzleIndex >= kNumTestCases) {
    wprintf("Invalid puzzle index!\n");
    return false;
  }

  const auto allTestCases = createTestCases();

  auto testCases = allTestCases[puzzleIndex];
  if (kernelCode.empty()) {
    printf("Kernel code is empty!\n");
    return false;
  }

  printf("dispatching puzzle with code:\n%s", kernelCode.c_str());

  CompilationInfo compilationInfo;

  bool allPassed = true;
  for (int caseIdx = 0; caseIdx < testCases.size(); ++caseIdx) {
    auto testCase = testCases[caseIdx];

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> output;
    switch (puzzleIndex) {
    case 0:
      output = runPuzzle1(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 1:
      output = runPuzzle2(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 2:
      output = runPuzzle3(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 3:
      output = runPuzzle4(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 4:
      output = runPuzzle5(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 5:
      output = runPuzzle6(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 6:
      output = runPuzzle7(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 7:
      output = runPuzzle8(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 8:
      output = runPuzzle9(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 9:
      output = runPuzzle10(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 10:
      output = runPuzzle11(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 11:
      output = runPuzzle12(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 12:
      output = runPuzzle13(ctx, testCase, kernelCode, compilationInfo);
      break;
    case 13:
      output = runPuzzle14(ctx, testCase, kernelCode, compilationInfo);
      break;
    // Add more cases for additional puzzles
    default:
      wprintf("Invalid puzzle index!");
      return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    const char *red = "\033[1;31m";
    const char *green = "\033[1;32m";
    const char *grey = "\033[1;30m";
    const char *reset = "\033[0m";

    // wprintf("Time taken: %f s\n", elapsed.count());

    bool compilePassed = true;
    int ptr = 0;
    constexpr size_t kBufSize = 1024 * 10;
    char buf[kBufSize];

    if (compilationInfo.messages.size() > 0) {
      if (caseIdx == 0) {
        // Don't print compilation errors more than once
        for (size_t idx = 0; idx < compilationInfo.messages.size(); ++idx) {
          ptr +=
              snprintf(buf, kBufSize, "%sError%s line %d, column %d:\n\r", red,
                       reset, static_cast<int>(compilationInfo.lineNums[idx]),
                       static_cast<int>(compilationInfo.linePos[idx]));
          ptr += snprintf(buf + ptr, kBufSize - ptr, "%s\n\n\r",
                          compilationInfo.messages[idx].c_str());
        }
        ptr += snprintf(buf + ptr, kBufSize,
                        "*   *   *   *   *   *   *   *\n\n\r");
      }
      allPassed = false;
      compilePassed = false;
    }

    // ptr = 0;
    if (compilePassed && checkOutput(output, testCase.expectedOutput)) {
      ptr += snprintf(buf + ptr, kBufSize, "Test case %d %sPASSED%s\n\n\r",
                      caseIdx + 1, green, reset);
    } else {
      ptr += snprintf(buf + ptr, kBufSize, "Test case %d %sFAILED%s\n\n\r",
                      caseIdx + 1, red, reset);
      allPassed = false;
    }

    ptr += snprintf(buf + ptr, kBufSize,
                    "\033[1;30mWorkgroup Size          ( %s )\n\r",
                    toString(testCase.workgroupSize).c_str());
    ptr += snprintf(buf + ptr, kBufSize,
                    "Number of Workgroups    ( %s )\n\033[0m\n\r",
                    toString(testCase.gridSize).c_str());

    wprintf("%s", buf);

    if (testCase.nInputs > 1) {
      for (size_t inp = 0; inp < testCase.nInputs; ++inp) {
        size_t sz = testCase.input.size() / testCase.nInputs;
        size_t offset = inp * sz;
        snprintf(buf, sizeof(buf), "%sInput  %zu%s", grey, inp, reset);
        printVec({begin(testCase.input) + offset,
                  begin(testCase.input) + offset + sz},
                 buf);
      }
    } else {
      snprintf(buf, sizeof(buf), "%sInput   %s", grey, reset);
      printVec(testCase.input, buf);
    }
    if (compilePassed) {
      wprintf("");
      snprintf(buf, sizeof(buf), "%sGot     %s", grey, reset);
      printVec(output, buf);
      wprintf("");
    }

    snprintf(buf, sizeof(buf), "%sExpected%s", grey, reset);
    printVec(testCase.expectedOutput, buf);
    wprintf("");
  }

  return allPassed;
}

#endif // EVALUATOR_H
