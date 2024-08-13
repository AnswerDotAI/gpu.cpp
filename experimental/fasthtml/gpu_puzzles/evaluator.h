#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "gpu.h"
#include "utils/array_utils.h"
#include <array>
#include <cstdio>
#include <future>
#include <vector>
#include <numeric>
#include <cstddef> // for size_t

using namespace gpu;

constexpr size_t kNumTestCases = 2;

struct TestCase {
    std::vector<float> input;
    std::vector<float> expectedOutput;
    Shape workgroupSize = {1, 1, 1};
    Shape gridSize = {1, 1, 1};
    Shape sharedMemorySize = {1, 1, 1};
};

typedef std::array<std::vector<TestCase>, kNumTestCases> TestCases;

// TODO: Use these spec functions in the evaluations

void map_spec(const float* a, size_t length, float* result) {
    for (size_t i = 0; i < length; ++i) {
        result[i] = a[i] + 10;
    }
}

// C++ version of zip_spec
void zip_spec(const float* a, const float* b, size_t length, float* result) {
    for (size_t i = 0; i < length; ++i) {
        result[i] = a[i] + b[i];
    }
}

// C++ version of pool_spec
void pool_spec(const float* a, size_t length, float* result) {
    for (size_t i = 0; i < length; ++i) {
        float sum = 0;
        for (size_t j = (i < 2 ? 0 : i - 2); j <= i; ++j) {
            sum += a[j];
        }
        result[i] = sum;
    }
}

// C++ version of dot_spec
void dot_spec(const float* a, const float* b, size_t length, float* result) {
    *result = std::inner_product(a, a + length, b, 0.0f);
}

// C++ version of conv_spec
void conv_spec(const float* a, const float* b, size_t a_length, size_t b_length, float* result) {
    for (size_t i = 0; i < a_length; ++i) {
        result[i] = 0.0f;
        for (size_t j = 0; j < b_length; ++j) {
            if (i + j < a_length) {
                result[i] += a[i + j] * b[j];
            }
        }
    }
}

// C++ version of sum_spec
void sum_spec(const float* a, size_t length, size_t TPB, float* result) {
    size_t out_size = (length + TPB - 1) / TPB;
    for (size_t j = 0, i = 0; i < length; i += TPB, ++j) {
        result[j] = 0.0f;
        for (size_t k = i; k < i + TPB && k < length; ++k) {
            result[j] += a[k];
        }
    }
}

// C++ version of matmul_spec
void matmul_spec(const float* a, const float* b, size_t N, float* result) {
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
std::vector<float> runPuzzle1(Context &ctx, const TestCase &testCase,
                              const std::string &kernelString) {
    std::printf("Puzzle 1\n");

    size_t N = testCase.input.size();

    Tensor a = createTensor(ctx, {N}, kf32, testCase.input.data());
    Tensor output = createTensor(ctx, {N}, kf32);

    Kernel op = createKernel(ctx, {kernelString, N}, Bindings{a, output},
                             testCase.gridSize);

    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    dispatchKernel(ctx, op, promise);

    std::vector<float> outputArr(N);
    wait(ctx, future);
    toCPU(ctx, output, outputArr.data(), outputArr.size() * sizeof(float));

    return outputArr;
}

// Function to run Puzzle 2
std::vector<float> runPuzzle2(Context &ctx, const TestCase &testCase,
                              const std::string &kernelString) {
    std::printf("Puzzle 2\n");

    size_t N = testCase.input.size() / 2;

    std::vector<float> aVec(testCase.input.begin(), testCase.input.begin() + N);
    std::vector<float> bVec(testCase.input.begin() + N, testCase.input.end());

    Tensor a = createTensor(ctx, {N}, kf32, aVec.data());
    Tensor b = createTensor(ctx, {N}, kf32, bVec.data());
    Tensor output = createTensor(ctx, {N}, kf32);

    Kernel op = createKernel(ctx, {kernelString, N}, Bindings{a, b, output},
                             testCase.gridSize);

    std::promise<void> promise;
    std::future<void> future = promise.get_future();

    dispatchKernel(ctx, op, promise);

    std::vector<float> outputArr(N);
    wait(ctx, future);
    toCPU(ctx, output, outputArr.data(), outputArr.size() * sizeof(float));

    return outputArr;
}

// Function to initialize the test cases
TestCases createTestCases() {
    TestCases testCases;

    // Initialize test cases for Puzzle 1
    testCases[0] = {
        {{0, 1, 2, 3}, {10, 11, 12, 13}}, // Test case 1
        {{0, 1, 2, 3, 4, 5, 6, 7, 8},
         {10, 11, 12, 13, 14, 15, 16, 17, 18}}, // Test case 2
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
         {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}} // Test case 3
    };

    // Initialize test cases for Puzzle 2
    testCases[1] = {
        {{0, 1, 2, 0}, {2, 1}},                             // Test case 1
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {5, 7, 9, 11, 13}} // Test case 2
    };

    return testCases;
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

// Helper function to print a vector
void printVec(const std::vector<float> &vec) {
    std::printf("[");
    for (size_t i = 0; i < vec.size(); ++i) {
        std::printf("%f", vec[i]);
        if (i != vec.size() - 1)
            std::printf(", ");
    }
    std::printf("]\n");
}

// Function to evaluate the test cases
bool evaluate(Context &ctx, TestCases &allTestCases,
              const std::string &kernelCode, int puzzleIndex) {
    if (puzzleIndex >= kNumTestCases) {
        std::printf("Invalid puzzle index!\n");
        return false;
    }

    const auto &testCases = allTestCases[puzzleIndex];

    bool allPassed = true;
    for (int i = 0; i < testCases.size(); ++i) {
        const auto &testCase = testCases[i];
        std::printf("\n<------------------------------->\n");
        std::printf("Running test case %d...\n", i + 1);

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<float> output;
        switch (puzzleIndex) {
        case 0:
            output = runPuzzle1(ctx, testCase, kernelCode);
            break;
        case 1:
            output = runPuzzle2(ctx, testCase, kernelCode);
            break;
        // Add more cases for additional puzzles
        default:
            std::printf("Invalid puzzle index!\n");
            return false;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::printf("Time taken: %f s\n", elapsed.count());

        if (checkOutput(output, testCase.expectedOutput)) {
            std::printf("Test case %d passed!\n", i + 1);
        } else {
            std::printf("Test case %d failed.\n", i + 1);
            std::printf("Expected: ");
            printVec(testCase.expectedOutput);
            std::printf("Got: ");
            printVec(output);
            allPassed = false;
        }
        std::printf("<------------------------------->\n");
    }
    return allPassed;
}

#endif // EVALUATOR_H
