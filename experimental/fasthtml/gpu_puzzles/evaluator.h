#ifndef TESTING_H
#define TESTING_H

#include "gpu.h"
#include "utils/array_utils.h"
#include <array>
#include <cstdio>
#include <future>
#include <iostream>

using namespace gpu;

struct TestCase {
  std::vector<float> input;
  std::vector<float> expectedOutput;
  std::array<int, 3> workgroupSize = {1, 1, 1};
  std::array<int, 3> gridSize = {1, 1, 1};
  std::array<int, 3> sharedMemorySize = {1, 1, 1};
};

struct Puzzle {
  std::vector<TestCase> testCases;
  std::function<std::vector<float>(
      Context &, std::vector<float>, std::vector<float>, std::array<int, 3>,
      std::array<int, 3>, std::array<int, 3>, std::string)>
      hostCode;
};

// Class to handle the evaluation
class Evaluator {
public:
  Evaluator() {
    // Initialize test cases for 14 puzzles
    initTestCases();
  }

  bool evaluate(Context &ctx, std::string kernelCode, int puzzleIndex) {
    if (puzzleIndex >= puzzles.size()) {
      std::cout << "Invalid puzzle index!\n";
      return false;
    }

    auto &puzzle = puzzles[puzzleIndex];

    // Optionally execute the custom function before running test cases

    bool allPassed = true;
    for (int i = 0; i < puzzle.testCases.size(); ++i) {
      auto &testCase = puzzle.testCases[i];
      std::cout << "\n<------------------------------->\n";
      std::cout << "Running test case " << i + 1 << "...\n";

      auto start = std::chrono::high_resolution_clock::now();

      std::vector<float> output = puzzle.hostCode(
          ctx, puzzle.testCases[i].input, puzzle.testCases[i].expectedOutput,
          puzzle.testCases[i].workgroupSize, puzzle.testCases[i].gridSize,
          puzzle.testCases[i].sharedMemorySize, kernelCode);

      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> elapsed = end - start;
      std::cout << "Time taken: " << elapsed.count() << " s\n";

      if (checkOutput(output, testCase.expectedOutput)) {
        std::cout << "Test case " << i + 1 << " passed!\n";
      } else {
        std::cout << "Test case " << i + 1 << " failed.\n";
        std::cout << "Expected: ";
        printVector(testCase.expectedOutput);
        std::cout << "Got: ";
        printVector(output);
        allPassed = false;
      }
      std::cout << "<------------------------------->\n";
    }
    return allPassed;
  }

private:
  std::vector<Puzzle> puzzles;

  // Initialize the test cases
  void initTestCases() {
    // Fill in your test cases here
    // Example:
    Puzzle puzzle1;
    // input, expected output, workgroup size, grid size, shared memory size
    puzzle1.testCases.push_back({{0, 1, 2, 3}, {10, 11, 12, 13}});
    puzzle1.testCases.push_back(
        {{0, 1, 2, 3, 4, 5, 6, 7, 8}, {10, 11, 12, 13, 14, 15, 16, 17, 18}});
    puzzle1.testCases.push_back({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                 {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}});

    puzzle1.hostCode = [](Context &ctx, std::vector<float> input,
                          std::vector<float> expectedOutput,
                          std::array<int, 3> workgroupSize,
                          std::array<int, 3> gridSize,
                          std::array<int, 3> sharedMemorySize,
                          std::string kernelString) {
      printf("Puzzle 1\n");

      size_t N = input.size();

      Tensor a = createTensor(ctx, {N}, kf32, input.data());
      Tensor output = createTensor(ctx, {N}, kf32);

      Kernel op = createKernel(ctx, {kernelString, N}, Bindings{a, output},
                               /*nWorkgroups */
                               {static_cast<unsigned long>(gridSize[0]),
                                static_cast<unsigned long>(gridSize[1]),
                                static_cast<unsigned long>(gridSize[2])});

      std::promise<void> promise;
      std::future<void> future = promise.get_future();

      dispatchKernel(ctx, op, promise);

      std::vector<float> outputArr(N);
      wait(ctx, future);
      toCPU(ctx, output, outputArr.data(), outputArr.size() * sizeof(float));

      return outputArr;
    };

    puzzles.push_back(puzzle1);

    Puzzle puzzle2;

    puzzle2.testCases.push_back({{{0, 1, 2, 0}}, {2, 1}});
    puzzle2.testCases.push_back(
        {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}, {5, 7, 9, 11, 13}});

    puzzle2.hostCode = [](Context &ctx, std::vector<float> input,
                          std::vector<float> expectedOutput,
                          std::array<int, 3> workgroupSize,
                          std::array<int, 3> gridSize,
                          std::array<int, 3> sharedMemorySize,
                          std::string kernelString) {
      printf("Puzzle 2\n");

      size_t N = input.size() / 2;

      std::vector<float> aVec(input.begin(), input.begin() + N);
      std::vector<float> bVec(input.begin() + N, input.end());

      Tensor a = createTensor(ctx, {N}, kf32, aVec.data());
      Tensor b = createTensor(ctx, {N}, kf32, bVec.data());
      Tensor output = createTensor(ctx, {N}, kf32);

      Kernel op = createKernel(ctx, {kernelString, N}, Bindings{a, b, output},
                               /*nWorkgroups */
                               {static_cast<unsigned long>(gridSize[0]),
                                static_cast<unsigned long>(gridSize[1]),
                                static_cast<unsigned long>(gridSize[2])});

      std::promise<void> promise;
      std::future<void> future = promise.get_future();

      dispatchKernel(ctx, op, promise);

      std::vector<float> outputArr(N);
      wait(ctx, future);
      toCPU(ctx, output, outputArr.data(), outputArr.size() * sizeof(float));

      return outputArr;
    };

    puzzles.push_back(puzzle2);
    // Add more test cases for the 14 puzzles
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
  void printVector(const std::vector<float> &vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      std::cout << vec[i];
      if (i != vec.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]\n";
  }
};

#endif // EVALUATOR_H
