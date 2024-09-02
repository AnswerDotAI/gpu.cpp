/*
 * array_utils.hpp
 *
 * This file contains utility functions for working with arrays. These are
 * mostly convenience functions for setting up and inspecting data for testing.
 * They are not optimized and are not intended for use in performance-critical
 * code.
 *
 */

#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

#include <algorithm> // std::max_element
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include "utils/logging.hpp"
#include "numeric_types/half.hpp"

namespace gpu {

static constexpr int kShowMaxRows = 8;
static constexpr int kShowMaxCols = 8;

/**
 * @brief Show a 2D array as a string, base implementation.
 *
 * @param a The array to show.
 * @param rows The number of rows in the array.
 * @param cols The number of columns in the array.
 * @param name The name of the array to show.
 * @return std::string The string representation of the array.
 * @code
 *   std::array<float, 4> a = {1.0, 2.0, 3.0, 4.0};
 *   printf("%s", show<float>(a.data(), 2, 2, "a").c_str());
 * @endcode
 */
template <typename numtype>
std::string show(const numtype *a, size_t rows, size_t cols,
                 const std::string &name = "") {
  std::string output = "\n";
  if (name != "") {
    output += "\n" + name + " (" + std::to_string(rows) + ", " +
              std::to_string(cols) + ")\n\n";
  } else {
    output +=
        "\n(" + std::to_string(rows) + ", " + std::to_string(cols) + ")\n\n";
  }
  // spacing as log10 of max value
  int spacing = 1;
  if constexpr (std::is_same<numtype, int>::value) {
    int max = *std::max_element(a, a + rows * cols);
    spacing = std::max(0, (int)log10(max + .01)) + 2;
  } else if constexpr (std::is_same<numtype, float>::value) {
    // spacing = std::max(0, (int)log10(max + .01)) + 1;
    spacing = 8; // scientific notation
  } else if constexpr (std::is_same<numtype, half>::value) {
    spacing = 8;
  } else {
    throw std::runtime_error("Unsupported number type for show()");
  }
  // print to stdout line break for each row
  for (size_t i = 0; i < rows; i++) {
    if (i == kShowMaxRows / 2 && rows > kShowMaxRows) {
      output += "...\n";
      i = rows - kShowMaxRows / 2;
    }
    for (size_t j = 0; j < cols; j++) {
      if (j == kShowMaxCols / 2 && cols > kShowMaxCols) {
        output += " ..";
        j = cols - kShowMaxCols / 2;
      }
      char buffer[50];
      if constexpr (std::is_same<numtype, int>::value) {
        snprintf(buffer, spacing, "%*d", spacing, a[i * cols + j]);
      } else if constexpr (std::is_same<numtype, float>::value) {
        if (std::abs(a[i * cols + j]) < 1000 &&
                std::abs(a[i * cols + j]) > 0.01 ||
            a[i * cols + j] == 0.0) {
          snprintf(buffer, 16, "%9.2f", a[i * cols + j]);
        } else
          snprintf(buffer, 16, "%10.2e", a[i * cols + j]);
      } else if constexpr (std::is_same<numtype, half>::value) {
	float tmp = halfToFloat(a[i * cols + j]);
        if (std::abs(tmp) < 1000 &&
                std::abs(tmp) > 0.01 ||
            tmp == 0.0) {
          snprintf(buffer, 16, "%9.2f", tmp);
        } else
          snprintf(buffer, 16, "%10.2e", tmp);
      } else {
        throw std::runtime_error("Unsupported number type for show()");
      }
      output += buffer;
    }
    output += "\n";
  }
  output += "\n";
  return output;
}

/**
 * @brief Overload of `show()` for std::array.
 *
 * @param a The array to show.
 * @param name The name of the array to show.
 * @return std::string The string representation of the array.
 * @code
 *  std::array<float, 4> a = {1.0, 2.0, 3.0, 4.0};
 *  printf("%s", show<float>(a, "a").c_str());
 * @endcode
 */
template <typename numtype, size_t rows, size_t cols>
std::string show(const std::array<numtype, rows * cols> &a,
                 const std::string &name = "") {
  return show<numtype>(a.data(), rows, cols, name);
}

/**
 * @brief Overload of `show()` for float std::array.
 * @param a The array to show.
 * @param name The name of the array to show.
 * @return std::string The string representation of the array.
 *
 * @code
 * std::array<float, 4> a = {1.0, 2.0, 3.0, 4.0};
 * printf("%s", show(a, "a").c_str());
 * @endcode
 @  
 */
template <size_t rows, size_t cols>
std::string show(const std::array<float, rows * cols> &a,
                 const std::string &name = "") {
  return show<float, rows, cols>(a, name);
}

/**
 * @brief Populate the array with a range of values. This is mostly for testing
 * purposes.
 * @param input The array to populate.
 * @param N The number of elements in the array.
 * @param start The starting value.
 * @param step The step size.
 */
void range(float *input, size_t N, float start = 0.0, float step = 1.0) {
  // TODO(avh): currently unused - check
  float curr = start;
  for (size_t i = 0; i < N; i++) {
    input[i] = curr;
    curr += step;
  }
}

/**
 * @brief Overload of `range()` for std::array.
 * @param input The array to populate.
 * @param start The starting value.
 * @param step The step size.
 */
template <size_t N>
void range(std::array<float, N> &input, float start = 0.0, float step = 1.0) {
  float curr = start;
  for (size_t i = start; i < N; i++) {
    input[i] = curr;
    curr += step;
  }
}

/**
 * @brief Populate the array with random integers.
 * @param a The array to populate.
 * @param N The number of elements in the array.
 * @param gen The random number generator.
 * @param min The minimum value for the random integers.
 * @param max The maximum value for the random integers.
 */
void randint(float *a, size_t N, std::mt19937 &gen, int min = -1, int max = 1) {
  std::uniform_int_distribution<> dist(min, max);
  for (int i = 0; i < N; i++) {
    a[i] = static_cast<float>(dist(gen));
  }
}

/**
 * @brief Overload of `randint()` for std::array.
 * @param a The array to populate.
 * @param gen The random number generator.
 * @param min The minimum value for the random integers.
 * @param max The maximum value for the random integers.
 */
template <typename numtype, size_t size>
void randint(std::array<numtype, size> &a, std::mt19937 &gen, int min = -1,
             int max = 1) {
  std::uniform_int_distribution<> dist(min, max);
  for (int i = 0; i < size; i++) {
    a[i] = static_cast<numtype>(dist(gen));
  }
}

/**
 * @brief Populate the array with random floats, generated from a Gaussian distribution.
 * @param a The array to populate.
 * @param N The number of elements in the array.
 * @param gen The random number generator.
 * @param mean The mean of the Gaussian distribution.
 * @param std The standard deviation of the Gaussian distribution.
 */
inline void randn(float *a, size_t N, std::mt19937 &gen, float mean = 0.0,
                  float std = 1.0) {
  std::normal_distribution<float> dist(mean, std);
  for (int i = 0; i < N; i++) {
    a[i] = static_cast<float>(dist(gen));
  }
}

inline void randn(half *a, size_t N, std::mt19937 &gen, float mean = 0.0,
                  float std = 1.0) {
  std::normal_distribution<float> dist(mean, std);
  for (int i = 0; i < N; i++) {
    a[i] = halfFromFloat(dist(gen));
  }
}

/**
 * @brief Overload of `randn()` for std::array.
 * @param a The array to populate.
 * @param gen The random number generator.
 * @param mean The mean of the Gaussian distribution.
 * @param std The standard deviation of the Gaussian distribution.
 */
template <size_t size>
void randn(std::array<float, size> &a, std::mt19937 &gen, float mean = 0.0,
           float std = 1.0) {
  std::normal_distribution<float> dist(mean, std);
  for (int i = 0; i < size; i++) {
    a[i] = static_cast<float>(dist(gen));
  }
}

/**
 * @brief Populate a square matrix with the identity matrix.
 * @param a The array to populate.
 * @param N The number of rows and columns in the square matrix.
 */
inline void eye(float *a, size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      a[i * N + j] = (i == j) ? 1.0 : 0.0;
    }
  }
}

// Note transformation operations here are purely for testing - they are not
// optimized to be used in hot paths.

/**
 * @brief Transpose a matrix.
 * @param input The input matrix.
 * @param output The output matrix.
 * @param M The number of rows in the input matrix.
 * @param N The number of columns in the input matrix.
 */
inline void transpose(float *input, float *output, size_t M, size_t N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      output[j * M + i] = input[i * N + j];
    }
  }
}

inline void transpose(half *input, half *output, size_t M, size_t N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      output[j * M + i] = input[i * N + j];
    }
  }
}

/**
 * @brief Flip a matrix horizontally or vertically.
 * @param a The matrix to flip.
 * @param R The number of rows in the matrix.
 * @param C The number of columns in the matrix.
 * @param horizontal Whether to flip horizontally (true) or vertically (false).
 */
inline void flip(float *a, size_t R, size_t C, bool horizontal = true) {
  if (horizontal) {
    for (size_t i = 0; i < R; i++) {
      for (size_t j = 0; j < C / 2; j++) {
        std::swap(a[i * C + j], a[i * C + C - j - 1]);
      }
    }
  } else {
    for (size_t i = 0; i < R / 2; i++) {
      for (size_t j = 0; j < C; j++) {
        std::swap(a[i * C + j], a[(R - i - 1) * C + j]);
      }
    }
  }
}

/**
 * @brief Determine if the values of two arrays are close to each other.
 * @param a The first array.
 * @param b The second array.
 * @param n The number of elements in the arrays.
 * @param tol The tolerance for closeness.
 * @return bool True if the arrays are close, false otherwise.
 */
inline bool isclose(float *a, float *b, size_t n, float tol = 1e-3) {
  for (size_t i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) > tol || std::isnan(a[i]) || std::isnan(b[i])) {
      LOG(kDefLog, kError, "Mismatch at index %d: %f != %f", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

inline bool isclose(half *a, half *b, size_t n, float tol = 1) {
  for (size_t i = 0; i < n; i++) {
    float ai = halfToFloat(a[i]);
    float bi = halfToFloat(b[i]);
    if (std::abs(ai - bi) > tol || std::isnan(ai) || std::isnan(bi)) {
      LOG(kDefLog, kError, "Mismatch at index %d: %f != %f", i, ai, bi);
      return false;
    }
  }
  return true;
}

} // namespace gpu

#endif
