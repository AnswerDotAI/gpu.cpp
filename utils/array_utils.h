/*
 * array_utils.h
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

#include "utils/logging.h"

namespace gpu {

static constexpr int kShowMaxRows = 14;
static constexpr int kShowMaxCols = 8;

template <typename numtype>
std::string show(const numtype* a, size_t rows, size_t cols, const std::string& name = "") {
  std::string output = "\n";
  if (name != "") {
    output += name + " (" + std::to_string(rows) + ", " + std::to_string(cols) + ")\n";
  } else {
    output += "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")\n";
  }
  // spacing as log10 of max value
  int spacing = 1;
  numtype max = *std::max_element(a, a + rows * cols);
  if constexpr (std::is_same<numtype, int>::value) {
    spacing = std::max(0, (int)log10(max + .01)) + 2;
  } else if constexpr (std::is_same<numtype, float>::value) {
    // spacing = std::max(0, (int)log10(max + .01)) + 1;
    spacing = 8; // scientific notation
  } else {
    throw std::runtime_error("Unsupported number type for show()");
  }
  // print to stdout line break for each row
  for (size_t i = 0; i < rows; i++) {
    if (i == kShowMaxRows / 2 && rows > kShowMaxRows) {
      output += "...\n";
      i = rows - kShowMaxRows  /2;
    }
    for (size_t j = 0; j < cols; j++) {
      if (j == kShowMaxCols / 2 && cols > kShowMaxCols) {
        output += " .. ";
        j = cols - kShowMaxCols/2;
      }
      char buffer[50];
      if constexpr (std::is_same<numtype, int>::value) {
        sprintf(buffer, "%*d", spacing, a[i * cols + j]);
      } else if constexpr (std::is_same<numtype, float>::value) {
        if (std::abs(a[i * cols + j]) < 1000 && std::abs(a[i * cols + j]) > 0.01 || a[i * cols + j] == 0.0) {
          sprintf(buffer, "%10.2f", a[i * cols + j]);
        } else
        sprintf(buffer, "%10.2e", a[i * cols + j]);
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

template <typename numtype, size_t rows, size_t cols>
std::string show(const std::array<numtype, rows * cols>& a, const std::string& name = "") {
  return show<numtype>(a.data(), rows, cols, name);
}

template <size_t rows, size_t cols>
std::string show(const std::array<float, rows * cols>& a, const std::string& name = "") {
  return show<float, rows, cols>(a, name);
}


// For testing only, not optimized
inline void transpose(float* input, float* output, size_t M, size_t N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      output[j * M + i] = input[i * N + j];
    }
  }
}

void range(float* input, size_t N, float start = 0.0, float step = 1.0) {
  // TODO(avh): currently unused - check
  float curr = start;
  for (size_t i = 0; i < N; i++) {
    input[i] = curr;
    curr += step;
  }
}

template <size_t N> void range(std::array<float, N> &input, float start = 0.0, float step = 1.0) {
  float curr = start;
  for (size_t i = start; i < N; i++) {
    input[i] = curr;
    curr += step;
  }
}

void randint(float* a, size_t N, std::mt19937 &gen, int min=-1, int max=1) {
  std::uniform_int_distribution<> dist(min, max);
  for (int i = 0; i < N; i++) {
    a[i] = static_cast<float>(dist(gen));
  }
}

template <typename numtype, size_t size>
void randint(std::array<numtype, size> &a, std::mt19937 &gen, int min=-1,
             int max=1) {
  std::uniform_int_distribution<> dist(min, max);
  for (int i = 0; i < size; i++) {
    a[i] = static_cast<numtype>(dist(gen));
  }
}

void randn(float* a, size_t N, std::mt19937 &gen, float mean = 0.0,
             float std=1.0) {
  std::normal_distribution<float> dist(mean, std);
  for (int i = 0; i < N; i++) {
    a[i] = static_cast<float>(dist(gen));
  }
}

template <size_t size>
void randn(std::array<float, size> &a, std::mt19937 &gen, float mean = 0.0,
             float std=1.0) {
  std::normal_distribution<float> dist(mean, std);
  for (int i = 0; i < size; i++) {
    a[i] = static_cast<float>(dist(gen));
  }
}


inline void eye(float* a, size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      a[i * N + j] = (i == j) ? 1.0 : 0.0;
    }
  }
}

inline void flip(float* a, size_t R, size_t C, bool horizontal = true) {
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

bool isclose(float *a, float *b, size_t n, float tol = 1e-3) {
  for (size_t i = 0; i < n; i++) {
    if (std::abs(a[i] - b[i]) > tol || std::isnan(a[i]) || std::isnan(b[i])) {
      LOG(kDefLog, kError, "Mismatch at index %d: %f != %f", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

} // namespace gpu

#endif 
