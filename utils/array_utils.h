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
#include <string>
#include <utility>

namespace gpu {

static constexpr int kShowMaxRows = 14;
static constexpr int kShowMaxCols = 8;

template <typename numtype, size_t rows, size_t cols>
std::string show(const std::array<numtype, rows * cols>& a, const std::string& name = "") {
  std::string output = "\n";
  if (name != "") {
    output += name + " (" + std::to_string(rows) + ", " + std::to_string(cols) + ")\n";
  } else {
    output += std::to_string(rows) + ", " + std::to_string(cols) + "\n";
  }
  // spacing as log10 of max value
  int spacing = 1;
  numtype max = *std::max_element(a.begin(), a.end());
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


// For testing only, not optimized
void transpose(float* input, float* output, size_t M, size_t N) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      output[j * M + i] = input[i * N + j];
    }
  }
}


template <size_t N> void range(std::array<float, N> &input, int start = 0) {
  for (size_t i = start; i < N; i++) {
    input[i] = static_cast<float>(i);
  }
}


template <typename numtype, size_t size>
void randint(std::array<numtype, size> &a, std::mt19937 &gen, int min,
             int max) {
  std::uniform_int_distribution<> dist(min, max);
  for (int i = 0; i < size; i++) {
    a[i] = static_cast<numtype>(dist(gen));
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


void eye(float* a, size_t N) {
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      a[i * N + j] = (i == j) ? 1.0 : 0.0;
    }
  }
}

void flip(float* a, size_t R, size_t C, bool horizontal = true) {
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

} // namespace gpu

#endif 
