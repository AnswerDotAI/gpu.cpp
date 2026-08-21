#ifndef GPU_CPP_UTILS_TUI_HPP
#define GPU_CPP_UTILS_TUI_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

// Work-in-progress - various terminal UI visualization functions

namespace gpu {

inline void cls() { printf("\033[2J\033[H"); }

template <size_t NROWS, size_t NCOLS>
void canvas(const std::array<char, NROWS * NCOLS> &raster) {
  printf("+");
  for (size_t col = 0; col < NCOLS; ++col) {
    printf("-");
  }
  printf("+\n");
  for (size_t row = 0; row < NROWS; ++row) {
    printf("|");
    for (size_t col = 0; col < NCOLS; ++col) {
      printf("%c", raster[row * NCOLS + col]);
    }
    printf("|\n");
  }
  printf("+");
  for (size_t col = 0; col < NCOLS; ++col) {
    printf("-");
  }
  printf("+\n");
}

// double pendulum rasterizer
inline void rasterize(const float *pos, size_t n, float maxX, float maxY,
                      std::string &screen, size_t screenWidth,
                      size_t screenHeight) {
  static const char intensity[] = " .`'^-+=*x17X$8#%@";
  const size_t eps = 1;
  // maximum number of simulations to display on the screen
  const size_t nShow = std::min(n, size_t{2000});
  for (size_t i = 0; i < screenHeight; ++i) {
    for (size_t j = 0; j < screenWidth - 2; ++j) {
      int count = 0;
      for (size_t k = 0; k < 2 * nShow; k += 2) {
        float nx =
            (1.0 + pos[k] / maxX) / 2.0 * static_cast<float>(screenWidth);
        // negate y since it extends from top to bottom
        float ny = (1.0 - (pos[k + 1] / maxY)) / 2.0 *
                   static_cast<float>(screenHeight);
        float length = std::sqrt((nx - j) * (nx - j) + (ny - i) * (ny - i));
        if (length < eps) {
          count++;
        }
      }
      count = std::min(count / 2, 17); // Need to adjust  /2 scaling for different n
      screen[i * screenWidth + j] = intensity[count];
    }
    screen[i * screenWidth + screenWidth - 1] = '\n';
  }
}




} // namespace gpu

#endif
