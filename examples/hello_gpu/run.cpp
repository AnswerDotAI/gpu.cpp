#include <cstdio>
#include "gpu.h"

using namespace gpu;

int main(int argc, char **argv) {
  GPUContext ctx = CreateGPUContext();
  fprintf(stdout, "Hello, World!\n");
  return 0;
}
