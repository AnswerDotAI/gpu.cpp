#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include "gpu.h"

using namespace gpu;

static const char* kAsciiBanner = R"(
   ____ _____  __  __ _________  ____ 
  / __ `/ __ \/ / / // ___/ __ \/ __ \
 / /_/ / /_/ / /_/ // /__/ /_/ / /_/ /
 \__, / .___/\__,_(_)___/ .___/ .___/ 
/____/_/               /_/   /_/
)";

int main(int argc, char **argv) {

  // Clear screen and print banner
  fprintf(stdout, "\033[2J\033[1;1H");
  fprintf(stdout, "%s\n", kAsciiBanner);

  // Creating a GPUContext
  

  static const char* kInstructions1 = R"(
Welcome! This program is a brief intro to the gpu.cpp library.

You can use the library by simply including the gpu.h header, starting with a
build template (see examples/hello_gpu/ for a template project that builds the
library).

  #include "gpu.h"

In your program, you can create a GPUContext like this:

  GPUContext ctx = gpu::GPUContext();

Let's try doing that in this program now.)";
  fprintf(stdout, "%s\n", kInstructions1);

  GPUContext ctx = CreateGPUContext();
  fprintf(stdout, "Successfully created a GPUContext.\n\n");
  fprintf(stdout, "Press Enter to continue ...\n");
  getchar();

  // TODO: Add more examples here.

  fprintf(stdout, "Goodbye!\n");
}
