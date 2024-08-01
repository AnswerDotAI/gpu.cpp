#include "wasm.h"

int main() {
  // Note: This calls createContext but this doesn't work to obtain the return value
  // due to async
  // Context* ctx = createContext();
  // destroyContext(ctx);

  LOG("Hello, World!");

  return 0;
}
