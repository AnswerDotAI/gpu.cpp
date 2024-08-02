// Hello world llvm wasm test

extern "C" {
  int add(int a, int b) { return a + b; }
  int mul(int a, int b) { return a * b; }
  int foo(int a, int b) { return a * a + b + 4; }
}
