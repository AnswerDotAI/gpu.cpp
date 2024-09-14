#include <array>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "gpu.hpp"
#include "numeric_types/half.hpp"

using namespace gpu;
using std::isinf;
using std::isnan;

#define EPSILON 0.01f
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"

int approximatelyEqual(float a, float b, float epsilon) {
  return fabsf(a - b) <= epsilon;
}

void printResult(bool passed, const char *message, float input, float output) {
  if (passed) {
    printf("[" COLOR_GREEN "PASSED" COLOR_RESET "]"
           " : %s (in: %.10f, out: %.10f)\n",
           message, input, output);
  } else {
    printf("[" COLOR_RED "FAILED" COLOR_RESET "]"
           " : %s (in: %.10f, out: %.10f)\n",
           message, input, output);
  }
}

void printResult(bool passed, const char *message, float input,
                 uint16_t output) {
  if (passed) {
    printf("[" COLOR_GREEN "PASSED" COLOR_RESET "]"
           " : %s (input: %.10f, output: 0x%04x)\n",
           message, input, output);
  } else {
    printf("[" COLOR_RED "FAILED" COLOR_RESET "]"
           " : %s (input: %.10f, output: 0x%04x)\n",
           message, input, output);
  }
}

void testRoundTrip(float value) {
  // half h = halfFromFloat(value);
  half h = half(value);
  float result = static_cast<float>(h);
  char message[1024];

  if (isnan(value)) {
    snprintf(message, sizeof(message), "NaN correctly round tripped");
    printResult(isnan(result), message, value, result);
  } else if (isinf(value)) {
    snprintf(message, sizeof(message), "Infinity correctly round tripped");
    printResult(isinf(result) &&
                    ((value > 0 && result > 0) || (value < 0 && result < 0)),
                message, value, result);
  } else {
    snprintf(message, sizeof(message), "%.10f correctly round tripped", value);
    printResult(approximatelyEqual(result, value, EPSILON), message, value,
                result);
  }
}

void testRoundTrip(uint16_t value) {
  half h;
  h.data = value;
  float f = halfToFloat(h);
  half result = halfFromFloat(f);
  char message[512];
  snprintf(message, sizeof(message), "half 0x%04x correctly round tripped", value);
  printResult(result.data == value, message, (float)value, result.data);
}

void testRoundTrip(half value) {
  float f = static_cast<float>(value);
  half result = half(f);
  char message[512];
  snprintf(message, sizeof(message), "half 0x%04x correctly round tripped", value.data);
  printResult(result.data == value.data, message, (float)value, result.data);
}

void testSpecialCases() {
  half h;
  char message[512];

  // Zero
  h.data = 0x0000;
  snprintf(message, sizeof(message), "0x0000 correctly converted to 0.0f");
  printResult(halfToFloat(h) == 0.0f, message, 0.0f, halfToFloat(h));

  // Negative zero
  h.data = 0x8000;
  snprintf(message, sizeof(message), "0x8000 correctly converted to -0.0f");
  printResult(halfToFloat(h) == -0.0f, message, -0.0f, halfToFloat(h));

  // Infinity
  h.data = 0x7c00;
  snprintf(message, sizeof(message), "0x7c00 correctly converted to positive infinity");
  printResult(isinf(halfToFloat(h)) && halfToFloat(h) > 0, message, INFINITY,
              halfToFloat(h));

  // Negative infinity
  h.data = 0xfc00;
  snprintf(message, sizeof(message), "0xfc00 correctly converted to negative infinity");
  printResult(isinf(halfToFloat(h)) && halfToFloat(h) < 0, message, -INFINITY,
              halfToFloat(h));

  // NaN
  h.data = 0x7e00;
  snprintf(message, sizeof(message), "0x7e00 correctly converted to NaN");
  printResult(isnan(halfToFloat(h)), message, NAN, halfToFloat(h));

  // Smallest positive normal number
  h.data = 0x0400;
  snprintf(message, sizeof(message), "0x0400 correctly converted to 6.10352e-05f");
  printResult(approximatelyEqual(halfToFloat(h), 6.10352e-05f, EPSILON),
              message, 6.10352e-05f, halfToFloat(h));

  // Largest denormalized number
  h.data = 0x03ff;
  snprintf(message, sizeof(message), "0x03ff correctly converted to 6.09756e-05f");
  printResult(approximatelyEqual(halfToFloat(h), 6.09756e-05f, EPSILON),
              message, 6.09756e-05f, halfToFloat(h));

  // Smallest positive denormalized number
  h.data = 0x0001;
  snprintf(message, sizeof(message), "0x0001 correctly converted to 5.96046e-08f");
  printResult(approximatelyEqual(halfToFloat(h), 5.96046e-08f, EPSILON),
              message, 5.96046e-08f, halfToFloat(h));

  // Zero
  h = halfFromFloat(0.0f);
  snprintf(message, sizeof(message), "0.0f correctly converted to 0x0000");
  printResult(h.data == 0x0000, message, 0.0f, h.data);

  // Negative zero
  h = halfFromFloat(-0.0f);
  snprintf(message, sizeof(message), "-0.0f correctly converted to 0x8000");
  printResult(h.data == 0x8000, message, -0.0f, h.data);

  // Infinity
  h = halfFromFloat(INFINITY);
  snprintf(message, sizeof(message), "positive infinity correctly converted to 0x7c00");
  printResult(h.data == 0x7c00, message, INFINITY, h.data);

  // Negative infinity
  h = halfFromFloat(-INFINITY);
  snprintf(message, sizeof(message), "negative infinity correctly converted to 0xfc00");
  printResult(h.data == 0xfc00, message, -INFINITY, h.data);

  // NaN
  h = halfFromFloat(NAN);
  snprintf(message, sizeof(message), "NaN correctly converted to NaN representation");
  printResult((h.data & 0x7c00) == 0x7c00 && (h.data & 0x03ff) != 0x0000,
              message, NAN, h.data);

  // Smallest positive normal number
  h = halfFromFloat(6.10352e-05f);
  snprintf(message, sizeof(message), "6.10352e-05f correctly converted to 0x0400");
  printResult(h.data == 0x0400, message, 6.10352e-05f, h.data);

  // Largest denormalized number
  h = halfFromFloat(6.09756e-05f);
  snprintf(message, sizeof(message), "6.09756e-05f correctly converted to 0x03ff");
  printResult(h.data == 0x03ff, message, 6.09756e-05f, h.data);

  // Smallest positive denormalized number
  h = halfFromFloat(5.96046448e-08f);
  snprintf(message, sizeof(message), "5.96046448e-08f correctly converted to 0x0001");
  printResult(h.data == 0x0001, message, 5.96046e-08f, h.data);
}

void testContainers() {
  {
    std::array<half, 4> h = {0.0f, -0.0f, INFINITY, NAN};
    testRoundTrip(h[0]);
    testRoundTrip(h[1]);
    testRoundTrip(h[2]);
    testRoundTrip(h[3]);
  }
  {
    Context ctx = createContext();
    std::array<half, 8> h = {1.0f, 0.5f, 2.0f, 3.14f, 1.0, 2.0, 3.0, 4.0};
    Tensor devH = createTensor(ctx, {h.size()}, kf16, h.data());
    std::array<half, 8> h2;
    toCPU(ctx, devH, h2.data(), sizeof(h2));
    for (int i = 0; i < 8; ++i) {
      printResult(h[i].data == h2[i].data, "Container round trip",
                  static_cast<float>(h[i]), static_cast<float>(h2[i]));
    }
  }
}

void testWGSL() {
  static const char *kGelu = R"(
const GELU_SCALING_FACTOR: f16 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> out: array<{{precision}}>;
@group(0) @binding(1) var<storage, read_write> dummy: array<{{precision}}>;
@compute @workgroup_size({{workgroupSize}})
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < arrayLength(&inp)) {
        let x: f16 = inp[i];
        out[i] = select(0.5 * x * (1.0 + tanh(GELU_SCALING_FACTOR 
                 * (x + .044715 * x * x * x))), x, x > 10.0);
    }
}
)";
  Context ctx = createContext(
      {}, {},
      /*device descriptor, enabling f16 in WGSL*/
      {
          .requiredFeatureCount = 1,
          .requiredFeatures = std::array{WGPUFeatureName_ShaderF16}.data(),
      });
  static constexpr size_t N = 10000;
  std::array<half, N> inputArr, outputArr;
  for (int i = 0; i < N; ++i) {
    inputArr[i] = half(static_cast<float>(i) / 10.0f); // dummy input data
  }
  Tensor input = createTensor(ctx, Shape{N}, kf16, inputArr.data());
  Tensor output = createTensor(ctx, Shape{N}, kf16);
  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  Kernel op = createKernel(ctx, {kGelu, 256, kf16}, Bindings{input, output},
                           {cdiv(N, 256), 1, 1});
  dispatchKernel(ctx, op, promise);
  wait(ctx, future);
  toCPU(ctx, output, outputArr.data(), sizeof(outputArr));
  for (int i = 0; i < 12; ++i) {
    printf("  gelu(%.2f) = %.2f\n", static_cast<float>(inputArr[i]),
           static_cast<float>(outputArr[i]));
  }
}

int main() {
  printf("\nHalf-precision float tests\n==========================\n");

  printf("\nRegular values float round trips\n\n");
  testRoundTrip(1.0f);
  testRoundTrip(0.5f);
  testRoundTrip(2.0f);
  testRoundTrip(3.14f);
  testRoundTrip(-1.0f);
  testRoundTrip(-0.5f);
  testRoundTrip(-2.0f);
  testRoundTrip(-3.14f);

  printf("\nEdge Case float round trips\n\n");
  testRoundTrip(0.0f);
  testRoundTrip(-0.0f);
  testRoundTrip(INFINITY);
  testRoundTrip(-INFINITY);
  testRoundTrip(NAN);
  // testRoundTrip(FLT_MAX); // since FLT_MAX is not representable as half it
  // is not expected to round-trip correctly testRoundTrip(FLT_MIN);
  testRoundTrip(FLT_TRUE_MIN);

  printf("\nSpecial half values\n\n");
  testSpecialCases();

  printf("\nContainers and CPU/GPU round trip\n\n");
  testContainers();

  printf("\nWGSL f16 extension test\n\n");
  testWGSL();

  printf("\nTests completed.\n");

  return 0;
}
