#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "half.h"

#define EPSILON 0.01f

#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"

int approximatelyEqual(float a, float b, float epsilon) {
  // return fabsf(a - b) <= epsilon * fmaxf(fabsf(a), fabsf(b));
  return fabsf(a - b) <= epsilon;
}

void printResult(int passed, const char *message, float input, float output) {
  if (passed) {
    printf(COLOR_GREEN "PASSED" COLOR_RESET
                       ": %s (input: %.10f, output: %.10f)\n",
           message, input, output);
  } else {
    printf(COLOR_RED "FAILED" COLOR_RESET
                     ": %s (input: %.10f, output: %.10f)\n",
           message, input, output);
  }
}

void printResultHalf(int passed, const char *message, float input,
                     uint16_t output) {
  if (passed) {
    printf(COLOR_GREEN "PASSED" COLOR_RESET
                       ": %s (input: %.10f, output: 0x%04x)\n",
           message, input, output);
  } else {
    printf(COLOR_RED "FAILED" COLOR_RESET
                     ": %s (input: %.10f, output: 0x%04x)\n",
           message, input, output);
  }
}

void testRoundTrip(float value) {
  half h = halfFromFloat(value);
  float result = halfToFloat(h);
  char message[256];

  if (isnan(value)) {
    sprintf(message, "NaN correctly round tripped");
    printResult(isnan(result), message, value, result);
  } else if (isinf(value)) {
    sprintf(message, "Infinity correctly round tripped");
    printResult(isinf(result) &&
                    ((value > 0 && result > 0) || (value < 0 && result < 0)),
                message, value, result);
  } else {
    sprintf(message, "%.10f correctly round tripped", value);
    printResult(approximatelyEqual(result, value, EPSILON), message, value,
                result);
  }
}

void testRoundTripHalf(uint16_t value) {
  half h;
  h.data = value;
  float f = halfToFloat(h);
  half result = halfFromFloat(f);
  char message[256];

  sprintf(message, "half 0x%04x correctly round tripped", value);
  printResult(result.data == value, message, (float)value, result.data);
}

void testSpecialCases() {
  half h;
  char message[256];

  printf("Special cases\n\n");
  // Zero
  h.data = 0x0000;
  sprintf(message, "0x0000 correctly converted to 0.0f");
  printResult(halfToFloat(h) == 0.0f, message, 0.0f, halfToFloat(h));

  // Negative zero
  h.data = 0x8000;
  sprintf(message, "0x8000 correctly converted to -0.0f");
  printResult(halfToFloat(h) == -0.0f, message, -0.0f, halfToFloat(h));

  // Infinity
  h.data = 0x7c00;
  sprintf(message, "0x7c00 correctly converted to positive infinity");
  printResult(isinf(halfToFloat(h)) && halfToFloat(h) > 0, message, INFINITY,
              halfToFloat(h));

  // Negative infinity
  h.data = 0xfc00;
  sprintf(message, "0xfc00 correctly converted to negative infinity");
  printResult(isinf(halfToFloat(h)) && halfToFloat(h) < 0, message, -INFINITY,
              halfToFloat(h));

  // NaN
  h.data = 0x7e00;
  sprintf(message, "0x7e00 correctly converted to NaN");
  printResult(isnan(halfToFloat(h)), message, NAN, halfToFloat(h));

  // Smallest positive normal number
  h.data = 0x0400;
  sprintf(message, "0x0400 correctly converted to 6.10352e-05f");
  printResult(approximatelyEqual(halfToFloat(h), 6.10352e-05f, EPSILON),
              message, 6.10352e-05f, halfToFloat(h));

  // Largest denormalized number
  h.data = 0x03ff;
  sprintf(message, "0x03ff correctly converted to 6.09756e-05f");
  printResult(approximatelyEqual(halfToFloat(h), 6.09756e-05f, EPSILON),
              message, 6.09756e-05f, halfToFloat(h));

  // Smallest positive denormalized number
  h.data = 0x0001;
  sprintf(message, "0x0001 correctly converted to 5.96046e-08f");
  printResult(approximatelyEqual(halfToFloat(h), 5.96046e-08f, EPSILON),
              message, 5.96046e-08f, halfToFloat(h));

  // Zero
  h = halfFromFloat(0.0f);
  sprintf(message, "0.0f correctly converted to 0x0000");
  printResultHalf(h.data == 0x0000, message, 0.0f, h.data);

  // Negative zero
  h = halfFromFloat(-0.0f);
  sprintf(message, "-0.0f correctly converted to 0x8000");
  printResultHalf(h.data == 0x8000, message, -0.0f, h.data);

  // Infinity
  h = halfFromFloat(INFINITY);
  sprintf(message, "positive infinity correctly converted to 0x7c00");
  printResultHalf(h.data == 0x7c00, message, INFINITY, h.data);

  // Negative infinity
  h = halfFromFloat(-INFINITY);
  sprintf(message, "negative infinity correctly converted to 0xfc00");
  printResultHalf(h.data == 0xfc00, message, -INFINITY, h.data);

  // NaN
  h = halfFromFloat(NAN);
  sprintf(message, "NaN correctly converted to NaN representation");
  printResultHalf((h.data & 0x7c00) == 0x7c00 && (h.data & 0x03ff) != 0x0000,
                  message, NAN, h.data);

  // Smallest positive normal number
  h = halfFromFloat(6.10352e-05f);
  sprintf(message, "6.10352e-05f correctly converted to 0x0400");
  printResultHalf(h.data == 0x0400, message, 6.10352e-05f, h.data);

  // Largest denormalized number
  h = halfFromFloat(6.09756e-05f);
  sprintf(message, "6.09756e-05f correctly converted to 0x03ff");
  printResultHalf(h.data == 0x03ff, message, 6.09756e-05f, h.data);

  // Smallest positive denormalized number
  h = halfFromFloat(5.96046448e-08f);
  sprintf(message, "5.96046448e-08f correctly converted to 0x0001");
  printResultHalf(h.data == 0x0001, message, 5.96046e-08f, h.data);
}

int main() {
  printf("Running tests...\n");

  // Regular values
  testRoundTrip(1.0f);
  testRoundTrip(0.5f);
  testRoundTrip(2.0f);
  testRoundTrip(3.14f);
  testRoundTrip(-1.0f);
  testRoundTrip(-0.5f);
  testRoundTrip(-2.0f);
  testRoundTrip(-3.14f);

  // Edge cases
  testRoundTrip(0.0f);
  testRoundTrip(-0.0f);
  testRoundTrip(INFINITY);
  testRoundTrip(-INFINITY);
  testRoundTrip(NAN);
  // testRoundTrip(FLT_MAX); // since FLT_MAX is not representable as half it
  // is not expected to round-trip correctly testRoundTrip(FLT_MIN);
  testRoundTrip(FLT_TRUE_MIN);

  // Special cases for half
  testSpecialCases();

  printf("Tests completed.\n");

  return 0;
}
