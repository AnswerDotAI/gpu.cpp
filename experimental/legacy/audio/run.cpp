#include <array>
#include <cmath>
#include <cstdio>

#include "gpu.hpp"
#include "portaudio.h"

#define SAMPLE_RATE (44100)
#define PA_SAMPLE_TYPE paFloat32
#define FRAMES_PER_BUFFER (1024)

typedef float SAMPLE;

struct Buffer {
  float *buffer; // non-owning pointer into buffer
  size_t size;
};


float sigmoid(float x) {
  return 1 / (1 + std::exp(-4 * x)) - 0.5;
}

static int gNumNoInputs = 0;
static int callback(const void *inputBuffer, void *outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo *timeInfo,
                        PaStreamCallbackFlags statusFlags, void *userData) {
  SAMPLE* out = (SAMPLE *)outputBuffer;
  const SAMPLE *in = (const SAMPLE *)inputBuffer;
  (void)timeInfo; /* Prevent unused variable warnings. */
  (void)statusFlags;
  (void)userData;

  Buffer *buffer = reinterpret_cast<Buffer *>(userData);
  size_t timeIndex = (timeInfo->currentTime /* in seconds */ * SAMPLE_RATE) /
                     FRAMES_PER_BUFFER * FRAMES_PER_BUFFER;

  int scale = 1;
  size_t reverseIndex = buffer->size - scale * timeIndex;

  if (inputBuffer == NULL) {
    for (int i = 0; i < framesPerBuffer; i++) {
      *out++ = sigmoid(0);
      *out++ = sigmoid(0);
    }
    gNumNoInputs += 1;
  } else {
    for (int i = 0; i < framesPerBuffer; i++) {
      size_t playHead0 = (timeIndex + i) % buffer->size;
      size_t playHead1 = (reverseIndex - scale * i ) % buffer->size; // reverse playhead

      SAMPLE sample = *in++; /* MONO input */

      float value = sigmoid(0.5 * sample + 0.5 * buffer->buffer[playHead1]);

      *out++ = value; /* LEFT */
      *out++ = value; /* RIGHT */
      buffer->buffer[playHead0] = sample;

      printf("\033[H\033[H\n\nTime = %f\nplayHead0 = %zu\nplayHead1 (reverse) "
             "index=%zu\noutput value=%.2f\ninput value=%.2f\nplayhead1 value =%.2f\n",
             timeInfo->currentTime, playHead0, playHead1, value,
             sample,
             buffer->buffer[playHead1]);
    }
  }

  return paContinue;
}

void check(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    Pa_Terminate();
    exit(1);
  }
}

int main(void) {
  PaStreamParameters inputParameters, outputParameters;
  PaStream *stream;
  PaError err;

  printf("\033[H\033[J");

  printf("Turn down volume before starting.\nPress Enter to start.");
  getchar();

  printf("\033[H\033[J");

  err = Pa_Initialize();
  check(err == paNoError, "Error: Pa_Initialize failed.");

  // Setup device

  inputParameters = {
      .device = Pa_GetDefaultInputDevice(),
      .channelCount = 1,
      .sampleFormat = PA_SAMPLE_TYPE,
      .suggestedLatency =
          Pa_GetDeviceInfo(Pa_GetDefaultInputDevice())->defaultLowInputLatency,
      .hostApiSpecificStreamInfo = NULL};

  check(inputParameters.device != paNoDevice,
        "Error: No default input device.");

  outputParameters = {.device = Pa_GetDefaultOutputDevice(),
                      .channelCount = 2,
                      .sampleFormat = PA_SAMPLE_TYPE,
                      .suggestedLatency =
                          Pa_GetDeviceInfo(Pa_GetDefaultOutputDevice())
                              ->defaultLowOutputLatency,
                      .hostApiSpecificStreamInfo = NULL};

  if (outputParameters.device == paNoDevice) {
    fprintf(stderr, "Error: No default output device.\n");
    exit(1);
  }

  constexpr size_t kBufferTime = 4; // seconds
  std::array<float, SAMPLE_RATE * kBufferTime> bufferAlloc;
  // zero
  for (size_t i = 0; i < bufferAlloc.size(); i++) {
    bufferAlloc[i] = 0;
  }

  Buffer buffer = {bufferAlloc.data(), bufferAlloc.size()};

  err = Pa_OpenStream(&stream, &inputParameters, &outputParameters, SAMPLE_RATE,
                      FRAMES_PER_BUFFER, 0,
                      /* paClipOff, */ /* we won't output out of range samples
                                          so don't bother clipping them */
                      callback, reinterpret_cast<void *>(&buffer));
  check(err == paNoError, "Error: Pa_OpenStream failed.");

  err = Pa_StartStream(stream);
  check(err == paNoError, "Error: Pa_StartStream failed.");

  printf("Hit Enter to stop program.\n");
  getchar();
  err = Pa_CloseStream(stream);
  check(err == paNoError, "Error: Pa_CloseStream failed.");

  printf("Finished. gNumNoInputs = %d\n", gNumNoInputs);
  Pa_Terminate();
  return 0;
}
