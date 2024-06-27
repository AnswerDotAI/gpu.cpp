#ifndef LOGGING_H
#define LOGGING_H

#include <cstdio>
#include <cstdarg>

namespace gpu {

enum LogLevel { kError = 0, kWarn = 1, kInfo = 2, kTrace = 3 };

static const char *kLevelStr[] = {"error", "warn", "info", "trace"};

struct Logger {
  FILE *stream;
  char buffer[1024];
  int level;
};

#ifndef NDEBUG
inline void LOG(Logger& logger, int level, const char *message, ...) {
  static const char *orange = "\033[0;33m";
  static const char *red = "\033[0;31m";
  static const char *white = "\033[0;37m";
  static const char *gray = "\033[0;90m";
  static const char *reset = "\033[0m";
  static const char *logColors[] = {red, red, orange, gray};
  if (level <= logger.level) {
    va_list(args);
    va_start(args, message);
    snprintf(logger.buffer, sizeof(logger.buffer), message, args);
    // Brackets and messages are white.
    // Log levells are red for error and warning, orange for info, and grey for trace.
    // Then the color is reset.
    fprintf(logger.stream, "%s[%s%s%s] ", white, logColors[level], kLevelStr[level],
            white);
    vfprintf(logger.stream, message, args);
    fprintf(logger.stream, "%s\n", reset);
    va_end(args);
  }
}
#else
#define LOG(logger, level, message, ...) ((void)0)
#endif

static Logger kDefLog = {stdout, "", kInfo};

} // namespace gpu

#endif
