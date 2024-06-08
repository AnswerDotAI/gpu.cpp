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

void log(Logger& logger, int level, const char *message, ...) {
  static const char *orange = "\033[0;33m";
  static const char *red = "\033[0;31m";
  static const char *white = "\033[0;37m";
  static const char *gray = "\033[0;90m";
  static const char *logColors[] = {red, red, orange, gray};
  if (level <= logger.level) {
    va_list(args);
    va_start(args, message);
    snprintf(logger.buffer, sizeof(logger.buffer), message, args);
    fprintf(logger.stream, "[%s%s%s] ", logColors[level], kLevelStr[level],
            white);
    vfprintf(logger.stream, message, args);
    fprintf(logger.stream, "\n");
    va_end(args);
  }
}

static Logger kDefLog = {stdout, "", kInfo};

} // namespace gpu

#endif
