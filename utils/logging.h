#ifndef LOGGING_H
#define LOGGING_H

#include <cstdio>
#include <cstdarg>

namespace gpu {

enum LogLevel { kError = 0, kWarn = 1, kInfo = 2, kTrace = 3 };

static const char *kLevelStr[] = {"error", "warn", "info", "trace"};

/**
 * @brief Logger struct for logging messages.
 * stream: The stream to log to.
 * buffer: A buffer to store the formatted message.
 * level: The log level to log messages at.
 */
struct Logger {
  FILE *stream;
  char buffer[32768]; // TODO(avh): Expand as needed or fail gracefully.
  int level;
};

#ifndef NDEBUG
/**
 * @brief Log a message to the logger. If NDEBUG is defined in a source or as a
 * compiler flag, this is a no-op.
 *
 * @param logger The logger to log to.
 * @param level The log level of the message.
 * @param message The message to log.
 */
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
    // Log levels are red for error and warning, orange for info, and grey for trace.
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

/**
 * @brief Default logger for logging messages to stdout at the info level.
 * Output stream and logging level for the default logger can be globally
 * changed on a per-program basis.
 */
static Logger kDefLog = {stdout, "", kInfo};

/**
 * @brief Set the log level of the default logger.
 * @param level The log level to set.
 */
static inline void setLogLevel(int level) {
  kDefLog.level = level;
}

} // namespace gpu

#endif
