#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <sstream>
#include <cstdarg>
#include <memory>

namespace gpu
{
  enum LogLevel
  {
    kNone = 0,
    kError = 1,
    kWarn = 2,
    kInfo = 3,
    kTrace = 4
  };

  static const char *kLevelStr[] = {"none", "error", "warn", "info", "trace"};

  /**
   * @brief Logger struct for logging messages.
   * stream: The stream to log to.
   * level: The log level to log messages at.
   */
  struct Logger
  {
    std::ostream &stream;
    int level;
  };

  template <typename T>
  inline std::string toString(const T &value)
  {
    std::ostringstream oss;
    oss << value;
    return oss.str();
  }

#ifndef NO_LOG
  /**
   * @brief Log a message to the logger. If NDEBUG is defined in a source or as a
   * compiler flag, this is a no-op.
   *
   * @param logger The logger to log to.
   * @param level The log level of the message.
   * @param message The message to log.
   */
  template <typename... Args>
  inline void LOG(Logger &logger, int level, const char *message, ...)
  {
    static const char *orange = "\033[0;33m";
    static const char *red = "\033[0;31m";
    static const char *white = "\033[0;37m";
    static const char *gray = "\033[0;90m";
    static const char *reset = "\033[0m";
    static const char *logColors[] = {red, red, orange, gray};
    if (level <= logger.level)
    {
      va_list args;
      va_start(args, message);

      int size = vsnprintf(nullptr, 0, message, args) + 1;
      std::unique_ptr<char[]> buffer(new char[size]);

#ifdef _WIN32
      _vsnprintf_s(buffer.get(), size, _TRUNCATE, message, args);
#else
      vsnprintf(buffer.get(), size, message, args);
#endif

      // Brackets and messages are white.
      // Log levels are red for error and warning, orange for info, and grey for trace.
      // Then the color is reset.
      logger.stream << white << "[" << logColors[level] << kLevelStr[level] << white << "] ";
      logger.stream << buffer.get();
      logger.stream << reset << std::endl;
      va_end(args);
    }
  }
#else
  template <typename... Args>
  inline void LOG(Logger &logger, int level, const char *message, Args... args)
  {
    (void)logger;
    (void)level;
    (void)message;
    (void)(std::initializer_list<int>{(static_cast<void>(args), 0)...});
  }
#endif

  /**
   * @brief Default logger for logging messages to stdout at the info level.
   * Output stream and logging level for the default logger can be globally
   * changed on a per-program basis.
   */
  static Logger kDefLog = {std::cout, kInfo};

  /**
   * @brief Set the log level of the default logger.
   * @param level The log level to set.
   */
  static inline void setLogLevel(int level)
  {
    kDefLog.level = level;
  }

} // namespace gpu

#endif
