#ifndef GPU_CPP_UTILS_WGSL_HPP
#define GPU_CPP_UTILS_WGSL_HPP

#include <regex>
#include <string>

namespace gpu {

// Unrolls flat WGSL loops of the form
// `for (var i: u32 = START; i < END; i++) { BODY }`.
inline std::string unrollLoops(const std::string &code, int threshold = 32) {
  static const std::regex loop(
      R"(for\s*\(\s*var\s+(\w+):\s*u32\s*=\s*(\d+)\s*;\s*\1\s*<\s*(\d+)\s*;\s*\1\+\+\s*\)\s*\{\s*([^{}]*)\})");
  std::string result;
  size_t previous = 0;
  for (std::sregex_iterator it(code.begin(), code.end(), loop), sentinel;
       it != sentinel; ++it) {
    const std::smatch &match = *it;
    result.append(code, previous, match.position() - previous);
    const auto name = match[1].str();
    const int start = std::stoi(match[2]);
    const int end = std::stoi(match[3]);
    if (end - start > threshold) {
      result += match.str();
    } else {
      const std::regex variable("\\b" + name + "\\b");
      for (int i = start; i < end; ++i)
        result +=
          std::regex_replace(match[4].str(), variable, std::to_string(i));
    }
    previous = match.position() + match.length();
  }
  result.append(code, previous, std::string::npos);
  return result;
}

} // namespace gpu

#endif
