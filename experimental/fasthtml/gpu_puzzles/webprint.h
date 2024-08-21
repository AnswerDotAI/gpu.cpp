#ifndef WEBPRINT_H
#define WEBPRINT_H


#include <emscripten/emscripten.h>

EM_JS(void, js_print, (const char *str), {
  if (typeof window != 'undefined' && window.customPrint) {
    window.customPrint(UTF8ToString(str));
  } else {
    console.log("window.customPrint is not defined.");
    console.log(UTF8ToString(str));
  }
});


// need to allow printf with variable arguments
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
template<typename... Args>
void wprintf(const char *str, Args... args) {
  char buffer[1024];
  snprintf(buffer, sizeof(buffer), str, args...);
  js_print(buffer);
}
#pragma clang diagnostic pop

void printVec(const std::vector<float> &vec, const char *name = "") {
    char buffer[1024];
    size_t pos = 0;
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "[ ");
    for (size_t i = 0; i < vec.size(); ++i) {
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "%.1f", vec[i]);
        if (i != vec.size() - 1) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, ", ");
        }
    }
    snprintf(buffer + pos, sizeof(buffer) - pos, " ]");
    wprintf("%s %s", name, buffer);
}


void printVecBuf(const std::vector<float> &vec, const char *name, char *buffer, size_t& pos) {
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "%s", name);
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, "[ ");
    for (size_t i = 0; i < vec.size(); ++i) {
        pos += snprintf(buffer + pos, sizeof(buffer) - pos, "%2.0f", vec[i]);
        if (i != vec.size() - 1) {
            pos += snprintf(buffer + pos, sizeof(buffer) - pos, ", ");
        }
    }
    pos += snprintf(buffer + pos, sizeof(buffer) - pos, " ]\n\r");
}


#endif // WEBPRINT_H
