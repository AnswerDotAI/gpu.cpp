#include <array>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/screen.hpp"
#include "ftxui/screen/string.hpp"

#include "gpu.h"


// test function - multiply by constant
const char *kShaderCMul = R"(
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let x = GlobalInvocationID.x;
    let y = GlobalInvocationID.y;
    if (idx < arrayLength(&input)) {
      output[idx] = x * y;
    }
  }
)";

std::string getCurrentTime() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

int main(int argc, char **argv) {
  using namespace ftxui;
  auto screen = ScreenInteractive::Fullscreen();

  auto component = Renderer([&] {

    std::string time_str = getCurrentTime();
    auto c = Canvas(80, 80);
    for (int x = 0; x < 80; x++) {
      for (int y = 0; y < 80; y++) {
        if (std::rand() % 100 > 95) {
          int rand_blue = std::rand() % 64;
          c.DrawBlock(x, y, true, Color::RGB(64, 64, 128 + rand_blue));
        }
      }
    }

    for (int i = 0; i < 5; i++) {
      int y = std::rand() % 80;
      int intensity = std::rand() % 64+ 64;
      c.DrawBlockLine(0, y, 79, y, Color::RGB(intensity, intensity, intensity));
    }
    int sec = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    if (sec % 2 == 0) {
      c.DrawText(10, 10, "Testing.");
    } else {
      c.DrawText(10, 10, "1, 2, 3 ...");
    }

    return vbox({
               text(L"Test app"), separator(),
               hbox({
                   canvas(std::move(c)) | border,
                   text(time_str) | border,
               }),
           }) |
           border;
  });

  std::thread updater([&screen]() {
    while (true) {
      screen.PostEvent(Event::Custom);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  screen.Loop(component);
  updater.join();

  return 0;
}
