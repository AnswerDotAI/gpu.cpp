cmake_minimum_required(VERSION 3.11)
project(raymarch)

include(FetchContent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch ftxui
FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG 7b1f4d435bedecc49abc3f935bd97dc997a41977
  )
FetchContent_MakeAvailable(ftxui)

add_executable(raymarch run.cc)
target_link_libraries(raymarch PRIVATE ftxui::component ftxui::screen ftxui::dom)
