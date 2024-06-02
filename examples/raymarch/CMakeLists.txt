cmake_minimum_required(VERSION 3.11)
project(raymarch)

include(FetchContent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For a standalone repo, remove this line and set the path to the repos own
# FetchContent cache directory. Alternatively, don't set FETCHCONTENT_BASE_DIR
# and the repos will be downloaded to the build directory.
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")

# Fetch ftxui
FetchContent_Declare(
  ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG 7b1f4d435bedecc49abc3f935bd97dc997a41977
  )
FetchContent_MakeAvailable(ftxui)

FetchContent_Declare(
  gpu
  # For standalone repo, replace GIT_REPOSITORY with the URL:
  # GIT_REPOSITORY https://github.com/AnswerDotAI/gpu.cpp
  GIT_REPOSITORY file://${CMAKE_CURRENT_SOURCE_DIR}/../../
  GIT_TAG main
)
FetchContent_MakeAvailable(gpu)

FetchContent_Declare(
  webgpu-backend-dawn
  GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
  GIT_TAG        dawn
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(webgpu-backend-dawn)

add_executable(raymarch run.cc)
target_link_libraries(raymarch PRIVATE ftxui::component ftxui::screen ftxui::dom gpu webgpu)
