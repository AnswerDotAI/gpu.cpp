cmake_minimum_required(VERSION 3.28)
project(wgpu_tutorial)

include(FetchContent)

FetchContent_Declare(
  webgpu-backend-dawn
  GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
  GIT_TAG        dawn-6376
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(webgpu-backend-dawn)

FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git 
  GIT_TAG 27cb4c76708608465c413f6d0e6b8d99a4d84302
)
FetchContent_MakeAvailable(spdlog)

add_executable(wgpu_tutorial run.cpp)
target_link_libraries(wgpu_tutorial webgpu spdlog)
