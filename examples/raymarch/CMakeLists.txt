cmake_minimum_required(VERSION 3.11)
project(raymarch)

include(FetchContent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For a standalone repo, remove this line and set the path to the repos own
# FetchContent cache directory. Alternatively, don't set FETCHCONTENT_BASE_DIR
# and the repos will be downloaded to the build directory.

set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party")

FetchContent_Declare(
  gpu
  # For standalone repo, replace GIT_REPOSITORY with the URL:
  # GIT_REPOSITORY https://github.com/AnswerDotAI/gpu.cpp
  GIT_REPOSITORY file://${CMAKE_CURRENT_SOURCE_DIR}/../../
  GIT_TAG main
  GIT_SHALLOW    TRUE
  USE_LOCAL_LIBS ON
)
FetchContent_MakeAvailable(gpu)

add_executable(raymarch run.cpp)
target_link_libraries(raymarch PRIVATE gpu webgpu)
target_include_directories(raymarch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ )
