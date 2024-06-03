cmake_minimum_required(VERSION 3.11)
project(gpu)

include(FetchContent)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

option(BUILD_MODE "'local' or 'remote' git fetch for builds")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # export compile_commands.json to use with LSP

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(IMPLEMENTATION "WebGPU implementation to use" "dawn")
IF(NOT IMPLEMENTATION)
  set(IMPLEMENTATION "dawn")
ENDIF()

IF(IMPLEMENTATION STREQUAL "dawn")

  message(STATUS "Using Dawn as WebGPU implementation")
  FetchContent_Declare(
    webgpu-backend-dawn
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
    # GIT_TAG        dawn-6376
    GIT_TAG        dawn
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(webgpu-backend-dawn)

ELSEIF(IMPLEMENTATION STREQUAL "wgpu")
  message(STATUS "Using WGPU as WebGPU implementation")
  FetchContent_Declare(
    webgpu-backend-wgpu
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
    GIT_TAG       wgpu
    GIT_SHALLOW   TRUE
  )
  FetchContent_MakeAvailable(webgpu-backend-wgpu)
ELSEIF(IMPLEMENTATION STREQUAL "emscripten")
  message(STATUS "Using Emscripten as WebGPU implementation")
  FetchContent_Declare(
    webgpu-backend-emscripten
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
    GIT_TAG       emscripten
    GIT_SHALLOW   TRUE
  )
  FetchContent_MakeAvailable(webgpu-backend-emscripten)
ENDIF()

FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git 
  # GIT_TAG 27cb4c76708608465c413f6d0e6b8d99a4d84302
  GIT_TAG c3aed4b68373955e1cc94307683d44dca1515d2b
)
FetchContent_MakeAvailable(spdlog)

option(FASTBUILD "Option to enable fast builds" OFF)
if(FASTBUILD)
  set(CMAKE_BUILD_TYPE None)  # Avoid default flags of predefined build types
  set(CMAKE_CXX_FLAGS "-O0")
endif()

option(DEBUG "Option to enable debug flags" OFF)
if(DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS "-O0 -g")
endif()

# dl for dlopen/dlysm/dlclose
find_library(LIBDL dl REQUIRED)
if(LIBDL)
  message(STATUS "Found libdl: ${LIBDL}")
else()
  message(FATAL_ERROR "libdl not found")
  exit()
endif()

# Hello world demo

set(SRC_DEMO run.cpp gpu.h kernels.h utils/array_utils.h)
add_executable(run_demo ${SRC_DEMO})
# target_link_libraries(run_demo PRIVATE ${LIBDL} ${CMAKE_DL_LIBS} webgpu spdlog)
target_link_libraries(run_demo PRIVATE ${LIBDL} ${CMAKE_DL_LIBS} webgpu spdlog::spdlog)

# message(STATUS "SPDLOG_HEADERS_DIR: ${SPDLOG_HEADERS_DIR}")
#target_include_directories(run_demo PRIVATE ${SPDLOG_HEADERS_DIR})

# Test of basic kernels

set(SRC_TESTS utils/test_kernels.cpp gpu.h kernels.h utils/array_utils.h)
add_executable(run_tests ${SRC_TESTS})
target_link_libraries(run_tests PRIVATE ${LIBDL} ${CMAKE_DL_LIBS} webgpu spdlog)
target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# Build the library target (libgpu)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(SRC_LIB gpu.h kernels.h utils/array_utils.h)
add_library(gpu SHARED ${SRC_LIB})
target_link_libraries(gpu PUBLIC ${LIBDL} ${CMAKE_DL_LIBS} webgpu spdlog)
