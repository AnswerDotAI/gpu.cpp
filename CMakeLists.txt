cmake_minimum_required(VERSION 3.11)
project(gpu)

include(FetchContent)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fetchcontent")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # export compile_commands.json to use with LSP
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(EXTRA_LIBS "-framework QuartzCore" "-framework Metal" "-framework IOSurface" "-framework CoreFoundation" "-framework Security" "-framework IOKit" "-framework AppKit" objc)
endif()

option(USE_DAWN_BINARY "Use prebuilt dawn" ON)

if(USE_DAWN_BINARY)
include_directories(./third_party/libdawn/include)
link_directories(./third_party/libdawn/lib)
else()
  option(USE_LOCAL_LIBS "Use local libraries instead of fetching from the internet" OFF)
  
  # Define paths for local and remote repositories
  set(WEBGPU_DIST_LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/local/WebGPU-distribution")
  
  # Conditional assignment based on USE_LOCAL_LIBS
  if(USE_LOCAL_LIBS)
    set(WEBGPU_DIST_GIT_REPO ${WEBGPU_DIST_LOCAL_PATH})
    message(STATUS "Using local WebGPU distribution: ${WEBGPU_DIST_LOCAL_PATH}")
  else()
    set(WEBGPU_DIST_GIT_REPO "https://github.com/eliemichel/WebGPU-distribution")
  endif()
  
  
  option(WEBGPU_TAG "WebGPU distribution tag to use")
  IF (NOT WEBGPU_TAG)
    set(WEBGPU_TAG "dawn")
  ENDIF()
  message(STATUS "Using WebGPU distribution tag: ${WEBGPU_TAG}")
  
  FetchContent_Declare(
    webgpu
    GIT_REPOSITORY  ${WEBGPU_DIST_GIT_REPO}
    GIT_TAG        ${WEBGPU_TAG}
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(webgpu)
endif()
  
if (WEBGPU_TAG STREQUAL "dawn")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWEBGPU_BACKEND_DAWN")
  message(STATUS "Using Dawn backend")
endif()

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

set(SRC_DEMO run.cpp gpu.h nn/shaders.h utils/array_utils.h utils/logging.h)
add_executable(run_demo ${SRC_DEMO})
target_link_libraries(run_demo PRIVATE ${LIBDL} ${CMAKE_DL_LIBS} webgpu ${EXTRA_LIBS})

# Test of basic kernels

set(SRC_TESTS utils/test_kernels.cpp gpu.h nn/shaders.h utils/array_utils.h utils/logging.h)
add_executable(run_tests ${SRC_TESTS})
target_link_libraries(run_tests PRIVATE ${LIBDL} ${CMAKE_DL_LIBS} webgpu ${EXTRA_LIBS})
target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# Build the library target (libgpu)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(SRC_LIB gpu.h nn/shaders.h utils/array_utils.h utils/logging.h)
add_library(gpu SHARED ${SRC_LIB})
set_target_properties(gpu PROPERTIES LINKER_LANGUAGE CXX)
