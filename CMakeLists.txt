cmake_minimum_required(VERSION 3.28)
project(gpu)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/webgpu.cmake")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # export compile_commands.json to use with
                                      # LSP
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_LOCAL_LIBS
       "Use local libraries instead of fetching from the internet" OFF)

# Ensure the build type is set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build: Debug or Release" FORCE)
endif()

option(FASTBUILD "Option to enable fast builds" OFF)
if(FASTBUILD)
    set(CMAKE_BUILD_TYPE None) # Avoid default flags of predefined build types
    set(CMAKE_CXX_FLAGS "-O0")
endif()

option(DEBUG "Option to enable debug flags" OFF)
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "-O0 -g")
endif()

if(WIN64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWEBGPU_BACKEND_DAWN")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu.cmake")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(
    STATUS
        "Include directories for wgpu: ${CMAKE_CURRENT_SOURCE_DIR}/third_party/headers"
)

add_library(gpud SHARED gpu.hpp)
set_target_properties(gpud PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(gpud PRIVATE wgpu)
target_link_libraries(gpud PRIVATE webgpu)
target_link_libraries(gpud PRIVATE gpu)
install(TARGETS gpud)
