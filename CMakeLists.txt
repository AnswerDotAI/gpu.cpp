cmake_minimum_required(VERSION 3.28)
project(gpu)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # export compile_commands.json to use with
                                      # LSP
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu.cmake")

add_library(gpud SHARED gpu.hpp)
set_target_properties(gpud PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(gpud PRIVATE gpu)