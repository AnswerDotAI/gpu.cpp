# This only builds a shared lib, see cmake/example.cmake
# and cmake/gpu.cmake for more details
cmake_minimum_required(VERSION 3.28)
project(gpu)
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
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

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/dawn.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu.cmake")

target_link_libraries(gpu PRIVATE webgpu_dawn)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)

add_executable(test_gpu ${CMAKE_CURRENT_SOURCE_DIR}/test/test_gpu.cpp)
target_link_libraries(test_gpu PRIVATE gpu)

# Platform-specific post-build actions (e.g. copying DLLs for MSVC)
if(MSVC)
    add_custom_command(
        TARGET test_gpu POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${DAWN_BUILD_DIR}/$<CONFIG>/webgpu_dawn.dll
                $<TARGET_FILE_DIR:test_gpu>
        COMMENT "Copying webgpu_dawn.dll to the build directory"
    )
endif()

add_library(gpud SHARED gpu.hpp)
set_target_properties(gpud PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(gpud PRIVATE gpu)
