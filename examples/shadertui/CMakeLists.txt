cmake_minimum_required(VERSION 3.11)
project(shadertui)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # export compile_commands.json to use with LSP
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)

message(STATUS "PROJECT_ROOT: ${PROJECT_ROOT}")

# Ensure the build type is set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug or Release" FORCE)
endif()

# Define architecture and build type directories or file names
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(ARCH "x64")
else()
  set(ARCH "x86")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(BUILD_TYPE "Debug")
else()
  set(BUILD_TYPE "Release")
endif()

if(NOT TARGET gpu)
  message(STATUS "GPU_LIB not found")
  include("${PROJECT_ROOT}/cmake/webgpu.cmake")
  include("${PROJECT_ROOT}/cmake/gpu.cmake")
endif()

add_executable(${PROJECT_NAME} run.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE gpu)
target_link_libraries(${PROJECT_NAME} PRIVATE wgpu)
target_link_libraries(${PROJECT_NAME} PRIVATE webgpu)

if(WIN32)
# Ensure DLL is copied if on Windows
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${DLL_PATH}
                   $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()