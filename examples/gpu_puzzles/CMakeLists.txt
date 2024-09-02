cmake_minimum_required(VERSION 3.28)
project(gpu_puzzles)

set(FILENAME "gpu.hpp")

get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)

# Construct potential paths
set(FILEPATH_CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
set(FILEPATH_PROJECT_ROOT "${PROJECT_ROOT}/${FILENAME}")

# Check if the file exists in the current directory
if(EXISTS ${FILEPATH_CURRENT_DIR})
    set(TARGET_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
elseif(EXISTS ${FILEPATH_PROJECT_ROOT})
    set(TARGET_FILE_PATH ${PROJECT_ROOT})
else()
    message(FATAL_ERROR "File ${FILENAME} not found in either ${CMAKE_CURRENT_SOURCE_DIR} or ${CMAKE_CURRENT_SOURCE_DIR}/../../")
endif()

include("${TARGET_FILE_PATH}/cmake/example.cmake")