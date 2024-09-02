# Specify the filename to search for
set(FILENAME "gpu.hpp")

get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)

# Construct potential paths
set(FILEPATH_CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
set(FILEPATH_PROJECT_ROOT "${PROJECT_ROOT}/${FILENAME}")

# Include file finding utility script
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_gpu.cmake")

# Check if the file exists in the current directory
find_project_root(${CMAKE_CURRENT_SOURCE_DIR} ${FILENAME} TARGET_FILE_PATH)
if("${TARGET_FILE_PATH}" STREQUAL "")
    find_project_root(${FILEPATH_CURRENT_DIR} ${FILENAME} TARGET_FILE_PATH)
    if("${TARGET_FILE_PATH}" STREQUAL "")
        message(
            FATAL_ERROR
                "File ${FILENAME} not found in either ${CMAKE_CURRENT_SOURCE_DIR} or ${CMAKE_CURRENT_SOURCE_DIR}/../../"
        )
    endif()
endif()

include(FetchContent)

set(FETCHCONTENT_BASE_DIR "${TARGET_FILE_PATH}/third_party/fetchcontent")
set(WEBGPU_DIST_LOCAL_PATH
    "${TARGET_FILE_PATH}/third_party/local/WebGPU-distribution")

if(USE_LOCAL_LIBS)
    set(WEBGPU_DIST_GIT_REPO ${WEBGPU_DIST_LOCAL_PATH})
    message(STATUS "Using local WebGPU distribution: ${WEBGPU_DIST_LOCAL_PATH}")
else()
    set(WEBGPU_DIST_GIT_REPO
        "https://github.com/eliemichel/WebGPU-distribution")
endif()

option(WEBGPU_TAG "WebGPU distribution tag to use")
if(NOT WEBGPU_TAG)
    set(WEBGPU_TAG "dawn")
endif()
message(STATUS "Using WebGPU distribution tag: ${WEBGPU_TAG}")

if(WEBGPU_TAG STREQUAL "dawn")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWEBGPU_BACKEND_DAWN")
    # use specific commit set(WEBGPU_TAG
    # "1025b977e1927b6d0327e67352f90feb4bcf8274") set(WEBGPU_TAG
    # "acf972b7b909f52e183bdae3971b93bb13d4a29e")
    # add_compile_options(-UABSL_INTERNAL_AT_LEAST_CXX20) set(CMAKE_CXX_FLAGS
    # "${CMAKE_CXX_FLAGS} -UABSL_INTERNAL_AT_LEAST_CXX20")
    message(STATUS "Using Dawn backend")
endif()

FetchContent_Declare(
    webgpu
    GIT_REPOSITORY ${WEBGPU_DIST_GIT_REPO}
    GIT_TAG ${WEBGPU_TAG}
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(webgpu)
