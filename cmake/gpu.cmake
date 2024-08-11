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

add_library(webgpulib SHARED IMPORTED)
add_library(gpu INTERFACE)
add_library(wgpu INTERFACE)
add_dependencies(gpu webgpulib)
# Define the header-only library
target_include_directories(gpu INTERFACE ${TARGET_FILE_PATH})

# Add headers webgpu.h
target_include_directories(wgpu
                           INTERFACE ${TARGET_FILE_PATH}/third_party/headers)
include(ExternalProject)

set(DAWN_EXT_PREFIX "${TARGET_FILE_PATH}/third_party/local/dawn")

ExternalProject_Add(
    dawn_project
    PREFIX ${DAWN_EXT_PREFIX}
    GIT_REPOSITORY "https://dawn.googlesource.com/dawn"
    GIT_TAG "main"
    SOURCE_DIR "${DAWN_EXT_PREFIX}/source"
    BINARY_DIR "${DAWN_EXT_PREFIX}/build"
    INSTALL_DIR "${DAWN_EXT_PREFIX}/install"
    GIT_SUBMODULES ""
    # setting cmake args doesn't work and I don't know why
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -S ${DAWN_EXT_PREFIX}/source -B
        ${DAWN_EXT_PREFIX}/build -DDAWN_FETCH_DEPENDENCIES=ON
        -DDAWN_ENABLE_INSTALL=ON -DDAWN_BUILD_MONOLITHIC_LIBRARY=ON
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -G ${CMAKE_GENERATOR}
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --prefix
                    ${DAWN_EXT_PREFIX}/install
    LOG_INSTALL ON)
find_library(LIBDAWN dawn PATHS "${DAWN_EXT_PREFIX}/install/lib")
target_link_libraries(webgpulib INTERFACE ${LIBDAWN})
