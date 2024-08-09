# Specify the filename to search for
set(FILENAME "gpu.h")

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
    message(
        FATAL_ERROR
            "File ${FILENAME} not found in either ${CMAKE_CURRENT_SOURCE_DIR} or ${CMAKE_CURRENT_SOURCE_DIR}/../../"
    )
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
if(WIN32)
    set(DLL_PATH
        "${TARGET_FILE_PATH}/third_party/lib/libdawn_${ARCH}_${BUILD_TYPE}.dll")
    if(EXISTS ${DLL_PATH})
        file(COPY ${DLL_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(webgpulib INTERFACE ${DLL_PATH})
    else()
        message(FATAL_ERROR "libdawn dll not found at: ${DLL_PATH}")
    endif()
else()
    find_library(LIBDAWN dawn PATHS "${TARGET_FILE_PATH}/third_party/lib")
    if(LIBDAWN)
        message(STATUS "Found libdawn: ${LIBDAWN}")
        # Link against libdawn
        target_link_libraries(webgpulib INTERFACE ${LIBDAWN})
        # if not found, try download from release
    else()
        message("libdawn not found, try downloading from the release")
        FetchContent_Declare(
            libdawn
            URL https://github.com/austinvhuang/dawn-artifacts/releases/download/prerelease/libdawn.dylib
                DOWNLOAD_DIR
                "${TARGET_FILE_PATH}/third_party/lib")
        FetchContent_MakeAvailable(libdawn)
        find_library(LIBDAWN dawn REQUIRED
                     PATHS "${TARGET_FILE_PATH}/third_party/lib")
        if(LIBDAWN)
            message(STATUS "Found libdawn: ${LIBDAWN}")
            # Link against libdawn
            target_link_libraries(webgpulib INTERFACE ${LIBDAWN})
        else()
            message(FATAL_ERROR "libdawn not found")
        endif()
    endif()
endif()
