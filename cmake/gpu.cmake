set(FILENAME "gpu.hpp")

# Setup project root here.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
    set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)
    set(PROJECT_ROOT "${PROJECT_ROOT}/")
endif()

message(STATUS "PROJECT_ROOT: ${PROJECT_ROOT}")


include(FetchContent)

set(FETCHCONTENT_BASE_DIR "${PROJECT_ROOT}/third_party/fetchcontent/_deps")
set(DAWN_INSTALL_PREFIX "${FETCHCONTENT_BASE_DIR}/dawn-build/out/${CMAKE_BUILD_TYPE}" CACHE INTERNAL "Dawn install location" FORCE)


# Before fetching, set configuration options for Dawn.
set(DCMAKE_INSTALL_PREFIX   ${DAWN_INSTALL_PREFIX} CACHE INTERNAL "Dawn install location" FORCE)

# Dawn options for more,
# see https://dawn.googlesource.com/dawn/+/refs/heads/main/CMakeLists.txt
set(DAWN_ALWAYS_ASSERT     OFF CACHE INTERNAL "Always assert in Dawn" FORCE)
set(DAWN_BUILD_MONOLITHIC_LIBRARY ON CACHE INTERNAL "Build Dawn monolithically" FORCE)
set(DAWN_BUILD_EXAMPLES      OFF CACHE INTERNAL "Build Dawn examples" FORCE)
set(DAWN_BUILD_SAMPLES      OFF CACHE INTERNAL "Build Dawn samples" FORCE)
set(DAWN_BUILD_TESTS         OFF CACHE INTERNAL "Build Dawn tests" FORCE)
set(DAWN_ENABLE_INSTALL      ON  CACHE INTERNAL "Enable Dawn installation" FORCE)
set(DAWN_FETCH_DEPENDENCIES ON  CACHE INTERNAL "Fetch Dawn dependencies" FORCE)

set(TINT_BUILD_TESTS        OFF CACHE INTERNAL "Build Tint Tests" FORCE)
set(TINT_BUILD_IR_BINARY    OFF CACHE INTERNAL "Build Tint IR binary" FORCE)
set(TINT_BUILD_CMD_TOOLS   OFF CACHE INTERNAL "Build Tint command line tools" FORCE)

set(BUILD_SHARED_LIBS       OFF CACHE INTERNAL "Build shared libraries" FORCE)


# Fetch Setup
# Add a commit hash to pin the version of Dawn.
# git fetch --depth=1 url <commit hash>
FetchContent_Declare(
    dawn
    DOWNLOAD_COMMAND
    cd ${FETCHCONTENT_BASE_DIR}/dawn-src &&
    git init &&
    git fetch --depth=1 https://dawn.googlesource.com/dawn &&
    git reset --hard FETCH_HEAD
)
 

# Download the repository and add it as a subdirectory.
FetchContent_MakeAvailable(dawn)

 
# Since we require Dawn to be built before linking against it, we need to configure it now.
execute_process(
    COMMAND ${CMAKE_COMMAND} ${FETCHCONTENT_BASE_DIR}/dawn-src 
        -B ${DAWN_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -G "${CMAKE_GENERATOR}"
)

# Build Dawn
execute_process(
    WORKING_DIRECTORY ${FETCHCONTENT_BASE_DIR}/dawn-src
    COMMAND ${CMAKE_COMMAND} --build ${DAWN_INSTALL_PREFIX} --config ${CMAKE_BUILD_TYPE}
)

# Add sources
set(GPU_SOURCES
    "${PROJECT_ROOT}/gpu.cpp"
    "${PROJECT_ROOT}/numeric_types/half.cpp"
)

# Add headers
set(GPU_HEADERS
    "${PROJECT_ROOT}/gpu.hpp"
    "${PROJECT_ROOT}/utils/logging.hpp"
    "${PROJECT_ROOT}/utils/array_utils.hpp"
    "${PROJECT_ROOT}/numeric_types/half.hpp"
)

# Emscripten includes a header automatically
if(EMSCRIPTEN)
    file(REMOVE "${PROJECT_ROOT}/webgpu/webgpu.h")
else()
    list(APPEND GPU_HEADERS "${PROJECT_ROOT}/third_party/headers/webgpu/webgpu.h")
endif()


# Create the STATIC library for gpu
add_library(gpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})
target_include_directories(gpu PUBLIC "${PROJECT_ROOT}")
target_include_directories(gpu PUBLIC "${PROJECT_ROOT}/third_party/headers")

# find_library, windows adds extra folder
if(MSVC)
    find_library(WEBGPU_DAWN_MONOLITHIC
    NAMES webgpu_dawn
    PATHS "${DAWN_INSTALL_PREFIX}/src/dawn/native/${CMAKE_BUILD_TYPE}"
    REQUIRED
    )
else()
    find_library(WEBGPU_DAWN_MONOLITHIC
    NAMES webgpu_dawn
    PATHS "${DAWN_INSTALL_PREFIX}/src/dawn/native"
    REQUIRED
    )
endif()

# Link the monolithic library
target_link_libraries(gpu PRIVATE ${WEBGPU_DAWN_MONOLITHIC})
