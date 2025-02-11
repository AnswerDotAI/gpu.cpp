set(FILENAME "gpu.hpp")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
    set(FILEPATH_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
else()
    get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)
    
    set(FILEPATH_PROJECT_ROOT "${PROJECT_ROOT}/")
endif()


include(FetchContent)

set(FETCHCONTENT_BASE_DIR "${FILEPATH_PROJECT_ROOT}/third_party/fetchcontent/_deps")
set(DAWN_INSTALL_PREFIX "${FETCHCONTENT_BASE_DIR}/dawn-build/out/${CMAKE_BUILD_TYPE}" CACHE INTERNAL "Dawn install location" FORCE)


# Before fetching, set configuration options for Dawn.
# These CMake variables are “global” (cached INTERNAL) so that Dawn’s own CMakeLists.txt
# will pick them up. Adjust them as needed.
set(DAWN_BUILD_TYPE          ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Dawn build type" FORCE)
set(DCMAKE_INSTALL_PREFIX   ${DAWN_INSTALL_PREFIX} CACHE INTERNAL "Dawn install location" FORCE)

# Dawn options
set(DAWN_FETCH_DEPENDENCIES ON  CACHE INTERNAL "Fetch Dawn dependencies" FORCE)
set(DAWN_ENABLE_INSTALL      ON  CACHE INTERNAL "Enable Dawn installation" FORCE)
set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF CACHE INTERNAL "Build Dawn monolithically" FORCE)
set(DAWN_BUILD_EXAMPLES      OFF CACHE INTERNAL "Build Dawn examples" FORCE)
set(DAWN_BUILD_SAMPLES      OFF CACHE INTERNAL "Build Dawn samples" FORCE)
set(DAWN_BUILD_TESTS         OFF CACHE INTERNAL "Build Dawn tests" FORCE)
set(DAWN_BUILD_UTILS         OFF CACHE INTERNAL "Build Dawn utilities" FORCE)
set(TINT_BUILD_TESTS        OFF CACHE INTERNAL "Build Tint Tests" FORCE)
set(TINT_BUILD_IR_BINARY    OFF CACHE INTERNAL "Build Tint IR binary" FORCE)
set(TINT_BUILD_CMD_TOOLS   OFF CACHE INTERNAL "Build Tint command line tools" FORCE)
set(BUILD_SHARED_LIBS       OFF CACHE INTERNAL "Build shared libraries" FORCE)


# Set up an install location for Dawn – you can change this to a specific location.


FetchContent_Declare(
    dawn
    DOWNLOAD_COMMAND
    cd ${FETCHCONTENT_BASE_DIR}/dawn-src &&
    git init &&
    git fetch --depth=1 https://dawn.googlesource.com/dawn &&
    git reset --hard FETCH_HEAD
)
 

# This call will download the repository and add it as a subdirectory.
FetchContent_MakeAvailable(dawn)

 
# At this point, assuming Dawn’s CMakeLists.txt is written so that an install step is available,
# we trigger a build of its install target. This custom target will build (and install) Dawn
# into ${DAWN_INSTALL_PREFIX}. (If Dawn already adds an install target, you may simply depend on it.)
add_custom_target(build_dawn_config ALL
    COMMAND ${CMAKE_COMMAND} ${FETCHCONTENT_BASE_DIR}/dawn-src 
        -B ${DAWN_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DDAWN_FETCH_DEPENDENCIES=ON
        -DDAWN_ENABLE_INSTALL=ON
        -DDAWN_BUILD_MONOLITHIC_LIBRARY=OFF
        -DDAWN_BUILD_EXAMPLES=OFF
        -DDAWN_BUILD_SAMPLES=OFF
        -DDAWN_BUILD_TESTS=OFF
        -DDAWN_BUILD_UTILS=OFF
        -DTINT_BUILD_TESTS=OFF
        -DTINT_BUILD_IR_BINARY=OFF
        -DTINT_BUILD_CMD_TOOLS=OFF
        -DBUILD_SHARED_LIBS=OFF
        -G "${CMAKE_GENERATOR}"
    COMMENT "Configuring Dawn build with custom options in ${DAWN_INSTALL_PREFIX}"
)

add_custom_target(build_dawn_install ALL
    COMMAND ${CMAKE_COMMAND} --build ${DAWN_INSTALL_PREFIX} --target install
    COMMENT "Installing Dawn into ${DAWN_INSTALL_PREFIX}"
)

include(${FETCHCONTENT_BASE_DIR}/dawn-build/cmake/DawnTargets.cmake)

set(GPU_SOURCES
    "${FILEPATH_PROJECT_ROOT}/gpu.cpp"
    "${FILEPATH_PROJECT_ROOT}/numeric_types/half.cpp"
)

set(GPU_HEADERS
    "${FILEPATH_PROJECT_ROOT}/gpu.hpp"
    "${FILEPATH_PROJECT_ROOT}/utils/logging.hpp"
    "${FILEPATH_PROJECT_ROOT}/utils/array_utils.hpp"
    "${FILEPATH_PROJECT_ROOT}/numeric_types/half.hpp"
)

if(EMSCRIPTEN)
    file(REMOVE "${FILEPATH_PROJECT_ROOT}/webgpu/webgpu.h")
else()
    list(APPEND GPU_HEADERS "${DAWN_INSTALL_PREFIX}/gen/webgpu-headers/webgpu.h")
endif()


# Create the INTERFACE library ‘gpu’
add_library(gpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})
target_include_directories(gpu PUBLIC "${FILEPATH_PROJECT_ROOT}")
target_include_directories(gpu PUBLIC "${FILEPATH_PROJECT_ROOT}/third_party/headers")

# Ensure that the gpu target is built only after Dawn has been installed.
add_dependencies(gpu build_dawn_install)

find_library(WEBGPU_DAWN
    NAMES webgpu_dawn
    HINTS "${DAWN_INSTALL_PREFIX}/src/dawn/native/Debug/"
)