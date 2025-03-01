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

# Add sources
set(GPU_SOURCES
    "${PROJECT_ROOT}/gpu.cpp"
    "${PROJECT_ROOT}/numeric_types/half.cpp"
    "${DAWN_BUILD_DIR}/gen/include/dawn/webgpu.h"
)

# Add headers
set(GPU_HEADERS
    "${PROJECT_ROOT}/gpu.hpp"
    "${PROJECT_ROOT}/utils/logging.hpp"
    "${PROJECT_ROOT}/utils/array_utils.hpp"
    "${PROJECT_ROOT}/numeric_types/half.hpp"
    
)

# Create the STATIC library for gpu
add_library(gpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})
set_target_properties(gpu PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(gpu PUBLIC "${PROJECT_ROOT}")
if(NOT EMSCRIPTEN)
    target_include_directories(gpu PUBLIC "${DAWN_BUILD_DIR}/gen/include/")
    target_include_directories(gpu PUBLIC "${DAWN_BUILD_DIR}/gen/include/dawn/")
    target_include_directories(gpu PUBLIC "${DAWN_DIR}/include/")
else()
    target_include_directories(gpu PUBLIC "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/")
    target_include_directories(gpu PUBLIC "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/webgpu/")
endif()
