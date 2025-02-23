# Getting Started with CMAKE
# Each example includes this and sets PROJECT_NAME.
#
# Example usage:
#   cd examples/hello_world
#   cmake -S . build/ -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/ --config Release
#   ./build/hello_world   (or serve the output .js/.wasm for Emscripten)
#   or for emscripten
#   emcmake cmake -S . -B ./build_web -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_web --config Release
#   python3 -m http.server 8080 --d build_web

if(NOT MSVC)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 20)
endif()

# Locate the project root (two levels up from the current source dir)
get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)

# Include external libraries and helper scripts (dawn and gpu)
include("${PROJECT_ROOT}/cmake/dawn.cmake")
include("${PROJECT_ROOT}/cmake/gpu.cmake")

# Create the executable
add_executable(${PROJECT_NAME} run.cpp)

# Platform-specific linking & build settings
if(EMSCRIPTEN)
    # Emscripten-specific configuration

    # Define a web output directory (adjust as needed)
    set(WEB_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/web_build")

    # If necessary, include the generated WebGPU include dirs first.
    include_directories(BEFORE "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/")

    # Create a helper library for WebGPU support.
    add_library(webgpu_web "${DAWN_DIR}/third_party/emdawnwebgpu/webgpu.cpp")
    target_link_libraries(${PROJECT_NAME} PRIVATE webgpu_web)

    # Set Emscripten-specific link flags that enable WASM output and expose certain symbols.
    # Needed to use updated version, emdawnwebgpu
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "\
        -O3 \
        -sUSE_WEBGPU=0 \
        -sWASM=1 \
        -DDAWN_EMSCRIPTEN_TOOLCHAIN=${EMSCRIPTEN_DIR} \
        -sEXPORTED_FUNCTIONS=_main,_malloc,_free,_memcpy \
        -sEXPORTED_RUNTIME_METHODS=ccall \
        -sUSE_GLFW=3 \
        -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=15MB \
        -sASYNCIFY \
        --js-library=${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_enum_tables.js \
        --js-library=${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_generated_struct_info.js \
        --js-library=${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_generated_sig_info.js \
        --js-library=${DAWN_DIR}/third_party/emdawnwebgpu/library_webgpu.js \
        --closure-args=--externs=${EMSCRIPTEN_DIR}/src/closure-externs/webgpu-externs.js \
    ")

else()
    # Non-Emscripten (desktop) linking
    if(MSVC)
        target_link_libraries(gpu 
            PRIVATE 
                $<$<CONFIG:Debug>:${WEBGPU_DAWN_DEBUG}>
                $<$<CONFIG:Release>:${WEBGPU_DAWN_RELEASE}>
        )
    else()
        target_link_libraries(gpu PRIVATE webgpu_dawn)
    endif()
endif()

# Link the gpu/dawn library to the executable.
target_link_libraries(${PROJECT_NAME} PRIVATE gpu)

# Platform-specific post-build actions (e.g. copying DLLs for MSVC)
if(MSVC)
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${DAWN_BUILD_DIR}/$<CONFIG>/webgpu_dawn.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying webgpu_dawn.dll to the build directory"
    )
endif()

if(EMSCRIPTEN)

    # Configure the HTML file by replacing @PROJECT_NAME@ with the actual target name.
    configure_file(${PROJECT_ROOT}cmake/templates/index.html.in
                   ${CMAKE_CURRENT_BINARY_DIR}/index.html
                   @ONLY)

endif()
