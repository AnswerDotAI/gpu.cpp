# Setup directories
set(FETCHCONTENT_BASE_DIR "${PROJECT_ROOT}/third_party")
set(DAWN_DIR "${FETCHCONTENT_BASE_DIR}/dawn" CACHE INTERNAL "")
set(DAWN_BUILD_DIR "${DAWN_DIR}/build" CACHE INTERNAL "")

if(EMSCRIPTEN)
    set(EM_SDK_DIR $ENV{EMSDK} CACHE INTERNAL "")
    set(DAWN_BUILD_DIR "${DAWN_DIR}/build_web" CACHE INTERNAL "")
    set(DAWN_EMSCRIPTEN_TOOLCHAIN ${EM_SDK_DIR}/upstream/emscripten CACHE INTERNAL "" FORCE)
else()
    add_compile_definitions(USE_DAWN_API)
endif()

# Enable find for no dawn rebuilds with flutter run
set(ENABLE_DAWN_FIND OFF CACHE BOOL "Enable finding Dawn" FORCE)
set(DAWN_BUILD_FOUND OFF CACHE BOOL "Dawn build found" FORCE)
if(ENABLE_DAWN_FIND)
    # find_library, windows adds extra folder
    if(MSVC)
        find_library(WEBGPU_DAWN_DEBUG webgpu_dawn
        NAMES webgpu_dawn
        HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Debug"
        )
        find_library(WEBGPU_DAWN_RELEASE webgpu_dawn
        NAMES webgpu_dawn
        HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Release"
        )
        set(DAWN_BUILD_FOUND ON)
    elseif(NOT EMSCRIPTEN AND NOT MSVC)
        find_library(WEBGPU_DAWN_LIB
        NAMES webgpu_dawn
        PATHS "${DAWN_BUILD_DIR}/src/dawn/native"
        REQUIRED
        )
        set(DAWN_BUILD_FOUND ON)
    else()
        set(DAWN_BUILD_FOUND ON)
    endif()
endif()

# Dawn options for more,
# see https://dawn.googlesource.com/dawn/+/refs/heads/main/CMakeLists.txt
set(DAWN_ALWAYS_ASSERT     OFF CACHE INTERNAL "Always assert in Dawn" FORCE)
set(DAWN_BUILD_MONOLITHIC_LIBRARY ON CACHE INTERNAL "Build Dawn monolithically" FORCE)
set(DAWN_BUILD_EXAMPLES      OFF CACHE INTERNAL "Build Dawn examples" FORCE)
set(DAWN_BUILD_SAMPLES      OFF CACHE INTERNAL "Build Dawn samples" FORCE)
set(DAWN_BUILD_TESTS         OFF CACHE INTERNAL "Build Dawn tests" FORCE)
set(DAWN_ENABLE_INSTALL      OFF  CACHE INTERNAL "Enable Dawn installation" FORCE)
set(DAWN_FETCH_DEPENDENCIES ON  CACHE INTERNAL "Fetch Dawn dependencies" FORCE)
set(TINT_BUILD_TESTS        OFF CACHE INTERNAL "Build Tint Tests" FORCE)
set(TINT_BUILD_IR_BINARY    OFF CACHE INTERNAL "Build Tint IR binary" FORCE)
set(TINT_BUILD_CMD_TOOLS   OFF CACHE INTERNAL "Build Tint command line tools" FORCE)

if(NOT DAWN_BUILD_FOUND)
    include(FetchContent)
    message("webgpu_dawn not found start building")
    if(EMSCRIPTEN)
        set(EMSCRIPTEN_DIR "${EM_SDK_DIR}/upstream/emscripten" CACHE INTERNAL "" FORCE)
        set(DAWN_EMSCRIPTEN_TOOLCHAIN ${EMSCRIPTEN_DIR} CACHE INTERNAL "" FORCE)
    endif()

    FetchContent_Declare(
        dawn
        DOWNLOAD_DIR ${DAWN_DIR}
        SOURCE_DIR ${DAWN_DIR}
        SUBBUILD_DIR ${DAWN_BUILD_DIR}/tmp
        BINARY_DIR ${DAWN_BUILD_DIR}
        DOWNLOAD_COMMAND
        cd ${DAWN_DIR} &&
        git init &&
        git fetch --depth=1 https://dawn.googlesource.com/dawn &&
        git reset --hard FETCH_HEAD
    )

    # Download the repository and add it as a subdirectory.
    FetchContent_MakeAvailable(dawn)

    # attempt fix flutter rebuilds
    set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${DAWN_DIR}/src" CACHE INTERNAL "")

    execute_process(
        WORKING_DIRECTORY ${DAWN_DIR}
        COMMAND ${CMAKE_COMMAND} -S ${DAWN_DIR}
            -B ${DAWN_BUILD_DIR}
    )

    # Build Dawn
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${DAWN_BUILD_DIR}
    )
    
    # find_library, windows adds extra folder
    if(MSVC)
        find_library(WEBGPU_DAWN_DEBUG webgpu_dawn
        NAMES webgpu_dawn
        HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Debug"
        )
        find_library(WEBGPU_DAWN_RELEASE webgpu_dawn
        NAMES webgpu_dawn
        HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Release"
        )
        set(DAWN_BUILD_FOUND ON)
    elseif(NOT EMSCRIPTEN AND NOT MSVC)
        find_library(WEBGPU_DAWN_LIB
        NAMES webgpu_dawn
        PATHS "${DAWN_BUILD_DIR}/src/dawn/native"
        REQUIRED
        )
        set(DAWN_BUILD_FOUND ON)
    else()
        set(DAWN_BUILD_FOUND ON)
    endif()
endif()

if(EMSCRIPTEN)
    add_library(webgpu_dawn INTERFACE IMPORTED)
    target_include_directories(webgpu_dawn INTERFACE ${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include)
    target_include_directories(webgpu_dawn INTERFACE ${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/webgpu/webgpu.h)
    target_link_libraries(webgpu_dawn INTERFACE ${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_enum_tables.js)
    target_link_libraries(webgpu_dawn INTERFACE ${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_generated_struct_info.js)
    target_link_libraries(webgpu_dawn INTERFACE ${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_generated_sig_info.js)
    target_link_libraries(webgpu_dawn INTERFACE ${DAWN_DIR}/third_party/emdawnwebgpu/library_webgpu.js)
else()
endif()
