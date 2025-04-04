cmake_minimum_required(VERSION 3.14)

include(ExternalProject)
include(FetchContent)

# include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/print_target.cmake")


# Setup directories and basic paths
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(DAWN_DIR           "${FETCHCONTENT_BASE_DIR}/dawn" CACHE INTERNAL "Dawn source directory")

# For Emscripten builds (if desired)
set(EM_SDK_DIR         $ENV{EMSDK} CACHE INTERNAL "")
set(EMSCRIPTEN_DIR     "${EM_SDK_DIR}/upstream/emscripten" CACHE INTERNAL "")

# Decide where to build Dawn’s build files.
if(EMSCRIPTEN)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_web" CACHE INTERNAL "web build directory" FORCE)
elseif(WIN32)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_win" CACHE INTERNAL "windows build directory" FORCE)
elseif(IOS)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_ios" CACHE INTERNAL "ios build directory" FORCE)
elseif(APPLE)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_mac" CACHE INTERNAL "mac build directory" FORCE)
elseif(ANDROID)
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_android" CACHE INTERNAL "android build directory" FORCE)
else()
  set(DAWN_BUILD_DIR "${DAWN_DIR}/build_unix" CACHE INTERNAL "linux build directory" FORCE)
endif()

# Add Dawn header include directories so that they are available later.
include_directories(BEFORE PUBLIC 
  "${DAWN_BUILD_DIR}/src/dawn/native/"
  "${DAWN_BUILD_DIR}/src/dawn/native/Debug"
  "${DAWN_BUILD_DIR}/src/dawn/native/Release"
)


# Optionally try to find an existing Dawn build.
set(ENABLE_DAWN_FIND ON CACHE BOOL "Attempt to find an existing Dawn build" FORCE)
set(DAWN_BUILD_FOUND OFF CACHE BOOL "Dawn build found" FORCE)

if(ENABLE_DAWN_FIND)
    message(STATUS "Attempting to find an existing Dawn build...")
  if(WIN32)
    find_library(WEBGPU_DAWN_DEBUG NAMES webgpu_dawn HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Debug")
    find_library(WEBGPU_DAWN_RELEASE NAMES webgpu_dawn HINTS "${DAWN_BUILD_DIR}/src/dawn/native/Release")
    
    if(WEBGPU_DAWN_DEBUG OR WEBGPU_DAWN_RELEASE)
    message(STATUS "Dawn build found on Windows. Debug: ${WEBGPU_DAWN_DEBUG}, Release: ${WEBGPU_DAWN_RELEASE}")
      set(DAWN_BUILD_FOUND ON)
    endif()
  elseif(NOT EMSCRIPTEN AND NOT WIN32)
    find_library(WEBGPU_DAWN_LIB NAMES webgpu_dawn.so PATHS "${DAWN_BUILD_DIR}/src/dawn/native")
    
    if(WEBGPU_DAWN_LIB)
    message(STATUS "Dawn build found on Linux/Unix. Library: ${WEBGPU_DAWN_LIB}")
      set(DAWN_BUILD_FOUND ON)
    endif()
  endif()
endif()


# Pre-build Dawn at configuration time if not already built.
if(NOT DAWN_BUILD_FOUND)
  message(STATUS "Dawn build not found - pre-building Dawn.")

  # Force Dawn build options.
  set(DAWN_ALWAYS_ASSERT           ON CACHE INTERNAL "Always assert in Dawn" FORCE)
  set(DAWN_BUILD_MONOLITHIC_LIBRARY ON CACHE INTERNAL "Build Dawn monolithically" FORCE)
  set(DAWN_BUILD_EXAMPLES          OFF CACHE INTERNAL "Build Dawn examples" FORCE)
  set(DAWN_BUILD_SAMPLES           OFF CACHE INTERNAL "Build Dawn samples" FORCE)
  set(DAWN_BUILD_TESTS             OFF CACHE INTERNAL "Build Dawn tests" FORCE)
  set(DAWN_ENABLE_INSTALL          OFF  CACHE INTERNAL "Enable Dawn installation" FORCE)
  set(DAWN_FETCH_DEPENDENCIES      ON  CACHE INTERNAL "Fetch Dawn dependencies" FORCE)
  set(TINT_BUILD_TESTS             OFF CACHE INTERNAL "Build Tint Tests" FORCE)
  set(TINT_BUILD_IR_BINARY         OFF CACHE INTERNAL "Build Tint IR binary" FORCE)
  set(TINT_BUILD_CMD_TOOLS         OFF CACHE INTERNAL "Build Tint command line tools" FORCE)
  set(DAWN_EMSCRIPTEN_TOOLCHAIN    ${EMSCRIPTEN_DIR} CACHE INTERNAL "Emscripten toolchain" FORCE)

  set(DAWN_COMMIT "66d57f910357befb441b91162f29a97f687af6d9" CACHE STRING "Dawn commit to checkout" FORCE)
  
  file(MAKE_DIRECTORY ${DAWN_DIR})
  # Initialize Git and set/update remote.
  execute_process(COMMAND git init
  WORKING_DIRECTORY "${DAWN_DIR}"
  )
  execute_process(
    COMMAND git remote add origin https://dawn.googlesource.com/dawn
    WORKING_DIRECTORY "${DAWN_DIR}"
  )
  # Fetch and checkout the specified commit.
  execute_process(
  COMMAND git fetch origin ${DAWN_COMMIT}
  WORKING_DIRECTORY "${DAWN_DIR}"
  )
  execute_process(
  COMMAND git checkout ${DAWN_COMMIT}
  WORKING_DIRECTORY "${DAWN_DIR}"
  )
  execute_process(
  COMMAND git reset --hard ${DAWN_COMMIT}
  WORKING_DIRECTORY "${DAWN_DIR}"
  )
  # Fetch the Dawn repository if not already present.
  FetchContent_Declare(
    dawn
    SOURCE_DIR   ${DAWN_DIR}
    SUBBUILD_DIR ${DAWN_BUILD_DIR}/tmp
    BINARY_DIR   ${DAWN_BUILD_DIR}
  )
  FetchContent_MakeAvailable(dawn)

  set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${DAWN_DIR}/src" CACHE INTERNAL "")

  set(DAWN_BUILD_FOUND ON)
endif()  # End pre-build Dawn

# Create an IMPORTED target for the Dawn library.
# Adjust the expected output name/extension per platform.
if(MSVC)
message(STATUS "Dawn build found on Windows.")
# MSVC: use separate debug and release dlls.
if((NOT WEBGPU_DAWN_DEBUG) OR (WEBGPU_DAWN_DEBUG MATCHES "NOTFOUND"))
  find_library(WEBGPU_DAWN_DEBUG NAMES webgpu_dawn PATHS "${DAWN_BUILD_DIR}/src/dawn/native/Debug")
endif()
if((NOT WEBGPU_DAWN_RELEASE) OR (WEBGPU_DAWN_RELEASE MATCHES "NOTFOUND"))
  find_library(WEBGPU_DAWN_RELEASE NAMES webgpu_dawn PATHS "${DAWN_BUILD_DIR}/src/dawn/native/Release")
endif()

if(WEBGPU_DAWN_DEBUG OR WEBGPU_DAWN_RELEASE)
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn INTERFACE)
    target_link_libraries(webgpu_dawn INTERFACE
      $<$<CONFIG:Debug>:${WEBGPU_DAWN_DEBUG}>
      $<$<CONFIG:Release>:${WEBGPU_DAWN_RELEASE}>
    )
  endif()
endif()
elseif(IOS)
  # On iOS, it is common to build a static library.
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn STATIC IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.a")
  endif()
elseif(APPLE)
  # On macOS (non-iOS), typically a dynamic library (.dylib) is built.
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn SHARED IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.dylib")
  endif()
elseif(ANDROID)
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn SHARED IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.so")
  endif()
elseif(NOT EMSCRIPTEN)  # For Linux and other Unix-like systems.
  if(NOT TARGET webgpu_dawn)
    add_library(webgpu_dawn SHARED IMPORTED)
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/webgpu_dawn.so")
  endif()
endif()