cmake_minimum_required(VERSION 3.14)

include(ExternalProject)
include(FetchContent)

# include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/print_target.cmake")

# Optionally try to find an existing Dawn build.
set(ENABLE_DAWN_FIND ON CACHE BOOL "Attempt to find an existing Dawn build" FORCE)
set(DAWN_BUILD_FOUND OFF CACHE BOOL "Dawn build found" FORCE)

# Setup directories and basic paths
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(DAWN_DIR           "${FETCHCONTENT_BASE_DIR}/dawn" CACHE INTERNAL "Dawn source directory")

# For Emscripten builds (if desired)
set(EM_SDK_DIR         $ENV{EMSDK} CACHE INTERNAL "")
set(EMSCRIPTEN_DIR     "${EM_SDK_DIR}/upstream/emscripten" CACHE INTERNAL "")

# Detect and normalize target architecture
# This will be used to make the Dawn build directory arch-specific.
set(_raw_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(EMSCRIPTEN)
  set(_raw_arch "wasm32")
elseif(APPLE)
  # Prefer CMAKE_OSX_ARCHITECTURES when provided (can be a list)
  if(DEFINED CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL "")
    list(LENGTH CMAKE_OSX_ARCHITECTURES _num_osx_archs)
    if(_num_osx_archs GREATER 1)
      message(WARNING "Multiple CMAKE_OSX_ARCHITECTURES set: ${CMAKE_OSX_ARCHITECTURES}. Using the first for Dawn build selection.")
    endif()
    list(GET CMAKE_OSX_ARCHITECTURES 0 _raw_arch)
  endif()
elseif(ANDROID)
  # Use the ABI name when available (e.g., arm64-v8a, armeabi-v7a, x86_64)
  if(DEFINED ANDROID_ABI AND NOT ANDROID_ABI STREQUAL "")
    set(_raw_arch "${ANDROID_ABI}")
  endif()
elseif(WIN32)
  set(DAWN_ENABLE_VULKAN           OFF CACHE INTERNAL "Always assert in Dawn" FORCE)
  set(DAWN_FORCE_SYSTEM_COMPONENT_LOAD            ON CACHE INTERNAL " " FORCE)
  # Prefer generator platform when present (e.g., x64, Win32, ARM64)
  if(DEFINED CMAKE_GENERATOR_PLATFORM AND NOT CMAKE_GENERATOR_PLATFORM STREQUAL "")
    set(_raw_arch "${CMAKE_GENERATOR_PLATFORM}")
  endif()
endif()

string(TOLOWER "${_raw_arch}" _arch)
# Normalize common variants
if(_arch STREQUAL "amd64" OR _arch STREQUAL "x64")
  set(_arch "x86_64")
elseif(_arch STREQUAL "aarch64")
  set(_arch "arm64")
elseif(_arch STREQUAL "armv7-a" OR _arch STREQUAL "armeabi-v7a")
  set(_arch "armv7")
elseif(_arch MATCHES "arm64[-_]?v8a")
  set(_arch "arm64-v8a")
elseif(_arch STREQUAL "" OR _arch STREQUAL "unknown")
  set(_arch "unknown")
endif()

set(DAWN_ARCH "${_arch}" CACHE INTERNAL "Target architecture for Dawn" FORCE)

# Decide where to build Dawn’s build files (now arch-aware).
if(EMSCRIPTEN)
  set(_dawn_build_os "web")
elseif(WIN32)
  set(_dawn_build_os "win")
elseif(IOS)
  set(_dawn_build_os "ios")
elseif(APPLE)
  set(_dawn_build_os "mac")
elseif(ANDROID)
  set(_dawn_build_os "android")
else()
  set(_dawn_build_os "unix")
endif()

set(DAWN_BUILD_DIR "${DAWN_DIR}/build_${_dawn_build_os}_${DAWN_ARCH}" CACHE INTERNAL "arch-specific build directory" FORCE)
message(STATUS "Dawn: target OS=${_dawn_build_os}, arch=${DAWN_ARCH}, build dir=${DAWN_BUILD_DIR}")

# Ensure Dawn/Tint inherit iOS 13+ (important for std::filesystem availability)
if(IOS)
  set(DAWN_USE_GLFW                      OFF CACHE INTERNAL "" FORCE)
  if(NOT DEFINED MINIGPU_IOS_DEPLOYMENT_TARGET)
    set(MINIGPU_IOS_DEPLOYMENT_TARGET "16.0" CACHE STRING "Minimum iOS version" FORCE)
  endif()
  set(CMAKE_OSX_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
  set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
  set(CMAKE_XCODE_ATTRIBUTE_IPHONESIMULATOR_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)

  # If not using the Xcode generator, also force min-version flags
  if(CMAKE_GENERATOR MATCHES "Unix Makefiles|Ninja")
    if(CMAKE_OSX_SYSROOT MATCHES "iphonesimulator")
      add_compile_options(-mios-simulator-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
      add_link_options(-mios-simulator-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
    else()
      add_compile_options(-miphoneos-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
      add_link_options(-miphoneos-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
    endif()
  endif()
endif()

# Add Dawn header include directories so that they are available later.
include_directories(BEFORE PUBLIC 
  "${DAWN_BUILD_DIR}/src/dawn/native/"
  "${DAWN_BUILD_DIR}/src/dawn/native/Debug"
  "${DAWN_BUILD_DIR}/src/dawn/native/Release"
)

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

  # Dawn options
  set(DAWN_ALWAYS_ASSERT               OFF   CACHE BOOL "" FORCE)
  set(DAWN_BUILD_EXAMPLES              OFF   CACHE BOOL "" FORCE)
  set(DAWN_BUILD_SAMPLES               OFF   CACHE BOOL "" FORCE)
  set(DAWN_BUILD_TESTS                 OFF   CACHE BOOL "" FORCE)
  set(DAWN_ENABLE_INSTALL              OFF   CACHE BOOL "" FORCE)
  set(DAWN_FETCH_DEPENDENCIES           ON   CACHE BOOL "" FORCE)
  set(TINT_BUILD_TESTS                 OFF   CACHE BOOL "" FORCE)
  set(TINT_BUILD_IR_BINARY             OFF   CACHE BOOL "" FORCE)
  set(TINT_BUILD_CMD_TOOLS             OFF   CACHE BOOL "" FORCE)
  set(DAWN_ENABLE_GLFW                 OFF   CACHE BOOL "" FORCE)
  set(DAWN_USE_GLFW                    OFF   CACHE BOOL "" FORCE)
  set(DAWN_BUILD_MONOLITHIC_LIBRARY  SHARED CACHE STRING "Monolithic library type" FORCE)

  # iOS minimum version (std::filesystem availability, simulator)
  if(IOS)
    if(NOT DEFINED MINIGPU_IOS_DEPLOYMENT_TARGET)
      set(MINIGPU_IOS_DEPLOYMENT_TARGET "13.0" CACHE STRING "" FORCE)
    endif()
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
    set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
    set(CMAKE_XCODE_ATTRIBUTE_IPHONESIMULATOR_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)

    # For non-Xcode generators, also force min-version flags
    if(CMAKE_GENERATOR MATCHES "Unix Makefiles|Ninja")
      if(CMAKE_OSX_SYSROOT MATCHES "iphonesimulator")
        add_compile_options(-mios-simulator-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
        add_link_options(-mios-simulator-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
      else()
        add_compile_options(-miphoneos-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
        add_link_options(-miphoneos-version-min=${MINIGPU_IOS_DEPLOYMENT_TARGET})
      endif()
    endif()
  endif()

  # Ensure source present on required commit (idempotent remote setup)
  if(NOT DEFINED DAWN_COMMIT OR DAWN_COMMIT STREQUAL "")
    set(DAWN_COMMIT "e1d6e12337080cf9f6d8726209e86df449bc6e9a" CACHE STRING "Dawn commit to checkout" FORCE)
  endif()
  file(MAKE_DIRECTORY ${DAWN_DIR})
  execute_process(COMMAND git init WORKING_DIRECTORY "${DAWN_DIR}")
  execute_process(
    COMMAND git remote get-url origin
    WORKING_DIRECTORY "${DAWN_DIR}"
    RESULT_VARIABLE _have_origin
    OUTPUT_QUIET ERROR_QUIET
  )
  if(_have_origin EQUAL 0)
    execute_process(COMMAND git remote set-url origin https://dawn.googlesource.com/dawn WORKING_DIRECTORY "${DAWN_DIR}")
  else()
    execute_process(COMMAND git remote add origin https://dawn.googlesource.com/dawn WORKING_DIRECTORY "${DAWN_DIR}")
  endif()
  execute_process(COMMAND git fetch origin ${DAWN_COMMIT} WORKING_DIRECTORY "${DAWN_DIR}")
  execute_process(COMMAND git checkout ${DAWN_COMMIT} WORKING_DIRECTORY "${DAWN_DIR}")
  execute_process(COMMAND git reset --hard ${DAWN_COMMIT} WORKING_DIRECTORY "${DAWN_DIR}")

  # Set kIOMainPortDefault to 0
  if(APPLE)
    set(PORTDEFAULT_FILE "${DAWN_DIR}/src/dawn/native/metal/PhysicalDeviceMTL.mm")
    if(EXISTS "${PORTDEFAULT_FILE}")
      file(READ  "${PORTDEFAULT_FILE}" PORTDEFAULT_CONTENT)
      string(REGEX REPLACE "kIOMainPortDefault" "0" PORTDEFAULT_CONTENT "${PORTDEFAULT_CONTENT}")
      file(WRITE "${PORTDEFAULT_FILE}" "${PORTDEFAULT_CONTENT}")
    endif()
  endif()

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

# Create an IMPORTED target that matches the monolithic output
if(TARGET webgpu_dawn)
  # Dawn already created it in this project; use it directly
else()
  if(IOS)
    # Xcode config suffix: Debug-iphoneos/Debug-iphonesimulator etc.
    if(CMAKE_OSX_SYSROOT MATCHES "iphonesimulator")
      set(_ios_conf_suffix "-iphonesimulator")
    else()
      set(_ios_conf_suffix "-iphoneos")
    endif()
    add_library(webgpu_dawn STATIC IMPORTED)
    # Monolithic static archive name is libwebgpu_dawn.a
    set_target_properties(webgpu_dawn PROPERTIES
      IMPORTED_LOCATION_DEBUG           "${DAWN_BUILD_DIR}/src/dawn/native/Debug${_ios_conf_suffix}/libwebgpu_dawn.a"
      IMPORTED_LOCATION_RELEASE         "${DAWN_BUILD_DIR}/src/dawn/native/Release${_ios_conf_suffix}/libwebgpu_dawn.a"
      IMPORTED_LOCATION_RELWITHDEBINFO  "${DAWN_BUILD_DIR}/src/dawn/native/RelWithDebInfo${_ios_conf_suffix}/libwebgpu_dawn.a"
      IMPORTED_LOCATION_MINSIZEREL      "${DAWN_BUILD_DIR}/src/dawn/native/MinSizeRel${_ios_conf_suffix}/libwebgpu_dawn.a"
    )
  elseif(APPLE)
    # macOS: prefer shared monolithic dylib; fallback to static if needed
    if(EXISTS "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.dylib")
      add_library(webgpu_dawn SHARED IMPORTED)
      set_target_properties(webgpu_dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.dylib"
      )
    elseif(EXISTS "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.a")
      add_library(webgpu_dawn STATIC IMPORTED)
      set_target_properties(webgpu_dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.a"
      )
    endif()
  elseif(ANDROID)
    if(EXISTS "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.so")
      add_library(webgpu_dawn SHARED IMPORTED)
      set_target_properties(webgpu_dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.so"
      )
    elseif(EXISTS "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.a")
      add_library(webgpu_dawn STATIC IMPORTED)
      set_target_properties(webgpu_dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.a"
      )
    endif()
  elseif(WIN32)
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
  else() # Linux/Unix
    if(EXISTS "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.so")
      add_library(webgpu_dawn SHARED IMPORTED)
      set_target_properties(webgpu_dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.so"
      )
    elseif(EXISTS "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.a")
      add_library(webgpu_dawn STATIC IMPORTED)
      set_target_properties(webgpu_dawn PROPERTIES
        IMPORTED_LOCATION "${DAWN_BUILD_DIR}/src/dawn/native/libwebgpu_dawn.a"
      )
    endif()
  endif()
endif()
