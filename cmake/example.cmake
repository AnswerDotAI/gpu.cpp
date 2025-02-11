set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # export compile_commands.json to use with
                                      # LSP
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)

# Construct potential paths
set(FILEPATH_CURRENT_DIR "${DIRECTORY}/")
set(FILEPATH_PROJECT_ROOT "${PROJECT_ROOT}/")

# Include file finding utility script
include("${FILEPATH_PROJECT_ROOT}/cmake/find_gpu.cmake")

# Check if the file exists in the current directory
find_project_root(${CMAKE_CURRENT_SOURCE_DIR} ${FILENAME}
                  TARGET_FILE_PATH)
if("${TARGET_FILE_PATH}" STREQUAL "")
    find_project_root(${FILEPATH_CURRENT_DIR} ${FILENAME}
                      TARGET_FILE_PATH)
    if("${TARGET_FILE_PATH}" STREQUAL "")
        message(
            FATAL_ERROR
                "File ${FILENAME} not found in either ${CMAKE_CURRENT_SOURCE_DIR} or ${CMAKE_CURRENT_SOURCE_DIR}/../../"
        )
    endif()
endif()

# Ensure the build type is set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build: Debug or Release" FORCE)
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

if(NOT TARGET gpu)
    message(STATUS "GPU_LIB not found")
    include("${TARGET_FILE_PATH}/cmake/gpu.cmake")
endif()
add_executable(${PROJECT_NAME} run.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE gpu)
target_link_libraries(${PROJECT_NAME} PRIVATE ${WEBGPU_DAWN})

if(MSVC)
# Copy webgpu_dawn.dll to the build directory
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${DAWN_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/webgpu_dawn.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

