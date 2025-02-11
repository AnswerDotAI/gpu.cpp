# Getting Started with CMAKE
# Each example includes this and sets PROJECT_NAME
# cd examples/hello_world
# cmake -S . build/ -DCMAKE_BUILD_TYPE=Release
# cmake --build build/ --config Release
# ./build/hello_world

if(NOT MSVC)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 20)
endif()

# Path finding logic to find our root recipes from nested folders
get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(PROJECT_ROOT ${PROJECT_ROOT} DIRECTORY)

# Ensure the build type is set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build: Debug or Release" FORCE)
endif()

# Include the gpu.cpp + Dawn library
include("${PROJECT_ROOT}/cmake/gpu.cmake")

# Create the executable
add_executable(${PROJECT_NAME} run.cpp)

# Link gpu + dawn library
target_link_libraries(${PROJECT_NAME} PRIVATE gpu)

# Certain platforms need to copy the library files to the build directory
if(MSVC)
    # Copy webgpu_dawn.dll to the build directory
    # CMake multigenerators like MSVC need --config Release on
    # the cmake --build command or they will output to /Debug
    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${DAWN_INSTALL_PREFIX}/${CMAKE_BUILD_TYPE}/webgpu_dawn.dll
                $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

