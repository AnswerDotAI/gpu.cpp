cmake_minimum_required(VERSION 3.28)
project(gpu.cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EMSCRIPTEN)
  set(GPUCPP_BUILD_PYTHON_DEFAULT OFF)
  set(GPUCPP_BUILD_EXAMPLES_DEFAULT OFF)
else()
  set(GPUCPP_BUILD_PYTHON_DEFAULT ${PROJECT_IS_TOP_LEVEL})
  set(GPUCPP_BUILD_EXAMPLES_DEFAULT ON)
endif()
option(GPUCPP_BUILD_PYTHON "Build the Python binding" ${GPUCPP_BUILD_PYTHON_DEFAULT})
option(GPUCPP_BUILD_EXAMPLES "Build gpu.cpp examples" ${GPUCPP_BUILD_EXAMPLES_DEFAULT})

add_library(gpucpp INTERFACE)
target_include_directories(gpucpp INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if(EMSCRIPTEN)
  set(GPUCPP_EMDAWN_ROOT
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/emdawnwebgpu" CACHE PATH
      "Staged Emdawnwebgpu package")
  set(EMDAWN_PORT
      "--use-port=${GPUCPP_EMDAWN_ROOT}/emdawnwebgpu.port.py")
  if(NOT EXISTS "${GPUCPP_EMDAWN_ROOT}/emdawnwebgpu.port.py")
    message(FATAL_ERROR "Emdawnwebgpu is not built; run tools/build_emdawn.sh")
  endif()
  target_compile_options(gpucpp INTERFACE "${EMDAWN_PORT}" "-fwasm-exceptions")
  target_link_options(gpucpp INTERFACE "${EMDAWN_PORT}" "-fwasm-exceptions"
                                           "-sJSPI=1" "--closure=1")
  if(CMAKE_HOST_APPLE)
    target_link_options(gpucpp INTERFACE "--closure-args=--platform=java")
  endif()
else()

set(GPUCPP_DAWN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dawn" CACHE PATH
    "Staged Dawn distribution")
set(GPUCPP_LOCAL_DAWN "${CMAKE_CURRENT_SOURCE_DIR}/third_party/local/dawn" CACHE PATH
    "Developer Dawn source/build tree")

if(EXISTS "${GPUCPP_DAWN_ROOT}/include/dawn/webgpu_cpp.h")
  set(DAWN_INCLUDE_DIRS "${GPUCPP_DAWN_ROOT}/include")
  if(APPLE)
    set(DAWN_LIBRARY "${GPUCPP_DAWN_ROOT}/lib/libwebgpu_dawn.dylib")
  elseif(UNIX)
    set(DAWN_LIBRARY "${GPUCPP_DAWN_ROOT}/lib/libwebgpu_dawn.so")
  endif()
  set(SPIRV_AS "${GPUCPP_DAWN_ROOT}/bin/spirv-as")
elseif(EXISTS "${GPUCPP_LOCAL_DAWN}/build-latest/gen/include/dawn/webgpu_cpp.h")
  set(DAWN_INCLUDE_DIRS
      "${GPUCPP_LOCAL_DAWN}/build-latest/gen/include"
      "${GPUCPP_LOCAL_DAWN}/source/include")
  if(APPLE)
    set(DAWN_LIBRARY
        "${GPUCPP_LOCAL_DAWN}/build-latest/src/dawn/native/libwebgpu_dawn.dylib")
  elseif(UNIX)
    set(DAWN_LIBRARY
        "${GPUCPP_LOCAL_DAWN}/build-latest/src/dawn/native/libwebgpu_dawn.so")
  endif()
  set(SPIRV_AS "${GPUCPP_LOCAL_DAWN}/build-spirv-tools/tools/spirv-as")
else()
  message(FATAL_ERROR "Dawn is not built; run tools/build_dawn.sh")
endif()

if(NOT EXISTS "${DAWN_LIBRARY}")
  message(FATAL_ERROR "Dawn library not found at ${DAWN_LIBRARY}")
endif()
if(NOT EXISTS "${SPIRV_AS}")
  message(FATAL_ERROR "spirv-as not found at ${SPIRV_AS}")
endif()

add_library(Dawn::webgpu SHARED IMPORTED)
set_target_properties(Dawn::webgpu PROPERTIES
  IMPORTED_LOCATION "${DAWN_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${DAWN_INCLUDE_DIRS}")

target_link_libraries(gpucpp INTERFACE Dawn::webgpu)
endif()

if(GPUCPP_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.4
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(pybind11)

  set(GPUCPP_PYTHON_PACKAGE_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/python/gpu_cpp")
  file(MAKE_DIRECTORY "${GPUCPP_PYTHON_PACKAGE_DIR}")
  configure_file(python/gpu_cpp/__init__.py
                 "${GPUCPP_PYTHON_PACKAGE_DIR}/__init__.py" COPYONLY)
  configure_file(python/gpu_cpp/_async.py
                 "${GPUCPP_PYTHON_PACKAGE_DIR}/_async.py" COPYONLY)

  pybind11_add_module(_gpu_cpp bindings/python/gpu_cpp.cpp)
  target_link_libraries(_gpu_cpp PRIVATE gpucpp)
  set_target_properties(_gpu_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${GPUCPP_PYTHON_PACKAGE_DIR}")
  if(APPLE)
    set_target_properties(_gpu_cpp PROPERTIES INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set_target_properties(_gpu_cpp PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()

  install(TARGETS _gpu_cpp LIBRARY DESTINATION gpu_cpp)
  install(FILES "${DAWN_LIBRARY}" DESTINATION gpu_cpp)
endif()

if(GPUCPP_BUILD_EXAMPLES)
  add_executable(hello_gpu examples/hello_world/run.cpp)
  add_executable(float16_gpu examples/float16/run.cpp)
  add_executable(gpu_puzzles examples/gpu_puzzles/run.cpp)
  add_executable(gpu_puzzles_key examples/gpu_puzzles/key.cpp)
  add_executable(matmul_gpu examples/matmul/run.cpp)
  add_executable(physics_gpu examples/physics/run.cpp)
  add_executable(render_gpu examples/render/run.cpp)
  add_executable(shadertui_gpu examples/shadertui/run.cpp)
  add_executable(transpose_gpu examples/transpose/run.cpp)

  foreach(target IN ITEMS hello_gpu float16_gpu gpu_puzzles gpu_puzzles_key
                          matmul_gpu physics_gpu render_gpu shadertui_gpu
                          transpose_gpu)
    target_link_libraries(${target} PRIVATE gpucpp)
  endforeach()
  target_include_directories(matmul_gpu PRIVATE third_party/headers)
  target_include_directories(transpose_gpu PRIVATE third_party/headers)

endif()

include(CTest)
if(EMSCRIPTEN)
  add_library(gpu_cpp_web_bindings OBJECT bindings/web/gpu_cpp.cpp)
  target_link_libraries(gpu_cpp_web_bindings PRIVATE gpucpp)

  add_executable(gpu_cpp_web $<TARGET_OBJECTS:gpu_cpp_web_bindings>)
  set_target_properties(gpu_cpp_web PROPERTIES SUFFIX ".mjs")
  target_link_libraries(gpu_cpp_web PRIVATE gpucpp)
  target_link_options(gpu_cpp_web PRIVATE "--bind" "--no-entry"
    "-sMODULARIZE=1" "-sEXPORT_ES6=1" "-sEXPORT_NAME=createGpuCpp"
    "-sENVIRONMENT=web" "-sALLOW_MEMORY_GROWTH=1")

  if(BUILD_TESTING)
    add_executable(web_binding_test tests/web_binding_test.cpp
                   $<TARGET_OBJECTS:gpu_cpp_web_bindings>)
    set_target_properties(web_binding_test PROPERTIES SUFFIX ".html")
    target_link_libraries(web_binding_test PRIVATE gpucpp)
    target_link_options(web_binding_test PRIVATE "--bind" "--emrun"
                                                "-sMODULARIZE=1" "-sEXPORT_ES6=1"
                                                "-sALLOW_MEMORY_GROWTH=1")
  endif()
elseif(BUILD_TESTING)
  set(TEST_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/write42.spv")
  add_custom_command(
    OUTPUT "${TEST_SPIRV}"
    COMMAND "${SPIRV_AS}" --target-env spv1.3
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/write42.spvasm" -o "${TEST_SPIRV}"
    DEPENDS tests/write42.spvasm
    VERBATIM)

  add_executable(gpu_test tests/gpu_test.cpp "${TEST_SPIRV}")
  target_link_libraries(gpu_test PRIVATE gpucpp)
  add_test(NAME gpu COMMAND gpu_test "${TEST_SPIRV}")
endif()
if(BUILD_TESTING AND GPUCPP_BUILD_PYTHON)
  add_test(NAME python
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/test_gpu_cpp.py")
  set_tests_properties(python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python")
endif()
