cmake_minimum_required(VERSION 3.13)
project(run)                       
set(CMAKE_CXX_STANDARD 20)        

add_executable(run "run.cpp")

set_target_properties(run PROPERTIES SUFFIX ".html")
set(SHELL_FILE_PATH "../custom_shell.html")

# target_link_options(run PRIVATE "-sUSE_WEBGPU=1" "-sUSE_GLFW=3")

# asyncify for emscripten_sleep()
# see https://emscripten.org/docs/api_reference/emscripten.h.html#pseudo-synchronous-functions
target_link_options(run PRIVATE "-sUSE_WEBGPU=1" "-sASYNCIFY=1" "--shell-file=${SHELL_FILE_PATH}")

target_include_directories(run PRIVATE "../../")
