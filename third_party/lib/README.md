# Release 0.2.0 (draft)

Switched from
https://github.com/jspanchu/webgpu-dawn-binaries
to building from the dawn repository:
https://github.com/google/dawn

Commit hash:
556f960f44690b3b808c779c08b44d48d4292925

Build with clang (assumes running from out/Release)

```
cmake -DBUILD_SHARED_LIBS=ON -DDAWN_BUILD_MONOLITHIC_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release ../..
```
Library artifact is at `src/dawn/native/libwebgpu_dawn.dylib` (7.4 MB)

Note that for OSX builds, needed to make modifications to the `CMakeLists.txt` file to get the build to work. Specifically, needed to add the following `FORCE_OBJECT` flags to dawn_glfw, dawn_wgpu_utils, and dawn_test_utils. Otherwise we get linker errors for missing symbols when building the shared library. This issue does not appear to be present on Linux builds.

```
(base) austinhuang@Austins-MacBook-Pro dawn % git diff 556f960f44690b3b808c779c08b44d48d4292925 5a00ab1fbc22d6ebbab39  
diff --git a/src/dawn/glfw/CMakeLists.txt b/src/dawn/glfw/CMakeLists.txt
index dc3f3ade03..d6d8d0ef4f 100644
--- a/src/dawn/glfw/CMakeLists.txt
+++ b/src/dawn/glfw/CMakeLists.txt
@@ -40,6 +40,7 @@ endif ()
 
 dawn_add_library(
   dawn_glfw
+  FORCE_OBJECT
   UTILITY_TARGET dawn_internal_config
   HEADERS
     "${headers}"
@@ -56,5 +57,5 @@ target_compile_definitions(dawn_glfw PRIVATE "WGPU_GLFW_IMPLEMENTATION")
 if(BUILD_SHARED_LIBS)
     target_compile_definitions(dawn_glfw PUBLIC "WGPU_GLFW_SHARED_LIBRARY")
 endif()
-
+# target_link_libraries(dawn_glfw PUBLIC webgpu_dawn)
 add_library(webgpu_glfw ALIAS dawn_glfw)
diff --git a/src/dawn/utils/CMakeLists.txt b/src/dawn/utils/CMakeLists.txt
index 5eb7120d99..3b00664829 100644
--- a/src/dawn/utils/CMakeLists.txt
+++ b/src/dawn/utils/CMakeLists.txt
@@ -36,6 +36,7 @@ endif()
 
 dawn_add_library(
   dawn_wgpu_utils
+  FORCE_OBJECT
   ENABLE_EMSCRIPTEN
   UTILITY_TARGET dawn_internal_config
   PRIVATE_HEADERS
@@ -55,6 +56,8 @@ dawn_add_library(
     ${private_wgpu_depends}
 )
 
+# target_link_libraries(dawn_wgpu_utils PUBLIC webgpu_dawn)
+
 # Needed by WGPUHelpers
 target_compile_definitions(dawn_wgpu_utils
   PUBLIC
@@ -66,6 +69,7 @@ target_compile_definitions(dawn_wgpu_utils
 ###############################################################################
 dawn_add_library(
   dawn_test_utils
+  FORCE_OBJECT
   UTILITY_TARGET dawn_internal_config
   PRIVATE_HEADERS
     "BinarySemaphore.h"
@@ -84,6 +88,9 @@ dawn_add_library(
     dawn::partition_alloc
 )
 
+# target_link_libraries(dawn_test_utils PUBLIC webgpu_dawn dawn_wgpu_utils dawn_proc)
+
+
 ###############################################################################
 # Dawn system utilities
 #   - Used in tests and samples
 ```

# Release 0.1.0

https://github.com/jspanchu/webgpu-dawn-binaries
commit hash:
c0602d5d0466040f6e080d6cb7209860538f9f8d

Built with clang:
cmake .. -DCMAKE_C_COMPILER=clang DCMAKE_CXX_COMPILER=clang++
