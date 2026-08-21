#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/tools/dependencies.sh"
readonly LOCAL="$ROOT/third_party/local/dawn"
readonly SOURCE="$LOCAL/source"
readonly BUILD="$LOCAL/build-latest"
readonly TOOLS_BUILD="$LOCAL/build-spirv-tools"
readonly STAGE="$ROOT/third_party/dawn"

"$ROOT/tools/fetch_dawn.sh"

backend_flags=(-DDAWN_ENABLE_NULL=OFF)
case "$(uname -s)" in
  Darwin) backend_flags+=(-DDAWN_ENABLE_METAL=ON -DDAWN_ENABLE_VULKAN=OFF) ;;
  Linux) backend_flags+=(-DDAWN_ENABLE_METAL=OFF -DDAWN_ENABLE_VULKAN=ON) ;;
  *) echo "gpu.cpp supports Dawn on macOS and Linux" >&2; exit 1 ;;
esac

cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED \
  -DDAWN_BUILD_SAMPLES=OFF \
  -DDAWN_BUILD_TESTS=OFF \
  -DDAWN_BUILD_BENCHMARKS=OFF \
  -DDAWN_BUILD_PROTOBUF=OFF \
  -DDAWN_FETCH_DEPENDENCIES=OFF \
  -DDAWN_USE_GLFW=OFF \
  -DTINT_BUILD_CMD_TOOLS=OFF \
  -DTINT_BUILD_SPV_READER=ON \
  -DTINT_BUILD_SPV_WRITER=OFF \
  "${backend_flags[@]}"
cmake --build "$BUILD" --target webgpu_dawn

cmake -S "$SOURCE/third_party/spirv-tools/src" -B "$TOOLS_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPIRV-Headers_SOURCE_DIR="$SOURCE/third_party/spirv-headers/src" \
  -DSPIRV_SKIP_TESTS=ON \
  -DSPIRV_SKIP_EXECUTABLES=OFF
cmake --build "$TOOLS_BUILD" --target spirv-as

cmake -E remove_directory "$STAGE"
cmake -E make_directory "$STAGE/include/dawn" "$STAGE/include/webgpu" \
  "$STAGE/lib" "$STAGE/bin"
cmake -E copy "$BUILD/gen/include/dawn/webgpu.h" \
  "$BUILD/gen/include/dawn/webgpu_cpp.h" "$STAGE/include/dawn"
cmake -E copy_directory "$SOURCE/include/webgpu" "$STAGE/include/webgpu"
cmake -E copy "$BUILD/gen/include/webgpu/webgpu_cpp_chained_struct.h" \
  "$STAGE/include/webgpu"
cmake -E copy "$TOOLS_BUILD/tools/spirv-as" "$STAGE/bin"
if [[ $(uname -s) == Darwin ]]; then
  cmake -E copy "$BUILD/src/dawn/native/libwebgpu_dawn.dylib" "$STAGE/lib"
else
  cmake -E copy "$BUILD/src/dawn/native/libwebgpu_dawn.so" "$STAGE/lib"
fi

echo "Staged Dawn $DAWN_REV in $STAGE"
