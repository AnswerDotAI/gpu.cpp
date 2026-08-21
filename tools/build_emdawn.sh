#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/tools/dependencies.sh"
readonly EMSDK_ROOT_DIR=${GPUCPP_EMSDK_ROOT:-"$ROOT/../emsdk"}
readonly SOURCE="$ROOT/third_party/local/dawn/source"
readonly BUILD="$ROOT/third_party/local/dawn/build-web"
readonly STAGE="$ROOT/third_party/emdawnwebgpu"

if [[ ! -d "$EMSDK_ROOT_DIR/.git" ]]; then
  echo "emsdk is not cloned at $EMSDK_ROOT_DIR" >&2
  exit 1
fi

if [[ $(git -C "$EMSDK_ROOT_DIR" rev-parse HEAD) != "$EMSDK_REV" ]]; then
  git -C "$EMSDK_ROOT_DIR" fetch --depth 1 origin "$EMSDK_REV"
  git -C "$EMSDK_ROOT_DIR" checkout --detach -q FETCH_HEAD
fi
export EMSDK_QUIET=1
"$EMSDK_ROOT_DIR/emsdk" install "$EMSCRIPTEN_VERSION"
"$EMSDK_ROOT_DIR/emsdk" activate "$EMSCRIPTEN_VERSION"
source "$EMSDK_ROOT_DIR/emsdk_env.sh" >/dev/null

"$ROOT/tools/fetch_dawn.sh"
emcmake cmake -S "$SOURCE" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAWN_BUILD_SAMPLES=OFF \
  -DDAWN_BUILD_TESTS=OFF \
  -DDAWN_BUILD_BENCHMARKS=OFF \
  -DDAWN_BUILD_PROTOBUF=OFF \
  -DDAWN_FETCH_DEPENDENCIES=OFF \
  -DDAWN_USE_GLFW=OFF \
  -DDAWN_SUPPORTS_CXX_MODULES=OFF \
  -DTINT_BUILD_CMD_TOOLS=OFF \
  -DTINT_BUILD_IR_BINARY=OFF
cmake --build "$BUILD" --target emdawnwebgpu_pkg

cmake -E remove_directory "$STAGE"
cmake -E copy_directory "$BUILD/emdawnwebgpu_pkg" "$STAGE"
echo "Staged Emdawnwebgpu from Dawn $DAWN_REV in $STAGE"
