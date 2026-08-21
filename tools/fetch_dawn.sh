#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/tools/dependencies.sh"
readonly SOURCE="$ROOT/third_party/local/dawn/source"

checkout() {
  local directory=$1 url=$2 revision=$3
  if [[ ! -d "$directory/.git" ]]; then
    mkdir -p "$directory"
    git -C "$directory" init -q
  fi
  if [[ $(git -C "$directory" rev-parse HEAD 2>/dev/null || true) != "$revision" ]]; then
    git -C "$directory" fetch --depth 1 "$url" "$revision"
    git -C "$directory" checkout --detach -q FETCH_HEAD
  fi
}

checkout "$SOURCE" https://dawn.googlesource.com/dawn "$DAWN_REV"
python3 "$SOURCE/tools/fetch_dawn_dependencies.py" --directory "$SOURCE" --shallow
checkout "$SOURCE/third_party/spirv-tools/src" \
  https://github.com/KhronosGroup/SPIRV-Tools.git "$SPIRV_TOOLS_REV"
checkout "$SOURCE/third_party/spirv-headers/src" \
  https://github.com/KhronosGroup/SPIRV-Headers.git "$SPIRV_HEADERS_REV"
