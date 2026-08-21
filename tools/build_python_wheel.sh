#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  echo "Local Python publishing currently supports macOS ARM64" >&2
  exit 1
fi

export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
uv build --wheel --clear --out-dir "$ROOT/dist" "$ROOT"
twine check "$ROOT"/dist/*.whl
