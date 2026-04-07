#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DIST_DIR="$ROOT/build/dist"
OUT_DIR="$ROOT/build/artifacts"

mkdir -p "$OUT_DIR"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
ASSET="statlibbackend-${OS}-${ARCH}.tar.gz"

if [ ! -d "$DIST_DIR" ]; then
  echo "Expected dist directory at $DIST_DIR" >&2
  exit 1
fi

tar -C "$DIST_DIR" -czf "$OUT_DIR/$ASSET" .
sha256sum "$OUT_DIR/$ASSET" > "$OUT_DIR/$ASSET.sha256"

echo "$OUT_DIR/$ASSET"
