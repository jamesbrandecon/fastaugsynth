#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DIST_DIR="$ROOT/build/dist"
OUT_DIR="$ROOT/build/artifacts"

mkdir -p "$OUT_DIR"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$OS" in
  linux*)
    OS="linux"
    ;;
  darwin*)
    OS="darwin"
    ;;
  msys*|mingw*|cygwin*)
    OS="windows"
    ;;
  *)
    echo "Unsupported OS name from uname: $OS" >&2
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64)
    ARCH="x86_64"
    ;;
  arm64|aarch64)
    ARCH="arm64"
    ;;
  *)
    echo "Unsupported architecture from uname: $ARCH" >&2
    exit 1
    ;;
esac

ASSET="statlibbackend-${OS}-${ARCH}.tar.gz"

if [ ! -d "$DIST_DIR" ]; then
  echo "Expected dist directory at $DIST_DIR" >&2
  exit 1
fi

tar -C "$DIST_DIR" -czf "$OUT_DIR/$ASSET" .

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUT_DIR/$ASSET" > "$OUT_DIR/$ASSET.sha256"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUT_DIR/$ASSET" > "$OUT_DIR/$ASSET.sha256"
else
  echo "Neither sha256sum nor shasum is available" >&2
  exit 1
fi

echo "$OUT_DIR/$ASSET"
