#!/usr/bin/env bash
# Build LibreDWG (open source, GPL) into tools/libredwg, for reading AutoCAD .dwg P&IDs in the
# app (src/pid_dexpi_aas/cad/dwg_read.py). Needs a C compiler and make (Linux: build-essential;
# macOS: xcode-select --install). No admin rights needed; takes a few minutes.
# Windows: use WSL2, or save the drawing as .dxf in AutoCAD (read directly, no LibreDWG needed).
set -euo pipefail
VERSION="${LIBREDWG_VERSION:-0.14}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/tools/src"
mkdir -p "$SRC"
cd "$SRC"
curl -fL -o "libredwg-$VERSION.tar.xz" "https://github.com/LibreDWG/libredwg/releases/download/$VERSION/libredwg-$VERSION.tar.xz"
tar xf "libredwg-$VERSION.tar.xz"
cd "libredwg-$VERSION"
./configure --prefix="$ROOT/tools/libredwg" --disable-bindings --disable-static > ../configure.log
make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" > ../make.log
make install > ../install.log
echo "LibreDWG $VERSION -> $ROOT/tools/libredwg/bin/dwg2dxf"
