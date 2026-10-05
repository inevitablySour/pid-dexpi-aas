#!/usr/bin/env bash
# Download the draw.io desktop app (Linux AppImage) to ml/tools/drawio.AppImage.
# The parser uses it to read each line's drawn route (tees, flow map). On macOS or
# Windows, install draw.io normally instead - it is found automatically.
set -euo pipefail
VERSION="${DRAWIO_VERSION:-31.7.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/ml/tools/drawio.AppImage"
mkdir -p "$(dirname "$OUT")"
curl -fL -o "$OUT" "https://github.com/jgraph/drawio-desktop/releases/download/v$VERSION/drawio-x86_64-$VERSION.AppImage"
chmod +x "$OUT"
echo "draw.io $VERSION -> $OUT"
