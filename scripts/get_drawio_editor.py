"""Install the draw.io web editor for the app's Editor tab (once, ~60 MB).

draw.io (Apache 2.0) ships its web editor inside the desktop app's resources/app.asar.
This copies it to src/pid_dexpi_aas/render/drawio_editor_files/webapp, where the app
serves it itself - drawings never leave this PC. Needs the desktop app: on Linux
ml/tools/drawio.AppImage (scripts/get_drawio.sh), else a normal draw.io install.

    .venv/bin/python scripts/get_drawio_editor.py
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "src" / "pid_dexpi_aas" / "render" / "drawio_editor_files" / "webapp"
PREFIX = "drawio/src/main/webapp/"


def find_asar(tmp: Path) -> Path:
    from pid_dexpi_aas.drawio.line_routes import DRAWIO_APP
    app = Path(DRAWIO_APP)
    if not app.exists():
        sys.exit("draw.io desktop app not found - run scripts/get_drawio.sh (Linux) or install draw.io.")
    if app.suffix == ".AppImage":
        subprocess.run([str(app), "--appimage-extract", "resources/app.asar"], cwd=tmp, check=True,
                       capture_output=True)
        return tmp / "squashfs-root" / "resources" / "app.asar"
    for cand in (app.parent / "resources" / "app.asar", app.parents[1] / "Resources" / "app.asar"):
        if cand.exists():  # Windows / Linux install; macOS bundle
            return cand
    sys.exit(f"no app.asar next to {app}")


def extract(asar: Path, prefix: str, out: Path) -> int:
    """Copy the files under `prefix` from an Electron .asar archive (a JSON header of
    offsets, then the file data) to `out`."""
    n = 0
    with open(asar, "rb") as f:
        _, header_size, _, json_size = struct.unpack("<4I", f.read(16))
        header = json.loads(f.read(json_size))
        base = 8 + header_size

        def walk(node: dict, path: str) -> None:
            nonlocal n
            for name, entry in node.get("files", {}).items():
                rel = f"{path}{name}"
                if "files" in entry:
                    walk(entry, rel + "/")
                elif rel.startswith(prefix) and "offset" in entry and not entry.get("unpacked"):
                    target = out / rel[len(prefix):]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    f.seek(base + int(entry["offset"]))
                    target.write_bytes(f.read(entry["size"]))
                    n += 1

        walk(header, "")
    return n


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        asar = find_asar(Path(tmp))
        if OUT.exists():
            shutil.rmtree(OUT)
        n = extract(asar, PREFIX, OUT)
    if not (OUT / "index.html").exists():
        sys.exit("the draw.io archive had no web editor in it")
    print(f"draw.io web editor: {n} files -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
