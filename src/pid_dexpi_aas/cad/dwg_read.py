"""A .dwg file as an ezdxf document, through LibreDWG's dwg2dxf (open source, GPL).

LibreDWG is built into tools/libredwg by scripts/get_libredwg.sh, or found on the PATH
($LIBREDWG_DWG2DXF points elsewhere). The file is converted in a temporary folder that is
deleted straight after (uploads are not kept on disk). A .dxf is read directly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def find_dwg2dxf() -> Path | None:
    for p in (os.environ.get("LIBREDWG_DWG2DXF"), ROOT / "tools" / "libredwg" / "bin" / "dwg2dxf",
              shutil.which("dwg2dxf")):
        if p and Path(p).exists():
            return Path(p)
    return None


def read_cad(data: bytes, name: str):
    """An ezdxf document from the bytes of a .dwg or .dxf file."""
    import ezdxf

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / ("drawing" + Path(name).suffix.lower())
        src.write_bytes(data)
        if src.suffix == ".dxf":
            return ezdxf.readfile(src)
        tool = find_dwg2dxf()
        if tool is None:
            raise RuntimeError("Reading .dwg files needs LibreDWG: run scripts/get_libredwg.sh (README).")
        out = Path(tmp) / "drawing.dxf"
        env = dict(os.environ)
        lib = tool.parent.parent / "lib"
        if lib.exists():
            env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
        r = subprocess.run([str(tool), "-y", "-o", str(out), str(src)], capture_output=True, text=True,
                           env=env, timeout=300)
        if not out.exists():
            raise RuntimeError(f"LibreDWG could not convert the file: {(r.stderr or r.stdout)[-500:]}")
        try:
            return ezdxf.readfile(out)
        except ezdxf.DXFStructureError:
            from ezdxf import recover
            doc, _ = recover.readfile(out)
            return doc
