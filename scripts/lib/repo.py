"""Shared repo paths for scripts living in subfolders."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
LIB = SCRIPTS / "lib"
SRC = ROOT / "src"


def bootstrap(extra: Path | None = None) -> None:
    for p in (SRC, LIB, extra):
        if p is None:
            continue
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
