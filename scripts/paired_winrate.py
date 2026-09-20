"""CLI forwarder — implementation in scripts/lib/paired_winrate.py."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "lib/paired_winrate.py"
raise SystemExit(runpy.run_path(str(_TARGET), run_name="__main__"))
