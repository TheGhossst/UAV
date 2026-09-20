"""Forwarder — see scripts/orchestration/run_full_regeneration_no_td3.py."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "orchestration/run_full_regeneration_no_td3.py"
raise SystemExit(runpy.run_path(str(_TARGET), run_name="__main__"))
