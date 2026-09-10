"""Backward-compatible entry point — see scripts/experiments/residual_on_sca/."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "experiments/residual_on_sca/run_residual_td3.py"
raise SystemExit(runpy.run_path(str(_TARGET), run_name="__main__"))
