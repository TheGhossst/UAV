"""Shared paths for docs/RESULTS.md §2.9 residual-on-SCA experiments."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
ROOT = _PKG.parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "results" / "residual_on_sca"
PROTECTED = ROOT / "results" / "campaign_8.8mhz_cap25_si12k.json"

OUT_MULTISTART = OUT_DIR / "multistart_n20.json"
OUT_RESIDUAL_TD3 = OUT_DIR / "residual_td3_heldout_21_40.json"
OUT_CMAES_POLISH = OUT_DIR / "cmaes_polish_heldout_21_40.json"


def bootstrap() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
