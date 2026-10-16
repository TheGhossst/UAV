"""Ledger for run_10mhz_td3_all.sh. Same commands as the cap-sweep helper.

Output root is results/run_10mhz_td3. The torch device is not part of the
settings fingerprint, so a CPU checkpoint can resume on CUDA.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "capsweepjobs",
    ROOT / "scripts" / "remote" / "run_10mhz_cap_sweep_jobs.py",
)
sweep = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(sweep)
sweep.OUT_NAME = "results/run_10mhz_td3"


def main(argv: list[str] | None = None) -> int:
    return int(sweep.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
