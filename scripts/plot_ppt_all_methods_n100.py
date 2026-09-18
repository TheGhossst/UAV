"""Backward-compatible wrapper: n100 100 m PPT figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from plot_ppt_all_methods_bank import main as bank_main

    return bank_main(
        [
            "--anchor",
            str(ROOT / "results" / "n100" / "eval_anchor.json"),
            "--td3",
            str(ROOT / "results" / "n100" / "eval_td3.json"),
            "--field-m",
            "100",
            "--n-layouts",
            "100",
            "--stem",
            "n100_mean_sum_rate",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
