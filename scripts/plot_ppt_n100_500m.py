"""PPT figures for n100 500 m bank (eval_anchor, no TD3 required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from plot_ppt_all_methods_bank import main as bank_main

    return bank_main(
        [
            "--anchor",
            str(ROOT / "results" / "n100_500m_cap25" / "eval_anchor.json"),
            "--td3",
            str(ROOT / "results" / "n100_500m_cap25" / "eval_anchor.json"),
            "--field-m",
            "500",
            "--n-layouts",
            "100",
            "--stem",
            "n100_500m_mean_sum_rate",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
