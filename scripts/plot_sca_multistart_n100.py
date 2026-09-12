"""Plot n100 multi-start eval JSON. Does not re-run solvers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.experiments.n100_plot import plot_n100_figures  # noqa: E402
from uavdt.experiments.scenario_bank import load_bank  # noqa: E402

JOBS = (
    (
        ROOT / "results" / "n100" / "eval_multistart.json",
        ROOT / "data" / "scenario_bank" / "n100_i10_j3_100m.json",
        ROOT / "results" / "figures" / "n100_multistart",
    ),
    (
        ROOT / "results" / "n100_500m_cap25" / "eval_multistart.json",
        ROOT / "data" / "scenario_bank" / "n100_i10_j3_500m.json",
        ROOT / "results" / "figures" / "n100_500m_multistart",
    ),
)


def main() -> int:
    for out, bank_path, fig in JOBS:
        payload = json.loads(out.read_text(encoding="utf-8"))
        bank = load_bank(bank_path)
        paths = plot_n100_figures(payload, bank, fig)
        print(f"{out.name}: {len(paths)} figures in {fig}")
        for p in paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
