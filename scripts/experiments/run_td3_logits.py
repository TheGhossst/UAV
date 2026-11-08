"""Second TD3 row: discount 0.99, critic sees the state, association and processing logits are used.

The +/- 10 m residual box is unchanged, so this row isolates those three settings.
Official score is the solver's true_eval (noise-free policy export, then the bandwidth LP).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from uavdt.config import headline_sim_config
from uavdt.scenario import generate_scenario
from uavdt.td3.settings import TD3Settings
from uavdt.td3.solve import solve_td3

OUT = Path("results/n100/td3_logits_g099.json")
N = 20


def main() -> None:
    done: dict = {}
    if OUT.exists():
        done = json.loads(OUT.read_text(encoding="utf-8"))
    cfg = headline_sim_config()
    settings = replace(
        TD3Settings(),
        discount=0.99,
        critic_use_obs=True,
        critic_move_only=False,
        assoc_mode="logits",
        process_mode="logits",
        log_every=0,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(1, N + 1):
        key = str(seed)
        if key in done:
            print(f"skip {seed}", flush=True)
            continue
        scenario = generate_scenario(seed, cfg)
        result = solve_td3(scenario, seed, settings=settings)
        diag = result.diagnostics
        done[key] = {
            "sum_rate_Mbps": float(result.true_eval.sum_rate_mbps),
            "feasible": bool(result.true_eval.feasible),
            "wall_clock_s": diag.get("wall_clock_s"),
            "policy_export_sum_rate_Mbps": diag.get("policy_export_sum_rate_Mbps"),
            "snapshot_export_sum_rate_Mbps": diag.get("snapshot_export_sum_rate_Mbps"),
            "origin_sum_rate_Mbps": diag.get("origin_sum_rate_Mbps"),
            "best_inner_sum_rate_Mbps": diag.get("best_inner_sum_rate_Mbps"),
        }
        OUT.write_text(json.dumps(done, indent=2), encoding="utf-8")
        row = done[key]
        print(
            f"seed {seed}  policy {row['sum_rate_Mbps']:.4f}  "
            f"snapshot {row['snapshot_export_sum_rate_Mbps']:.4f}  "
            f"origin {row['origin_sum_rate_Mbps']:.4f}  "
            f"{row['wall_clock_s']:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
