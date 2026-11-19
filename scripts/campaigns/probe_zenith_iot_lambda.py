"""Time one zenith-subset solve and compare two task rates at I=30."""

from __future__ import annotations

import time
from dataclasses import replace

from uavdt.config import headline_sim_config
from uavdt.experiments.grids import config_for_iot_sweep
from uavdt.experiments.methods import run_method
from uavdt.scenario import generate_scenario
from uavdt.sca_anchor import AnchorSettings

ANCHOR = AnchorSettings(max_enumerate=10000, top_k=3, beam_width=10)


def once(num_iot: int, lam: float, seed: int) -> None:
    base = replace(
        headline_sim_config(area_m=100.0, max_bw_share=None),
        dynamic_cap_alpha=1.0,
    )
    cfg = replace(config_for_iot_sweep(num_iot, base), lambda_i_per_s=lam)
    scenario = generate_scenario(seed, cfg)
    t0 = time.perf_counter()
    run = run_method(scenario, "sca_anchor", seed, anchor_settings=ANCHOR)
    dt = time.perf_counter() - t0
    print(
        f"I={num_iot} lambda={lam:g} seed={seed} "
        f"{run.sum_rate_mbps:.4f} Mbps feas={run.feasible} {dt:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    for seed in (2, 3):
        once(30, 1.0, seed)
        once(30, 3.5, seed)
    once(15, 1.0, 1)
    once(15, 3.5, 1)
