"""Multi-seed summaries. Paper §VII averages each plotted point over 20 runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.config import SimConfig
from uavdt.evaluator import EvalResult, evaluate
from uavdt.experiments.campaign import CampaignSettings, run_campaign, write_campaign
from uavdt.experiments.methods import METHODS, run_method
from uavdt.experiments.spot import spot_validate_sca
from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.random import place_random
from uavdt.scenario import generate_scenario


@dataclass
class SeedMetrics:
    seeds: tuple[int, ...]
    sum_rate_bit_per_s: np.ndarray
    aodt_s: np.ndarray  # (n_seeds, K)
    feasible: np.ndarray

    @property
    def mean_sum_rate_bit_per_s(self) -> float:
        return float(np.mean(self.sum_rate_bit_per_s))

    @property
    def std_sum_rate_bit_per_s(self) -> float:
        return float(np.std(self.sum_rate_bit_per_s, ddof=1)) if self.sum_rate_bit_per_s.size > 1 else 0.0

    @property
    def mean_aodt_s(self) -> np.ndarray:
        return np.mean(self.aodt_s, axis=0)

    @property
    def std_aodt_s(self) -> np.ndarray:
        if self.aodt_s.shape[0] < 2:
            return np.zeros(self.aodt_s.shape[1])
        return np.std(self.aodt_s, axis=0, ddof=1)


def run_one(
    seed: int,
    cfg: SimConfig,
    placement: str,
) -> tuple[np.ndarray, EvalResult]:
    scenario = generate_scenario(seed, cfg)
    if placement == "random":
        uav = place_random(cfg.num_uav, seed, cfg)
    elif placement == "kmeans":
        uav = place_kmeans(scenario, seed)
    else:
        raise ValueError("placement must be 'random' or 'kmeans'")
    return uav, evaluate(scenario, uav)


def run_seeds(
    seeds: tuple[int, ...] | list[int],
    cfg: SimConfig,
    placement: str = "random",
) -> SeedMetrics:
    rates = []
    aodts = []
    feas = []
    for seed in seeds:
        _, result = run_one(int(seed), cfg, placement)
        rates.append(result.sum_rate_bit_per_s)
        aodts.append(result.aodt_s)
        feas.append(result.feasible)
    return SeedMetrics(
        seeds=tuple(int(s) for s in seeds),
        sum_rate_bit_per_s=np.asarray(rates, dtype=float),
        aodt_s=np.vstack(aodts),
        feasible=np.asarray(feas, dtype=bool),
    )
