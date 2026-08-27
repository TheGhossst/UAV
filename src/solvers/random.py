"""Random UAV placement baseline."""

from __future__ import annotations

import time

import numpy as np

from src.evaluator import EvalResult, evaluate
from src.repair import complete_solution
from src.scenario import Scenario


def solve_random(scenario: Scenario, seed: int = 0, n_uav: int | None = None) -> tuple[np.ndarray, EvalResult, float]:
    cfg = scenario.cfg
    j = n_uav if n_uav is not None else cfg.num_uav
    rng = np.random.default_rng(seed)
    uav_xy = np.column_stack(
        [rng.uniform(0.0, cfg.area_x, j), rng.uniform(0.0, cfg.area_y, j)]
    )
    t0 = time.perf_counter()
    xy, a, b, bw = complete_solution(scenario, uav_xy)
    result = evaluate(scenario, xy, a, b, bw)
    return xy, result, time.perf_counter() - t0
