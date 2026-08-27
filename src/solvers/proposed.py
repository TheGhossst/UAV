"""Placeholder for a later proposed method. Not selected yet."""

from __future__ import annotations

from src.evaluator import EvalResult
from src.scenario import Scenario


def solve_proposed(scenario: Scenario, seed: int = 0):
    raise NotImplementedError(
        "Proposed method is intentionally empty until Random/K-means/PSO/SCA/TD3 "
        "share one evaluator and metrics exist."
    )
