"""SCA solver for Problem (P). Physics stay in the core evaluator."""

from uavdt.sca.algorithm import SCAResult, classify_stop_reason, solve_sca
from uavdt.sca.reporting import gap_vs_uncapped
from uavdt.sca.settings import SCASettings

__all__ = [
    "SCAResult",
    "SCASettings",
    "classify_stop_reason",
    "gap_vs_uncapped",
    "solve_sca",
]
