"""UAV-aided digital twin IoT simulator (core model)."""

from uavdt.config import SimConfig, BANDWIDTH_PRESETS
from uavdt.evaluator import EvalResult, evaluate
from uavdt.models import Allocation, PhysicalProcess, Scenario
from uavdt.scenario import generate_scenario

__all__ = [
    "Allocation",
    "BANDWIDTH_PRESETS",
    "EvalResult",
    "PhysicalProcess",
    "Scenario",
    "SimConfig",
    "evaluate",
    "generate_scenario",
]
