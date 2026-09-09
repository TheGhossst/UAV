"""TD3 solver for Problem (P). Algorithm knobs are TD3Settings, not SimConfig."""

from uavdt.td3.settings import TD3Settings

__all__ = ["TD3Settings", "solve_td3"]


def __getattr__(name: str):
    if name == "solve_td3":
        from uavdt.td3.solve import solve_td3

        return solve_td3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
