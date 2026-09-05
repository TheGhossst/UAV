"""Paper §VII sweep axes. Grids are DERIVED from figure captions.

Exact tick lists are IMPLEMENTATION CHOICE where the PDF does not print
them. Headline B_sys is this reproduction's 8.8 MHz (paper 20 kHz as the
(27) cap is infeasible; model-free ceiling 0.997 Mbps at SNR_max=1e15).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from uavdt.config import SimConfig

# PAPER: Fig. 6 I=10, vary J; text discusses five UAVs.
UAV_COUNTS = (1, 2, 3, 4, 5)

# PAPER: Fig. 7 J=3, I up to 32. Intermediate ticks not printed.
IOT_COUNTS = (10, 16, 20, 24, 28, 32)

# PAPER: Fig. 8 λ from 1 to 3.5 tasks/s, I=10, J=3.
LAMBDA_PER_S = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)

# PAPER: Fig. 9 T_k from 0.8 s, relaxed to 3 s, I=10, J=3.
AODT_THRESHOLD_S = (0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0)

# PAPER: Fig. 10 UAV CPU, text quotes 250 MHz at the high end.
# Table II main value is 2e8 cycles/s. Treated as Hz = cycles/s.
CPU_CYCLES_PER_S = (0.5e8, 1.0e8, 1.5e8, 2.0e8, 2.5e8)

AXES = ("uavs", "iots", "lambda", "aodt", "cpu")


@dataclass(frozen=True)
class SweepPoint:
    axis: str
    x_name: str
    x_value: float
    cfg: SimConfig
    label: str


def config_for_counts(
    num_iot: int,
    num_uav: int,
    cfg: SimConfig | None = None,
    num_processes: int = 2,
) -> SimConfig:
    """K=2 processes, even IoT split. IMPLEMENTATION CHOICE for I ≠ 10."""
    base = cfg or SimConfig()
    if num_iot < num_processes:
        raise ValueError("num_iot must be >= num_processes")
    if num_iot % num_processes != 0:
        raise ValueError(
            "this campaign keeps equal |N_k|; num_iot must be divisible by "
            f"{num_processes}"
        )
    return replace(
        base,
        num_iot=int(num_iot),
        num_uav=int(num_uav),
        num_processes=int(num_processes),
        iots_per_process=int(num_iot // num_processes),
    )


def iter_axis(axis: str, cfg: SimConfig) -> list[SweepPoint]:
    name = axis.lower().strip()
    if name == "uavs":
        return [
            SweepPoint(
                "uavs",
                "num_uav",
                float(j),
                config_for_counts(10, j, cfg),
                f"PAPER Fig. 6  I=10  J={j}",
            )
            for j in UAV_COUNTS
        ]
    if name == "iots":
        return [
            SweepPoint(
                "iots",
                "num_iot",
                float(i),
                config_for_counts(i, 3, cfg),
                f"PAPER Fig. 7  I={i}  J=3",
            )
            for i in IOT_COUNTS
        ]
    if name == "lambda":
        return [
            SweepPoint(
                "lambda",
                "lambda_i_per_s",
                float(lam),
                replace(config_for_counts(10, 3, cfg), lambda_i_per_s=float(lam)),
                f"PAPER Fig. 8  lambda={lam:g}/s  I=10  J=3",
            )
            for lam in LAMBDA_PER_S
        ]
    if name == "aodt":
        return [
            SweepPoint(
                "aodt",
                "aodt_threshold_s",
                float(tk),
                replace(config_for_counts(10, 3, cfg), aodt_threshold_s=float(tk)),
                f"PAPER Fig. 9  Tk={tk:g}s  I=10  J=3",
            )
            for tk in AODT_THRESHOLD_S
        ]
    if name == "cpu":
        return [
            SweepPoint(
                "cpu",
                "uav_cpu_cycles_per_s",
                float(fj),
                replace(
                    config_for_counts(10, 3, cfg),
                    uav_cpu_cycles_per_s=float(fj),
                ),
                f"PAPER Fig. 10  fj={fj:g}  I=10  J=3",
            )
            for fj in CPU_CYCLES_PER_S
        ]
    raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
