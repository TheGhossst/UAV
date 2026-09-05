"""Simulation parameters.

Paper values, the 100×100 m modification, bandwidth experiments, and
external (not-in-paper) quantities are kept distinct. Do not treat
`task_size_bits` or `task_cycles` as Table II.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


# --- Paper scenario (IEEE TNSM 2026, §VII / Table II) ---
PAPER_AREA_X_M = 500.0
PAPER_AREA_Y_M = 500.0
NUM_IOT = 10
NUM_UAV = 3
NUM_PROCESSES = 2
IOTS_PER_PROCESS = 5
UAV_HEIGHT_M = 100.0
UAV_MIN_SEPARATION_M = 10.0
R_MIN_BIT_PER_S = 10_000.0
PAPER_B_SYS_HZ = 20_000.0
T_U2U_S = 0.3
F_C_HZ = 1.0e6
LAMBDA_I_PER_S = 2.0
UAV_CPU_CYCLES_PER_S = 2.0e8
C_LIGHT_M_PER_S = 3.0e8
AODT_THRESHOLD_S = 2.8
ETA_LOS = 1.0
ETA_NLOS = 21.0
SIGMA = 0.01  # Table II; Eq. (6) uses sigma**2
P_I_W = 0.2
ENV_A = 9.61
ENV_B = 0.16

# --- Intentional modification ---
AREA_X_M = 100.0
AREA_Y_M = 100.0

# Bandwidth experiments (Hz). Table II lists 20_000 Hz. That paper value
# is infeasible here (QoS/AoDT floors need ~102 kHz) and cannot produce
# the paper's 7–14 Mbps plots (Eq. (6) bounds 20 kHz at ~0.13 Mbps).
# Headline results: 2.4 MHz and 8.8 MHz, each with and without a 25%
# per-link cap. 20 kHz is kept only as a diagnostic of Table II.
BANDWIDTH_PRESETS: dict[str, float] = {
    "20khz": 20_000.0,
    "2.4mhz": 2_400_000.0,
    "8.8mhz": 8_800_000.0,
}

# --- External (not in Table II). Settled experimental choices. ---
# Paper labels S_i in bytes (Eq. 11); store bits so D = S/r is in seconds.
# S_i = 12_000 bytes, L = 3.75e6 cycles/task (μ = f_j/L ≈ 53.3 /s).
EXTERNAL_TASK_SIZE_BYTES = 12_000.0
EXTERNAL_TASK_SIZE_BITS = EXTERNAL_TASK_SIZE_BYTES * 8.0  # 96_000 bit
EXTERNAL_TASK_CYCLES = 3.75e6

PAPER_N_RUNS = 20

LosAngleUnit = Literal["rad", "deg"]


@dataclass(frozen=True)
class SimConfig:
    """All SI: metres, hertz, watts, bit/s, cycles/s, seconds."""

    area_x_m: float = AREA_X_M
    area_y_m: float = AREA_Y_M
    num_iot: int = NUM_IOT
    num_uav: int = NUM_UAV
    num_processes: int = NUM_PROCESSES
    iots_per_process: int = IOTS_PER_PROCESS
    uav_height_m: float = UAV_HEIGHT_M
    uav_min_separation_m: float = UAV_MIN_SEPARATION_M
    r_min_bit_per_s: float = R_MIN_BIT_PER_S
    b_sys_hz: float = PAPER_B_SYS_HZ
    t_u2u_s: float = T_U2U_S
    f_c_hz: float = F_C_HZ
    lambda_i_per_s: float = LAMBDA_I_PER_S
    uav_cpu_cycles_per_s: float = UAV_CPU_CYCLES_PER_S
    c_light_m_per_s: float = C_LIGHT_M_PER_S
    aodt_threshold_s: float = AODT_THRESHOLD_S
    eta_los: float = ETA_LOS
    eta_nlos: float = ETA_NLOS
    sigma: float = SIGMA
    p_i_w: float = P_I_W
    env_a: float = ENV_A
    env_b: float = ENV_B
    # Eq. (4) writes arcsin without a degree conversion. Default is radians.
    los_angle_unit: LosAngleUnit = "rad"
    # External — not paper Table II.
    task_size_bits: float = EXTERNAL_TASK_SIZE_BITS
    task_cycles: float = EXTERNAL_TASK_CYCLES
    # Per-link cap as a fraction of B_sys. None = only (26)–(27) as written.
    # 0.25 is an EXTERNAL PARAMETER (experimental restriction), not in
    # Problem (P) and not in Table II.
    max_bw_share: float | None = None
    # Eq. (12) UAV→BS download Z_l. Paper neglects this (processed payload
    # is small); default 0 keeps Problem (P) / Eq. (17) unchanged.
    download_time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.num_iot != self.num_processes * self.iots_per_process:
            raise ValueError(
                "num_iot must equal num_processes * iots_per_process "
                f"({self.num_processes}*{self.iots_per_process}!={self.num_iot})"
            )
        if self.b_sys_hz <= 0.0:
            raise ValueError("b_sys_hz must be positive")
        if self.task_size_bits <= 0.0:
            raise ValueError("task_size_bits is external and must be positive")
        if self.task_cycles <= 0.0:
            raise ValueError("task_cycles is external and must be positive")
        if self.los_angle_unit not in ("rad", "deg"):
            raise ValueError("los_angle_unit must be 'rad' or 'deg'")
        if self.max_bw_share is not None and not (0.0 < self.max_bw_share <= 1.0):
            raise ValueError("max_bw_share must be in (0, 1] or None")
        if self.download_time_s < 0.0:
            raise ValueError("download_time_s must be >= 0")

    @property
    def noise_power_w(self) -> float:
        """Gaussian white-noise power in Eq. (6): sigma**2."""
        return self.sigma ** 2

    @property
    def service_rate_per_s(self) -> float:
        """mu_j = f_j / L. L is external."""
        return self.uav_cpu_cycles_per_s / self.task_cycles

    @property
    def link_bandwidth_cap_hz(self) -> float:
        """Per-link B_ij cap. Paper (26)–(27): full B_sys.

        Optional max_bw_share is an EXTERNAL PARAMETER, not Problem (P).
        """
        if self.max_bw_share is None:
            return self.b_sys_hz
        return float(self.max_bw_share) * self.b_sys_hz

    def with_bandwidth_hz(self, b_sys_hz: float) -> "SimConfig":
        return replace(self, b_sys_hz=float(b_sys_hz))


DEFAULT = SimConfig()
