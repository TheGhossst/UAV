"""Paper Table II parameters and algorithm knobs.

Values marked experimental are NOT in the paper and must not be reported as
Table II constants. TASK_SIZE_BYTES and TASK_CYCLES default to None; solvers
that need AoDT/CPU use EXPERIMENTAL_* only when explicitly enabled.

Radio settings come from a named profile (see RADIO_PROFILES). "table2" is the
literal Table II reading; "calibrated" is the default and is what reproduces
the Mbps-scale figures. See docs/calibration.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional


# --- Scenario (paper setup) ---
AREA_X = 500.0
AREA_Y = 500.0

NUM_IOT = 10
NUM_UAV = 3
NUM_PROCESSES = 2
IOTS_PER_PROCESS = 5

# --- UAV ---
UAV_HEIGHT = 100.0
UAV_MIN_DISTANCE = 10.0

# --- Communication (Table II) ---
R_MIN = 10_000.0  # bit/s
F_C = 1e6  # Hz
C_LIGHT = 3e8  # m/s

ETA_LOS = 1.0
ETA_NLOS = 21.0

P_I = 0.2  # W

SIGMA = 0.01  # Table II label: "noise power sigma = 10 x 10^-3 W"

ENV_A = 9.61
ENV_B = 0.16
LOS_ANGLE_UNIT: Literal["rad", "deg"] = "deg"

# --- Radio profiles ---
# Table II read literally (B_sys = 20 kHz, sigma^2 = 1e-4) puts the Shannon
# bound of this channel at ~0.13 Mbps, so Eq. (6) cannot reach the Mbps-scale
# sum rates plotted in Figs. 6-10 no matter how well the solvers do, and every
# link sits so far into the high-SNR regime that moving a UAV barely changes
# log2(1 + SNR) -- which is why all methods bunch together under that reading.
#
# The "calibrated" profile changes three things and nothing else:
#   * noise_power = sigma, i.e. Table II's "noise power 10 x 10^-3 W" is used
#     directly as the sigma^2 of Eq. (6) instead of being squared. This is the
#     literal reading of the table label and it puts the links in the low-SNR
#     regime, where rate is roughly proportional to d^-2 and placement matters.
#   * b_sys is fitted (one free scale knob) so the J sweep spans the published
#     3-9 Mbps range. Sum rate is exactly linear in b_sys, so this only sets the
#     y-axis scale; it cannot change any ranking.
#   * max_bw_share caps how much of a pool one link may take, see
#     repair.link_bandwidth_cap.
# Constraint (27) stays system-wide as written in the paper; bandwidth_scope =
# "per_uav" is available for the alternative reading where each UAV owns a band.
#
# Neither profile is bit-exact with the paper; "table2" is kept so the literal
# reading stays runnable and reportable.


@dataclass(frozen=True)
class RadioProfile:
    name: str
    b_sys: float  # Hz, per bandwidth pool (see bandwidth_scope)
    noise_power: float  # W, the sigma^2 of Eq. (6)
    bandwidth_scope: Literal["system", "per_uav"]
    max_bw_share: Optional[float]  # per-link cap as a fraction of its pool


RADIO_PROFILES: dict[str, RadioProfile] = {
    "table2": RadioProfile(
        name="table2",
        b_sys=20_000.0,
        noise_power=SIGMA**2,
        bandwidth_scope="system",
        max_bw_share=None,
    ),
    "calibrated": RadioProfile(
        name="calibrated",
        b_sys=8.8e6,
        noise_power=SIGMA,
        bandwidth_scope="system",
        max_bw_share=0.25,
    ),
}

RADIO_PROFILE = "calibrated"
_PROFILE = RADIO_PROFILES[RADIO_PROFILE]

B_SYS = _PROFILE.b_sys
NOISE_POWER = _PROFILE.noise_power
BANDWIDTH_SCOPE = _PROFILE.bandwidth_scope
MAX_BW_SHARE = _PROFILE.max_bw_share

# --- Computing (Table II) ---
LAMBDA_I = 2.0  # tasks/s
UAV_CPU = 2e8  # cycles/s

# --- AoDT (Table II) ---
AODT_THRESHOLD = 2.8  # s
T_U2U = 0.3  # s

# Paper-unspecified; keep None unless compute/AoDT is enabled.
TASK_SIZE_BYTES: Optional[float] = None
TASK_CYCLES: Optional[float] = None

# Documented experimental defaults (NOT Table II). Used only with use_compute_model.
# Chosen so AoDT can bind at T_k = 2.8 s given Table II rates (kbps-scale):
# S_i = 2000 bytes => 16000 bits; D = 1.6 s at R_min so AoDT ≈ 2.1 s can be feasible,
# while weak links (r << R_min) still violate T_k = 2.8 s.
EXPERIMENTAL_TASK_SIZE_BYTES = 2000.0
# L = 2e6 cycles => mu = 100 requests/s at f_j = 2e8 (stable for 10 IoTs at λ=2).
EXPERIMENTAL_TASK_CYCLES = 2e6

# --- Seeds ---
DEV_SCENARIO_SEEDS = (100, 101, 102, 103, 104)
PAPER_N_RUNS = 20
PAPER_SCENARIO_SEEDS = tuple(range(100, 100 + PAPER_N_RUNS))

# --- PSO (algorithm choices, not Table II) ---
PSO_N_PARTICLES = 20
PSO_N_ITER = 100
PSO_W = 0.7
PSO_C1 = 1.5
PSO_C2 = 1.5
PSO_V_MAX_FRAC = 0.2
PSO_PENALTY = 1e7

# --- SCA ---
SCA_MAX_ITER = 30
SCA_TRUST = 25.0  # m per iteration
SCA_TOL = 1e-4
SCA_STEP = 1.0

# --- TD3 (not in Table II; Alg. 2 structure is paper) ---
TD3_GAMMA = 0.99
TD3_TAU = 0.005
TD3_POLICY_DELAY = 2
TD3_ACTOR_LR = 1e-3
TD3_CRITIC_LR = 1e-3
TD3_BATCH_SIZE = 128
TD3_BUFFER_SIZE = 50_000
TD3_NOISE = 0.1
TD3_TARGET_NOISE = 0.2
TD3_NOISE_CLIP = 0.5
TD3_HIDDEN = 256
TD3_WARMUP = 500
TD3_TOTAL_STEPS = 7000
TD3_POS_SCALE = 10.0  # Alg. 2: Δx, Δy × 10
# bit/s. Not in the paper. Sized so sum_rate/R_MAX is the same order as the
# TD3_W_* penalty weights; at 1e7 the rate term is ~0.5 against penalties of 5
# to 50 and the critic cannot resolve it.
TD3_R_MAX = 1e6
TD3_EPISODE_LEN = 50  # steps before the env is re-initialised during training
# Association/processing logits are offsets on top of -d/area_x, so a zero
# action means "nearest UAV". Learning a 3*I*J logit block from scratch inside
# TD3_TOTAL_STEPS does not work: the policy collapses onto associations that
# violate R_min and never recovers.
TD3_ASSOC_ACTION_SCALE = 0.25
TD3_W_AODT = 10.0
TD3_W_DIST = 5.0
TD3_W_VIOL = 5.0

FROZEN_UAV_XY = (250.0, 250.0)


@dataclass(frozen=True)
class SimConfig:
    area_x: float = AREA_X
    area_y: float = AREA_Y
    num_iot: int = NUM_IOT
    num_uav: int = NUM_UAV
    num_processes: int = NUM_PROCESSES
    iots_per_process: int = IOTS_PER_PROCESS
    uav_height: float = UAV_HEIGHT
    uav_min_distance: float = UAV_MIN_DISTANCE
    r_min: float = R_MIN
    b_sys: float = B_SYS
    bandwidth_scope: Literal["system", "per_uav"] = BANDWIDTH_SCOPE
    max_bw_share: Optional[float] = MAX_BW_SHARE
    f_c: float = F_C
    c_light: float = C_LIGHT
    eta_los: float = ETA_LOS
    eta_nlos: float = ETA_NLOS
    p_i: float = P_I
    sigma: float = SIGMA
    noise_power: float = NOISE_POWER
    env_a: float = ENV_A
    env_b: float = ENV_B
    los_angle_unit: Literal["rad", "deg"] = LOS_ANGLE_UNIT
    lambda_i: float = LAMBDA_I
    uav_cpu: float = UAV_CPU
    aodt_threshold: float = AODT_THRESHOLD
    t_u2u: float = T_U2U
    task_size_bytes: Optional[float] = TASK_SIZE_BYTES
    task_cycles: Optional[float] = TASK_CYCLES
    use_compute_model: bool = False

    def with_compute(self) -> "SimConfig":
        """Enable AoDT/CPU using documented experimental S_i and L."""
        return replace(
            self,
            use_compute_model=True,
            task_size_bytes=EXPERIMENTAL_TASK_SIZE_BYTES
            if self.task_size_bytes is None
            else self.task_size_bytes,
            task_cycles=EXPERIMENTAL_TASK_CYCLES
            if self.task_cycles is None
            else self.task_cycles,
        )

    def with_radio_profile(self, name: str) -> "SimConfig":
        """Swap the radio profile (see RADIO_PROFILES)."""
        try:
            p = RADIO_PROFILES[name]
        except KeyError:
            raise ValueError(f"unknown radio profile {name!r}; have {sorted(RADIO_PROFILES)}") from None
        return replace(
            self,
            b_sys=p.b_sys,
            noise_power=p.noise_power,
            bandwidth_scope=p.bandwidth_scope,
            max_bw_share=p.max_bw_share,
        )


DEFAULT = SimConfig()
DEFAULT_COMPUTE = DEFAULT.with_compute()
TABLE_II = DEFAULT.with_radio_profile("table2")
