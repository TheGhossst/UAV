"""Paper Table II parameters and algorithm knobs.

Values marked experimental are NOT in the paper and must not be reported as
Table II constants. TASK_SIZE_BYTES and TASK_CYCLES default to None; solvers
that need AoDT/CPU use EXPERIMENTAL_* only when explicitly enabled.
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
B_SYS = 20_000.0  # Hz
F_C = 1e6  # Hz
C_LIGHT = 3e8  # m/s

ETA_LOS = 1.0
ETA_NLOS = 21.0

P_I = 0.2  # W

SIGMA = 0.01  # Table II label
NOISE_POWER = SIGMA ** 2  # Eq. (6) uses sigma^2

ENV_A = 9.61
ENV_B = 0.16
LOS_ANGLE_UNIT: Literal["rad", "deg"] = "deg"

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
TD3_R_MAX = 2e5  # bit/s; match observed sum-rate scale so reward is not dominated by penalties only
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


DEFAULT = SimConfig()
DEFAULT_COMPUTE = DEFAULT.with_compute()
