"""Sequential SCA-style knobs. Not Table II / paper physics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SCASettings:
    """Algorithm knobs for the sequential SCA-style solver.

    None of these are paper Table II values. The paper does not publish
    the complete convexified program.
    """

    max_iterations: int = 30
    epsilon: float = 1e-4
    improvement_tolerance: float = 1.0  # bit/s; larger than typical CVXPY noise
    fd_step_m: float = 1e-3
    # L-inf cap on each position candidate. IMPLEMENTATION CHOICE.
    step_size_m: float = 20.0
    min_step_size_m: float = 1e-3
    step_size_shrink: float = 0.5
    # None: CVXPY HiGHS/CLARABEL (unit tests). "matlab"/"MOSEK": one MATLAB
    # CVX+MOSEK session for the sequential convex LPs.
    solver: str | None = None
    verbose: bool = False
    # If False (default), a_ij and b_ij stay at initialization. If True,
    # after each convex (q, B) step the solver may reassign association
    # and processing. Accepted only when evaluate()/true_gate_ok() pass
    # and the true sum rate improves (or the point becomes newly feasible).
    # MATLAB CVX is frozen-only; dynamic_assignment is CVXPY.
    dynamic_assignment: bool = False
    # Kept only so older call sites that passed trust_region_m=0 still
    # mean "do not move UAVs". Not a joint Taylor trust region.
    trust_region_m: float | None = None
    # SCA-joint only. Frozen SCA ignores this. When True, the discrete
    # rematch also tries process-cohesive a_ij (N_k on one UAV) as a
    # candidate beside best-SE. Default False keeps the recorded
    # best-SE-only SCA-joint probe.
    process_cohesive_candidate: bool = False
