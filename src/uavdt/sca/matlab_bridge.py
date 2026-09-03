"""MATLAB CVX + MOSEK bridge for sequential SCA convex subproblems.

Physics stay in the Python core. MATLAB only solves the convex LPs.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from uavdt.aodt import queueing_term_s
from uavdt.computation import service_rate_per_s
from uavdt.models import Scenario
from uavdt.sca.cvx_problem import BandwidthSolveResult, aodt_upload_slacks_s, bandwidth_floors_hz
from uavdt.sca.linearize import se_jacobian, spectral_efficiency
from uavdt.sca.settings import SCASettings

MATLAB_EXE = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")
MATLAB_DIR = Path(__file__).resolve().parents[3] / "matlab"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def matlab_available() -> bool:
    return (
        MATLAB_EXE.is_file()
        and (MATLAB_DIR / "run_sca_seq.m").is_file()
        and (MATLAB_DIR / "run_convex_step.m").is_file()
    )


def _vec(mat: np.ndarray) -> list[float]:
    return np.asarray(mat, dtype=float).ravel(order="F").tolist()


def _mat(vec, i: int, j: int) -> np.ndarray:
    return np.asarray(vec, dtype=float).reshape((i, j), order="F")


def _run_matlab_entry(entry: str, payload: dict) -> dict:
    if not matlab_available():
        raise RuntimeError("MATLAB/CVX/MOSEK bridge is not available")
    MATLAB_DIR.mkdir(parents=True, exist_ok=True)
    results = Path(__file__).resolve().parents[3] / "results"
    results.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uavdt_cvx_") as tmp:
        in_path = Path(tmp) / "in.json"
        out_path = Path(tmp) / "out.json"
        in_path.write_text(json.dumps(payload, cls=_NumpyEncoder), encoding="utf-8")
        (results / "matlab_last_in.json").write_text(
            in_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        cmd = [
            str(MATLAB_EXE),
            "-batch",
            f"addpath('{MATLAB_DIR.as_posix()}'); "
            f"{entry}('{in_path.as_posix()}', '{out_path.as_posix()}')",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=600,
                cwd=str(MATLAB_DIR),
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            raise RuntimeError(
                "MATLAB CVX step timed out after 600s:\n"
                f"stdout:\n{out[-4000:]}\n"
                f"stderr:\n{err[-4000:]}"
            ) from exc
        if proc.stdout:
            (results / "matlab_last_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        if proc.stderr:
            (results / "matlab_last_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(
                "MATLAB CVX step failed:\n"
                f"stdout:\n{(proc.stdout or '')[-4000:]}\n"
                f"stderr:\n{(proc.stderr or '')[-4000:]}"
            )
        text = out_path.read_text(encoding="utf-8")
        (results / "matlab_last_out.json").write_text(text, encoding="utf-8")
        return json.loads(text)


def _run_matlab(payload: dict) -> dict:
    return _run_matlab_entry("run_convex_step", payload)


def solve_bandwidth_matlab(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    settings: SCASettings | None = None,
) -> BandwidthSolveResult:
    settings = settings or SCASettings()
    se = spectral_efficiency(scenario.iot_xyz_m, uav_xyz_m, scenario.cfg)
    floors = bandwidth_floors_hz(scenario, association, processing, se)
    if np.any(~np.isfinite(floors[association > 0.5])):
        return BandwidthSolveResult(
            status="aodt_slack_nonpositive",
            solver_name="MATLAB-CVX-MOSEK",
            bandwidth_hz=np.zeros_like(se),
            se=se,
            objective=float("nan"),
            solve_time_s=0.0,
            infeasible=True,
            message="AoDT slack non-positive or SE=0",
        )
    a = (association > 0.5).astype(float)
    payload = {
        "task": "bandwidth",
        "I": int(a.shape[0]),
        "J": int(a.shape[1]),
        "A": _vec(a),
        "SE": _vec(se),
        "floors": _vec(floors),
        "B_sys": float(scenario.cfg.b_sys_hz),
        "B_cap": float(scenario.cfg.link_bandwidth_cap_hz),
    }
    sol = _run_matlab(payload)
    infeas = bool(sol.get("infeasible"))
    status = str(sol.get("status", "unknown"))
    if infeas:
        return BandwidthSolveResult(
            status=status,
            solver_name="MATLAB-CVX-MOSEK",
            bandwidth_hz=np.zeros_like(se),
            se=se,
            objective=float("nan"),
            solve_time_s=0.0,
            infeasible=True,
            message=status,
        )
    bw = _mat(sol["B"], a.shape[0], a.shape[1])
    bw = np.maximum(bw, 0.0)
    bw[a < 0.5] = 0.0
    if bw.sum() > scenario.cfg.b_sys_hz + 1e-6:
        bw *= scenario.cfg.b_sys_hz / bw.sum()
    obj = sol.get("objective")
    return BandwidthSolveResult(
        status=status,
        solver_name="MATLAB-CVX-MOSEK",
        bandwidth_hz=bw,
        se=se,
        objective=float(obj) if obj is not None else float(np.sum(a * se * bw)),
        solve_time_s=0.0,
        infeasible=False,
        message=status,
    )


def solve_position_matlab(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    bandwidth_hz: np.ndarray,
    association: np.ndarray,
    step_m: float,
    settings: SCASettings | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Return candidate xy (J,2), status, infeasible."""
    settings = settings or SCASettings()
    cfg = scenario.cfg
    se, gx, gy = se_jacobian(
        scenario.iot_xyz_m, uav_xyz_m, cfg, step_m=settings.fd_step_m
    )
    a = (association > 0.5).astype(float)
    payload = {
        "task": "position",
        "I": int(a.shape[0]),
        "J": int(a.shape[1]),
        "A": _vec(a),
        "SE": _vec(se),
        "gx": _vec(gx),
        "gy": _vec(gy),
        "B0": _vec(bandwidth_hz),
        "x0": np.asarray(uav_xyz_m[:, 0], dtype=float).tolist(),
        "y0": np.asarray(uav_xyz_m[:, 1], dtype=float).tolist(),
        "step_m": float(step_m),
        "area_x": float(cfg.area_x_m),
        "area_y": float(cfg.area_y_m),
        "theta": float(cfg.uav_min_separation_m),
    }
    sol = _run_matlab(payload)
    status = str(sol.get("status", "unknown"))
    if bool(sol.get("infeasible")):
        return uav_xyz_m[:, :2].copy(), status, True
    xy = np.column_stack(
        [
            np.clip(np.asarray(sol["x"], dtype=float).ravel(), 0.0, cfg.area_x_m),
            np.clip(np.asarray(sol["y"], dtype=float).ravel(), 0.0, cfg.area_y_m),
        ]
    )
    return xy, status, False


def sequential_matlab_payload(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    settings: SCASettings,
) -> dict:
    """Pack frozen physics + algorithm knobs for matlab/sca_seq.m."""
    cfg = scenario.cfg
    a = (association > 0.5).astype(float)
    b = (processing > 0.5).astype(float)
    mu = float(service_rate_per_s(cfg))
    slacks = aodt_upload_slacks_s(scenario, a, b)
    q_terms = [
        float(queueing_term_s(scenario.lambdas_per_s[proc.iot_indices], mu))
        for proc in scenario.processes
    ]
    se = spectral_efficiency(scenario.iot_xyz_m, uav_xyz_m, cfg)
    step = float(settings.step_size_m)
    if settings.trust_region_m is not None and settings.trust_region_m <= 0.0:
        step = 0.0
    return {
        "I": int(a.shape[0]),
        "J": int(a.shape[1]),
        "iot_xyz": _vec(scenario.iot_xyz_m),
        "uav_xyz": _vec(uav_xyz_m),
        "A": _vec(a),
        "processing": _vec(b),
        "se_python": _vec(se),
        "slacks": np.asarray(slacks, dtype=float).tolist(),
        "Q": q_terms,
        "members": [
            [int(i) for i in proc.iot_indices.tolist()] for proc in scenario.processes
        ],
        "lambdas": np.asarray(scenario.lambdas_per_s, dtype=float).tolist(),
        "params": {
            "area_x": float(cfg.area_x_m),
            "area_y": float(cfg.area_y_m),
            "height": float(cfg.uav_height_m),
            "theta": float(cfg.uav_min_separation_m),
            "B_sys": float(cfg.b_sys_hz),
            "B_cap": float(cfg.link_bandwidth_cap_hz),
            "R_min": float(cfg.r_min_bit_per_s),
            "S": float(cfg.task_size_bits),
            "T_k": float(cfg.aodt_threshold_s),
            "t_u2u": float(cfg.t_u2u_s),
            "mu": mu,
            "f_c": float(cfg.f_c_hz),
            "c_light": float(cfg.c_light_m_per_s),
            "sigma": float(cfg.sigma),
            "p_i": float(cfg.p_i_w),
            "eta_los": float(cfg.eta_los),
            "eta_nlos": float(cfg.eta_nlos),
            "env_a": float(cfg.env_a),
            "env_b": float(cfg.env_b),
        },
        "step_m": step,
        "min_step_m": float(settings.min_step_size_m),
        "step_shrink": float(settings.step_size_shrink),
        "max_iterations": int(settings.max_iterations),
        "epsilon": float(settings.epsilon),
        "improvement_tolerance": float(settings.improvement_tolerance),
        "fd_step_m": float(settings.fd_step_m),
    }


def run_sequential_matlab(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    settings: SCASettings | None = None,
) -> dict:
    """One MATLAB session: sequential SCA LPs. Returns the JSON struct."""
    settings = settings or SCASettings()
    payload = sequential_matlab_payload(
        scenario, uav_xyz_m, association, processing, settings
    )
    return _run_matlab_entry("run_sca_seq", payload)
