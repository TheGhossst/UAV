"""SCA with the convexified LP solved by MATLAB CVX + MOSEK.

Same Algorithm 1 loop and same linearized (P) as ``src.solvers.sca``, but each
joint (q, B) LP is handed to CVX with ``cvx_solver mosek`` — the stack the
IEEE TNSM 2026 paper states in §V. SciPy HiGHS remains the default
``--mode sca`` backend.

Requires a local MATLAB install with CVX on the path (run ``cvx_setup`` once)
and a MOSEK license CVX can see. Override the executable with env ``UAV_MATLAB``.

    python -m src.main --mode sca-cvx --seed 100
    python -m src.main --mode sca-cvx --compute --seed 100
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from src.evaluator import EvalResult
from src.logutil import log
from src.scenario import Scenario
from src.solvers.sca import (
    _assemble_convexified_lp,
    _decode_convexified_solution,
    solve_sca,
)

MATLAB_DIR = Path(__file__).resolve().parents[2] / "matlab"
_READY_TIMEOUT_S = 180.0
_SOLVE_TIMEOUT_S = 120.0


def find_matlab() -> Path:
    """Return the MATLAB executable. Raises FileNotFoundError if missing."""
    for key in ("UAV_MATLAB", "MATLAB"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw)
            if p.is_file():
                return p
            candidate = p / "bin" / "matlab.exe" if os.name == "nt" else p / "bin" / "matlab"
            if candidate.is_file():
                return candidate
    which = shutil.which("matlab")
    if which:
        return Path(which)
    roots = [Path(r"C:\Program Files\MATLAB"), Path(r"C:\Program Files (x86)\MATLAB")]
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(root.glob("R*/bin/matlab.exe"))
        found.extend(root.glob("R*/bin/matlab"))
    if found:
        return sorted(found)[-1]
    raise FileNotFoundError(
        "MATLAB not found. Install MATLAB, or set UAV_MATLAB to matlab.exe."
    )


def matlab_is_available() -> bool:
    try:
        find_matlab()
        return True
    except FileNotFoundError:
        return False


def _bounds_to_vectors(
    bounds: list[tuple[float | None, float | None]],
) -> tuple[np.ndarray, np.ndarray]:
    lb = np.array([-np.inf if lo is None else float(lo) for lo, _ in bounds], dtype=float)
    ub = np.array([np.inf if hi is None else float(hi) for _, hi in bounds], dtype=float)
    return lb, ub


class MatlabCvxSession:
    """One MATLAB process reused for every SCA iteration."""

    def __init__(self) -> None:
        self._eng = None
        self._proc = None
        self._workdir: Path | None = None
        self.backend = "unknown"
        self._start()

    def _start(self) -> None:
        matlab_dir = str(MATLAB_DIR)
        try:
            import matlab.engine  # type: ignore[import-untyped]

            log.info("sca-cvx: starting MATLAB Engine …")
            eng = matlab.engine.start_matlab()
            eng.addpath(matlab_dir, nargout=0)
            self._eng = eng
            self.backend = "matlab.engine"
            probe = self._solve_engine(
                np.array([1.0]),
                np.array([[-1.0]]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([1e9]),
            )
            if probe is None:
                raise RuntimeError("CVX+MOSEK probe via MATLAB Engine failed")
            log.info("sca-cvx: MATLAB Engine ready (CVX+MOSEK)")
            return
        except Exception as exc:
            if self._eng is not None:
                try:
                    self._eng.quit()
                except Exception:
                    pass
                self._eng = None
            log.info("sca-cvx: MATLAB Engine not used (%s); falling back to a live MATLAB process", exc)

        exe = find_matlab()
        workdir = Path(tempfile.mkdtemp(prefix="uav_sca_cvx_"))
        self._workdir = workdir
        workdir_posix = str(workdir).replace("\\", "/")
        env = os.environ.copy()
        env["UAV_CVX_WORKDIR"] = workdir_posix
        r_cmd = (
            "try, convexified_lp_server(getenv('UAV_CVX_WORKDIR')); "
            "catch e, fid=fopen(fullfile(getenv('UAV_CVX_WORKDIR'),'error.txt'),'w'); "
            r"fprintf(fid,'%s',getReport(e,'extended','hyperlinks','off')); "
            "fclose(fid); end; exit;"
        )
        cmd = [str(exe), "-nodesktop", "-nosplash"]
        if os.name == "nt":
            cmd.append("-wait")
        cmd.extend(["-r", r_cmd])
        log.info("sca-cvx: launching %s (CVX+MOSEK server)", exe)
        import subprocess

        log_path = workdir / "matlab.log"
        with log_path.open("w", encoding="utf-8") as log_f:
            self._proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(MATLAB_DIR),
                env=env,
            )
        self.backend = "matlab-process"
        self._wait_ready()

    def _wait_ready(self) -> None:
        assert self._workdir is not None
        ready = self._workdir / "ready.flag"
        err = self._workdir / "error.txt"
        t0 = time.time()
        while time.time() - t0 < _READY_TIMEOUT_S:
            if err.is_file() and err.stat().st_size > 0:
                raise RuntimeError(f"MATLAB CVX/MOSEK setup failed:\n{err.read_text(encoding='utf-8', errors='replace')}")
            if ready.is_file():
                log.info("sca-cvx: MATLAB CVX+MOSEK server ready")
                return
            if self._proc is not None and self._proc.poll() is not None:
                extra = err.read_text(encoding="utf-8", errors="replace") if err.is_file() else ""
                log_txt = ""
                log_file = self._workdir / "matlab.log"
                if log_file.is_file():
                    log_txt = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(
                    f"MATLAB exited before becoming ready (code {self._proc.returncode}). {extra}\n{log_txt}"
                )
            time.sleep(0.2)
        raise TimeoutError(
            f"MATLAB did not become ready in {_READY_TIMEOUT_S:.0f}s. "
            "Confirm CVX is on the path (cvx_setup) and MOSEK is licensed."
        )

    def solve_lp(self, c: np.ndarray, a_ub: np.ndarray, b_ub: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray | None:
        if self._eng is not None:
            return self._solve_engine(c, a_ub, b_ub, lb, ub)
        return self._solve_files(c, a_ub, b_ub, lb, ub)

    def _solve_engine(
        self,
        c: np.ndarray,
        a_ub: np.ndarray,
        b_ub: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray | None:
        import matlab  # type: ignore[import-untyped]

        eng = self._eng
        c_m = matlab.double(np.asarray(c, dtype=float).reshape(-1, 1).tolist())
        if a_ub.size == 0:
            a_m = matlab.double(np.zeros((0, c.size)).tolist())
        else:
            a_m = matlab.double(np.asarray(a_ub, dtype=float).tolist())
        b_m = matlab.double(np.asarray(b_ub, dtype=float).reshape(-1, 1).tolist())
        lb_m = matlab.double(np.asarray(lb, dtype=float).reshape(-1, 1).tolist())
        ub_m = matlab.double(np.asarray(ub, dtype=float).reshape(-1, 1).tolist())
        x, ok, st = eng.convexified_lp(c_m, a_m, b_m, lb_m, ub_m, nargout=3)
        if int(ok) != 1:
            log.debug("sca-cvx: CVX status %s", st)
            return None
        return np.asarray(x, dtype=float).ravel()

    def _solve_files(
        self,
        c: np.ndarray,
        a_ub: np.ndarray,
        b_ub: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray | None:
        assert self._workdir is not None
        in_mat = self._workdir / "in.mat"
        out_mat = self._workdir / "out.mat"
        req = self._workdir / "request.flag"
        resp = self._workdir / "response.flag"
        if resp.exists():
            resp.unlink()
        if out_mat.exists():
            out_mat.unlink()
        n = int(c.size)
        a = np.zeros((0, n)) if a_ub.size == 0 else np.asarray(a_ub, dtype=float)
        savemat(
            in_mat,
            {
                "c": np.asarray(c, dtype=float).reshape(-1, 1),
                "A_ub": a,
                "b_ub": np.asarray(b_ub, dtype=float).reshape(-1, 1),
                "lb": np.asarray(lb, dtype=float).reshape(-1, 1),
                "ub": np.asarray(ub, dtype=float).reshape(-1, 1),
            },
            oned_as="column",
        )
        req.write_text("1", encoding="ascii")
        t0 = time.time()
        while time.time() - t0 < _SOLVE_TIMEOUT_S:
            if resp.is_file() and out_mat.is_file():
                # MATLAB fclose may still be flushing; retry a couple of loads.
                for _ in range(20):
                    try:
                        payload = loadmat(out_mat, squeeze_me=True)
                        break
                    except Exception:
                        time.sleep(0.05)
                else:
                    raise RuntimeError("MATLAB wrote out.mat but Python could not read it")
                ok = int(np.ravel(payload.get("ok", 0))[0])
                if ok != 1:
                    st = payload.get("st", "failed")
                    log.debug("sca-cvx: CVX status %s", st)
                    return None
                return np.asarray(payload["x"], dtype=float).ravel()
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"MATLAB process died (code {self._proc.returncode})")
            time.sleep(0.05)
        raise TimeoutError(f"CVX+MOSEK LP timed out after {_SOLVE_TIMEOUT_S:.0f}s")

    def close(self) -> None:
        if self._eng is not None:
            try:
                self._eng.quit()
            except Exception:
                pass
            self._eng = None
        if self._workdir is not None:
            stop = self._workdir / "stop.flag"
            try:
                stop.write_text("1", encoding="ascii")
            except OSError:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def __enter__(self) -> MatlabCvxSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _convexified_lp_cvx(
    scenario: Scenario,
    xy: np.ndarray,
    bw: np.ndarray,
    association: np.ndarray,
    processing: np.ndarray,
    trust: float,
    session: MatlabCvxSession,
) -> tuple[np.ndarray, np.ndarray] | None:
    lp = _assemble_convexified_lp(scenario, xy, bw, association, processing, trust)
    if lp is None:
        return None
    lb, ub = _bounds_to_vectors(lp.bounds)
    x = session.solve_lp(lp.c, lp.a_ub, lp.b_ub, lb, ub)
    if x is None:
        return None
    return _decode_convexified_solution(x, lp, bw)


def solve_sca_cvx(
    scenario: Scenario,
    seed: int = 0,
    n_uav: int | None = None,
    max_iter: int | None = None,
    trust: float | None = None,
) -> tuple[np.ndarray, EvalResult, float]:
    """Algorithm 1 with the convexified LP solved by CVX + MOSEK."""
    from src.config import SCA_MAX_ITER, SCA_TRUST

    kwargs: dict = {}
    if max_iter is not None:
        kwargs["max_iter"] = max_iter
    else:
        kwargs["max_iter"] = SCA_MAX_ITER
    if trust is not None:
        kwargs["trust"] = trust
    else:
        kwargs["trust"] = SCA_TRUST

    with MatlabCvxSession() as session:
        def lp_solver(sc, q, b, a, proc, tr):
            return _convexified_lp_cvx(sc, q, b, a, proc, tr, session)

        return solve_sca(
            scenario,
            seed=seed,
            n_uav=n_uav,
            lp_solver=lp_solver,
            backend=f"cvx-mosek/{session.backend}",
            **kwargs,
        )
