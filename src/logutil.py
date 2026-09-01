"""Process-wide progress logging for the CLI and long solvers."""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

LOGGER_NAME = "uav"
log = logging.getLogger(LOGGER_NAME)

_TD3_LOG_EVERY = 500


class _UtcLocalFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return time.strftime("%H:%M:%S", time.localtime(record.created))


def configure_logging(
    level: str = "INFO",
    *,
    quiet: bool = False,
    log_file: str | None = None,
    td3_log_every: int = 500,
) -> None:
    """Attach a stderr handler once. Safe to call from tests."""
    global _TD3_LOG_EVERY
    _TD3_LOG_EVERY = max(1, int(td3_log_every))
    root = logging.getLogger(LOGGER_NAME)
    if quiet:
        root.setLevel(logging.WARNING)
    else:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_UtcLocalFormatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    root.addHandler(handler)
    root.propagate = False
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(_UtcLocalFormatter("%(asctime)s  %(levelname)-7s  %(message)s"))
        root.addHandler(fh)


def td3_log_every() -> int:
    return _TD3_LOG_EVERY


def mbps(rate_bps: float) -> str:
    return f"{rate_bps / 1e6:6.3f} Mbps"


def duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0 or total <= done:
        return "-"
    return duration(elapsed / done * (total - done))


def result_bits(result: Any, runtime: float | None = None) -> str:
    feas = "ok" if result.feasible else "INFEAS"
    bits = f"{mbps(result.sum_rate)}  {feas:6}  qos={result.qos_violations}"
    if getattr(result, "compute_available", False):
        bits += f"  aodt_viol={result.aodt_violations}"
    if runtime is not None:
        bits += f"  {runtime:.1f}s"
    return bits


def banner(title: str) -> None:
    log.info("--- %s", title)


def log_run(mode: str, cfg, args) -> None:
    profile = getattr(args, "radio_profile", None) or "calibrated"
    radio = f"{profile}  Bsys={cfg.b_sys:g} Hz"
    compute = "on" if cfg.use_compute_model else "off"
    seeds = "paper-20" if getattr(args, "paper_runs", False) else "dev-5"
    methods = ["random", "kmeans", "pso", "sca"]
    if getattr(args, "with_td3", False) or mode in ("td3", "aodt-compare"):
        methods = methods + ["td3"]
    banner(f"mode={mode}")
    log.info("  radio     %s  noise=%g W  compute=%s", radio, cfg.noise_power, compute)
    log.info(
        "  scene     I=%d  J=%d  lambda=%g  Tk=%g s  fj=%g  area=%.0fx%.0f",
        cfg.num_iot,
        cfg.num_uav,
        cfg.lambda_i,
        cfg.aodt_threshold,
        cfg.uav_cpu,
        cfg.area_x,
        cfg.area_y,
    )
    log.info(
        "  run       seeds=%s  methods=%s  pso=%dx%d  td3_steps=%d  device=%s",
        seeds if mode in ("compare", "sweeps", "aodt-compare") else str(getattr(args, "seed", "")),
        ",".join(methods) if mode in ("compare", "sweeps", "aodt-compare") else mode,
        getattr(args, "particles", 0),
        getattr(args, "iters", 0),
        getattr(args, "td3_steps", 0),
        getattr(args, "device", "auto"),
    )
    if cfg.use_compute_model:
        log.info(
            "  compute   S_i=%s bytes  L=%s cycles  (experimental, not Table II)",
            cfg.task_size_bytes,
            cfg.task_cycles,
        )


class Counter:
    """1-based job counter with ETA for nested experiment loops."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(int(total), 1)
        self.i = 0
        self.t0 = time.perf_counter()
        log.info("%s  %d job%s", label, self.total, "" if self.total == 1 else "s")

    def tick(self, what: str, result: Any | None = None, runtime: float | None = None) -> None:
        self.i += 1
        elapsed = time.perf_counter() - self.t0
        extra = result_bits(result, runtime) if result is not None else ""
        log.info(
            "  [%3d/%d]  %s  %s  eta %s",
            self.i,
            self.total,
            what,
            extra,
            eta(elapsed, self.i, self.total),
        )


def log_td3_step(
    tag: str,
    step: int,
    total: int,
    t0: float,
    reward: float,
    sum_rate: float,
    *,
    force: bool = False,
) -> None:
    interval = td3_log_every()
    last = step + 1 >= total
    first = step == 0
    if not force and not first and not last and (step + 1) % interval != 0:
        return
    elapsed = time.perf_counter() - t0
    done = step + 1
    log.info(
        "  td3 %-8s  [%5d/%d]  %3.0f%%  reward=%7.3f  %s  elapsed %s  eta %s",
        tag,
        done,
        total,
        100.0 * done / max(total, 1),
        reward,
        mbps(sum_rate),
        duration(elapsed),
        eta(elapsed, done, total),
    )
