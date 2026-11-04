"""Ledger for run_10mhz_missing_all.sh.

Queues the 10 MHz results that are not already under the headline trees.
Finished JSON is skipped. Placement jobs resume from ``*.checkpoint.json``.
Analysis scripts write one JSON at the end, so a kill restarts that one job.

Outputs stay under ``results/run_10mhz_missing/``. This file refuses paths in
``run_10mhz_cap25``, ``run_10mhz_cap_sweep``, ``run_10mhz_td3``, and
``run_10mhz_controls``.

The AoDT axis starts at 1.2 s. T_k = 0.8 s is not queued.

Default groups, in run order:

  mechanism      side/H, proportional noise, proportional polish (20 seeds)
  fig11          Eq. (17) patterns, 100 m and 500 m, cap 25%, 20 seeds
  frozen-sca     frozen-assignment SCA bank, cap 25%, both fields
  pso            n100 at 10/18/20/22%, both fields (equal-share fitness)
  controls       medoid, frozen random-anchor, continuous at no-cap and 10%,
                 both fields, plus 100 m cap 25% random-anchor and continuous
  anchor-dyn     dynamic zenith n100 at no-cap and 10/12/15/18/20/22%, both
                 fields, plus the 500 m cap 25% campaign
  residual-td3   residual-on-sca n100, cap 25%, both fields
  td3-n100       Algorithm 2 n100 at every cap except 25%, both fields

``td3-campaign`` is off unless ``--groups all`` or ``--groups td3-campaign``.
It is the same caps as ``td3-n100``, campaign axes, both fields.

Not queued: PSO with an LP fitness (the code has no such switch), CMA-ES,
association search, the learned proposal policy, and any T_k = 0.8 s repair.
Cap 25% dynamic zenith n100, the 100 m dynamic zenith campaign, cap 25% TD3,
and the controls that already exist are not repeated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_NAME = "results/run_10mhz_missing"
B_SYS_HZ = 10_000_000.0
PROTECTED = (
    "results/run_10mhz_cap25/",
    "results/run_10mhz_cap_sweep/",
    "results/run_10mhz_td3/",
    "results/run_10mhz_controls/",
)
GROUP_ORDER = (
    "mechanism",
    "fig11",
    "frozen-sca",
    "pso",
    "controls",
    "anchor-dyn",
    "residual-td3",
    "td3-n100",
    "td3-campaign",
)
DEFAULT_GROUPS = GROUP_ORDER[:-1]
GPU_GROUPS = frozenset({"td3-n100", "td3-campaign", "residual-td3"})
AREAS = (100, 500)
# Caps other than the finished 25% headline cell.
OTHER_CAPS: tuple[tuple[str, float | None], ...] = (
    ("nocap", None),
    ("cap10", 0.10),
    ("cap12", 0.12),
    ("cap15", 0.15),
    ("cap18", 0.18),
    ("cap20", 0.20),
    ("cap22", 0.22),
)
PSO_CAPS = (("cap10", 0.10), ("cap18", 0.18), ("cap20", 0.20), ("cap22", 0.22))
ANCHOR_FLAGS = (
    "--anchor-max-enumerate",
    "10000",
    "--anchor-top-k",
    "3",
    "--anchor-beam-width",
    "10",
)


def _load(alias: str, filename: str):
    path = ROOT / "scripts" / "remote" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cap25 = _load("cap25jobs_missing", "run_10mhz_cap25_jobs.py")
_sweep = _load("capsweepjobs_missing", "run_10mhz_cap_sweep_jobs.py")


@dataclass(frozen=True)
class Job:
    id: str
    group: str
    out: str
    check: str
    argv: tuple[str, ...]
    settings: dict
    needs_device: bool = False
    method: str = ""
    area: int = 0
    share: float | None = None
    n: int = 0
    extras: tuple[str, ...] = ()

    def with_runtime(self, *, device: str, force: bool) -> tuple[str, ...]:
        cmd = list(self.argv)
        if self.needs_device:
            if not device:
                raise SystemExit(f"{self.id} needs --device")
            cmd += ["--device", device]
        if force and self.check.startswith("n100"):
            cmd.append("--no-resume")
        return tuple(cmd)


def _posix(path: Path) -> str:
    return path.as_posix()


def _suffix(rel: str, suffix: str) -> str:
    return _posix(Path(rel).with_suffix(suffix))


def _guard(rel: str) -> str:
    text = rel.replace("\\", "/")
    if not text.startswith(OUT_NAME + "/"):
        raise SystemExit(f"refusing output outside {OUT_NAME}: {text}")
    for prefix in PROTECTED:
        if text.startswith(prefix):
            raise SystemExit(f"refusing protected output {text}")
    return text


def _settings(base: dict) -> dict:
    if "device" in base:
        raise SystemExit("device must not be part of the settings fingerprint")
    return base


def _placement(
    *,
    group: str,
    job_id: str,
    kind: str,
    area: int,
    method: str,
    share: float | None,
    out: str,
    n: int,
    check: str,
    extra: tuple[str, ...] = (),
    needs_device: bool = False,
    td3_preset: str | None = None,
    settings_extra: dict | None = None,
) -> Job:
    out = _guard(out)
    common = [
        "--bandwidth-preset",
        "10mhz",
        "--area-m",
        str(int(area)),
        "--methods",
        method,
        "--seed-start",
        "1",
        "--solver",
        "cvxpy",
        "--max-iterations",
        "30",
        "--epsilon",
        "1e-4",
        "--step-size",
        "20",
    ]
    if share is None:
        common.append("--no-per-link-cap")
    else:
        common += ["--max-bw-share", f"{float(share):.2f}"]
    ckpt = _suffix(out, ".checkpoint.json")
    if kind == "n100":
        argv = [
            "-m",
            "uavdt",
            "n100",
            *common,
            "--n-scenarios",
            str(int(n)),
            "--num-iot",
            "10",
            "--num-uav",
            "3",
            "--bank",
            f"data/scenario_bank/n100_i10_j3_{int(area)}m.json",
            "--out",
            out,
            "--checkpoint",
            ckpt,
            "--fig-dir",
            _posix(Path(out).with_suffix("")),
            "--skip-plot",
        ]
        extras = (ckpt, _suffix(out, ".csv"))
    elif kind == "campaign":
        argv = [
            "-m",
            "uavdt",
            "campaign",
            "--axis",
            "all",
            *common,
            "--n-runs",
            str(int(n)),
            "--out",
            out,
        ]
        extras = (ckpt, _suffix(out, ".csv"))
    else:
        raise SystemExit(f"unknown kind {kind}")
    argv += list(extra)
    settings = {
        "group": group,
        "check": check,
        "bandwidth_preset": "10mhz",
        "max_bw_share": share,
        "area_m": int(area),
        "method": method,
        "kind": kind,
        "n": int(n),
        "seed_start": 1,
        "solver": "cvxpy",
        "max_iterations": 30,
        "epsilon": 1e-4,
        "step_size_m": 20.0,
    }
    if td3_preset is not None:
        settings["td3_preset"] = td3_preset
    if settings_extra:
        settings.update(settings_extra)
    return Job(
        id=job_id,
        group=group,
        out=out,
        check=check,
        argv=tuple(argv),
        settings=_settings(settings),
        needs_device=needs_device,
        method=method,
        area=int(area),
        share=share,
        n=int(n),
        extras=extras,
    )


def _script_job(
    *,
    group: str,
    job_id: str,
    out: str,
    check: str,
    argv: tuple[str, ...],
    n: int,
    extras: tuple[str, ...] = (),
    settings_extra: dict | None = None,
    area: int = 0,
) -> Job:
    out = _guard(out)
    settings = {"group": group, "check": check, "n": int(n), "out": out}
    if area:
        settings["area_m"] = int(area)
    if settings_extra:
        settings.update(settings_extra)
    return Job(
        id=job_id,
        group=group,
        out=out,
        check=check,
        argv=argv,
        settings=_settings(settings),
        area=int(area),
        n=int(n),
        extras=extras,
    )


def parse_groups(text: str) -> tuple[str, ...]:
    raw = [part.strip() for part in str(text).split(",") if part.strip()]
    if not raw:
        raise SystemExit("no groups selected")
    if "all" in raw:
        return GROUP_ORDER
    unknown = [name for name in raw if name not in GROUP_ORDER]
    if unknown:
        names = ", ".join(GROUP_ORDER)
        raise SystemExit(f"unknown group {unknown}; expected {names}, or all")
    return tuple(name for name in GROUP_ORDER if name in set(raw))


def build_jobs(
    *,
    n_runs: int,
    analysis_n: int,
    groups: tuple[str, ...],
    td3_preset: str,
) -> list[Job]:
    if int(n_runs) < 1 or int(analysis_n) < 1:
        raise SystemExit("n_runs and analysis_n must be positive")
    n_runs = int(n_runs)
    analysis_n = int(analysis_n)
    preset = td3_preset or "alg2"
    jobs: list[Job] = []

    side = f"{OUT_NAME}/mechanism/side_over_h.json"
    side_png = f"{OUT_NAME}/mechanism/side_over_h.png"
    prop = f"{OUT_NAME}/mechanism/bw_proportional_noise.json"
    polish = f"{OUT_NAME}/mechanism/polish_proportional.json"
    jobs.append(
        _script_job(
            group="mechanism",
            job_id="mechanism/side-over-h",
            out=side,
            check="side",
            n=analysis_n,
            extras=(side_png,),
            argv=(
                "scripts/analyze/mechanism_side_over_h.py",
                "--n-runs",
                str(analysis_n),
                "--out",
                side,
                "--plot",
                side_png,
            ),
            settings_extra={"areas": "heights and sides"},
        )
    )
    jobs.append(
        _script_job(
            group="mechanism",
            job_id="mechanism/proportional-noise",
            out=prop,
            check="proportional",
            n=analysis_n,
            argv=(
                "scripts/analyze/compare_bw_proportional_noise.py",
                "--n-runs",
                str(analysis_n),
                "--areas",
                "100",
                "500",
                "--max-bw-share",
                "0.25",
                "--out",
                prop,
            ),
            settings_extra={"max_bw_share": 0.25, "areas": [100, 500]},
        )
    )
    jobs.append(
        _script_job(
            group="mechanism",
            job_id="mechanism/polish",
            out=polish,
            check="polish",
            n=analysis_n,
            argv=(
                "scripts/analyze/polish_proportional_radio.py",
                "--n-runs",
                str(analysis_n),
                "--areas",
                "500",
                "100",
                "--out",
                polish,
            ),
            settings_extra={"areas": [500, 100]},
        )
    )

    for area in AREAS:
        out = f"{OUT_NAME}/fig11/{area}m/fig11.json"
        jobs.append(
            _script_job(
                group="fig11",
                job_id=f"fig11/{area}m",
                out=out,
                check="fig11",
                n=analysis_n,
                extras=(_suffix(out, ".csv"),),
                argv=(
                    "-m",
                    "uavdt",
                    "fig11",
                    "--bandwidth-preset",
                    "10mhz",
                    "--max-bw-share",
                    "0.25",
                    "--area-m",
                    str(area),
                    "--n-runs",
                    str(analysis_n),
                    "--seed-start",
                    "1",
                    "--out",
                    out,
                ),
                area=area,
                settings_extra={
                    "bandwidth_preset": "10mhz",
                    "max_bw_share": 0.25,
                    "area_m": area,
                },
            )
        )

    for area in AREAS:
        out = f"{OUT_NAME}/frozen-sca/cap25/{area}m/n100_frozen_sca.json"
        jobs.append(
            _placement(
                group="frozen-sca",
                job_id=f"frozen-sca/cap25/{area}m/n100",
                kind="n100",
                area=area,
                method="frozen_sca",
                share=0.25,
                out=out,
                n=n_runs,
                check="n100-frozen",
            )
        )

    for slug, share in PSO_CAPS:
        for area in AREAS:
            out = f"{OUT_NAME}/pso/{slug}/{area}m/n100_pso.json"
            jobs.append(
                _placement(
                    group="pso",
                    job_id=f"pso/{slug}/{area}m/n100",
                    kind="n100",
                    area=area,
                    method="pso",
                    share=share,
                    out=out,
                    n=n_runs,
                    check="n100",
                )
            )

    control_caps = (("nocap", None), ("cap10", 0.10))
    for slug, share in control_caps:
        for area in AREAS:
            jobs.append(
                _placement(
                    group="controls",
                    job_id=f"controls/{slug}/{area}m/medoid",
                    kind="n100",
                    area=area,
                    method="sca_medoid",
                    share=share,
                    out=f"{OUT_NAME}/controls/{slug}/{area}m/n100_sca_medoid.json",
                    n=n_runs,
                    check="n100-frozen",
                    extra=ANCHOR_FLAGS,
                    settings_extra={"anchor_selection": "medoid"},
                )
            )
            jobs.append(
                _placement(
                    group="controls",
                    job_id=f"controls/{slug}/{area}m/random-anchor",
                    kind="n100",
                    area=area,
                    method="sca_anchor_frozen",
                    share=share,
                    out=f"{OUT_NAME}/controls/{slug}/{area}m/n100_sca_anchor_random.json",
                    n=n_runs,
                    check="n100-random",
                    extra=ANCHOR_FLAGS
                    + (
                        "--anchor-selection",
                        "random",
                        "--anchor-n-random",
                        "1",
                        "--anchor-top-k",
                        "1",
                    ),
                    settings_extra={
                        "anchor_selection": "random",
                        "anchor_n_random": 1,
                        "anchor_top_k": 1,
                    },
                )
            )
            jobs.append(
                _placement(
                    group="controls",
                    job_id=f"controls/{slug}/{area}m/continuous",
                    kind="n100",
                    area=area,
                    method="sca_continuous",
                    share=share,
                    out=f"{OUT_NAME}/controls/{slug}/{area}m/n100_sca_continuous.json",
                    n=n_runs,
                    check="n100-frozen",
                    extra=ANCHOR_FLAGS,
                    settings_extra={"anchor_selection": "continuous"},
                )
            )
    for method, stem, selection in (
        ("sca_anchor_frozen", "random-anchor", "random"),
        ("sca_continuous", "continuous", "continuous"),
    ):
        extra = ANCHOR_FLAGS
        more: dict = {"anchor_selection": selection}
        if selection == "random":
            extra = ANCHOR_FLAGS + (
                "--anchor-selection",
                "random",
                "--anchor-n-random",
                "1",
                "--anchor-top-k",
                "1",
            )
            more["anchor_n_random"] = 1
            more["anchor_top_k"] = 1
        out_name = (
            "n100_sca_anchor_random.json"
            if selection == "random"
            else "n100_sca_continuous.json"
        )
        jobs.append(
            _placement(
                group="controls",
                job_id=f"controls/cap25/100m/{stem}",
                kind="n100",
                area=100,
                method=method,
                share=0.25,
                out=f"{OUT_NAME}/controls/cap25/100m/{out_name}",
                n=n_runs,
                check="n100-random" if selection == "random" else "n100-frozen",
                extra=extra,
                settings_extra=more,
            )
        )

    for slug, share in OTHER_CAPS:
        for area in AREAS:
            jobs.append(
                _placement(
                    group="anchor-dyn",
                    job_id=f"anchor-dyn/{slug}/{area}m/n100",
                    kind="n100",
                    area=area,
                    method="sca_anchor",
                    share=share,
                    out=f"{OUT_NAME}/anchor-dyn/{slug}/{area}m/n100_sca_anchor.json",
                    n=n_runs,
                    check="n100-dynamic",
                    extra=ANCHOR_FLAGS,
                    settings_extra={
                        "anchor_max_enumerate": 10000,
                        "anchor_top_k": 3,
                        "anchor_beam_width": 10,
                    },
                )
            )
    jobs.append(
        _placement(
            group="anchor-dyn",
            job_id="anchor-dyn/cap25/500m/campaign",
            kind="campaign",
            area=500,
            method="sca_anchor",
            share=0.25,
            out=f"{OUT_NAME}/anchor-dyn/cap25/500m/campaign_sca_anchor.json",
            n=n_runs,
            check="campaign-dynamic",
            extra=ANCHOR_FLAGS,
            settings_extra={
                "anchor_max_enumerate": 10000,
                "anchor_top_k": 3,
                "anchor_beam_width": 10,
            },
        )
    )

    for area in AREAS:
        jobs.append(
            _placement(
                group="residual-td3",
                job_id=f"residual-td3/cap25/{area}m/n100",
                kind="n100",
                area=area,
                method="td3",
                share=0.25,
                out=f"{OUT_NAME}/residual-td3/cap25/{area}m/n100_td3.json",
                n=n_runs,
                check="n100",
                needs_device=True,
                td3_preset="residual-on-sca",
                extra=("--td3-preset", "residual-on-sca"),
            )
        )

    for slug, share in OTHER_CAPS:
        for area in AREAS:
            jobs.append(
                _placement(
                    group="td3-n100",
                    job_id=f"td3-n100/{slug}/{area}m",
                    kind="n100",
                    area=area,
                    method="td3",
                    share=share,
                    out=f"{OUT_NAME}/td3/{slug}/{area}m/n100_td3.json",
                    n=n_runs,
                    check="n100",
                    needs_device=True,
                    td3_preset=preset,
                    extra=("--td3-preset", preset),
                )
            )
            jobs.append(
                _placement(
                    group="td3-campaign",
                    job_id=f"td3-campaign/{slug}/{area}m",
                    kind="campaign",
                    area=area,
                    method="td3",
                    share=share,
                    out=f"{OUT_NAME}/td3/{slug}/{area}m/campaign_td3.json",
                    n=n_runs,
                    check="campaign",
                    needs_device=True,
                    td3_preset=preset,
                    extra=("--td3-preset", preset),
                )
            )

    selected = set(groups)
    return [job for job in jobs if job.group in selected]


def _find(jobs: list[Job], job_id: str) -> Job:
    for job in jobs:
        if job.id == job_id:
            return job
    raise SystemExit(f"unknown job {job_id}")


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _dynamic_ok(diags: object, want: bool) -> bool:
    if not isinstance(diags, list) or not diags:
        return False
    seen = 0
    for item in diags:
        if not isinstance(item, dict) or "dynamic_assignment" not in item:
            continue
        seen += 1
        if bool(item["dynamic_assignment"]) is not want:
            return False
    return seen > 0


def _n100_diags(payload: dict, method: str) -> list:
    found = []
    for rec in payload.get("runs") or []:
        if not isinstance(rec, dict) or rec.get("method") != method:
            continue
        diag = rec.get("diagnostics")
        if isinstance(diag, dict):
            found.append(diag)
    return found


def _campaign_diags(stats: object) -> list:
    if not isinstance(stats, dict):
        return []
    diags = stats.get("diagnostics")
    if not isinstance(diags, list):
        return []
    return [item for item in diags if isinstance(item, dict)]


def _placement_complete(job: Job, path: Path) -> bool:
    kind = "campaign" if job.check == "campaign-dynamic" or job.check == "campaign" else "n100"
    if not _sweep.output_complete(
        path,
        kind=kind,
        method=job.method,
        area=float(job.area),
        share=job.share,
        n_runs=job.n,
        n_scenarios=job.n,
    ):
        return False
    payload = _read_json(path)
    if payload is None:
        return False
    if job.check in {"n100", "campaign"}:
        return True
    want = job.check in {"n100-dynamic", "campaign-dynamic"}
    if kind == "n100":
        # n100 stores per-seed diagnostics on runs, not on by_method.
        diags = _n100_diags(payload, job.method)
        if not _dynamic_ok(diags, want):
            return False
        if job.check == "n100-random":
            selections = [d.get("selection") for d in diags if "selection" in d]
            if not selections or any(item != "random" for item in selections):
                return False
        return True
    for pt in payload.get("points") or []:
        if not isinstance(pt, dict):
            return False
        stats = (pt.get("by_method") or {}).get(job.method)
        if not _dynamic_ok(_campaign_diags(stats), want):
            return False
    return True


def _len_ok(rows: object, n: int) -> bool:
    return isinstance(rows, list) and len(rows) == int(n)


def job_complete(job: Job, root: Path | None = None) -> bool:
    path = (root or ROOT) / job.out
    payload = _read_json(path)
    if payload is None or payload.get("checkpoint") is True:
        return False
    if job.check.startswith("n100") or job.check.startswith("campaign"):
        return _placement_complete(job, path)
    if job.check == "side":
        if int(payload.get("n_runs") or -1) != job.n:
            return False
        points = payload.get("points") or []
        if not isinstance(points, list) or len(points) < 9:
            return False
        if not all(isinstance(pt, dict) and _len_ok(pt.get("seeds"), job.n) for pt in points):
            return False
        return ((root or ROOT) / job.extras[0]).is_file()
    if job.check == "proportional":
        areas = payload.get("areas") or []
        if not isinstance(areas, list):
            return False
        found = set()
        for block in areas:
            if not isinstance(block, dict) or not _len_ok(block.get("seeds"), job.n):
                return False
            if abs(float(block.get("b_sys_hz") or 0) - B_SYS_HZ) > 1.0:
                return False
            if abs(float(block.get("max_bw_share") or -1) - 0.25) > 1e-9:
                return False
            found.add(int(round(float(block.get("area_m") or -1))))
        return found == {100, 500}
    if job.check == "polish":
        areas = payload.get("areas") or []
        if not isinstance(areas, list):
            return False
        found = set()
        for block in areas:
            if not isinstance(block, dict) or not _len_ok(block.get("seeds"), job.n):
                return False
            found.add(int(round(float(block.get("area_m") or -1))))
        return found == {100, 500}
    if job.check == "fig11":
        if abs(float(payload.get("b_sys_hz") or 0) - B_SYS_HZ) > 1.0:
            return False
        if abs(float(payload.get("max_bw_share") or -1) - 0.25) > 1e-9:
            return False
        if int(payload.get("n_runs") or -1) != job.n:
            return False
        raw_area = payload.get("area_m")
        if not isinstance(raw_area, (list, tuple)) or len(raw_area) < 2:
            return False
        if abs(float(raw_area[0]) - job.area) > 1e-6:
            return False
        points = payload.get("points") or []
        if not isinstance(points, list) or len(points) != 5:
            return False
        for pt in points:
            if not isinstance(pt, dict):
                return False
            patterns = pt.get("by_pattern") or {}
            if not isinstance(patterns, dict):
                return False
            for name in ("uniform_fast", "uniform_slow", "heterogeneous"):
                stats = patterns.get(name) or {}
                if not _len_ok(stats.get("per_seed_eq17_max_s"), job.n):
                    return False
        return True
    raise SystemExit(f"unknown check {job.check}")


def discard_job(job: Job, root: Path | None = None) -> None:
    base = root or ROOT
    path = base / job.out
    if path.is_file():
        dest = path.with_name(path.name + ".prev")
        if dest.exists():
            dest.unlink()
        path.replace(dest)
    for rel in job.extras:
        extra = base / rel
        if extra.is_file():
            extra.unlink()
    settings = _suffix(job.out, ".settings.json")
    settings_path = base / settings
    if settings_path.is_file():
        settings_path.unlink()


def _selector(args: argparse.Namespace) -> list[Job]:
    return build_jobs(
        n_runs=int(args.n_runs),
        analysis_n=int(args.analysis_n),
        groups=parse_groups(args.groups),
        td3_preset=str(args.td3_preset),
    )


def _add_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--analysis-n", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--td3-preset", default="alg2")


def _invocation_namespace(path: Path, fallback: argparse.Namespace) -> argparse.Namespace:
    payload = _read_json(path)
    if payload is None:
        raise SystemExit(f"unreadable invocation {path}")
    ns = argparse.Namespace(
        groups=",".join(payload.get("groups") or []),
        n_runs=int(payload.get("n_runs") or fallback.n_runs),
        analysis_n=int(payload.get("analysis_n") or fallback.analysis_n),
        device=str(payload.get("device") or fallback.device),
        td3_preset=str(payload.get("td3_preset") or fallback.td3_preset),
    )
    return ns


def cmd_plan(args: argparse.Namespace) -> int:
    for job in _selector(args):
        print(f"{job.id}\t{job.group}\t{job.out}\t{job.check}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    jobs = _selector(args)
    print(f"{len(jobs)} jobs")
    for job in jobs:
        print(f"{job.group}\t{job.id}\t{job.out}")
    return 0


def cmd_argv(args: argparse.Namespace) -> int:
    job = _find(_selector(args), args.job)
    for arg in job.with_runtime(device=str(args.device), force=bool(args.force)):
        print(arg)
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    job = _find(_selector(args), args.job)
    return 0 if job_complete(job) else 1


def cmd_bind(args: argparse.Namespace) -> int:
    job = _find(_selector(args), args.job)
    path = _cap25.resolve(_suffix(job.out, ".settings.json"))
    return _cap25.bind_settings(path, job.settings, force=bool(args.force))


def cmd_discard(args: argparse.Namespace) -> int:
    job = _find(_selector(args), args.job)
    discard_job(job)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    _cap25.append_record(_cap25.resolve(args.ledger), args.job, args.status, args.field or [])
    return 0


def cmd_last(args: argparse.Namespace) -> int:
    print(_cap25.last_status(_cap25.resolve(args.ledger), args.job))
    return 0


def cmd_needs_cuda(args: argparse.Namespace) -> int:
    return 0 if GPU_GROUPS.intersection(parse_groups(args.groups)) else 1


def cmd_save_invocation(args: argparse.Namespace) -> int:
    path = _cap25.resolve(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "groups": list(parse_groups(args.groups)),
        "n_runs": int(args.n_runs),
        "analysis_n": int(args.analysis_n),
        "device": str(args.device),
        "td3_preset": str(args.td3_preset),
        "out": OUT_NAME,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ns = args
    if args.from_invocation:
        ns = _invocation_namespace(_cap25.resolve(args.from_invocation), args)
        ns.ledger = args.ledger
        ns.root = args.root
        ns.pidfile = args.pidfile
        ns.current_job = args.current_job
        ns.progress = args.progress
    jobs = _selector(ns)
    ledger = _cap25.resolve(args.ledger)
    last = _cap25.ledger_map(ledger)
    current = ""
    current_path = _cap25.resolve(args.current_job) if args.current_job else None
    if current_path is not None and current_path.is_file():
        current = current_path.read_text(encoding="utf-8").strip()
    pid = _cap25.pid_state(_cap25.resolve(args.pidfile) if args.pidfile else None)
    lines = [f"pid: {pid}", f"jobs: {len(jobs)}", ""]
    counts = {"ok": 0, "fail": 0, "running": 0, "pending": 0}
    for job in jobs:
        done = job_complete(job)
        rec = last.get(job.id) or {}
        state = str(rec.get("status") or "")
        if done:
            label = "ok"
        elif job.id == current and state == "start":
            label = "running"
        elif state in {"fail", "interrupted"}:
            label = state
        elif state == "start":
            label = "running"
        else:
            label = "pending"
        bucket = "ok" if label == "ok" else label if label in counts else "pending"
        if label == "interrupted":
            bucket = "fail"
        counts[bucket] = counts.get(bucket, 0) + 1
        lines.append(f"{label:<12} {job.id}")
    lines.append("")
    lines.append(
        "ok={ok}  running={running}  fail={fail}  pending={pending}".format(**counts)
    )
    text = "\n".join(lines) + "\n"
    if args.progress:
        progress = _cap25.resolve(args.progress)
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    default = build_jobs(
        n_runs=100, analysis_n=20, groups=DEFAULT_GROUPS, td3_preset="alg2"
    )
    full = build_jobs(n_runs=100, analysis_n=20, groups=GROUP_ORDER, td3_preset="alg2")
    if len(default) != 60:
        raise SystemExit(f"expected 60 default jobs, got {len(default)}")
    if len(full) != 74:
        raise SystemExit(f"expected 74 jobs with td3-campaign, got {len(full)}")
    ids = [job.id for job in full]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate job id")
    for job in full:
        if not job.argv or job.argv[0] not in {"-m", "scripts/analyze/mechanism_side_over_h.py", "scripts/analyze/compare_bw_proportional_noise.py", "scripts/analyze/polish_proportional_radio.py"}:
            if not (job.argv and (job.argv[0] == "-m" or job.argv[0].startswith("scripts/"))):
                raise SystemExit(f"bad argv for {job.id}")
        if "0.8" in " ".join(job.argv):
            raise SystemExit(f"T_k 0.8 leaked into {job.id}")
        _guard(job.out)
        if "device" in job.settings:
            raise SystemExit(f"device leaked into settings for {job.id}")
        if job.needs_device:
            runtime = job.with_runtime(device="cuda", force=False)
            if "--device" not in runtime or "cuda" not in runtime:
                raise SystemExit(f"missing device for {job.id}")
            if "--device" in job.argv:
                raise SystemExit(f"device baked into argv for {job.id}")
        if job.group == "td3-campaign" and job.check != "campaign":
            raise SystemExit(job.id)
        if job.group == "td3-n100" and "cap25" in job.id:
            raise SystemExit(f"cap25 TD3 should stay in the existing tree: {job.id}")
        if job.check == "fig11" and job.area not in (100, 500):
            raise SystemExit(f"fig11 area missing for {job.id}")
    if any(job.group == "td3-campaign" for job in default):
        raise SystemExit("td3-campaign must stay out of the default queue")

    import tempfile

    n = 2
    runs = []
    for seed in range(1, n + 1):
        runs.append(
            {
                "seed": seed,
                "method": "sca_anchor_frozen",
                "sum_rate_Mbps": 1.0,
                "feasible": True,
                "diagnostics": {
                    "dynamic_assignment": False,
                    "selection": "random",
                },
            }
        )
    payload = {
        "b_sys_hz": B_SYS_HZ,
        "max_bw_share": 0.25,
        "area_m": [100.0, 100.0],
        "methods": ["sca_anchor_frozen"],
        "n_scenarios": n,
        "by_method": {"sca_anchor_frozen": {"n": n, "per_seed_Mbps": [1.0, 1.0]}},
        "runs": runs,
    }
    probe = Job(
        id="probe",
        group="controls",
        out=f"{OUT_NAME}/probe.json",
        check="n100-random",
        argv=("-m", "uavdt", "n100"),
        settings={},
        method="sca_anchor_frozen",
        area=100,
        share=0.25,
        n=n,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "n100.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        if not _placement_complete(probe, path):
            raise SystemExit("n100 diagnostics on runs should count as complete")
        payload["runs"] = [
            {**rec, "diagnostics": {"dynamic_assignment": False}} for rec in runs
        ]
        path.write_text(json.dumps(payload), encoding="utf-8")
        if _placement_complete(probe, path):
            raise SystemExit("random-anchor without selection=random should not pass")
    print(f"self-check ok  default={len(default)}  with-td3-campaign={len(full)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    _add_selector(plan)
    plan.set_defaults(func=cmd_plan)

    listed = sub.add_parser("list")
    _add_selector(listed)
    listed.set_defaults(func=cmd_list)

    argv_p = sub.add_parser("argv")
    _add_selector(argv_p)
    argv_p.add_argument("--job", required=True)
    argv_p.add_argument("--force", action="store_true")
    argv_p.set_defaults(func=cmd_argv)

    complete = sub.add_parser("complete")
    _add_selector(complete)
    complete.add_argument("--job", required=True)
    complete.set_defaults(func=cmd_complete)

    bind = sub.add_parser("bind-settings")
    _add_selector(bind)
    bind.add_argument("--job", required=True)
    bind.add_argument("--force", action="store_true")
    bind.set_defaults(func=cmd_bind)

    discard = sub.add_parser("discard")
    _add_selector(discard)
    discard.add_argument("--job", required=True)
    discard.set_defaults(func=cmd_discard)

    record = sub.add_parser("record")
    record.add_argument("--ledger", required=True)
    record.add_argument("--job", required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--field", action="append")
    record.set_defaults(func=cmd_record)

    last = sub.add_parser("last")
    last.add_argument("--ledger", required=True)
    last.add_argument("--job", required=True)
    last.set_defaults(func=cmd_last)

    cuda = sub.add_parser("needs-cuda")
    cuda.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    cuda.set_defaults(func=cmd_needs_cuda)

    save = sub.add_parser("save-invocation")
    _add_selector(save)
    save.add_argument("--path", required=True)
    save.set_defaults(func=cmd_save_invocation)

    status = sub.add_parser("status")
    _add_selector(status)
    status.add_argument("--from-invocation", default="")
    status.add_argument("--ledger", required=True)
    status.add_argument("--root", default=OUT_NAME)
    status.add_argument("--pidfile", default="")
    status.add_argument("--current-job", default="")
    status.add_argument("--progress", default="")
    status.set_defaults(func=cmd_status)

    check = sub.add_parser("self-check")
    check.set_defaults(func=cmd_self_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
