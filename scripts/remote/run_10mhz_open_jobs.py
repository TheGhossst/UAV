"""Resumable ledger for the 10 MHz rows that are still open.

Writes only under results/run_10mhz_open/. Refuses the finished trees
results/run_10mhz_cap25/, run_10mhz_cap_sweep/, run_10mhz_td3/,
run_10mhz_controls/, and the finished missing-results ledger
results/run_10mhz_missing/.

Groups, in order:
  anchor-dyn-campaign  dynamic zenith campaigns, caps other than 25%, both fields
  pso-lp               PSO whose inner fitness is the frozen-q LP, every cap,
                       both fields, plus the 25% campaign on both fields
  los-deg              25% n100 with --los-angle-unit deg, both fields,
                       random / k-means / SCA / multi-start / dynamic zenith
  cmaes                CMA-ES residual + SCA polish at 10 MHz, 25%, both fields,
                       seeds 1..analysis-n (default 20)
  td3-campaign         Algorithm 2 TD3 campaigns, caps other than 25%, both fields

The learned proposal policy is not queued. That trainer is not in this
repository; the 8.8 MHz rows in observations O32 have no script to retarget.
T_k = 0.8 s is not queued. The 25% TD3 campaign and the 25% dynamic-anchor
campaigns already exist and are not repeated.

Launch with scripts/remote/run_10mhz_open_all.sh. Do not point that launcher
at the finished run_10mhz_missing ledger.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_NAME = "results/run_10mhz_open"
B_SYS_HZ = 10_000_000.0
PROTECTED = (
    "results/run_10mhz_cap25/",
    "results/run_10mhz_cap_sweep/",
    "results/run_10mhz_td3/",
    "results/run_10mhz_controls/",
    "results/run_10mhz_missing/",
    "results/residual_on_sca/",
)
GROUP_ORDER = (
    "anchor-dyn-campaign",
    "pso-lp",
    "los-deg",
    "cmaes",
    "td3-campaign",
)
DEFAULT_GROUPS = GROUP_ORDER
GPU_GROUPS = frozenset({"td3-campaign"})
AREAS = (100, 500)
OTHER_CAPS: tuple[tuple[str, float | None], ...] = (
    ("nocap", None),
    ("cap10", 0.10),
    ("cap12", 0.12),
    ("cap15", 0.15),
    ("cap18", 0.18),
    ("cap20", 0.20),
    ("cap22", 0.22),
)
ALL_CAPS: tuple[tuple[str, float | None], ...] = OTHER_CAPS + (("cap25", 0.25),)
LOS_METHODS = (
    "random",
    "kmeans",
    "sca",
    "sca_multistart",
    "sca_anchor",
)
ANCHOR_FLAGS = (
    "--anchor-max-enumerate",
    "10000",
    "--anchor-top-k",
    "3",
    "--anchor-beam-width",
    "10",
)
MULTISTART_FLAGS = (
    "--multistart-random",
    "2",
    "--multistart-kmeans",
    "2",
)
CMAES_SCRIPT = "scripts/experiments/residual_on_sca/run_cmaes_polish.py"


def _load(alias: str, filename: str):
    path = ROOT / "scripts" / "remote" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cap25 = _load("cap25jobs_open", "run_10mhz_cap25_jobs.py")
_sweep = _load("capsweepjobs_open", "run_10mhz_cap_sweep_jobs.py")


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


def _los_extra(method: str) -> tuple[tuple[str, ...], dict]:
    extra: list[str] = ["--los-angle-unit", "deg"]
    settings = {"los_angle_unit": "deg"}
    if method == "sca_multistart":
        extra += list(MULTISTART_FLAGS)
        settings["multistart_random"] = 2
        settings["multistart_kmeans"] = 2
    if method == "sca_anchor":
        extra += list(ANCHOR_FLAGS)
        settings["anchor_max_enumerate"] = 10000
        settings["anchor_top_k"] = 3
        settings["anchor_beam_width"] = 10
        settings["dynamic_assignment"] = True
    return tuple(extra), settings


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

    for slug, share in OTHER_CAPS:
        for area in AREAS:
            jobs.append(
                _placement(
                    group="anchor-dyn-campaign",
                    job_id=f"anchor-dyn-campaign/{slug}/{area}m",
                    kind="campaign",
                    area=area,
                    method="sca_anchor",
                    share=share,
                    out=(
                        f"{OUT_NAME}/anchor-dyn/{slug}/{area}m/"
                        "campaign_sca_anchor_dyn.json"
                    ),
                    n=n_runs,
                    check="campaign-dynamic",
                    extra=ANCHOR_FLAGS,
                    settings_extra={
                        "anchor_max_enumerate": 10000,
                        "anchor_top_k": 3,
                        "anchor_beam_width": 10,
                        "dynamic_assignment": True,
                    },
                )
            )

    for slug, share in ALL_CAPS:
        for area in AREAS:
            jobs.append(
                _placement(
                    group="pso-lp",
                    job_id=f"pso-lp/{slug}/{area}m/n100",
                    kind="n100",
                    area=area,
                    method="pso",
                    share=share,
                    out=f"{OUT_NAME}/pso-lp/{slug}/{area}m/n100_pso_lp.json",
                    n=n_runs,
                    check="n100-lp",
                    extra=("--pso-fitness", "lp"),
                    settings_extra={"pso_fitness": "lp"},
                )
            )
    for area in AREAS:
        jobs.append(
            _placement(
                group="pso-lp",
                job_id=f"pso-lp/cap25/{area}m/campaign",
                kind="campaign",
                area=area,
                method="pso",
                share=0.25,
                out=f"{OUT_NAME}/pso-lp/cap25/{area}m/campaign_pso_lp.json",
                n=n_runs,
                check="campaign-lp",
                extra=("--pso-fitness", "lp"),
                settings_extra={"pso_fitness": "lp"},
            )
        )

    for method in LOS_METHODS:
        extra, settings_extra = _los_extra(method)
        for area in AREAS:
            jobs.append(
                _placement(
                    group="los-deg",
                    job_id=f"los-deg/cap25/{area}m/{method}",
                    kind="n100",
                    area=area,
                    method=method,
                    share=0.25,
                    out=f"{OUT_NAME}/los-deg/cap25/{area}m/n100_{method}_deg.json",
                    n=n_runs,
                    check="n100-deg",
                    extra=extra,
                    settings_extra=settings_extra,
                )
            )

    for area in AREAS:
        out = f"{OUT_NAME}/cmaes/cap25/{area}m/cmaes_polish.json"
        jobs.append(
            Job(
                id=f"cmaes/cap25/{area}m",
                group="cmaes",
                out=_guard(out),
                check="cmaes",
                argv=(
                    CMAES_SCRIPT,
                    "--n-runs",
                    str(analysis_n),
                    "--seed-start",
                    "1",
                    "--max-evals",
                    "400",
                    "--sigma0",
                    "5",
                    "--bandwidth-preset",
                    "10mhz",
                    "--max-bw-share",
                    "0.25",
                    "--area-m",
                    str(area),
                    "--los-angle-unit",
                    "rad",
                    "--out",
                    out,
                ),
                settings=_settings(
                    {
                        "group": "cmaes",
                        "check": "cmaes",
                        "bandwidth_preset": "10mhz",
                        "max_bw_share": 0.25,
                        "area_m": int(area),
                        "los_angle_unit": "rad",
                        "n": int(analysis_n),
                        "seed_start": 1,
                        "max_evals": 400,
                        "sigma0_m": 5.0,
                    }
                ),
                area=area,
                share=0.25,
                n=analysis_n,
            )
        )

    for slug, share in OTHER_CAPS:
        for area in AREAS:
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


def _dynamic_ok(diags: object) -> bool:
    if not isinstance(diags, list) or not diags:
        return False
    seen = 0
    for item in diags:
        if not isinstance(item, dict) or "dynamic_assignment" not in item:
            continue
        seen += 1
        if not bool(item["dynamic_assignment"]):
            return False
    return seen > 0


def _tagged_ok(diags: object, key: str, want: str) -> bool:
    if not isinstance(diags, list) or not diags:
        return False
    seen = 0
    for item in diags:
        if not isinstance(item, dict) or key not in item:
            continue
        seen += 1
        if str(item[key]) != want:
            return False
    return seen > 0


def _n100_diags(payload: dict, method: str) -> list:
    found = []
    for rec in payload.get("runs") or []:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("method") or "") not in {"", method}:
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
    kind = "campaign" if job.check.startswith("campaign") else "n100"
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
    if job.check == "campaign":
        return True
    if job.check == "n100-deg":
        cfg = payload.get("cfg") or {}
        return str(cfg.get("los_angle_unit") or "") == "deg"
    if job.check == "n100-lp":
        return _tagged_ok(_n100_diags(payload, job.method), "pso_fitness", "lp")
    if job.check == "campaign-lp":
        points = payload.get("points") or []
        if not points:
            return False
        for pt in points:
            if not isinstance(pt, dict):
                return False
            stats = (pt.get("by_method") or {}).get(job.method)
            if not _tagged_ok(_campaign_diags(stats), "pso_fitness", "lp"):
                return False
        return True
    if job.check == "campaign-dynamic":
        points = payload.get("points") or []
        if not points:
            return False
        for pt in points:
            if not isinstance(pt, dict):
                return False
            stats = (pt.get("by_method") or {}).get(job.method)
            if not _dynamic_ok(_campaign_diags(stats)):
                return False
        return True
    raise SystemExit(f"unknown placement check {job.check}")


def _cmaes_complete(job: Job, path: Path) -> bool:
    payload = _read_json(path)
    if payload is None:
        return False
    if abs(float(payload.get("b_sys_hz") or 0) - B_SYS_HZ) > 1.0:
        return False
    if abs(float(payload.get("max_bw_share") or -1) - 0.25) > 1e-9:
        return False
    area = payload.get("area_m")
    if not isinstance(area, (list, tuple)) or len(area) < 2:
        return False
    if abs(float(area[0]) - job.area) > 1e-6 or abs(float(area[1]) - job.area) > 1e-6:
        return False
    if str(payload.get("los_angle_unit") or "") != "rad":
        return False
    if int(payload.get("seed_start") or -1) != 1:
        return False
    rows = payload.get("per_seed") or []
    if not isinstance(rows, list) or len(rows) != int(job.n):
        return False
    seeds = []
    for row in rows:
        if not isinstance(row, dict) or "seed" not in row:
            return False
        seeds.append(int(row["seed"]))
    return seeds == list(range(1, int(job.n) + 1))


def job_complete(job: Job, root: Path | None = None) -> bool:
    path = (root or ROOT) / job.out
    if job.check == "cmaes":
        return _cmaes_complete(job, path)
    if job.check.startswith("n100") or job.check.startswith("campaign"):
        return _placement_complete(job, path)
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
    settings_path = base / _suffix(job.out, ".settings.json")
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
    return argparse.Namespace(
        groups=",".join(payload.get("groups") or []),
        n_runs=int(payload.get("n_runs") or fallback.n_runs),
        analysis_n=int(payload.get("analysis_n") or fallback.analysis_n),
        device=str(payload.get("device") or fallback.device),
        td3_preset=str(payload.get("td3_preset") or fallback.td3_preset),
    )


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
    jobs = build_jobs(
        n_runs=100, analysis_n=20, groups=GROUP_ORDER, td3_preset="alg2"
    )
    expected = (
        len(OTHER_CAPS) * len(AREAS)  # dynamic-anchor campaigns
        + len(ALL_CAPS) * len(AREAS)  # PSO LP banks
        + len(AREAS)  # PSO LP 25% campaigns
        + len(LOS_METHODS) * len(AREAS)
        + len(AREAS)  # CMA-ES
        + len(OTHER_CAPS) * len(AREAS)  # TD3 campaigns
    )
    if len(jobs) != expected:
        raise SystemExit(f"expected {expected} jobs, got {len(jobs)}")
    ids = [job.id for job in jobs]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate job id")
    if "learned-proposal" in GROUP_ORDER:
        raise SystemExit("learned proposal policy has no trainer in this repo")
    for job in jobs:
        if "0.8" in " ".join(job.argv):
            raise SystemExit(f"T_k 0.8 leaked into {job.id}")
        if "8.8" in " ".join(job.argv):
            raise SystemExit(f"8.8 MHz leaked into {job.id}")
        _guard(job.out)
        if "device" in job.settings:
            raise SystemExit(f"device leaked into settings for {job.id}")
        if job.group in {"td3-campaign", "anchor-dyn-campaign"} and "cap25" in job.id:
            raise SystemExit(f"cap25 already exists and must stay out: {job.id}")
        if job.needs_device:
            runtime = job.with_runtime(device="cuda", force=False)
            if "--device" not in runtime or "cuda" not in runtime:
                raise SystemExit(f"missing device for {job.id}")
            if "--device" in job.argv:
                raise SystemExit(f"device baked into argv for {job.id}")
        else:
            if "--device" in job.argv:
                raise SystemExit(f"device on a CPU job {job.id}")
        if job.group == "pso-lp" and "--pso-fitness" not in job.argv:
            raise SystemExit(job.id)
        if job.group == "los-deg" and "deg" not in job.argv:
            raise SystemExit(job.id)
        if job.group == "cmaes" and job.argv[0] != CMAES_SCRIPT:
            raise SystemExit(job.id)
        if job.group == "td3-campaign" and job.check != "campaign":
            raise SystemExit(job.id)

    import tempfile

    n = 2
    payload = {
        "b_sys_hz": B_SYS_HZ,
        "max_bw_share": 0.25,
        "area_m": [100.0, 100.0],
        "methods": ["pso"],
        "n_scenarios": n,
        "cfg": {"los_angle_unit": "rad"},
        "by_method": {"pso": {"n": n, "per_seed_Mbps": [1.0, 1.0]}},
        "runs": [
            {
                "seed": 1,
                "method": "pso",
                "diagnostics": {"pso_fitness": "lp"},
            },
            {
                "seed": 2,
                "method": "pso",
                "diagnostics": {"pso_fitness": "lp"},
            },
        ],
    }
    probe = Job(
        id="probe",
        group="pso-lp",
        out=f"{OUT_NAME}/probe.json",
        check="n100-lp",
        argv=("-m", "uavdt", "n100"),
        settings={},
        method="pso",
        area=100,
        share=0.25,
        n=n,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "n100.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        if not _placement_complete(probe, path):
            raise SystemExit("LP fitness on runs should count as complete")
        payload["runs"][0]["diagnostics"] = {"pso_fitness": "equal_share"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        if _placement_complete(probe, path):
            raise SystemExit("equal-share PSO must not satisfy the LP check")
        deg = Job(
            id="deg",
            group="los-deg",
            out=f"{OUT_NAME}/deg.json",
            check="n100-deg",
            argv=("-m", "uavdt", "n100"),
            settings={},
            method="sca",
            area=100,
            share=0.25,
            n=n,
        )
        payload["methods"] = ["sca"]
        payload["by_method"] = {"sca": {"n": n, "per_seed_Mbps": [1.0, 1.0]}}
        payload["cfg"] = {"los_angle_unit": "rad"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        if _placement_complete(deg, path):
            raise SystemExit("radian file must not satisfy the degree check")
        payload["cfg"] = {"los_angle_unit": "deg"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        if not _placement_complete(deg, path):
            raise SystemExit("degree file should count as complete")
        cma = Job(
            id="cma",
            group="cmaes",
            out=f"{OUT_NAME}/cma.json",
            check="cmaes",
            argv=(CMAES_SCRIPT,),
            settings={},
            area=500,
            share=0.25,
            n=n,
        )
        cma_payload = {
            "b_sys_hz": B_SYS_HZ,
            "max_bw_share": 0.25,
            "area_m": [500.0, 500.0],
            "los_angle_unit": "rad",
            "seed_start": 1,
            "per_seed": [{"seed": 1}, {"seed": 2}],
        }
        path.write_text(json.dumps(cma_payload), encoding="utf-8")
        if not _cmaes_complete(cma, path):
            raise SystemExit("10 MHz CMA-ES payload should count as complete")
        cma_payload["b_sys_hz"] = 8.8e6
        path.write_text(json.dumps(cma_payload), encoding="utf-8")
        if _cmaes_complete(cma, path):
            raise SystemExit("8.8 MHz CMA-ES must not count")
    try:
        _guard("results/run_10mhz_cap25/100m/campaign.json")
    except SystemExit:
        pass
    else:
        raise SystemExit("protected cap25 path was accepted")
    print(f"self-check ok  jobs={len(jobs)}")
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
