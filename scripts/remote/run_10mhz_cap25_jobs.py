"""Ledger, completeness, and merged outputs for run_10mhz_cap25_all.sh.

Jobs are one method at a time so a crash in zenith does not throw away a
finished random/k-means/SCA run. Campaign checkpoints still resume inside
an unfinished job; this file decides whether the job itself is done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT_NAME = "results/run_10mhz_cap25"
B_SYS_HZ = 10_000_000.0
MAX_BW_SHARE = 0.25

METHOD_ALIASES = {
    "random": "random",
    "kmean": "kmeans",
    "kmeans": "kmeans",
    "sca": "sca",
    "multistep": "sca_multistart",
    "multi-step": "sca_multistart",
    "multistart": "sca_multistart",
    "multi-start": "sca_multistart",
    "sca_multistart": "sca_multistart",
    "zenith": "sca_anchor",
    "zenith-anchor": "sca_anchor",
    "anchor": "sca_anchor",
    "sca_anchor": "sca_anchor",
}
AREA_ALIASES = {
    "100": 100,
    "100m": 100,
    "100x100": 100,
    "500": 500,
    "500m": 500,
    "500x500": 500,
}
EXP_ALIASES = {
    "n100": "n100",
    "bank": "n100",
    "campaign": "campaign",
    "sweep": "campaign",
    "sweeps": "campaign",
}


def default_root() -> Path:
    return ROOT / OUT_NAME


def parse_csv(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def normalize_methods(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parse_csv(raw):
        name = METHOD_ALIASES.get(part)
        if name is None:
            known = ", ".join(sorted(set(METHOD_ALIASES)))
            raise SystemExit(f"unknown method {part!r}; expected one of: {known}")
        if name not in seen:
            seen.add(name)
            out.append(name)
    if not out:
        raise SystemExit("no methods selected")
    return out


def normalize_areas(raw: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in parse_csv(raw):
        if part not in AREA_ALIASES:
            raise SystemExit(
                f"unknown area {part!r}; use 100, 500, 100x100, or 500x500"
            )
        area = AREA_ALIASES[part]
        if area not in seen:
            seen.add(area)
            out.append(area)
    if not out:
        raise SystemExit("no areas selected")
    return out


def normalize_experiments(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in parse_csv(raw):
        if part == "all":
            names = ("n100", "campaign")
        else:
            name = EXP_ALIASES.get(part)
            if name is None:
                raise SystemExit(
                    f"unknown experiment {part!r}; use n100, campaign, or all"
                )
            names = (name,)
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
    if not out:
        raise SystemExit("no experiments selected")
    return out


def job_id(area: int, kind: str, method: str) -> str:
    return f"{area}m/{kind}/{method}"


def method_output(root: Path, kind: str, area: int, method: str) -> Path:
    stem = "campaign" if kind == "campaign" else "n100"
    return root / f"{area}m" / f"{stem}_{method}.json"


def rel_to_repo(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _grids():
    from uavdt.config import headline_sim_config
    from uavdt.experiments.grids import AXES, iter_axis

    return headline_sim_config, AXES, iter_axis


def point_key(axis: object, x: object) -> tuple[str, float]:
    return (str(axis), round(float(x), 6))


def expected_point_keys(area: float) -> set[tuple[str, float]]:
    headline_sim_config, axes, iter_axis = _grids()
    cfg = headline_sim_config(area_m=float(area))
    keys: set[tuple[str, float]] = set()
    for axis in axes:
        for point in iter_axis(axis, cfg):
            keys.add(point_key(point.axis, point.x_value))
    return keys


def _close(got: object, want: float, tol: float) -> bool:
    try:
        return abs(float(got) - want) < tol
    except (TypeError, ValueError):
        return False


def _area_ok(payload: dict, area: float) -> bool:
    raw = payload.get("area_m")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return False
    return _close(raw[0], area, 1e-6) and _close(raw[1], area, 1e-6)


def output_complete(
    path: Path,
    *,
    kind: str,
    method: str,
    area: float,
    n_runs: int,
    n_scenarios: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("checkpoint") is True:
        return False
    if not _close(payload.get("b_sys_hz"), B_SYS_HZ, 1.0):
        return False
    if not _close(payload.get("max_bw_share"), MAX_BW_SHARE, 1e-9):
        return False
    if not _area_ok(payload, area):
        return False
    if method not in list(payload.get("methods") or []):
        return False
    if kind == "campaign":
        if int(payload.get("n_runs") or -1) != int(n_runs):
            return False
        points = payload.get("points") or []
        seen: set[tuple[str, float]] = set()
        for pt in points:
            if not isinstance(pt, dict):
                return False
            seen.add(point_key(pt.get("axis"), pt.get("x")))
            stats = (pt.get("by_method") or {}).get(method)
            if not isinstance(stats, dict):
                return False
            if int(stats.get("n") or -1) != int(n_runs):
                return False
        return seen == expected_point_keys(area)
    if kind == "n100":
        if int(payload.get("n_scenarios") or -1) != int(n_scenarios):
            return False
        stats = (payload.get("by_method") or {}).get(method)
        if not isinstance(stats, dict):
            return False
        if int(stats.get("n") or -1) != int(n_scenarios):
            return False
        per = stats.get("per_seed_Mbps") or []
        return len(per) == int(n_scenarios)
    raise SystemExit(f"unknown kind {kind!r}")


def require_banks(areas: list[int], experiments: list[str]) -> None:
    if "n100" not in experiments:
        return
    for area in areas:
        path = ROOT / "data" / "scenario_bank" / f"n100_i10_j3_{area}m.json"
        if not path.is_file():
            raise SystemExit(f"missing frozen bank {rel_to_repo(path)}")
        try:
            geo = json.loads(path.read_text(encoding="utf-8"))["geometry"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SystemExit(f"unreadable bank {path}: {exc}") from exc
        if not _close(geo.get("area_x_m"), area, 1e-6) or not _close(
            geo.get("area_y_m"), area, 1e-6
        ):
            raise SystemExit(
                f"bank {rel_to_repo(path)} is "
                f"{geo.get('area_x_m')}x{geo.get('area_y_m')}, not {area}x{area}"
            )


def build_plan(
    methods: list[str], areas: list[int], experiments: list[str]
) -> list[tuple[str, str, int, str, str]]:
    require_banks(areas, experiments)
    root = default_root()
    rows: list[tuple[str, str, int, str, str]] = []
    for area in areas:
        for kind in experiments:
            for method in methods:
                path = method_output(root, kind, area, method)
                rows.append(
                    (job_id(area, kind, method), kind, area, method, rel_to_repo(path))
                )
    return rows


def job_settings(
    *,
    kind: str,
    area: int,
    method: str,
    n_runs: int,
    n_scenarios: int,
) -> dict:
    settings = {
        "bandwidth_preset": "10mhz",
        "max_bw_share": MAX_BW_SHARE,
        "area_m": int(area),
        "method": method,
        "kind": kind,
        "seed_start": 1,
        "solver": "cvxpy",
        "max_iterations": 30,
        "epsilon": 1e-4,
        "step_size_m": 20.0,
    }
    if kind == "campaign":
        settings["n_runs"] = int(n_runs)
        settings["axes"] = ["uavs", "iots", "lambda", "aodt", "cpu"]
    else:
        settings["n_scenarios"] = int(n_scenarios)
        settings["bank"] = f"data/scenario_bank/n100_i10_j3_{int(area)}m.json"
        settings["num_iot"] = 10
        settings["num_uav"] = 3
    if method == "sca_anchor":
        settings["anchor_max_enumerate"] = 10000
        settings["anchor_top_k"] = 3
        settings["anchor_beam_width"] = 10
    if method == "sca_multistart":
        settings["multistart_random"] = 2
        settings["multistart_kmeans"] = 2
    return settings


def build_argv(
    *,
    kind: str,
    area: int,
    method: str,
    out_rel: str,
    n_runs: int,
    n_scenarios: int,
    plots: bool,
    force: bool,
) -> list[str]:
    common = [
        "--bandwidth-preset",
        "10mhz",
        "--max-bw-share",
        "0.25",
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
    if kind == "campaign":
        cmd = [
            "-m",
            "uavdt",
            "campaign",
            "--axis",
            "all",
            *common,
            "--n-runs",
            str(int(n_runs)),
            "--out",
            out_rel,
        ]
    elif kind == "n100":
        cmd = [
            "-m",
            "uavdt",
            "n100",
            *common,
            "--n-scenarios",
            str(int(n_scenarios)),
            "--num-iot",
            "10",
            "--num-uav",
            "3",
            "--bank",
            f"data/scenario_bank/n100_i10_j3_{int(area)}m.json",
            "--out",
            out_rel,
            "--checkpoint",
            str(Path(out_rel).with_name(Path(out_rel).stem + ".checkpoint.json")).replace(
                "\\", "/"
            ),
            "--fig-dir",
            f"{OUT_NAME}/{int(area)}m/figures/n100_{method}",
        ]
        if not plots:
            cmd.append("--skip-plot")
        if force:
            cmd.append("--no-resume")
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    if method == "sca_anchor":
        cmd += [
            "--anchor-max-enumerate",
            "10000",
            "--anchor-top-k",
            "3",
            "--anchor-beam-width",
            "10",
        ]
    if method == "sca_multistart":
        cmd += ["--multistart-random", "2", "--multistart-kmeans", "2"]
    return cmd


def _lock_append(path: Path, line: str) -> None:
    # One writer. Avoid flock: on NFS it can sit forever after a job finishes.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()


def append_record(ledger: Path, job: str, status: str, fields: list[str]) -> None:
    rec: dict[str, object] = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job": job,
        "status": status,
    }
    for item in fields:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        rec[key] = value
    _lock_append(ledger, json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")


def _iter_records(ledger: Path) -> list[dict]:
    if not ledger.is_file():
        return []
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def last_status(ledger: Path, job: str) -> str:
    last = ""
    for rec in _iter_records(ledger):
        if rec.get("job") == job:
            last = str(rec.get("status") or "")
    return last


def ledger_map(ledger: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for rec in _iter_records(ledger):
        if rec.get("job"):
            found[str(rec["job"])] = rec
    return found


def pid_state(pidfile: Path | None) -> str:
    if pidfile is None or not pidfile.is_file():
        return "not running"
    raw = pidfile.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        return "not running"
    pid = int(raw)
    try:
        os.kill(pid, 0)
    except OSError:
        return f"not running (stale pid {pid})"
    except SystemError:
        return f"pid {pid}"
    return f"running pid {pid}"


def render_status(
    *,
    methods: list[str],
    areas: list[int],
    experiments: list[str],
    n_runs: int,
    n_scenarios: int,
    ledger: Path,
    pidfile: Path | None,
    current_job: str,
    root: Path,
) -> str:
    last = ledger_map(ledger)
    running = pid_state(pidfile).startswith("running")
    lines = [
        pid_state(pidfile),
        f"output: {root}",
        (
            f"protocol: 10 MHz, cap 25%, n_runs={n_runs}, "
            f"n_scenarios={n_scenarios}"
        ),
        f"{'job':<36} {'state':<14} detail",
    ]
    for job, kind, area, method, out_rel in build_plan(methods, areas, experiments):
        path = resolve(out_rel)
        done = output_complete(
            path,
            kind=kind,
            method=method,
            area=float(area),
            n_runs=n_runs,
            n_scenarios=n_scenarios,
        )
        rec = last.get(job) or {}
        prev = str(rec.get("status") or "")
        reason = str(rec.get("reason") or "")
        ckpt = path.with_name(path.stem + ".checkpoint.json")
        if done:
            state, detail = "ok", rel_to_repo(path)
        elif prev == "fail" and reason == "settings_mismatch":
            state, detail = "mismatch", "delete .settings.json and the checkpoint to change flags"
        elif running and current_job == job:
            state, detail = "running", rel_to_repo(ckpt) if ckpt.is_file() else "started"
        elif prev == "interrupted":
            state = "interrupted"
            detail = "rerun the same command to resume"
        elif prev == "fail":
            state, detail = "fail", "rerun the same command to retry"
        elif prev == "start" or ckpt.is_file():
            state = "incomplete"
            detail = "checkpoint will resume" if ckpt.is_file() else "started, no checkpoint yet"
        else:
            state, detail = "pending", ""
        lines.append(f"{job:<36} {state:<14} {detail}")
    return "\n".join(lines) + "\n"


def _stable(obj: dict) -> dict:
    return json.loads(json.dumps(obj, sort_keys=True))


def bind_settings(path: Path, expected: dict, *, force: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not force:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"unreadable settings {path}: {exc}", file=sys.stderr)
            return 3
        if _stable(old) != _stable(expected):
            print(f"settings mismatch for {path}", file=sys.stderr)
            print(f"  saved: {json.dumps(old, sort_keys=True)}", file=sys.stderr)
            print(f"  now:   {json.dumps(expected, sort_keys=True)}", file=sys.stderr)
            ckpt = path.with_name(path.name.replace(".settings.json", ".checkpoint.json"))
            print(
                "Refusing to mix a partial run with different flags. "
                f"Delete {path.name} and {ckpt.name} before changing n_runs or method knobs.",
                file=sys.stderr,
            )
            return 3
        return 0
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return payload


def merge_campaigns(
    root: Path, area: int, methods: list[str], n_runs: int
) -> Path | None:
    present: list[tuple[str, dict]] = []
    for method in methods:
        path = method_output(root, "campaign", area, method)
        if output_complete(
            path,
            kind="campaign",
            method=method,
            area=float(area),
            n_runs=n_runs,
            n_scenarios=1,
        ):
            present.append((method, _load_json(path)))
    if not present:
        return None
    from uavdt.experiments.campaign import write_campaign

    by_key: dict[tuple[str, float], dict] = {}
    order: list[tuple[str, float]] = []
    for _method, payload in present:
        for pt in payload.get("points") or []:
            key = point_key(pt.get("axis"), pt.get("x"))
            if key not in by_key:
                copy = json.loads(json.dumps(pt))
                copy["by_method"] = {}
                by_key[key] = copy
                order.append(key)
            by_key[key]["by_method"].update(pt.get("by_method") or {})
    header = dict(present[0][1])
    names = [method for method, _payload in present]
    header["methods"] = names
    header["sca_multistart"] = "sca_multistart" in names
    header["sca_anchor"] = "sca_anchor" in names
    header["points"] = [by_key[key] for key in order]
    header.pop("checkpoint", None)
    header["note"] = (
        "Merged per-method 10 MHz / 25% campaigns "
        f"({', '.join(names)}). " + str(header.get("note") or "")
    )
    out = root / f"{area}m" / "campaign_all.json"
    write_campaign(header, out)
    return out


def merge_n100(
    root: Path, area: int, methods: list[str], n_scenarios: int
) -> Path | None:
    present: list[tuple[str, dict]] = []
    for method in methods:
        path = method_output(root, "n100", area, method)
        if output_complete(
            path,
            kind="n100",
            method=method,
            area=float(area),
            n_runs=1,
            n_scenarios=n_scenarios,
        ):
            present.append((method, _load_json(path)))
    if not present:
        return None
    from uavdt.experiments.n100 import _paired_delta, _paired_stats, write_eval

    header = dict(present[0][1])
    names = [method for method, _payload in present]
    by_method = {}
    runs = []
    for method, payload in present:
        by_method[method] = (payload.get("by_method") or {})[method]
        runs.extend(
            r
            for r in (payload.get("runs") or [])
            if isinstance(r, dict) and r.get("method") == method
        )
    header["methods"] = names
    header["by_method"] = by_method
    header["runs"] = runs
    header["sca_minus_baseline_Mbps"] = _paired_stats(by_method)
    header["sca_minus_frozen_sca_Mbps"] = _paired_delta(by_method, "sca", "frozen_sca")
    header.pop("checkpoint", None)
    header["note"] = (
        "Merged per-method 10 MHz / 25% bank evals "
        f"({', '.join(names)}). " + str(header.get("note") or "")
    )
    out = root / f"{area}m" / "n100_all.json"
    write_eval(header, out)
    return out


def merge_all(
    root: Path, areas: list[int], methods: list[str], n_runs: int, n_scenarios: int
) -> list[Path]:
    wrote: list[Path] = []
    for area in areas:
        camp = merge_campaigns(root, area, methods, n_runs)
        bank = merge_n100(root, area, methods, n_scenarios)
        if camp is not None:
            wrote.append(camp)
        if bank is not None:
            wrote.append(bank)
    return wrote


def save_invocation(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_csv(value: object) -> str:
    """Join a JSON list. str(['random', ...]) is not a method name."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return str(value)


def load_invocation(path: Path) -> dict:
    payload = _load_json(path)
    return {
        "methods": normalize_methods(_as_csv(payload["methods"])),
        "areas": normalize_areas(_as_csv(payload["areas"])),
        "experiments": normalize_experiments(_as_csv(payload["experiments"])),
        "n_runs": int(payload["n_runs"]),
        "n_scenarios": int(payload["n_scenarios"]),
    }


def _write_status_text(args: argparse.Namespace, text: str) -> None:
    sys.stdout.write(text)
    if args.progress:
        progress = resolve(args.progress)
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(text, encoding="utf-8")


def _selection_from_args(args: argparse.Namespace) -> tuple[list[str], list[int], list[str], int, int]:
    if getattr(args, "from_invocation", None):
        inv = load_invocation(resolve(args.from_invocation))
        return (
            inv["methods"],
            inv["areas"],
            inv["experiments"],
            inv["n_runs"],
            inv["n_scenarios"],
        )
    return (
        normalize_methods(args.methods),
        normalize_areas(args.areas),
        normalize_experiments(args.experiments),
        int(args.n_runs),
        int(args.n_scenarios),
    )


def cmd_plan(args: argparse.Namespace) -> int:
    methods, areas, experiments, _n_runs, _n_scenarios = _selection_from_args(args)
    for row in build_plan(methods, areas, experiments):
        print("\t".join(str(part) for part in row))
    return 0


def cmd_argv(args: argparse.Namespace) -> int:
    method = normalize_methods(args.method)[0]
    for part in build_argv(
        kind=args.kind,
        area=int(args.area),
        method=method,
        out_rel=args.out.replace("\\", "/"),
        n_runs=int(args.n_runs),
        n_scenarios=int(args.n_scenarios),
        plots=bool(args.plots),
        force=bool(args.force),
    ):
        print(part)
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    method = normalize_methods(args.method)[0]
    ok = output_complete(
        resolve(args.path),
        kind=args.kind,
        method=method,
        area=float(args.area),
        n_runs=int(args.n_runs),
        n_scenarios=int(args.n_scenarios),
    )
    return 0 if ok else 1


def cmd_record(args: argparse.Namespace) -> int:
    append_record(resolve(args.ledger), args.job, args.status, list(args.field or []))
    return 0


def cmd_last(args: argparse.Namespace) -> int:
    print(last_status(resolve(args.ledger), args.job))
    return 0


def cmd_bind(args: argparse.Namespace) -> int:
    method = normalize_methods(args.method)[0]
    expected = job_settings(
        kind=args.kind,
        area=int(args.area),
        method=method,
        n_runs=int(args.n_runs),
        n_scenarios=int(args.n_scenarios),
    )
    return bind_settings(resolve(args.path), expected, force=bool(args.force))


def cmd_merge(args: argparse.Namespace) -> int:
    methods, areas, _experiments, n_runs, n_scenarios = _selection_from_args(args)
    wrote = merge_all(resolve(args.root), areas, methods, n_runs, n_scenarios)
    if not wrote:
        print("merge: no complete method files yet")
        return 0
    for path in wrote:
        print(f"merged {rel_to_repo(path)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    methods, areas, experiments, n_runs, n_scenarios = _selection_from_args(args)
    current = ""
    if args.current_job:
        current_path = resolve(args.current_job)
        if current_path.is_file():
            current = current_path.read_text(encoding="utf-8").strip()
    text = render_status(
        methods=methods,
        areas=areas,
        experiments=experiments,
        n_runs=n_runs,
        n_scenarios=n_scenarios,
        ledger=resolve(args.ledger),
        pidfile=resolve(args.pidfile) if args.pidfile else None,
        current_job=current,
        root=resolve(args.root),
    )
    _write_status_text(args, text)
    return 0


def cmd_save_invocation(args: argparse.Namespace) -> int:
    methods, areas, experiments, n_runs, n_scenarios = _selection_from_args(args)
    save_invocation(
        resolve(args.path),
        {
            "methods": methods,
            "areas": areas,
            "experiments": experiments,
            "n_runs": n_runs,
            "n_scenarios": n_scenarios,
        },
    )
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    import tempfile

    methods = normalize_methods("kmean,random,sca,multistep,zenith")
    if methods != ["kmeans", "random", "sca", "sca_multistart", "sca_anchor"]:
        raise SystemExit(f"alias order wrong: {methods}")
    areas = normalize_areas("100x100,500x500")
    experiments = normalize_experiments("all")
    plan = build_plan(methods, areas, experiments)
    if len(plan) != 20:
        raise SystemExit(f"expected 20 jobs, got {len(plan)}")
    if plan[0][0] != "100m/n100/kmeans":
        raise SystemExit(f"unexpected first job {plan[0][0]}")
    if plan[-1][0] != "500m/campaign/sca_anchor":
        raise SystemExit(f"unexpected last job {plan[-1][0]}")
    argv = build_argv(
        kind="campaign",
        area=500,
        method="sca_anchor",
        out_rel="results/run_10mhz_cap25/500m/campaign_sca_anchor.json",
        n_runs=100,
        n_scenarios=100,
        plots=False,
        force=False,
    )
    if "--bandwidth-preset" not in argv or "10mhz" not in argv:
        raise SystemExit("campaign argv missing 10mhz")
    if "--max-bw-share" not in argv or "0.25" not in argv:
        raise SystemExit("campaign argv missing cap 0.25")
    if argv[argv.index("--anchor-max-enumerate") + 1] != "10000":
        raise SystemExit("zenith enumerate knob missing")
    n100_argv = build_argv(
        kind="n100",
        area=100,
        method="sca_multistart",
        out_rel="results/run_10mhz_cap25/100m/n100_sca_multistart.json",
        n_runs=100,
        n_scenarios=100,
        plots=False,
        force=False,
    )
    if "--skip-plot" not in n100_argv or "--multistart-random" not in n100_argv:
        raise SystemExit(f"bad n100 argv: {n100_argv}")

    keys = expected_point_keys(100)
    if len(keys) != 28:
        raise SystemExit(f"expected 28 sweep points, got {len(keys)}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        n_runs = 2
        n_scenarios = 3
        for method, rate in (("random", 1.0), ("kmeans", 2.0)):
            payload = _fake_campaign(100, method, n_runs, rate)
            path = method_output(root, "campaign", 100, method)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            if not output_complete(
                path,
                kind="campaign",
                method=method,
                area=100,
                n_runs=n_runs,
                n_scenarios=n_scenarios,
            ):
                raise SystemExit(f"fake campaign {method} should be complete")
            bank = _fake_n100(100, method, n_scenarios, rate)
            bpath = method_output(root, "n100", 100, method)
            bpath.write_text(json.dumps(bank), encoding="utf-8")
            if not output_complete(
                bpath,
                kind="n100",
                method=method,
                area=100,
                n_runs=n_runs,
                n_scenarios=n_scenarios,
            ):
                raise SystemExit(f"fake n100 {method} should be complete")
        broken = dict(_fake_campaign(100, "sca", n_runs, 3.0))
        broken["checkpoint"] = True
        bork = method_output(root, "campaign", 100, "sca")
        bork.write_text(json.dumps(broken), encoding="utf-8")
        if output_complete(
            bork, kind="campaign", method="sca", area=100, n_runs=n_runs, n_scenarios=1
        ):
            raise SystemExit("checkpoint file must not count as done")
        wrote = merge_all(root, [100], ["random", "kmeans", "sca"], n_runs, n_scenarios)
        names = {path.name for path in wrote}
        if names != {"campaign_all.json", "n100_all.json"}:
            raise SystemExit(f"unexpected merge outputs {names}")
        merged = json.loads((root / "100m" / "campaign_all.json").read_text(encoding="utf-8"))
        if merged["methods"] != ["random", "kmeans"]:
            raise SystemExit(f"merge included incomplete sca: {merged['methods']}")
        settings = job_settings(
            kind="campaign", area=100, method="random", n_runs=n_runs, n_scenarios=n_scenarios
        )
        sp = root / "random.settings.json"
        if bind_settings(sp, settings, force=False) != 0:
            raise SystemExit("first settings write failed")
        if bind_settings(sp, settings, force=False) != 0:
            raise SystemExit("matching settings should pass")
        changed = dict(settings)
        changed["n_runs"] = n_runs + 1
        if bind_settings(sp, changed, force=False) != 3:
            raise SystemExit("changed n_runs should mismatch")
        ledger = root / "jobs.jsonl"
        append_record(ledger, "100m/n100/random", "start", ["out=x"])
        append_record(ledger, "100m/n100/random", "ok", [])
        if last_status(ledger, "100m/n100/random") != "ok":
            raise SystemExit("ledger last status failed")
        inv_path = root / "invocation.json"
        save_invocation(
            inv_path,
            {
                "methods": ["random", "kmeans", "sca", "sca_multistart", "sca_anchor"],
                "areas": [100, 500],
                "experiments": ["n100", "campaign"],
                "n_runs": 100,
                "n_scenarios": 100,
            },
        )
        loaded = load_invocation(inv_path)
        if loaded["methods"][0] != "random" or len(loaded["methods"]) != 5:
            raise SystemExit(f"invocation methods did not round-trip: {loaded['methods']}")
        if loaded["areas"] != [100, 500] or loaded["experiments"] != ["n100", "campaign"]:
            raise SystemExit(f"invocation selection did not round-trip: {loaded}")
    print("self-check ok")
    return 0


def _fake_method_stats(n: int, rate: float) -> dict:
    return {
        "n": n,
        "mean_sum_rate_Mbps": rate,
        "std_sum_rate_Mbps": 0.0,
        "feasible_fraction": 1.0,
        "mean_max_AoDT_s": 0.5,
        "std_max_AoDT_s": 0.0,
        "per_seed_Mbps": [rate] * n,
        "seeds": list(range(1, n + 1)),
        "per_seed_feasible": [True] * n,
        "per_seed_max_AoDT_s": [0.5] * n,
    }


def _fake_campaign(area: int, method: str, n_runs: int, rate: float) -> dict:
    headline_sim_config, axes, iter_axis = _grids()
    cfg = headline_sim_config(area_m=float(area))
    points = []
    for axis in axes:
        for point in iter_axis(axis, cfg):
            points.append(
                {
                    "axis": point.axis,
                    "x_name": point.x_name,
                    "x": point.x_value,
                    "label": point.label,
                    "b_sys_hz": B_SYS_HZ,
                    "max_bw_share": MAX_BW_SHARE,
                    "num_iot": point.cfg.num_iot,
                    "num_uav": point.cfg.num_uav,
                    "by_method": {method: _fake_method_stats(n_runs, rate)},
                }
            )
    return {
        "n_runs": n_runs,
        "seed_start": 1,
        "methods": [method],
        "b_sys_hz": B_SYS_HZ,
        "max_bw_share": MAX_BW_SHARE,
        "area_m": [float(area), float(area)],
        "points": points,
    }


def _fake_n100(area: int, method: str, n_scenarios: int, rate: float) -> dict:
    runs = [
        {
            "scenario_id": i,
            "seed": i,
            "method": method,
            "sum_rate_Mbps": rate,
            "feasible": True,
            "max_AoDT_s": 0.5,
        }
        for i in range(1, n_scenarios + 1)
    ]
    return {
        "n_scenarios": n_scenarios,
        "methods": [method],
        "b_sys_hz": B_SYS_HZ,
        "max_bw_share": MAX_BW_SHARE,
        "area_m": [float(area), float(area)],
        "by_method": {method: _fake_method_stats(n_scenarios, rate)},
        "runs": runs,
        "note": "fake",
    }


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods",
        default="random,kmeans,sca,sca_multistart,sca_anchor",
    )
    parser.add_argument("--areas", default="100,500")
    parser.add_argument("--experiments", default="n100,campaign")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--from-invocation", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    _add_selection(plan)
    plan.set_defaults(func=cmd_plan)

    argv = sub.add_parser("argv")
    argv.add_argument("--kind", required=True, choices=("campaign", "n100"))
    argv.add_argument("--area", required=True, type=int)
    argv.add_argument("--method", required=True)
    argv.add_argument("--out", required=True)
    argv.add_argument("--n-runs", required=True, type=int)
    argv.add_argument("--n-scenarios", required=True, type=int)
    argv.add_argument("--plots", action="store_true")
    argv.add_argument("--force", action="store_true")
    argv.set_defaults(func=cmd_argv)

    complete = sub.add_parser("complete")
    complete.add_argument("--path", required=True)
    complete.add_argument("--kind", required=True, choices=("campaign", "n100"))
    complete.add_argument("--method", required=True)
    complete.add_argument("--area", required=True, type=float)
    complete.add_argument("--n-runs", required=True, type=int)
    complete.add_argument("--n-scenarios", required=True, type=int)
    complete.set_defaults(func=cmd_complete)

    record = sub.add_parser("record")
    record.add_argument("--ledger", required=True)
    record.add_argument("--job", required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--field", action="append", default=[])
    record.set_defaults(func=cmd_record)

    last = sub.add_parser("last")
    last.add_argument("--ledger", required=True)
    last.add_argument("--job", required=True)
    last.set_defaults(func=cmd_last)

    bind = sub.add_parser("bind-settings")
    bind.add_argument("--path", required=True)
    bind.add_argument("--kind", required=True, choices=("campaign", "n100"))
    bind.add_argument("--area", required=True, type=int)
    bind.add_argument("--method", required=True)
    bind.add_argument("--n-runs", required=True, type=int)
    bind.add_argument("--n-scenarios", required=True, type=int)
    bind.add_argument("--force", action="store_true")
    bind.set_defaults(func=cmd_bind)

    merge = sub.add_parser("merge")
    _add_selection(merge)
    merge.add_argument("--root", default=str(default_root()))
    merge.set_defaults(func=cmd_merge)

    status = sub.add_parser("status")
    _add_selection(status)
    status.add_argument("--ledger", required=True)
    status.add_argument("--root", default=str(default_root()))
    status.add_argument("--pidfile", default=None)
    status.add_argument("--current-job", default=None)
    status.add_argument("--progress", default=None)
    status.set_defaults(func=cmd_status)

    save = sub.add_parser("save-invocation")
    _add_selection(save)
    save.add_argument("--path", required=True)
    save.set_defaults(func=cmd_save_invocation)

    check = sub.add_parser("self-check")
    check.set_defaults(func=cmd_self_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
