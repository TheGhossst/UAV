"""Ledger and job planning for run_10mhz_cap_sweep_all.sh (10 MHz, many caps).

Caps are per-link fractions of B_sys. Use cap token ``none`` / ``0`` / ``0%`` for
no per-link cap (CLI ``--no-per-link-cap``). Default grid matches the fine
cap search: none, 10%, 12%, 15%, 18%, 20%, 22%, 25%.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT_NAME = "results/run_10mhz_cap_sweep"
B_SYS_HZ = 10_000_000.0
DEFAULT_CAPS = "none,10,12,15,18,20,22,25"

_spec = importlib.util.spec_from_file_location(
    "cap25jobs", ROOT / "scripts" / "remote" / "run_10mhz_cap25_jobs.py"
)
_cap25 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cap25)

METHOD_ALIASES = _cap25.METHOD_ALIASES
normalize_methods = _cap25.normalize_methods
normalize_areas = _cap25.normalize_areas
normalize_experiments = _cap25.normalize_experiments
require_banks = _cap25.require_banks
expected_point_keys = _cap25.expected_point_keys
point_key = _cap25.point_key
rel_to_repo = _cap25.rel_to_repo
resolve = _cap25.resolve
append_record = _cap25.append_record
last_status = _cap25.last_status
ledger_map = _cap25.ledger_map
pid_state = _cap25.pid_state
_load_json = _cap25._load_json
_as_csv = _cap25._as_csv
_stable = _cap25._stable
_bind_settings = _cap25.bind_settings  # noqa: SLF001 — shared helper name in cap25


def default_root() -> Path:
    return ROOT / OUT_NAME


def parse_cap_token(raw: str) -> float | None:
    p = raw.strip().lower()
    if p.startswith("cap") and len(p) > 3 and p[3:].isdigit():
        return float(p[3:]) / 100.0
    if p in {"none", "nocap", "no-cap", "0", "0%", "0.0", "uncapped", "no_cap"}:
        return None
    if p.endswith("%"):
        p = p[:-1].strip()
    val = float(p)
    if val > 1.0:
        val /= 100.0
    if not (0.0 < val <= 1.0):
        raise SystemExit(f"cap {raw!r} must be none or a fraction in (0, 1]")
    return float(val)


def normalize_caps(raw: str) -> list[float | None]:
    out: list[float | None] = []
    seen: set[str] = set()
    for part in _cap25.parse_csv(raw):
        share = parse_cap_token(part)
        key = "none" if share is None else f"{share:.6f}"
        if key in seen:
            continue
        seen.add(key)
        out.append(share)
    if not out:
        raise SystemExit("no caps selected")
    return out


def cap_slug(share: float | None) -> str:
    if share is None:
        return "nocap"
    pct = int(round(float(share) * 100))
    return f"cap{pct}"


def cap_out_root(root: Path, share: float | None) -> Path:
    return root / cap_slug(share)


def job_id(share: float | None, area: int, kind: str, method: str) -> str:
    return f"{cap_slug(share)}/{area}m/{kind}/{method}"


def method_output(root: Path, share: float | None, kind: str, area: int, method: str) -> Path:
    stem = "campaign" if kind == "campaign" else "n100"
    return cap_out_root(root, share) / f"{area}m" / f"{stem}_{method}.json"


def _share_matches(got: object, want: float | None) -> bool:
    if want is None:
        return got is None
    try:
        return abs(float(got) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return False


def _area_ok(payload: dict, area: float) -> bool:
    raw = payload.get("area_m")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return False
    return abs(float(raw[0]) - area) < 1e-6 and abs(float(raw[1]) - area) < 1e-6


def output_complete(
    path: Path,
    *,
    kind: str,
    method: str,
    area: float,
    share: float | None,
    n_runs: int,
    n_scenarios: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(payload, dict) or payload.get("checkpoint") is True:
        return False
    if abs(float(payload.get("b_sys_hz") or 0) - B_SYS_HZ) > 1.0:
        return False
    if not _share_matches(payload.get("max_bw_share"), share):
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
            if not isinstance(stats, dict) or int(stats.get("n") or -1) != int(n_runs):
                return False
        return seen == expected_point_keys(area)
    if kind == "n100":
        if int(payload.get("n_scenarios") or -1) != int(n_scenarios):
            return False
        stats = (payload.get("by_method") or {}).get(method)
        if not isinstance(stats, dict) or int(stats.get("n") or -1) != int(n_scenarios):
            return False
        per = stats.get("per_seed_Mbps") or []
        return len(per) == int(n_scenarios)
    raise SystemExit(f"unknown kind {kind!r}")


def build_plan(
    methods: list[str],
    areas: list[int],
    experiments: list[str],
    caps: list[float | None],
) -> list[tuple[str, str, str, int, str, str]]:
    require_banks(areas, experiments)
    root = default_root()
    rows: list[tuple[str, str, str, int, str, str]] = []
    for share in caps:
        slug = cap_slug(share)
        for area in areas:
            for kind in experiments:
                for method in methods:
                    path = method_output(root, share, kind, area, method)
                    rows.append(
                        (
                            job_id(share, area, kind, method),
                            slug,
                            kind,
                            area,
                            method,
                            rel_to_repo(path),
                        )
                    )
    return rows


def job_settings(
    *,
    kind: str,
    area: int,
    method: str,
    share: float | None,
    n_runs: int,
    n_scenarios: int,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "bandwidth_preset": "10mhz",
        "max_bw_share": share,
        "cap_slug": cap_slug(share),
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
    share: float | None,
    out_rel: str,
    n_runs: int,
    n_scenarios: int,
    plots: bool,
    force: bool,
) -> list[str]:
    slug = cap_slug(share)
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
        common.extend(["--max-bw-share", str(float(share))])
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
            f"{OUT_NAME}/{slug}/{int(area)}m/figures/n100_{method}",
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


def bind_settings(path: Path, expected: dict, *, force: bool) -> int:
    return _cap25.bind_settings(path, expected, force=force)


def merge_campaigns(
    root: Path, share: float | None, area: int, methods: list[str], n_runs: int
) -> Path | None:
    cap_root = cap_out_root(root, share)
    present: list[tuple[str, dict]] = []
    for method in methods:
        path = method_output(root, share, "campaign", area, method)
        if output_complete(
            path,
            kind="campaign",
            method=method,
            area=float(area),
            share=share,
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
        f"Merged 10 MHz / {cap_slug(share)} per-method campaigns "
        f"({', '.join(names)}). " + str(header.get("note") or "")
    )
    out = cap_root / f"{area}m" / "campaign_all.json"
    write_campaign(header, out)
    return out


def merge_n100(
    root: Path, share: float | None, area: int, methods: list[str], n_scenarios: int
) -> Path | None:
    present: list[tuple[str, dict]] = []
    for method in methods:
        path = method_output(root, share, "n100", area, method)
        if output_complete(
            path,
            kind="n100",
            method=method,
            area=float(area),
            share=share,
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
        f"Merged 10 MHz / {cap_slug(share)} bank evals ({', '.join(names)}). "
        + str(header.get("note") or "")
    )
    out = cap_out_root(root, share) / f"{area}m" / "n100_all.json"
    write_eval(header, out)
    return out


def merge_all(
    root: Path,
    caps: list[float | None],
    areas: list[int],
    methods: list[str],
    n_runs: int,
    n_scenarios: int,
) -> list[Path]:
    wrote: list[Path] = []
    for share in caps:
        for area in areas:
            camp = merge_campaigns(root, share, area, methods, n_runs)
            bank = merge_n100(root, share, area, methods, n_scenarios)
            if camp is not None:
                wrote.append(camp)
            if bank is not None:
                wrote.append(bank)
    return wrote


def save_invocation(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_invocation(path: Path) -> dict:
    payload = _load_json(path)
    caps_raw = payload["caps"]
    if isinstance(caps_raw, list):
        caps = [parse_cap_token(str(c)) for c in caps_raw]
    else:
        caps = normalize_caps(_as_csv(str(caps_raw)))
    def _list_csv(value: object) -> str:
        if isinstance(value, list):
            return ",".join(str(x) for x in value)
        return _as_csv(value)

    return {
        "methods": normalize_methods(_list_csv(payload["methods"])),
        "areas": normalize_areas(_list_csv(payload["areas"])),
        "experiments": normalize_experiments(_list_csv(payload["experiments"])),
        "caps": caps,
        "n_runs": int(payload["n_runs"]),
        "n_scenarios": int(payload["n_scenarios"]),
    }


def render_status(
    *,
    methods: list[str],
    areas: list[int],
    experiments: list[str],
    caps: list[float | None],
    n_runs: int,
    n_scenarios: int,
    ledger: Path,
    pidfile: Path | None,
    current_job: str,
    root: Path,
) -> str:
    last = ledger_map(ledger)
    running = pid_state(pidfile).startswith("running")
    cap_labels = ", ".join(cap_slug(c) for c in caps)
    lines = [
        pid_state(pidfile),
        f"output: {root}",
        f"protocol: 10 MHz, caps=({cap_labels}), n_runs={n_runs}, n_scenarios={n_scenarios}",
        f"{'job':<44} {'state':<14} detail",
    ]
    for job, slug, kind, area, method, out_rel in build_plan(
        methods, areas, experiments, caps
    ):
        share = parse_cap_token(slug)
        path = resolve(out_rel)
        done = output_complete(
            path,
            kind=kind,
            method=method,
            area=float(area),
            share=share,
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
            state, detail = "mismatch", "delete .settings.json and checkpoint to change flags"
        elif running and current_job == job:
            state, detail = "running", rel_to_repo(ckpt) if ckpt.is_file() else "started"
        elif prev == "interrupted":
            state, detail = "interrupted", "rerun to resume"
        elif prev == "fail":
            state, detail = "fail", "rerun to retry"
        elif prev == "start" or ckpt.is_file():
            state = "incomplete"
            detail = "checkpoint will resume" if ckpt.is_file() else "started"
        else:
            state, detail = "pending", ""
        lines.append(f"{job:<44} {state:<14} {detail}")
    return "\n".join(lines) + "\n"


def _selection_from_args(args: argparse.Namespace) -> tuple[
    list[str], list[int], list[str], list[float | None], int, int
]:
    if getattr(args, "from_invocation", None):
        inv = load_invocation(resolve(args.from_invocation))
        return (
            inv["methods"],
            inv["areas"],
            inv["experiments"],
            inv["caps"],
            inv["n_runs"],
            inv["n_scenarios"],
        )
    return (
        normalize_methods(args.methods),
        normalize_areas(args.areas),
        normalize_experiments(args.experiments),
        normalize_caps(args.caps),
        int(args.n_runs),
        int(args.n_scenarios),
    )


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods",
        default="random,kmeans,sca,sca_multistart,sca_anchor",
    )
    parser.add_argument("--areas", default="100,500")
    parser.add_argument("--experiments", default="n100,campaign")
    parser.add_argument("--caps", default=DEFAULT_CAPS)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--from-invocation", default=None)


def cmd_plan(args: argparse.Namespace) -> int:
    methods, areas, experiments, caps, _nr, _ns = _selection_from_args(args)
    for row in build_plan(methods, areas, experiments, caps):
        print("\t".join(str(part) for part in row))
    return 0


def cmd_argv(args: argparse.Namespace) -> int:
    method = normalize_methods(args.method)[0]
    share = parse_cap_token(args.cap)
    for part in build_argv(
        kind=args.kind,
        area=int(args.area),
        method=method,
        share=share,
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
    share = parse_cap_token(args.cap)
    ok = output_complete(
        resolve(args.path),
        kind=args.kind,
        method=method,
        area=float(args.area),
        share=share,
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
    share = parse_cap_token(args.cap)
    expected = job_settings(
        kind=args.kind,
        area=int(args.area),
        method=method,
        share=share,
        n_runs=int(args.n_runs),
        n_scenarios=int(args.n_scenarios),
    )
    return bind_settings(resolve(args.path), expected, force=bool(args.force))


def cmd_merge(args: argparse.Namespace) -> int:
    methods, areas, experiments, caps, n_runs, n_scenarios = _selection_from_args(args)
    wrote = merge_all(resolve(args.root), caps, areas, methods, n_runs, n_scenarios)
    if not wrote:
        print("merge: no complete method files yet")
        return 0
    for path in wrote:
        print(f"merged {rel_to_repo(path)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    methods, areas, experiments, caps, n_runs, n_scenarios = _selection_from_args(args)
    current = ""
    if args.current_job:
        current_path = resolve(args.current_job)
        if current_path.is_file():
            current = current_path.read_text(encoding="utf-8").strip()
    text = render_status(
        methods=methods,
        areas=areas,
        experiments=experiments,
        caps=caps,
        n_runs=n_runs,
        n_scenarios=n_scenarios,
        ledger=resolve(args.ledger),
        pidfile=resolve(args.pidfile) if args.pidfile else None,
        current_job=current,
        root=resolve(args.root),
    )
    sys.stdout.write(text)
    if args.progress:
        progress = resolve(args.progress)
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(text, encoding="utf-8")
    return 0


def cmd_save_invocation(args: argparse.Namespace) -> int:
    methods, areas, experiments, caps, n_runs, n_scenarios = _selection_from_args(args)
    save_invocation(
        resolve(args.path),
        {
            "methods": methods,
            "areas": areas,
            "experiments": experiments,
            "caps": [cap_slug(c) for c in caps],
            "n_runs": n_runs,
            "n_scenarios": n_scenarios,
        },
    )
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    caps = normalize_caps("0,15%,25,nocap")
    if caps != [None, 0.15, 0.25]:
        raise SystemExit(f"cap normalize: {caps}")
    argv = build_argv(
        kind="n100",
        area=100,
        method="sca",
        share=None,
        out_rel="results/run_10mhz_cap_sweep/nocap/100m/n100_sca.json",
        n_runs=100,
        n_scenarios=100,
        plots=False,
        force=False,
    )
    if "--no-per-link-cap" not in argv:
        raise SystemExit("nocap argv missing --no-per-link-cap")
    argv2 = build_argv(
        kind="campaign",
        area=500,
        method="sca",
        share=0.15,
        out_rel="results/run_10mhz_cap_sweep/cap15/500m/campaign_sca.json",
        n_runs=20,
        n_scenarios=100,
        plots=False,
        force=False,
    )
    if "0.15" not in argv2:
        raise SystemExit("cap15 argv missing share")
    plan = build_plan(["random"], [100], ["n100"], [None, 0.25])
    if len(plan) != 2:
        raise SystemExit(f"expected 2 jobs, got {len(plan)}")
    print("self-check ok")
    return 0


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
    argv.add_argument("--cap", required=True, help="cap token: none, nocap, cap15, 15, 0.15, ...")
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
    complete.add_argument("--cap", required=True)
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
    bind.add_argument("--cap", required=True)
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


def cap_token_for_cli(share: float | None) -> str:
    return "none" if share is None else cap_slug(share)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
