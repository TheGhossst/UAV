"""Rename sca/sca_dynamic keys in eval JSON to frozen_sca/sca."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uavdt.experiments.method_labels import normalize_by_method


def migrate_payload(payload: dict) -> dict:
    by = payload.get("by_method")
    if by:
        payload["by_method"] = normalize_by_method(by)
    runs = payload.get("runs")
    if isinstance(runs, list) and any(r.get("method") == "sca_dynamic" for r in runs):
        for row in runs:
            m = row.get("method")
            if m == "sca_dynamic":
                row["method"] = "sca"
            elif m == "sca":
                row["method"] = "frozen_sca"
    for old_key, new_key in (
        ("sca_dynamic_minus_sca_Mbps", "sca_minus_frozen_sca_Mbps"),
        ("sca_dynamic_minus_baseline_Mbps", None),
    ):
        if old_key in payload and new_key:
            if new_key not in payload:
                payload[new_key] = payload.pop(old_key)
    if "sca_dynamic_minus_baseline_Mbps" in payload:
        payload.pop("sca_dynamic_minus_baseline_Mbps", None)
    if "methods" in payload and isinstance(payload["methods"], list):
        ms = list(payload["methods"])
        if "sca_dynamic" in ms:
            if "sca" in ms and "frozen_sca" not in ms:
                ms = ["frozen_sca" if x == "sca" else x for x in ms]
            ms = ["sca" if x == "sca_dynamic" else x for x in ms]
            payload["methods"] = ms
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = migrate_payload(payload)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"migrated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
