# Completed 2026-09-01 — no-TD3 audit under results/run_20260901/

Re-run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_full_audit_no_td3.ps1
```

Refresh doc only:

```bash
python -m src.status_sync --results results/run_20260901
```

TD3 batch (slow, ~6 h) archived at `results/run_20260831/` — use `scripts/run_full_audit.ps1` if needed.
