# Scripts layout

Run from the **repository root** (`C:\code\UAV`). Paths below are relative to `scripts/`.

| Folder | Purpose |
|--------|---------|
| `lib/` | Shared helpers (`paired_winrate.py`, `aodt_plot_utils.py`, `repo.py`) |
| `orchestration/` | Full pipelines, PPT refresh, PowerShell resume helpers |
| `campaigns/` | Campaign and bank eval runners (`run_*`) |
| `analyze/` | Readouts and paired statistics (`analyze_*`) |
| `plot/` | Figures for paper and PPT (`plot_*`) |
| `td3/` | TD3 training, export, diagnostics |
| `tools/` | One-off diagnostics, bandwidth search, comparisons |
| `sync/` | Laptop ↔ college zip/SSH sync (PowerShell) |
| `remote/` | Shell helpers on the college machine |
| `experiments/` | Isolated experiment packages (e.g. residual-on-SCA) |

## Common entry points

```powershell
python scripts/orchestration/run_full_regeneration_no_td3.py --resume
python scripts/orchestration/update_ppt_from_pulled.py --latexmk
.\scripts\sync\pull_uav_from_college.ps1 -ResultsOnly
```

Root-level `run_full_regeneration_no_td3.py` and `sync_*.ps1` / `pull_*.ps1` are thin forwarders to the paths above.
