"""Rewrite measured tables in docs/PROJECT_STATUS_AND_PAPER_ANALYSIS.md.

    python -m scripts.update_status_doc
    python -m scripts.update_status_doc --results results/run_20260830_rng_fixed
"""

from src.status_sync import main

if __name__ == "__main__":
    raise SystemExit(main())
