"""Reporting helpers. Keep arithmetic explicit so Mbps/bit/s cannot slip 10x."""

from __future__ import annotations


def gap_vs_uncapped(uncapped_rate: float, capped_rate: float) -> tuple[float, float]:
    """gap = uncapped − capped, and percent of the uncapped rate.

    Both rates must already be in the same unit (bit/s or Mbps).
    Do not rescale by 10 or 1000 here.
    """
    gap = float(uncapped_rate) - float(capped_rate)
    if abs(float(uncapped_rate)) < 1e-30:
        return gap, float("nan")
    pct = 100.0 * gap / float(uncapped_rate)
    return gap, pct
