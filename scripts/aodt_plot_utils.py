"""Shared AoDT ($T_k$) sweep plotting: $T_k \\geq 1.2$ only, zoomed y-axis."""

from __future__ import annotations

from typing import Callable

AODT_X_MIN = 1.2
ZERO_EPS = 1e-6
ZERO_TICK_HALF_WIDTH = 0.08


def filter_aodt_points(points: list[dict], method: str) -> list[dict]:
    del method  # same sweep grid for every method
    return [p for p in points if float(p["x"]) >= AODT_X_MIN]


def collect_aodt_series(
    points: list[dict],
    method: str,
    mean_at: Callable[[dict, str], float | None],
) -> tuple[list[float], list[float], list[float]]:
    xs_ok: list[float] = []
    ys_ok: list[float] = []
    xs_zero: list[float] = []
    for p in filter_aodt_points(points, method):
        m = mean_at(p, method)
        if m is None:
            continue
        x = float(p["x"])
        if m <= ZERO_EPS:
            xs_zero.append(x)
        else:
            xs_ok.append(x)
            ys_ok.append(float(m))
    return xs_ok, ys_ok, xs_zero


def aodt_ylim(ys: list[float]) -> tuple[float, float]:
    if not ys:
        return 8.0, 8.5
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 0.02)
    return lo - 0.08 * span, hi + 0.14 * span


def aodt_xticks(points: list[dict], methods: tuple[str, ...] | list[str]) -> list[float]:
    del methods
    return sorted({float(p["x"]) for p in points if float(p["x"]) >= AODT_X_MIN})


def infeasible_xs_for_method(xs_zero: list[float], method: str) -> list[float]:
    del method
    return [x for x in xs_zero if float(x) >= AODT_X_MIN - 1e-9]


def draw_infeasible_ticks(
    ax,
    xs_zero: list[float],
    y_floor: float,
    *,
    color: str = "0.5",
) -> None:
    for x in xs_zero:
        ax.hlines(
            y_floor,
            x - ZERO_TICK_HALF_WIDTH,
            x + ZERO_TICK_HALF_WIDTH,
            colors=color,
            linestyles=(0, (4, 3)),
            linewidth=1.1,
            zorder=2,
        )
