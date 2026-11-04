"""Place a framed legend inside the axes, off the plotted lines."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt


def place_inside_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    ncol_options: tuple[int, ...],
    fontsize: float,
    markerscale: float,
    handlelength: float,
    columnspacing: float,
    handletextpad: float,
    borderpad: float,
    labelspacing: float,
    frame_linewidth: float,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    style = dict(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        fontsize=fontsize,
        markerscale=markerscale,
        handlelength=handlelength,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        borderpad=borderpad,
        labelspacing=labelspacing,
    )

    best: tuple[float, float, float, int, list, list] | None = None
    marker_pad = 8.0
    for ncol in ncol_options:
        ordered_h, ordered_l = _row_major(handles, labels, ncol)
        probe = ax.legend(ordered_h, ordered_l, loc="upper left", ncols=ncol, **style)
        probe.set_in_layout(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bb = ax.get_window_extent(renderer)
        leg_bb = probe.get_window_extent(renderer)
        probe.remove()
        if leg_bb.width > ax_bb.width - 8 or leg_bb.height > ax_bb.height * 0.7:
            continue
        segments = _segments(ax)
        marker_pad = _marker_pad_px(ax, ax_bb)
        found = _search(ax_bb, leg_bb.width, leg_bb.height, segments, marker_pad)
        if found is None:
            continue
        clearance, x_frac, y_frac = found
        if best is None or clearance > best[0]:
            best = (clearance, x_frac, y_frac, ncol, ordered_h, ordered_l)

    if best is not None and best[0] >= marker_pad:
        _clearance, x_frac, y_frac, ncol, ordered_h, ordered_l = best
    else:
        # No open patch big enough for the box. Open a blank band and sit in it.
        ncol = ncol_options[-1]
        ordered_h, ordered_l = _row_major(handles, labels, ncol)
        probe = ax.legend(ordered_h, ordered_l, loc="upper left", ncols=ncol, **style)
        probe.set_in_layout(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bb = ax.get_window_extent(renderer)
        leg_bb = probe.get_window_extent(renderer)
        probe.remove()
        x_frac, y_frac = _open_band(ax, ax_bb, leg_bb.width, leg_bb.height, _segments(ax))

    legend = ax.legend(
        ordered_h,
        ordered_l,
        loc="lower left",
        bbox_to_anchor=(x_frac, y_frac),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        ncols=ncol,
        **style,
    )
    legend.set_in_layout(False)
    legend.set_zorder(10)
    legend.get_frame().set_linewidth(frame_linewidth)


def _row_major(handles: list, labels: list, ncol: int) -> tuple[list, list]:
    """Matplotlib fills legends down columns. Reorder so each row reads left to right."""
    nrow = math.ceil(len(handles) / ncol)
    order = [
        row * ncol + col
        for col in range(ncol)
        for row in range(nrow)
        if row * ncol + col < len(handles)
    ]
    return [handles[i] for i in order], [labels[i] for i in order]


def _segments(ax: plt.Axes) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for line in ax.get_lines():
        xs = list(line.get_xdata())
        ys = list(line.get_ydata())
        points = [tuple(ax.transData.transform((x, y))) for x, y in zip(xs, ys)]
        segments.extend(zip(points, points[1:]))
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue
        for x, y in offsets:
            point = tuple(ax.transData.transform((x, y)))
            segments.append((point, point))
    return segments


def _marker_pad_px(ax: plt.Axes, ax_bb) -> float:
    radius_pt = 0.0
    for line in ax.get_lines():
        marker = line.get_marker()
        if marker is None or marker in ("None", "none", ""):
            continue
        radius_pt = max(radius_pt, 0.5 * float(line.get_markersize()))
    for coll in ax.collections:
        sizes = coll.get_sizes()
        if sizes is None or len(sizes) == 0:
            continue
        radius_pt = max(radius_pt, math.sqrt(float(max(sizes)) / math.pi))
    # Window extents are in display pixels; one point is dpi/72 pixels.
    dpi = ax.figure.dpi
    radius_px = radius_pt * dpi / 72.0
    # Stay a few pixels off the marker, and off the frame.
    return max(radius_px + 4.0, 6.0, 0.012 * min(ax_bb.width, ax_bb.height))


def _clearance(
    x: float,
    y: float,
    width: float,
    height: float,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in segments:
        steps = 1 if (x1, y1) == (x2, y2) else 20
        for step in range(steps + 1):
            t = step / steps
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            dx = max(x - px, 0.0, px - (x + width))
            dy = max(y - py, 0.0, py - (y + height))
            dist = math.hypot(dx, dy)
            if dist < best:
                best = dist
    return best


def _search(ax_bb, leg_w: float, leg_h: float, segments, marker_pad: float):
    margin = max(4.0, 0.012 * min(ax_bb.width, ax_bb.height))
    x_lo = ax_bb.x0 + margin
    x_hi = ax_bb.x1 - leg_w - margin
    y_lo = ax_bb.y0 + margin
    y_hi = ax_bb.y1 - leg_h - margin
    if x_hi < x_lo or y_hi < y_lo:
        return None

    def score_at(x: float, y: float) -> tuple[float, float]:
        clearance = _clearance(x, y, leg_w, leg_h, segments)
        cx = (x + leg_w / 2.0 - ax_bb.x0) / ax_bb.width
        cy = (y + leg_h / 2.0 - ax_bb.y0) / ax_bb.height
        corner = min(
            math.hypot(cx, cy),
            math.hypot(cx - 1.0, cy),
            math.hypot(cx, cy - 1.0),
            math.hypot(cx - 1.0, cy - 1.0),
        )
        # Mild pull toward a corner when several spots are equally clear of the lines.
        return clearance - 8.0 * corner, clearance

    best_score = -math.inf
    best: tuple[float, float, float] | None = None
    steps = 18
    for i in range(steps):
        x = x_lo + (x_hi - x_lo) * i / (steps - 1)
        for j in range(steps):
            y = y_lo + (y_hi - y_lo) * j / (steps - 1)
            score, clearance = score_at(x, y)
            if clearance < marker_pad:
                continue
            if score > best_score:
                best_score = score
                best = (
                    clearance,
                    (x - ax_bb.x0) / ax_bb.width,
                    (y - ax_bb.y0) / ax_bb.height,
                )
    return best


def _open_band(ax: plt.Axes, ax_bb, leg_w: float, leg_h: float, segments) -> tuple[float, float]:
    """Grow the y-limit so a blank band appears, then center the legend in that band."""
    ys = []
    for (x1, y1), (x2, y2) in segments:
        ys.append((y1 - ax_bb.y0) / ax_bb.height)
        ys.append((y2 - ax_bb.y0) / ax_bb.height)
    mean_y = sum(ys) / len(ys) if ys else 0.5
    # Put the band on the side the curves are farther from.
    side = "lower" if mean_y >= 0.5 else "upper"

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if span <= 0.0:
        span = 1.0
    frac = min(0.4, leg_h / ax_bb.height + 0.07)
    extra = span * frac / (1.0 - frac)
    if side == "upper":
        ax.set_ylim(y0, y1 + extra)
        band_bottom = 1.0 - frac
    else:
        ax.set_ylim(y0 - extra, y1)
        band_bottom = 0.0

    leg_h_frac = leg_h / ax_bb.height
    leg_w_frac = leg_w / ax_bb.width
    y_frac = band_bottom + max(0.0, (frac - leg_h_frac) / 2.0)
    x_frac = max(0.01, (1.0 - leg_w_frac) / 2.0)
    return x_frac, y_frac
