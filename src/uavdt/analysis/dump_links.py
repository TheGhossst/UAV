"""Leftover-dump link SE quality at a frozen geometry (mechanism probes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uavdt.models import Scenario
from uavdt.resources import nearest_association
from uavdt.sca.linearize import spectral_efficiency
from uavdt.sca_anchor import leftover_dump_slots
from uavdt.scenario import make_uav_xyz_m


def zenith_se_max_bit_per_hz(scenario: Scenario) -> float:
    """Max SE achievable by hovering a UAV above any single IoT (H fixed)."""
    cfg = scenario.cfg
    iot = scenario.iot_xyz_m
    h = float(cfg.uav_height_m)
    best = 0.0
    for idx in range(iot.shape[0]):
        uav = make_uav_xyz_m(iot[idx : idx + 1, :2], h)
        se = float(spectral_efficiency(iot[idx : idx + 1], uav, cfg)[0, 0])
        best = max(best, se)
    return best


@dataclass(frozen=True)
class DumpLinkSEAudit:
    k_dump: int
    se_max_bit_per_hz: float
    se_threshold_bit_per_hz: float
    n_good_dump_links: int
    dump_links: list[dict]
    """Each entry: i, j, se_bit_per_hz, good, rank (1..k)."""

    @property
    def frac_good(self) -> float:
        if self.k_dump <= 0:
            return float("nan")
        return float(self.n_good_dump_links) / float(self.k_dump)


def dump_link_se_audit(
    scenario: Scenario,
    uav_xyz_m: np.ndarray,
    association: np.ndarray | None = None,
    *,
    se_quality_frac: float = 0.98,
) -> DumpLinkSEAudit:
    """At fixed ``q``, rank associated links by SE and audit the top ``k`` dump slots.

    ``k = ceil(1 / max_bw_share)`` matches the leftover-dump cap story. A dump
    link is *good* when ``SE_ij(q) >= se_quality_frac * SE_max``, with
    ``SE_max`` the zenith-over-IoT ceiling from ``zenith_se_max_bit_per_hz``.
    """
    if not (0.0 < float(se_quality_frac) <= 1.0):
        raise ValueError("se_quality_frac must be in (0, 1]")
    cfg = scenario.cfg
    uav = np.asarray(uav_xyz_m, dtype=float)
    a = (
        np.asarray(association, dtype=float)
        if association is not None
        else nearest_association(scenario.iot_xyz_m, uav)
    )
    assoc = a > 0.5
    se = spectral_efficiency(scenario.iot_xyz_m, uav, cfg)
    k = leftover_dump_slots(cfg.max_bw_share)
    se_max = zenith_se_max_bit_per_hz(scenario)
    thr = float(se_quality_frac) * se_max

    pairs: list[tuple[float, int, int]] = []
    for i in range(se.shape[0]):
        for j in range(se.shape[1]):
            if assoc[i, j]:
                pairs.append((float(se[i, j]), i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    top = pairs[: max(0, min(k, len(pairs)))]

    dump_links: list[dict] = []
    n_good = 0
    for rank, (se_ij, i, j) in enumerate(top, start=1):
        good = bool(se_ij >= thr - 1e-15)
        n_good += int(good)
        dump_links.append(
            {
                "rank": rank,
                "i": int(i),
                "j": int(j),
                "se_bit_per_hz": se_ij,
                "good": good,
            }
        )
    return DumpLinkSEAudit(
        k_dump=int(k),
        se_max_bit_per_hz=se_max,
        se_threshold_bit_per_hz=thr,
        n_good_dump_links=int(n_good),
        dump_links=dump_links,
    )
