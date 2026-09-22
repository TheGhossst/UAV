"""Campaign method ids and display labels for plots / PPT."""

from __future__ import annotations

# LaTeX / PPT display strings (hyphenated frozen-SCA per project naming).
METHOD_LABELS: dict[str, str] = {
    "sca": "SCA",
    "frozen_sca": "frozen-SCA",
    "sca_joint": "SCA-joint",
    "sca_multistart": "SCA multi-start",
    "sca_anchor": "SCA zenith-anchor",
    "sca_medoid": "SCA medoid",
    "sca_continuous": "SCA continuous",
    "td3": "TD3",
    "random": "Random",
    "kmeans": "K-means",
    "pso": "PSO",
    # Legacy eval JSON keys (pre-rename).
    "sca_dynamic": "SCA",
}


def normalize_by_method(by_method: dict) -> dict:
    """Map legacy eval keys to current names.

    Old bank files used ``sca`` for frozen association/processing and
    ``sca_dynamic`` for the block-coordinate extension. Current names:
    ``frozen_sca`` and ``sca`` (dynamic assignment).
    """
    if not by_method:
        return by_method
    out = dict(by_method)
    if "sca_dynamic" in out:
        if "sca" in out and "frozen_sca" not in out:
            out["frozen_sca"] = out["sca"]
        out["sca"] = out.pop("sca_dynamic")
    return out


def label_for(method: str) -> str:
    return METHOD_LABELS.get(method, method)
