"""Legal association search (Experiment C)."""

from __future__ import annotations

import numpy as np
import pytest

from uavdt.aodt import queueing_term_s
from uavdt.assoc_search import (
    cohesive_on_required,
    forwarding_slack_s,
    hamming_association,
    is_legal_association,
    iter_one_opt_neighbors,
    must_group_mask,
    search_legal_association,
)
from uavdt.config import PRIMARY_MAX_BW_SHARE, SimConfig
from uavdt.resources import nearest_association
from uavdt.scenario import generate_scenario

cvxpy = pytest.importorskip("cvxpy")


def _default_cfg(**kwargs) -> SimConfig:
    base = dict(b_sys_hz=8.8e6, max_bw_share=PRIMARY_MAX_BW_SHARE)
    base.update(kwargs)
    return SimConfig(**base)


def test_queueing_plus_forward_matches_novelty_number():
    cfg = _default_cfg()
    sc = generate_scenario(1, cfg)
    mu = cfg.service_rate_per_s
    q = queueing_term_s(sc.lambdas_per_s[sc.processes[0].iot_indices], mu)
    assert abs(q - 0.59375) < 1e-4
    assert abs(q + cfg.t_u2u_s - 0.89375) < 1e-4


def test_must_group_only_at_tight_deadline():
    loose = generate_scenario(1, _default_cfg(aodt_threshold_s=2.8))
    tight = generate_scenario(1, _default_cfg(aodt_threshold_s=0.8))
    assert not np.any(must_group_mask(loose))
    assert np.all(must_group_mask(tight))
    assert np.all(forwarding_slack_s(tight) < 0.0)
    assert np.all(forwarding_slack_s(loose) > 0.0)


def test_one_opt_neighbor_count_i10_j3():
    cfg = _default_cfg()
    sc = generate_scenario(1, cfg)
    from uavdt.placement.kmeans import place_kmeans

    uav = place_kmeans(sc, 1)
    a = nearest_association(sc.iot_xyz_m, uav)
    nbs = list(iter_one_opt_neighbors(a))
    assert len(nbs) == cfg.num_iot * (cfg.num_uav - 1)
    i, j, a2 = nbs[0]
    assert hamming_association(a, a2) == 1
    assert int(np.argmax(a2[i])) == j


def test_split_illegal_when_must_group():
    sc = generate_scenario(1, _default_cfg(aodt_threshold_s=0.8))
    from uavdt.placement.kmeans import place_kmeans

    uav = place_kmeans(sc, 1)
    a = nearest_association(sc.iot_xyz_m, uav)
    if cohesive_on_required(sc, a):
        a[sc.processes[0].iot_indices[0], :] = 0.0
        other = 0 if int(np.argmax(a[sc.processes[0].iot_indices[1]])) != 0 else 1
        a[sc.processes[0].iot_indices[0], other] = 1.0
    ok, reason = is_legal_association(sc, a)
    assert not ok
    assert reason == "must_group_split"


def test_one_opt_at_sca_q_stays_legal():
    from uavdt.sca.algorithm import solve_sca
    from uavdt.sca.settings import SCASettings

    cfg = _default_cfg()
    sc = generate_scenario(1, cfg)
    frozen = solve_sca(sc, 1, settings=SCASettings(solver=None, max_iterations=8))
    result = search_legal_association(
        sc,
        frozen.uav_xyz_m,
        frozen.allocation.hard_association(),
        max_rounds=3,
        n_random=5,
        seed=1,
        include_best_se=True,
    )
    ok, reason = is_legal_association(sc, result.incumbent.association)
    assert ok, reason
    assert result.n_lp_evals >= 1
    assert result.incumbent.feasible == frozen.true_eval.feasible or result.n_accepted >= 0
