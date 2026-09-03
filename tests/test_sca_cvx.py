"""CVX+MOSEK SCA backend. MATLAB tests skip when MATLAB is not installed."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import DEFAULT
from src.repair import complete_solution
from src.scenario import generate_scenario
from src.solvers.sca import _assemble_convexified_lp, _convexified_lp
from src.solvers.sca_cvx import _bounds_to_vectors, matlab_is_available


def test_assembled_lp_matches_highs_decode():
    s = generate_scenario(100, DEFAULT)
    xy, a, proc, _ = complete_solution(
        s, np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    )
    from src.solvers.sca import _allocate_bandwidth

    bw = _allocate_bandwidth(s, xy, a, proc)
    lp = _assemble_convexified_lp(s, xy, bw, a, proc, trust=25.0)
    assert lp is not None
    assert lp.c.ndim == 1
    assert lp.a_ub.shape[1] == lp.c.size
    assert lp.b_ub.shape == (lp.a_ub.shape[0],)
    assert len(lp.bounds) == lp.c.size
    solved = _convexified_lp(s, xy, bw, a, proc, trust=25.0)
    assert solved is not None


def test_bounds_to_vectors_maps_none_to_inf():
    lb, ub = _bounds_to_vectors([(0.0, 1.0), (0.0, None), (None, 5.0)])
    assert lb[0] == 0.0 and ub[0] == 1.0
    assert np.isinf(ub[1]) and ub[1] > 0
    assert np.isinf(lb[2]) and lb[2] < 0


@pytest.mark.skipif(not matlab_is_available(), reason="MATLAB not installed")
def test_cvx_mosek_agrees_with_highs_on_one_lp():
    from src.solvers.sca import _allocate_bandwidth
    from src.solvers.sca_cvx import MatlabCvxSession, _convexified_lp_cvx

    s = generate_scenario(100, DEFAULT)
    xy, a, proc, _ = complete_solution(
        s, np.array([[120.0, 130.0], [380.0, 200.0], [250.0, 400.0]])
    )
    bw = _allocate_bandwidth(s, xy, a, proc)
    highs = _convexified_lp(s, xy, bw, a, proc, trust=25.0)
    assert highs is not None
    with MatlabCvxSession() as session:
        cvx = _convexified_lp_cvx(s, xy, bw, a, proc, 25.0, session)
    assert cvx is not None
    xy_h, bw_h = highs
    xy_c, bw_c = cvx
    np.testing.assert_allclose(xy_c, xy_h, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(bw_c, bw_h, rtol=1e-4, atol=1.0)


@pytest.mark.skipif(not matlab_is_available(), reason="MATLAB not installed")
def test_sca_cvx_runs():
    from src.solvers.sca_cvx import solve_sca_cvx

    s = generate_scenario(100, DEFAULT)
    xy, result, _ = solve_sca_cvx(s, seed=0, max_iter=3)
    assert xy.shape == (3, 2)
    assert result.sum_rate > 0
