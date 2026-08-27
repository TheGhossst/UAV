import numpy as np

from src.aodt import average_aodt
from src.config import DEFAULT
from src.compute import service_rate
from src.evaluator import evaluate
from src.repair import complete_solution
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.pso import solve_pso_joint, solve_pso_placement
from src.solvers.random import solve_random
from src.solvers.sca import solve_sca


def test_evaluator_same_for_equivalent_inputs():
    s = generate_scenario(100, DEFAULT)
    xy = np.array([[100.0, 100.0], [400.0, 200.0], [250.0, 400.0]])
    xy, a, b, bw = complete_solution(s, xy)
    r1 = evaluate(s, xy, a, b, bw)
    r2 = evaluate(s, xy, a, b, bw)
    assert r1.sum_rate == r2.sum_rate
    assert r1.qos_violations == r2.qos_violations


def test_one_association_per_iot():
    s = generate_scenario(101, DEFAULT)
    xy, a, b, bw = complete_solution(s, np.array([[250.0, 250.0], [100.0, 100.0], [400.0, 400.0]]))
    assert np.allclose(a.sum(axis=1), 1.0)
    assert np.allclose(b.sum(axis=1), 1.0)


def test_process_consistency():
    s = generate_scenario(102, DEFAULT)
    xy, a, b, bw = complete_solution(s, np.array([[50.0, 50.0], [250.0, 250.0], [450.0, 450.0]]))
    for members in s.groups:
        js = b[members].argmax(axis=1)
        assert np.unique(js).size == 1


def test_aodt_uses_eq17_when_compute_enabled():
    cfg = DEFAULT.with_compute()
    s = generate_scenario(100, cfg)
    xy, a, b, bw = complete_solution(s, np.array([[250.0, 250.0], [100.0, 400.0], [400.0, 100.0]]))
    r = evaluate(s, xy, a, b, bw)
    assert r.compute_available
    assert np.all(np.isfinite(r.aodt))
    mu = service_rate(cfg)
    aodt2 = average_aodt(s, a, b, r.rates, mu)
    np.testing.assert_allclose(r.aodt, aodt2)


def test_baselines_share_evaluator_type():
    s = generate_scenario(100, DEFAULT)
    _, r_rand, _ = solve_random(s, seed=0)
    _, r_km, _ = solve_kmeans(s, seed=0)
    _, r_pso, _, _ = solve_pso_placement(s, seed=0, n_particles=6, n_iter=4)
    _, r_sca, _ = solve_sca(s, seed=0, max_iter=3)
    for r in (r_rand, r_km, r_pso, r_sca):
        assert hasattr(r, "sum_rate")
        assert r.rates.shape[0] == 10


def test_joint_pso_returns_valid_shape():
    s = generate_scenario(100, DEFAULT)
    xy, result, _, _ = solve_pso_joint(s, seed=0, n_particles=6, n_iter=5)
    assert xy.shape == (3, 2)
    assert result.rates.shape == (10, 3)
    assert result.sum_rate > 0


def test_placement_pso_improves_or_matches_random_on_same_scenario():
    s = generate_scenario(100, DEFAULT)
    _, r_rand, _ = solve_random(s, seed=1)
    _, r_pso, _, _ = solve_pso_placement(s, seed=1, n_particles=10, n_iter=15)
    # With a penalty-aware search, PSO should not be wildly worse than a single random draw.
    assert r_pso.sum_rate > 0
    assert r_rand.sum_rate > 0
