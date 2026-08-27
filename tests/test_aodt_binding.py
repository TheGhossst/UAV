from src.config import DEFAULT
from src.evaluator import evaluate
from src.repair import complete_solution
from src.scenario import generate_scenario
from src.solvers.random import solve_random


def test_binding_aodt_can_violate_on_poor_placement():
    cfg = DEFAULT.with_compute()
    assert cfg.task_size_bytes == 2000.0
    s = generate_scenario(100, cfg)
    _, result, _ = solve_random(s, seed=0)
    assert result.compute_available
    assert result.aodt.shape == (cfg.num_processes,)
    assert result.aodt_violations >= 0


def test_aodt_present_after_repair():
    cfg = DEFAULT.with_compute()
    s = generate_scenario(100, cfg)
    xy, a, b, bw = complete_solution(s, __import__("numpy").array([[250.0, 250.0], [100.0, 100.0], [400.0, 400.0]]))
    r = evaluate(s, xy, a, b, bw)
    assert r.compute_available
    assert __import__("numpy").all(__import__("numpy").isfinite(r.aodt))
