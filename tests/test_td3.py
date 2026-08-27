from dataclasses import replace as dc_replace

import numpy as np

from src.config import DEFAULT
from src.evaluator import evaluate
from src.repair import nearest_association
from src.scenario import generate_scenario
from src.solvers.td3 import UAVAoDTEnv, solve_td3, train_td3


def test_td3_env_uses_evaluator_and_alg2_action_parse():
    s = generate_scenario(100, DEFAULT)
    env = UAVAoDTEnv(s, n_uav=3, seed=0)
    obs = env.reset()
    assert obs.shape == (env.state_dim,)
    action = np.zeros(env.action_dim)
    ns, reward, result = env.step(action)
    assert ns.shape == (env.state_dim,)
    assert np.ndim(reward) == 0
    assert result.rates.shape == (10, 3)


def test_td3_short_train_runs():
    s = generate_scenario(100, DEFAULT)
    agent, env, log = train_td3(s, seed=0, n_uav=3, total_steps=8)
    assert len(log.rewards) == 8
    assert agent.action_dim == env.action_dim
    assert log.best_result is not None and log.best_xy is not None


def test_zero_action_means_nearest_uav_association():
    s = generate_scenario(100, DEFAULT)
    env = UAVAoDTEnv(s, n_uav=3, seed=0)
    env.reset()
    xy, a, _b, _bw = env.parse_action(np.zeros(env.action_dim))
    np.testing.assert_array_equal(a, nearest_association(s.iot_xy, xy))


def test_actions_stay_qos_feasible_where_equal_split_is():
    """The agent's bandwidth request must not starve links below R_min."""
    s = generate_scenario(100, DEFAULT)
    env = UAVAoDTEnv(s, n_uav=3, seed=0)
    env.reset()
    assert env.last_result is not None and env.last_result.qos_violations == 0
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = np.zeros(env.action_dim)
        # Perturb only the bandwidth block, so placement stays where it was.
        action[2 * env.j + 2 * env.i * env.j :] = rng.uniform(-1.0, 1.0, size=env.i * env.j)
        _xy, a, b, bw = env.parse_action(action)
        assert evaluate(s, env.uav_xy, a, b, bw).qos_violations == 0


def test_solve_td3_reports_greedy_eval_not_training_archive(monkeypatch):
    """Training-search best must not be copied through as the reported result."""
    s = generate_scenario(100, DEFAULT)
    archive_xy = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    real_train = train_td3

    def wrapped(*args, **kwargs):
        agent, env, log = real_train(*args, **kwargs)
        log.best_xy = archive_xy.copy()
        log.best_result = dc_replace(
            env.last_result,
            sum_rate=1e12,
            qos_violations=0,
            bw_excess=0.0,
            sep_violations=0,
            cpu_unstable=0,
            aodt_violations=0,
            association_violations=0,
            processing_violations=0,
            process_consistency_violations=0,
        )
        return agent, env, log

    monkeypatch.setattr("src.solvers.td3.train_td3", wrapped)
    xy, result, _, log = solve_td3(
        s, seed=0, n_uav=3, total_steps=8, greedy_steps=2, n_restarts=2
    )
    assert log is not None
    assert not np.allclose(xy, archive_xy)
    assert result.sum_rate != 1e12
    assert xy.shape == (3, 2)
    assert np.all(xy >= 0.0)


def test_state_carries_per_link_spectral_efficiency():
    s = generate_scenario(100, DEFAULT)
    env = UAVAoDTEnv(s, n_uav=3, seed=0)
    env.reset()
    assert env.state_dim >= env.i * env.j
    near = env.spectral_efficiency(np.array([[250.0, 250.0]] * 3))
    far = env.spectral_efficiency(np.array([[2500.0, 2500.0]] * 3))
    assert np.all(near > far)
