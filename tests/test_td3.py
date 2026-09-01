from dataclasses import replace as dc_replace

import numpy as np

from src.config import DEFAULT, TD3_ASSOC_ACTION_SCALE
from src.evaluator import evaluate
from src.repair import nearest_association
from src.scenario import generate_scenario
from src.solvers.kmeans import solve_kmeans
from src.solvers.td3 import TD3Agent, UAVAoDTEnv, solve_td3, train_td3


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
    agent, env, log = train_td3(s, seed=0, n_uav=3, total_steps=8, device="cpu")
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
        kwargs.setdefault("device", "cpu")
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
        s, seed=0, n_uav=3, total_steps=8, greedy_steps=2, n_restarts=2, device="cpu"
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


def test_resolve_device_auto_prefers_cuda_when_available():
    from src.solvers.td3 import resolve_device

    cpu = resolve_device("cpu")
    assert cpu.type == "cpu"
    auto = resolve_device("auto")
    assert auto.type in ("cpu", "cuda")
    if __import__("torch").cuda.is_available():
        assert resolve_device("cuda").type == "cuda"


def test_overflow_bandwidth_action_does_not_explode_aodt_reward():
    """Infeasible-floor fallback must not treat 1+tanh as Hz (AoDT 1e16)."""
    cfg = DEFAULT.with_compute().with_radio_profile("table2")
    s = generate_scenario(100, cfg)
    env = UAVAoDTEnv(s, n_uav=3, seed=100)
    env.reset(
        uav_xy=np.array(
            [
                [cfg.area_x, cfg.area_y],
                [cfg.area_x - 10.0, cfg.area_y],
                [cfg.area_x, cfg.area_y - 10.0],
            ]
        )
    )
    action = np.zeros(env.action_dim)
    action[2 * env.j + 2 * env.i * env.j :] = -1.0
    action[2 * env.j + 2 * env.i * env.j] = 1.0
    _ns, reward, result = env.step(action)
    assert result.compute_available
    assert np.all(np.isfinite(result.aodt))
    assert float(np.max(result.aodt)) < 1e6
    assert result.aodt_excess < 1e6
    assert reward > -1e8
    assert result.min_assoc_rate > 1e-6


def test_train_log_records_in_episode_step_and_start_kind():
    s = generate_scenario(100, DEFAULT)
    _, _, log = train_td3(s, seed=0, n_uav=3, total_steps=8, episode_len=50, device="cpu")
    assert log.in_episode_steps == list(range(8))
    assert log.start_kinds == ["kmeans"] * 8


def test_train_log_alternates_start_kind_across_episodes():
    s = generate_scenario(100, DEFAULT)
    _, _, log = train_td3(s, seed=0, n_uav=3, total_steps=6, episode_len=3, device="cpu")
    assert log.in_episode_steps == [0, 1, 2, 0, 1, 2]
    assert log.start_kinds == ["kmeans"] * 3 + ["random"] * 3


def test_solve_td3_eval_trace_marks_pre_rollout_as_step_zero():
    s = generate_scenario(100, DEFAULT)
    trace: dict = {}
    solve_td3(
        s,
        seed=0,
        n_uav=3,
        total_steps=4,
        greedy_steps=2,
        n_restarts=2,
        device="cpu",
        eval_trace=trace,
    )
    assert trace["n_restarts"] == 2
    assert trace["greedy_steps"] == 2
    assert trace["winner"]["start_index"] in (0, 1)
    assert trace["winner"]["source"] in ("pre_rollout", "actor")
    pre = [r for r in trace["records"] if r["source"] == "pre_rollout"]
    assert len(pre) == 2
    assert all(r["step"] == 0 for r in pre)
    assert any(r["kind"] == "kmeans" and r["start_index"] == 0 for r in pre)


def test_default_env_keeps_distance_prior_and_assoc_scale():
    s = generate_scenario(100, DEFAULT)
    env = UAVAoDTEnv(s, n_uav=3, seed=0)
    assert env.distance_prior is True
    assert env.assoc_action_scale == TD3_ASSOC_ACTION_SCALE
    scaled = UAVAoDTEnv(s, n_uav=3, seed=0, assoc_action_scale=2.5, distance_prior=False)
    assert scaled.assoc_action_scale == 2.5
    assert scaled.distance_prior is False


def test_noop_kmeans_reset_matches_solve_kmeans():
    """n_restarts=1 / greedy_steps=0 control is k-means + complete_solution."""
    s = generate_scenario(100, DEFAULT)
    env = UAVAoDTEnv(s, n_uav=3, seed=100)
    env.reset()
    xy_k, res_k, _ = solve_kmeans(s, seed=100, n_uav=3)
    assert env.last_result is not None
    np.testing.assert_allclose(env.uav_xy, xy_k)
    assert env.last_result.sum_rate == res_k.sum_rate
    assert env.last_result.feasible == res_k.feasible


def test_solve_td3_eval_kmeans_matches_solve_kmeans_after_training():
    """Eval start_index=0 must use the same k-means draw as solve_kmeans.

    Training depletes env.rng (warmup uniforms + episode-reset k-means). The
    reported k-means start must still match the standalone baseline on the
    same seed, not whatever clustering the depleted generator produces.
    """
    for seed in (100, 101, 102, 103, 104):
        s = generate_scenario(seed, DEFAULT)
        xy_k, res_k, _ = solve_kmeans(s, seed=seed, n_uav=3)
        trace: dict = {}
        xy, result, _, _ = solve_td3(
            s,
            seed=seed,
            n_uav=3,
            total_steps=8,
            greedy_steps=0,
            n_restarts=1,
            device="cpu",
            eval_trace=trace,
        )
        assert trace["winner"]["kind"] == "kmeans"
        assert trace["winner"]["start_index"] == 0
        assert trace["winner"]["source"] == "pre_rollout"
        np.testing.assert_allclose(xy, xy_k)
        assert result.sum_rate == res_k.sum_rate

        env = UAVAoDTEnv(s, n_uav=3, seed=seed)
        agent = TD3Agent(env.state_dim, env.action_dim, seed=seed, device="cpu")
        xy2, result2, _, _ = solve_td3(
            s,
            seed=seed,
            n_uav=3,
            greedy_steps=0,
            n_restarts=1,
            agent=agent,
            device="cpu",
        )
        np.testing.assert_allclose(xy2, xy_k)
        assert result2.sum_rate == res_k.sum_rate


def test_eval_kmeans_does_not_advance_training_rng():
    """eval_kmeans=True must not consume self.rng; training resets still do."""
    s = generate_scenario(100, DEFAULT)
    env_train = UAVAoDTEnv(s, n_uav=3, seed=100)
    env_mixed = UAVAoDTEnv(s, n_uav=3, seed=100)
    env_mixed.reset(eval_kmeans=True)
    env_train.reset()
    env_mixed.reset()
    np.testing.assert_allclose(env_train.uav_xy, env_mixed.uav_xy)

    env_seq = UAVAoDTEnv(s, n_uav=3, seed=100)
    env_seq.reset()
    first = env_seq.uav_xy.copy()
    env_seq.reset()
    second = env_seq.uav_xy.copy()
    assert not np.allclose(first, second)


def test_fidelity_mode_is_continuous_unmediated_and_opt_in():
    """Alg. 2 variant: no episode resets, no distance prior, scale 1.0."""
    s = generate_scenario(100, DEFAULT)
    agent, env, log = train_td3(
        s, seed=0, n_uav=3, total_steps=6, device="cpu", fidelity_mode=True
    )
    assert env.distance_prior is False
    assert env.assoc_action_scale == 1.0
    assert log.start_kinds == ["continuous"] * 6
    assert log.in_episode_steps == list(range(6))
    assert len(log.rewards) == 6

    default_env = UAVAoDTEnv(s, n_uav=3, seed=0)
    assert default_env.distance_prior is True
    assert default_env.assoc_action_scale == TD3_ASSOC_ACTION_SCALE


def test_fidelity_mode_actor_acts_from_step_zero(monkeypatch):
    s = generate_scenario(100, DEFAULT)
    n = {"act": 0}
    real = TD3Agent.act

    def counting(self, state, noise=0.0):
        n["act"] += 1
        return real(self, state, noise)

    monkeypatch.setattr(TD3Agent, "act", counting)
    train_td3(s, seed=0, n_uav=3, total_steps=3, device="cpu", fidelity_mode=True)
    assert n["act"] == 3


def test_default_train_still_uses_warmup_instead_of_actor(monkeypatch):
    """Default path (compare/sweeps) must still uniform-sample during warmup."""
    s = generate_scenario(100, DEFAULT)
    n = {"act": 0}
    real = TD3Agent.act

    def counting(self, state, noise=0.0):
        n["act"] += 1
        return real(self, state, noise)

    monkeypatch.setattr(TD3Agent, "act", counting)
    train_td3(s, seed=0, n_uav=3, total_steps=3, device="cpu")
    assert n["act"] == 0


def test_fidelity_keeps_old_prior_available_via_flag():
    s = generate_scenario(100, DEFAULT)
    _, env, log = train_td3(
        s,
        seed=0,
        n_uav=3,
        total_steps=4,
        device="cpu",
        fidelity_mode=True,
        distance_prior=True,
        assoc_action_scale=TD3_ASSOC_ACTION_SCALE,
    )
    assert env.distance_prior is True
    assert env.assoc_action_scale == TD3_ASSOC_ACTION_SCALE
    assert log.start_kinds == ["continuous"] * 4

