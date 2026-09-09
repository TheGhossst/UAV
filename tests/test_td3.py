"""TD3 Algorithm 2 fill-in. SimConfig is not modified."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from uavdt.config import DEFAULT, SimConfig
from uavdt.experiments.cli import _cfg_from_args, _parse_methods, build_parser
from uavdt.experiments.methods import KNOWN_METHODS, METHODS
from uavdt.scenario import generate_scenario
from uavdt.td3.decode import action_size, decode_action, observation_size
from uavdt.td3.env import UAVAoDTEnv, step_reward
from uavdt.td3.settings import TD3Settings


def _tiny() -> TD3Settings:
    return TD3Settings(
        total_steps=30,
        horizon=10,
        hidden=32,
        batch_size=8,
        warmup_steps=8,
        buffer_size=64,
        log_every=0,
    )


def test_td3_is_opt_in_not_default_campaign():
    assert METHODS == ("random", "kmeans", "pso", "sca")
    assert "td3" not in METHODS
    assert "td3" in KNOWN_METHODS
    assert _parse_methods("td3") == ("td3",)


def test_simconfig_defaults_unchanged_by_td3_import():
    cfg = SimConfig()
    assert asdict(cfg) == asdict(DEFAULT)
    assert cfg.area_x_m == 100.0
    assert cfg.b_sys_hz == 20_000.0
    assert not hasattr(cfg, "td3")
    assert "actor_lr" not in SimConfig.__dataclass_fields__


def test_decode_hard_association_and_process_consistent_b():
    cfg = SimConfig()
    sc = generate_scenario(1, cfg)
    uav = np.array(
        [[10.0, 10.0, 100.0], [50.0, 50.0, 100.0], [80.0, 20.0, 100.0]],
        dtype=float,
    )
    act = np.zeros(action_size(cfg), dtype=float)
    act[0] = 1.0
    act[1] = -1.0
    j = cfg.num_uav
    n_move = 2 * j
    for iot in range(cfg.num_iot):
        act[n_move + iot * j + 1] = 1.0
    n_assoc = cfg.num_iot * j
    proc0 = n_move + n_assoc
    act[proc0 + 0] = 1.0
    act[proc0 + j + 2] = 1.0
    settings = TD3Settings(
        log_every=0,
        move_mode="delta",
        assoc_distance_coef=0.0,
        assoc_mode="logits",
        process_mode="logits",
    )
    new_uav, alloc = decode_action(act, uav, sc, settings)
    assert new_uav.shape == (3, 3)
    np.testing.assert_allclose(new_uav[0, 0], 20.0)
    np.testing.assert_allclose(new_uav[0, 1], 0.0)
    np.testing.assert_allclose(new_uav[:, 2], 100.0)
    assert np.all(new_uav[:, 0] >= 0.0) and np.all(new_uav[:, 0] <= cfg.area_x_m)
    a = alloc.hard_association()
    b = alloc.hard_processing()
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    np.testing.assert_array_equal(np.argmax(a, axis=1), 1)
    np.testing.assert_allclose(b.sum(axis=1), 1.0)
    np.testing.assert_array_equal(np.argmax(b[:5], axis=1), 0)
    np.testing.assert_array_equal(np.argmax(b[5:], axis=1), 2)
    np.testing.assert_allclose(alloc.bandwidth_hz.sum(), cfg.b_sys_hz, atol=1.0)


def test_decode_setpoint_steps_toward_field_target():
    cfg = SimConfig()
    sc = generate_scenario(1, cfg)
    uav = np.array(
        [[10.0, 10.0, 100.0], [50.0, 50.0, 100.0], [80.0, 20.0, 100.0]],
        dtype=float,
    )
    act = np.zeros(action_size(cfg), dtype=float)
    act[: 2 * cfg.num_uav] = 1.0
    settings = TD3Settings(log_every=0, move_mode="setpoint", assoc_distance_coef=0.0)
    new_uav, _alloc = decode_action(act, uav, sc, settings)
    np.testing.assert_allclose(new_uav[0, :2], [20.0, 20.0])
    np.testing.assert_allclose(new_uav[1, :2], [60.0, 60.0])
    np.testing.assert_allclose(new_uav[2, :2], [90.0, 30.0])
    stay = np.zeros(action_size(cfg), dtype=float)
    # tanh 0 maps to field center; UAV already at center should hover.
    center = np.array(
        [[50.0, 50.0, 100.0], [50.0, 50.0, 100.0], [50.0, 50.0, 100.0]],
        dtype=float,
    )
    hovered, _ = decode_action(stay, center, sc, settings)
    np.testing.assert_allclose(hovered[:, :2], center[:, :2])


def test_decode_residual_offsets_frozen_origin():
    cfg = SimConfig()
    sc = generate_scenario(1, cfg)
    origin = np.array(
        [[40.0, 40.0, 100.0], [50.0, 50.0, 100.0], [60.0, 30.0, 100.0]],
        dtype=float,
    )
    elsewhere = origin.copy()
    elsewhere[:, 0] += 15.0
    act = np.zeros(action_size(cfg), dtype=float)
    act[0] = 1.0
    settings = TD3Settings(log_every=0, move_mode="residual", assoc_distance_coef=0.0)
    new_uav, _ = decode_action(act, elsewhere, sc, settings, origin_xyz_m=origin)
    np.testing.assert_allclose(new_uav[0, :2], [50.0, 40.0])
    np.testing.assert_allclose(new_uav[1, :2], [50.0, 50.0])


def test_env_obs_action_shapes_and_finite_reward():
    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(1, cfg)
    env = UAVAoDTEnv(sc, _tiny(), seed=1)
    obs = env.reset()
    q0 = np.asarray(env._uav, dtype=float).copy()
    assert obs.shape == (observation_size(cfg),)
    assert np.all(np.isfinite(obs))
    rng = np.random.default_rng(0)
    action = rng.uniform(-1.0, 1.0, size=env.act_dim)
    assert action.shape == (action_size(cfg),)
    next_obs, reward, done, info = env.step(action)
    assert next_obs.shape == obs.shape
    assert np.isfinite(reward)
    assert "eval" in info
    assert "p_aodt" in info
    assert "p_dist" in info
    assert "v_viol" in info
    ev = info["eval"]
    r2 = step_reward(ev, env._uav, cfg, _tiny(), env.r_max)
    assert np.isfinite(r2)
    assert env.best is not None
    assert asdict(SimConfig()) == asdict(DEFAULT)
    env.step(rng.uniform(-1.0, 1.0, size=env.act_dim))
    obs2 = env.reset()
    assert obs2.shape == obs.shape
    np.testing.assert_allclose(env._uav, q0)


def test_cli_td3_help_and_cfg_from_args_untouched():
    parser = build_parser()
    args = parser.parse_args(["td3", "--seed", "1"])
    cfg = _cfg_from_args(args)
    assert cfg.area_x_m == DEFAULT.area_x_m
    assert cfg.b_sys_hz == DEFAULT.b_sys_hz
    assert cfg.task_size_bits == DEFAULT.task_size_bits
    assert cfg.task_cycles == DEFAULT.task_cycles
    help_text = parser.format_help()
    assert "td3" in help_text
    td_help = build_parser().parse_args(["td3", "--total-steps", "30"])
    assert td_help.total_steps == 30
    assert td_help.log_every == 250


def test_format_eta_and_log_every_default():
    from uavdt.td3.settings import TD3Settings
    from uavdt.td3.solve import format_eta

    assert format_eta(12) == "12s"
    assert format_eta(75) == "1m15s"
    assert TD3Settings().log_every == 250
    assert TD3Settings().export_mode == "policy"
    assert TD3Settings().export_avg_steps == 10
    assert TD3Settings().discount == 0.0
    assert TD3Settings().actor_preact_l2 == 0.05
    assert TD3Settings().actor_logit_l2 == 0.05
    assert TD3Settings().logit_clip == 3.0
    assert TD3Settings().actor_layer_norm is True
    assert TD3Settings().move_mode == "residual"
    assert TD3Settings().assoc_mode == "nearest"
    assert TD3Settings().assoc_distance_coef == 2.0
    assert TD3Settings().process_mode == "cpu_stable"
    assert TD3Settings().critic_use_obs is False
    assert TD3Settings().critic_move_only is True
    assert TD3Settings().export_actor == "best_checkpoint"
    assert TD3Settings().uav_init == "kmeans"
    assert TD3Settings().inner_bandwidth == "leftover"
    assert _tiny().log_every == 0


def test_actor_last_layer_near_zero_at_init():
    torch = pytest.importorskip("torch")
    from uavdt.td3.networks import Actor

    torch.manual_seed(0)
    actor = Actor(12, 6, 32)
    y = actor(torch.zeros(8, 12)).detach()
    assert float(y.abs().mean()) < 0.05
    assert float(y.abs().max()) < 0.2
    split = Actor(12, 10, 32, n_move=4)
    z = split(torch.zeros(8, 12)).detach()
    assert z.shape == (8, 10)
    assert float(z[:, :4].abs().max()) <= 1.0 + 1e-5
    pre = split.forward_pre_tanh(torch.zeros(8, 12)).detach()
    assert pre.shape == (8, 4)
    td_help = build_parser().parse_args(
        ["td3", "--export-mode", "best_snapshot", "--export-avg-steps", "5"]
    )
    assert td_help.export_mode == "best_snapshot"
    assert td_help.export_avg_steps == 5


def test_agent_warmup_disc_scale_and_critic_pack():
    torch = pytest.importorskip("torch")
    from uavdt.td3.agent import TD3Agent

    _ = torch
    settings = _tiny()
    agent = TD3Agent(8, 6, settings, seed=0, n_move=2)
    rng = np.random.default_rng(0)
    warm = np.stack([agent.sample_warmup_action(rng) for _ in range(80)])
    assert np.all(np.abs(warm[:, :2]) <= 1.0 + 1e-9)
    assert float(np.max(np.abs(warm[:, 2:]))) > 1.0
    a = torch.tensor([[0.5, -0.5, 3.0, -3.0, 0.0, 1.5]], dtype=torch.float32)
    packed = agent.pack_for_critic(a).detach().cpu().numpy()[0]
    np.testing.assert_allclose(packed, [0.5, -0.5])
    obs = np.zeros(8, dtype=np.float32)
    act = agent.select_action(obs, noise=False)
    assert act.shape == (6,)
    assert np.all(np.abs(act[:2]) <= 1.0 + 1e-5)
    assert np.all(np.abs(act[2:]) <= settings.logit_clip + 1e-5)


def test_run_method_tiny_td3():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cvxpy")
    _ = torch
    from uavdt.experiments.methods import run_method

    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(1, cfg)
    run = run_method(sc, "td3", seed=1, td3_settings=_tiny())
    assert run.method == "td3"
    assert run.uav_xyz_m.shape == (cfg.num_uav, 3)
    assert np.allclose(run.uav_xyz_m[:, 2], cfg.uav_height_m)
    assert run.true_eval.sum_rate_bit_per_s >= 0.0
    assert run.allocation.hard_association().shape == (cfg.num_iot, cfg.num_uav)
    assert run.diagnostics.get("method") == "td3"
    assert run.diagnostics.get("export_bandwidth") == "frozen_q_lp"
    assert run.diagnostics.get("export_mode") == "policy"
    assert str(run.diagnostics.get("export_rule", "")).startswith("deterministic_policy")
    assert "snapshot_export_sum_rate_Mbps" in run.diagnostics
    assert "policy_export_sum_rate_Mbps" in run.diagnostics
    assert abs(
        run.sum_rate_mbps - float(run.diagnostics["policy_export_sum_rate_Mbps"])
    ) < 1e-9
    assert run.diagnostics.get("n_move") == 2 * cfg.num_uav
    assert run.diagnostics.get("actor_layer_norm") is True
    assert run.diagnostics.get("export_actor") == "best_checkpoint"
    assert METHODS == ("random", "kmeans", "pso", "sca")


def test_best_snapshot_export_mode_records_both():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cvxpy")
    _ = torch
    from dataclasses import replace

    from uavdt.experiments.methods import run_method

    cfg = SimConfig(b_sys_hz=2.4e6)
    sc = generate_scenario(1, cfg)
    settings = replace(_tiny(), export_mode="best_snapshot")
    run = run_method(sc, "td3", seed=1, td3_settings=settings)
    d = run.diagnostics
    assert d.get("export_mode") == "best_snapshot"
    assert str(d.get("export_rule", "")).startswith("best_snapshot")
    assert abs(run.sum_rate_mbps - float(d["snapshot_export_sum_rate_Mbps"])) < 1e-9
    assert "policy_export_sum_rate_Mbps" in d
