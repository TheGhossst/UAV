import numpy as np

from src.config import DEFAULT
from src.scenario import generate_scenario
from src.solvers.td3 import UAVAoDTEnv, train_td3


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
