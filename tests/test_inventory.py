"""Unit tests: transitions, costs, simulator vs environment, optimal costs, and reproducibility."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inventory_rl.config import PUBLISHED_OPTIMUM, REWARD_SCALE  # noqa: E402
from inventory_rl.dp import policy_order_fn, relative_value_iteration  # noqa: E402
from inventory_rl.env import LostSalesEnv  # noqa: E402
from inventory_rl.simulate import demand_paths, simulate  # noqa: E402


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    check_env(LostSalesEnv(L=4, p=9.0), skip_render_check=True)


def test_one_step_by_hand():
    env = LostSalesEnv(L=2, p=9.0)
    env.reset(seed=0)
    env.s = np.array([3.0, 7.0])                     # 3 units available now, 7 arriving next period
    env.np_random = np.random.default_rng(0)
    d = np.random.default_rng(0).poisson(5.0)        # the demand the environment will draw
    obs, r, term, trunc, info = env.step(4)
    left, lost = max(3 - d, 0), max(d - 3, 0)
    assert info["demand"] == d
    assert info["cost"] == pytest.approx(1.0 * left + 9.0 * lost)
    assert r == pytest.approx(-info["cost"] / REWARD_SCALE)
    assert np.allclose(obs * 20, [left + 7, 4])       # pipeline shifts and the new order joins the end
    assert not term and not trunc


def test_truncation_after_horizon():
    env = LostSalesEnv(L=2, p=4.0, horizon=5)
    env.reset(seed=1)
    flags = [env.step(5)[3] for _ in range(5)]
    assert flags == [False, False, False, False, True]


def test_simulator_matches_environment():
    """The fast simulator and the Gymnasium environment give identical costs on the same demand."""
    L, p, T = 4, 9.0, 300
    env = LostSalesEnv(L=L, p=p)
    env.reset(seed=3)
    rng = np.random.default_rng(42)
    orders = rng.integers(0, 21, T)
    env_costs, demands = [], []
    for q in orders:
        _, _, _, _, info = env.step(int(q))
        env_costs.append(info["cost"])
        demands.append(info["demand"])
    it = iter(orders)
    r = simulate(lambda x: np.array([next(it)]), np.array([demands]), L, p)
    assert r["cost"][0] == pytest.approx(np.mean(env_costs))


@pytest.mark.parametrize("name,L,p", [("L2_p4", 2, 4.0), ("L2_p9", 2, 9.0)])
def test_optimal_cost_matches_published_and_simulation(name, L, p):
    dp = relative_value_iteration(L, p)
    assert round(dp["g"], 2) == PUBLISHED_OPTIMUM[name]
    D = demand_paths(3, 30_000, 777)
    sim = simulate(policy_order_fn(dp["policy"], L), D, L, p, warmup=1_000)["cost"].mean()
    assert sim == pytest.approx(dp["g"], rel=0.01)


@pytest.mark.parametrize("algo", ["A2C", "DQN"])
def test_training_is_reproducible(algo):
    """Same seed, same machine -> identical network weights."""
    import torch
    from inventory_rl.agents import make_model
    torch.set_num_threads(1)
    params = []
    for _ in range(2):
        m = make_model(algo, 2, 4.0, seed=0)
        m.learn(total_timesteps=2_000)
        params.append(torch.cat([t.flatten() for t in m.policy.state_dict().values()]).numpy())
    assert np.array_equal(params[0], params[1])
