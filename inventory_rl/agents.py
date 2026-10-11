"""Build, train, and evaluate Stable-Baselines3 agents on one instance."""
import time

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from .config import (DEFAULT_LR, EVAL_PATHS, EVAL_PERIODS, EVAL_SEED_BASE, GAMMA, N_ENVS, TEST_PATHS,
                     TEST_PERIODS, TEST_SEED_BASE, TEST_WARMUP)
from .env import LostSalesEnv
from .simulate import demand_paths, model_order_fn, simulate

ALGO_CLASSES = {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def make_model(algo, L, p, seed, lr_scale=1.0):
    """Stable-Baselines3 defaults (two hidden layers of 64 units, gamma 0.99) with eight
    environment copies for PPO and A2C. Only the learning rate is scaled during tuning."""
    env = make_vec_env(LostSalesEnv, n_envs=N_ENVS[algo], seed=seed, env_kwargs=dict(L=L, p=p))
    return ALGO_CLASSES[algo]("MlpPolicy", env, learning_rate=DEFAULT_LR[algo] * lr_scale, gamma=GAMMA,
                              seed=seed, device="cpu", verbose=0)


def evaluate(model, D, L, p, warmup=0):
    return simulate(model_order_fn(model), D, L, p, warmup=warmup)


class LearningCurve(BaseCallback):
    """Every `every` environment steps, run the greedy policy on the validation demand paths."""

    def __init__(self, L, p, every):
        super().__init__()
        self.L, self.p, self.every = L, p, every
        self.D = demand_paths(EVAL_PATHS, EVAL_PERIODS, EVAL_SEED_BASE)
        self.points = []
        self.next_eval = every

    def _record(self):
        t = time.time()
        cost = float(evaluate(self.model, self.D, self.L, self.p)["cost"].mean())
        self.eval_seconds += time.time() - t
        self.points.append([int(self.num_timesteps), cost])

    def _on_training_start(self):
        self.eval_seconds = 0.0
        self._record()                       # untrained policy at step 0

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._record()
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.every
        return True


def train_and_test(algo, L, p, seed, steps, every, lr_scale=1.0, final_test=True):
    """Train one run, record its learning curve, then test the final greedy policy on held-out demand."""
    import torch
    torch.set_num_threads(1)
    model = make_model(algo, L, p, seed, lr_scale)
    cb = LearningCurve(L, p, every)
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    seconds = time.time() - t0
    out = dict(algo=algo, L=L, p=p, seed=seed, lr=DEFAULT_LR[algo] * lr_scale, lr_scale=lr_scale,
               steps=int(model.num_timesteps), train_seconds=round(seconds - cb.eval_seconds, 1),
               eval_seconds=round(cb.eval_seconds, 1), curve=cb.points)
    if final_test:
        D = demand_paths(TEST_PATHS, TEST_PERIODS + TEST_WARMUP, TEST_SEED_BASE)
        r = evaluate(model, D, L, p, warmup=TEST_WARMUP)
        out["test_cost"] = r["cost"].tolist()
        out["test_fill_rate"] = r["fill_rate"].tolist()
    return model, out
