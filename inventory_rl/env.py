"""Gymnasium environment for a single-item, periodic-review, lost-sales inventory system."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import H, LAM, QMAX, S0_CAP, HORIZON, OBS_SCALE, REWARD_SCALE


class LostSalesEnv(gym.Env):
    """State s = (s0, s1, ..., s_{L-1}).

    s0 is stock on hand plus the order arriving this period; s1..s_{L-1} are orders still in transit
    (oldest first). Each period the agent orders q in {0..20}, which arrives L periods later. Demand
    D ~ Poisson(5) is served from s0; leftover stock costs h per unit and each lost sale costs p.
    """

    metadata = {"render_modes": []}

    def __init__(self, L=4, p=9.0, h=H, lam=LAM, horizon=HORIZON):
        super().__init__()
        self.L, self.p, self.h, self.lam, self.horizon = int(L), float(p), float(h), float(lam), int(horizon)
        self.observation_space = spaces.Box(0.0, S0_CAP / OBS_SCALE, shape=(self.L,), dtype=np.float32)
        self.action_space = spaces.Discrete(QMAX + 1)

    def _obs(self):
        return (self.s / OBS_SCALE).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = np.zeros(self.L)
        self.s[0] = self.lam
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        q = int(action)
        d = int(self.np_random.poisson(self.lam))
        s0 = self.s[0]
        left = max(s0 - d, 0.0)
        lost = max(d - s0, 0.0)
        cost = self.h * left + self.p * lost
        nxt = np.empty(self.L)
        nxt[0] = min(left + self.s[1], S0_CAP)
        nxt[1:-1] = self.s[2:]
        nxt[-1] = q
        self.s = nxt
        self.t += 1
        info = {"cost": cost, "demand": d, "lost": lost}
        return self._obs(), -cost / REWARD_SCALE, False, self.t >= self.horizon, info
