"""Exact optimal policy by relative value iteration (average-cost criterion)."""
import time

import numpy as np
from scipy.stats import poisson

from .config import H, LAM, QMAX, DP_S0_MAX, DP_TOL


def relative_value_iteration(L, p, s0_max=DP_S0_MAX, tol=DP_TOL, max_iter=5_000):
    """Solve the lost-sales MDP on a truncated state space.

    States: s0 in 0..s0_max and pipeline orders in 0..QMAX. Returns the optimal average cost g,
    the optimal order for every state (an integer array indexed by the state), iterations, and seconds.
    """
    t0 = time.time()
    dmax = int(poisson.ppf(1 - 1e-10, LAM))
    pd = poisson.pmf(np.arange(dmax + 1), LAM)
    pd[-1] += 1.0 - pd.sum()
    s0 = np.arange(s0_max + 1)
    d = np.arange(dmax + 1)
    left = np.maximum(s0[:, None] - d[None, :], 0)
    c0 = (H * left + p * np.maximum(d[None, :] - s0[:, None], 0)) @ pd     # expected period cost given s0
    dims = (s0_max + 1,) + (QMAX + 1,) * (L - 1)
    nxt0 = np.minimum(left[:, None, :] + np.arange(QMAX + 1)[None, :, None], s0_max)  # (s0, s1, d)
    c0b = c0.reshape((-1,) + (1,) * (L - 1))
    V = np.zeros(dims)
    g = 0.0
    for it in range(1, max_iter + 1):
        Q = np.empty(dims + (QMAX + 1,))
        for q in range(QMAX + 1):
            # next state = (min(left + s1, s0_max), s2, ..., s_{L-1}, q)
            Q[..., q] = c0b + np.tensordot(V[..., q][nxt0], pd, axes=([2], [0]))
        Vn = Q.min(axis=-1)
        g = Vn.flat[0] - V.flat[0]
        Vn -= Vn.flat[0]
        done = np.max(np.abs(Vn - V)) < tol
        V = Vn
        if done:
            break
    policy = Q.argmin(axis=-1).astype(np.int16)
    return dict(g=float(g), policy=policy, iterations=it, seconds=time.time() - t0)


def policy_order_fn(policy, L, s0_max=DP_S0_MAX):
    """Look up the optimal order for raw states (rows, L)."""
    caps = np.array([s0_max] + [QMAX] * (L - 1))

    def fn(x):
        idx = np.minimum(np.rint(x).astype(int), caps)
        return policy[tuple(idx.T)]
    return fn
