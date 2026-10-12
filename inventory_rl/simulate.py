"""Fast vectorized simulation with common random numbers (every policy sees the same demand)."""
import numpy as np

from .config import H, LAM, QMAX, S0_CAP, OBS_SCALE


def demand_paths(n_paths, n_periods, seed_base):
    """One Poisson(5) demand path per seed: seed_base, seed_base + 1, ..."""
    return np.stack([np.random.default_rng(seed_base + k).poisson(LAM, n_periods) for k in range(n_paths)])


def simulate(order_fn, D, L, p, n_rows=None, warmup=0):
    """Run a policy (or many policies side by side) through demand D.

    D has shape (rows, T) for one path per row, or (T,) to share one path across n_rows policies.
    order_fn maps the raw state array (rows, L) to integer orders (rows,).
    Returns a dict of per-row average cost, fill rate, and totals over the periods after `warmup`.
    """
    shared = D.ndim == 1
    T = D.shape[-1]
    n = n_rows if shared else D.shape[0]
    x = np.zeros((n, L))
    x[:, 0] = LAM
    cost = np.zeros(n)
    lost = np.zeros(n)
    demand = np.zeros(n)
    for t in range(T):
        q = np.clip(np.rint(np.asarray(order_fn(x), dtype=float).reshape(n)), 0, QMAX)
        d = D[t] if shared else D[:, t]
        s0 = x[:, 0]
        left = np.maximum(s0 - d, 0.0)
        short = np.maximum(d - s0, 0.0)
        if t >= warmup:
            cost += H * left + p * short
            lost += short
            demand += d
        x[:, 0] = np.minimum(left + x[:, 1], S0_CAP) if L > 1 else 0.0
        x[:, 1:-1] = x[:, 2:]
        x[:, -1] = q
    periods = T - warmup
    return {"cost": cost / periods, "fill_rate": 1.0 - lost / np.maximum(demand, 1.0)}


def trajectory(order_fn, d, L, p):
    """Period-by-period record of one policy on one demand path (used by the demo)."""
    x = np.zeros((1, L))
    x[0, 0] = LAM
    rows = []
    for t, dt in enumerate(d):
        state = x[0].copy()
        q = int(np.clip(np.rint(np.asarray(order_fn(x), dtype=float).reshape(1)[0]), 0, QMAX))
        s0 = state[0]
        left = max(s0 - dt, 0.0)
        short = max(dt - s0, 0.0)
        rows.append(dict(t=t, state=state, order=q, demand=int(dt), sold=float(min(s0, dt)), left=left,
                         lost=short, cost=H * left + p * short))
        x[0, 0] = min(left + x[0, 1], S0_CAP)
        x[0, 1:-1] = x[0, 2:]
        x[0, -1] = q
    return rows


def model_order_fn(model):
    """Greedy (deterministic) orders from a trained Stable-Baselines3 model."""
    def fn(x):
        action, _ = model.predict((x / OBS_SCALE).astype(np.float32), deterministic=True)
        return action
    return fn


def visited_states(order_fn, d, L, p, warmup=1_000):
    """Distinct states a policy visits on demand path d, with their visit shares."""
    x = np.zeros((1, L))
    x[0, 0] = LAM
    counts = {}
    for t, dt in enumerate(d):
        if t >= warmup:
            key = tuple(int(v) for v in x[0])
            counts[key] = counts.get(key, 0) + 1
        q = int(np.clip(np.rint(np.asarray(order_fn(x), dtype=float).reshape(1)[0]), 0, QMAX))
        left = max(x[0, 0] - dt, 0.0)
        x[0, 0] = min(left + x[0, 1], S0_CAP)
        x[0, 1:-1] = x[0, 2:]
        x[0, -1] = q
    states = np.array(list(counts), dtype=float)
    w = np.array(list(counts.values()), dtype=float)
    return states, w / w.sum()
