"""Interquartile mean (IQM) with stratified bootstrap confidence intervals (Agarwal et al., 2021).

Uses the rliable library when it is installed and falls back to the same computation in NumPy.
"""
import numpy as np

try:
    from rliable import library as rly
    from rliable import metrics as rmetrics
    HAVE_RLIABLE = True
except Exception:  # pragma: no cover - fallback path
    HAVE_RLIABLE = False

REPS = 2_000


def iqm(x):
    """Mean of the middle 50% of all values (25% trimmed from each end, as in rliable)."""
    from scipy.stats import trim_mean
    return float(trim_mean(np.asarray(x).ravel(), 0.25))


def iqm_ci(scores, reps=REPS, seed=0):
    """scores: array (runs, tasks). Returns (IQM, low, high) with a 95% stratified bootstrap interval."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    np.random.seed(seed)
    if HAVE_RLIABLE:
        point, ci = rly.get_interval_estimates({"x": scores}, lambda s: np.array([rmetrics.aggregate_iqm(s)]), reps=reps)
        return float(point["x"][0]), float(ci["x"][0, 0]), float(ci["x"][1, 0])
    rng = np.random.default_rng(seed)
    n, k = scores.shape
    boots = [iqm(scores[rng.integers(0, n, (n, k)), np.arange(k)]) for _ in range(reps)]   # resample runs within each task
    return iqm(scores), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def performance_profile(gaps_by_algo, taus, reps=REPS, seed=0):
    """Share of runs with optimality gap below each threshold tau, with 95% bootstrap bands."""
    np.random.seed(seed)
    if HAVE_RLIABLE:
        scores = {a: -np.asarray(g, dtype=float) for a, g in gaps_by_algo.items()}
        point, ci = rly.create_performance_profile(scores, -np.asarray(taus), reps=reps)
        return {a: (point[a], ci[a][0], ci[a][1]) for a in scores}
    out = {}
    rng = np.random.default_rng(seed)
    for a, g in gaps_by_algo.items():
        g = np.asarray(g, dtype=float)
        n, k = g.shape
        prof = lambda m: np.array([(m < t).mean() for t in taus])
        boots = np.array([prof(g[rng.integers(0, n, (n, k)), np.arange(k)]) for _ in range(reps)])
        out[a] = (prof(g), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0))
    return out
