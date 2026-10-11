"""Classical heuristics: base-stock and capped base-stock (Xin, 2021), tuned by grid search."""
import numpy as np

from .config import QMAX, SEARCH_PERIODS, SEARCH_SEED, TEST_WARMUP
from .simulate import demand_paths, simulate


def base_stock_fn(S):
    return lambda x: np.clip(S - x.sum(axis=1), 0, QMAX)


def capped_base_stock_fn(S, cap):
    return lambda x: np.minimum(np.clip(S - x.sum(axis=1), 0, QMAX), cap)


def search(L, p):
    """Grid-search the best base-stock level S and the best capped base-stock pair (S, cap)
    on a dedicated demand path (seed 30000) that the final test never uses."""
    D = demand_paths(1, SEARCH_PERIODS + TEST_WARMUP, SEARCH_SEED)[0]
    S_grid = np.arange(0, 61)
    c = simulate(lambda x: np.clip(S_grid - x.sum(1), 0, QMAX), D, L, p, n_rows=len(S_grid), warmup=TEST_WARMUP)["cost"]
    S_best = int(S_grid[c.argmin()])
    SS, RR = np.meshgrid(np.arange(0, 61), np.arange(1, QMAX + 1), indexing="ij")
    SS, RR = SS.ravel(), RR.ravel()
    c2 = simulate(lambda x: np.minimum(np.clip(SS - x.sum(1), 0, QMAX), RR), D, L, p, n_rows=len(SS),
                  warmup=TEST_WARMUP)["cost"]
    k = int(c2.argmin())
    return dict(base_stock=dict(S=S_best), capped_base_stock=dict(S=int(SS[k]), cap=int(RR[k])))
