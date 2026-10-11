"""Step 1: exact optimal policies and tuned heuristics for the four instances.

Writes results/baselines.json and the optimal policy tables in results/dp/.
Usage: python scripts/01_baselines.py
"""
import json
import time

import _paths  # noqa: F401
import numpy as np

from inventory_rl.baselines import base_stock_fn, capped_base_stock_fn, search
from inventory_rl.config import (DP_DIR, INSTANCES, PUBLISHED_OPTIMUM, RESULTS, TEST_PATHS, TEST_PERIODS,
                                 TEST_SEED_BASE, TEST_WARMUP, WORKERS)
from inventory_rl.dp import policy_order_fn, relative_value_iteration
from inventory_rl.runner import check_python, log, run_jobs
from inventory_rl.simulate import demand_paths, simulate


def job(name):
    L, p = INSTANCES[name]
    dp = relative_value_iteration(L, p)
    np.save(DP_DIR / f"optimal_policy_{name}.npy", dp["policy"])
    best = search(L, p)
    D = demand_paths(TEST_PATHS, TEST_PERIODS + TEST_WARMUP, TEST_SEED_BASE)
    policies = {
        "optimal": policy_order_fn(dp["policy"], L),
        "base_stock": base_stock_fn(best["base_stock"]["S"]),
        "capped_base_stock": capped_base_stock_fn(best["capped_base_stock"]["S"], best["capped_base_stock"]["cap"]),
    }
    test = {}
    for k, fn in policies.items():
        r = simulate(fn, D, L, p, warmup=TEST_WARMUP)
        test[k] = dict(cost=r["cost"].tolist(), fill_rate=r["fill_rate"].tolist())
    g = dp["g"]
    gap = {k: 100 * (np.mean(v["cost"]) / g - 1) for k, v in test.items()}
    return dict(name=name, L=L, p=p, optimal_cost=g, published=PUBLISHED_OPTIMUM[name],
                matches_published=bool(round(g, 2) == PUBLISHED_OPTIMUM[name]), dp_iterations=dp["iterations"],
                dp_seconds=round(dp["seconds"], 1), params=best, test=test,
                summary=f"optimal {g:.3f} vs published {PUBLISHED_OPTIMUM[name]:.2f}; base-stock "
                        f"{gap['base_stock']:+.1f}%, capped {gap['capped_base_stock']:+.1f}%, simulated optimal "
                        f"{gap['optimal']:+.2f}%")


if __name__ == "__main__":
    check_python()
    if (RESULTS / "baselines.json").exists() and len(list(DP_DIR.glob("optimal_policy_*.npy"))) == len(INSTANCES):
        log("Step 1 already done (results/baselines.json exists); skipping.")
        raise SystemExit(0)
    DP_DIR.mkdir(parents=True, exist_ok=True)
    logfile = RESULTS / "run_log.txt"
    t0 = time.time()
    log("STEP 1/6: optimal policies (relative value iteration) and heuristic baselines", logfile)
    out = {r["name"]: r for r in run_jobs(job, list(INSTANCES), min(WORKERS, 4), logfile, label="instance")}
    for name in INSTANCES:
        r = out[name]
        if not r["matches_published"]:
            log(f"WARNING: {name} optimal cost {r['optimal_cost']:.3f} does not match the published "
                f"{r['published']:.2f}", logfile)
        r.pop("summary")
    with open(RESULTS / "baselines.json", "w") as f:
        json.dump({k: out[k] for k in INSTANCES}, f, indent=1)
    log(f"Step 1 finished in {(time.time() - t0) / 60:.1f} min -> results/baselines.json", logfile)
