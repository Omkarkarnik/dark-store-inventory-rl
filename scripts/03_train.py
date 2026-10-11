"""Step 3: main experiment. 3 algorithms x 4 instances x 5 seeds = 60 runs of 2 million steps.

Each run records a learning curve (greedy policy on five 5,000-period validation paths every 100,000 steps),
then tests its final greedy policy for 100,000 periods on ten held-out demand paths shared by every policy.
Finished runs are skipped, so the script can be stopped and restarted safely.
Usage: python scripts/03_train.py            (full experiment)
       python scripts/03_train.py --steps 1000000   (fallback if time is short: fewer steps, same seeds)
"""
import argparse
import json
import time

import _paths  # noqa: F401

from inventory_rl.config import (ALGOS, EVAL_EVERY, INSTANCES, MODELS_DIR, RESULTS, RUNS_DIR, TOTAL_STEPS,
                                 TRAIN_SEEDS, WORKERS)
from inventory_rl.runner import check_python, log, run_jobs


def job(spec):
    from inventory_rl.agents import train_and_test
    algo, inst, seed, scale, steps = spec
    L, p = INSTANCES[inst]
    name = f"{algo}_{inst}_s{seed}"
    model, out = train_and_test(algo, L, p, seed, steps, EVAL_EVERY, lr_scale=scale)
    out.update(name=name, instance=inst)
    model.save(MODELS_DIR / f"{name}.zip")
    with open(RUNS_DIR / f"{name}.json", "w") as f:
        json.dump(out, f)
    import numpy as np
    return dict(name=name, summary=f"test cost {np.mean(out['test_cost']):.3f}, {out['train_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    check_python()
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logfile = RESULTS / "run_log.txt"
    tuning = RESULTS / "tuning.json"
    if tuning.exists():
        scales = json.load(open(tuning))["chosen_scale"]
    else:
        scales = {a: 1.0 for a in ALGOS}
        log("No results/tuning.json found: using the default learning rates.", logfile)
    t0 = time.time()
    log(f"STEP 3/6: main experiment, 60 runs of {args.steps:,} steps; learning-rate scales {scales}", logfile)
    order = {"DQN": 0, "PPO": 1, "A2C": 2}                       # longest runs first
    specs = sorted([(a, i, s, scales[a], args.steps) for a in ALGOS for i in INSTANCES for s in TRAIN_SEEDS],
                   key=lambda x: order[x[0]])
    todo = [sp for sp in specs if not (RUNS_DIR / f"{sp[0]}_{sp[1]}_s{sp[2]}.json").exists()]
    run_jobs(job, todo, args.workers, logfile, label="run")
    log(f"Step 3 finished in {(time.time() - t0) / 60:.1f} min -> results/runs/ and results/models/", logfile)
