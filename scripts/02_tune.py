"""Step 2: equal tuning effort for every algorithm.

Learning-rate sweep (0.3, 1, and 3 times the default) on the hardest instance (L = 4, p = 9) with
three seeds of 500,000 steps each: 3 algorithms x 3 rates x 3 seeds = 27 runs. The rate with the lowest
mean validation cost at the end of training is frozen for the main experiment.
Writes results/tuning.json. Usage: python scripts/02_tune.py
"""
import json
import time

import _paths  # noqa: F401
import numpy as np

from inventory_rl.config import (ALGOS, EVAL_EVERY, INSTANCES, LR_SCALES, RESULTS, TUNE_DIR, TUNE_INSTANCE,
                                 TUNE_SEEDS, TUNE_STEPS, WORKERS)
from inventory_rl.runner import check_python, log, run_jobs


def job(spec):
    from inventory_rl.agents import train_and_test
    algo, scale, seed = spec
    L, p = INSTANCES[TUNE_INSTANCE]
    name = f"{algo}_lr{scale:g}_s{seed}"
    _, out = train_and_test(algo, L, p, seed, TUNE_STEPS, EVAL_EVERY, lr_scale=scale, final_test=False)
    out["name"] = name
    with open(TUNE_DIR / f"{name}.json", "w") as f:
        json.dump(out, f)
    return dict(name=name, summary=f"validation cost {out['curve'][-1][1]:.3f}, {out['train_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    check_python()
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    logfile = RESULTS / "run_log.txt"
    t0 = time.time()
    log(f"STEP 2/6: learning-rate sweep ({len(ALGOS) * len(LR_SCALES) * len(TUNE_SEEDS)} runs of {TUNE_STEPS:,} steps on {TUNE_INSTANCE})", logfile)
    order = {"DQN": 0, "PPO": 1, "A2C": 2}                       # longest runs first
    specs = sorted([(a, s, seed) for a in ALGOS for s in LR_SCALES for seed in TUNE_SEEDS], key=lambda x: order[x[0]])
    todo = [sp for sp in specs if not (TUNE_DIR / f"{sp[0]}_lr{sp[1]:g}_s{sp[2]}.json").exists()]
    run_jobs(job, todo, WORKERS, logfile, label="tuning run")
    table, chosen = {}, {}
    for a in ALGOS:
        table[a] = {}
        for s in LR_SCALES:
            finals = [json.load(open(TUNE_DIR / f"{a}_lr{s:g}_s{seed}.json"))["curve"][-1][1] for seed in TUNE_SEEDS]
            table[a][f"{s:g}"] = dict(final_validation_cost=finals, mean=float(np.mean(finals)))
        chosen[a] = float(min(LR_SCALES, key=lambda s: table[a][f"{s:g}"]["mean"]))
        log(f"{a}: mean validation cost by learning-rate scale "
            + ", ".join(f"{s:g}x={table[a][f'{s:g}']['mean']:.3f}" for s in LR_SCALES) + f" -> chose {chosen[a]:g}x", logfile)
    with open(RESULTS / "tuning.json", "w") as f:
        json.dump(dict(instance=TUNE_INSTANCE, steps=TUNE_STEPS, seeds=TUNE_SEEDS, table=table, chosen_scale=chosen), f, indent=1)
    log(f"Step 2 finished in {(time.time() - t0) / 60:.1f} min -> results/tuning.json", logfile)
