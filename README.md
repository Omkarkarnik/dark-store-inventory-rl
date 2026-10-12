# Deep RL for Dark-Store Replenishment (Lost-Sales Inventory Control)

RL Project Part 2 for PHDAI-830-A02, Applied Machine Intelligence & Reinforcement Learning
(University of the Cumberlands). Author: Omkar Karnik.

This project trains three deep reinforcement learning algorithms (PPO, A2C, and DQN) to reorder stock for a
single fast-moving item in a quick-commerce dark store, and measures how close each comes to the exact optimal
policy. Unmet demand is lost (a customer who expects delivery in minutes switches apps rather than waits), and
each order arrives L periods after it is placed. Under lost sales, the best order depends on every order still
in transit, which is why simple rules fall short and learning is worth trying.

## What the project does

| Part | File | Purpose |
|---|---|---|
| Environment | `inventory_rl/env.py` | Gymnasium environment. State = (stock on hand plus the order arriving now, orders in transit); action = order 0 to 20 units; reward = minus (holding + lost-sales cost) / 10 |
| Exact optimum | `inventory_rl/dp.py` | Relative value iteration for the average-cost optimal policy, the benchmark every agent is measured against |
| Heuristics | `inventory_rl/baselines.py` | Best base-stock and best capped base-stock policies, found by grid search |
| Agents | `inventory_rl/agents.py` | Stable-Baselines3 PPO, A2C (synchronous A3C), and DQN; learning curves; final test |
| Simulator | `inventory_rl/simulate.py` | Fast simulation in which every policy faces the same demand (common random numbers) |
| Statistics | `inventory_rl/stats.py` | Interquartile mean (IQM) with 95% stratified bootstrap intervals and performance profiles (rliable) |
| Tests | `tests/test_inventory.py` | Transitions and costs by hand, simulator equals environment, optimal costs match the published values, training is reproducible |

### Experimental design (as planned in Part 1)

* **Instances:** Poisson demand with mean 5, holding cost h = 1, no ordering cost, lost-sales penalty p in {4, 9},
  lead time L in {2, 4}: four instances (Zipkin, 2008; Gijsbrechts et al., 2022).
* **Tuning:** the same effort for every algorithm: learning rate at 0.3, 1, and 3 times the default on the hardest
  instance (L = 4, p = 9), three seeds of 500,000 steps (27 runs). The best rate is frozen.
* **Main runs:** 3 algorithms x 4 instances x 5 seeds (0 to 4) = 60 runs of 2 million environment steps, each on
  one CPU core, four at a time.
* **Validation:** every 100,000 steps the greedy policy runs five 5,000-period demand paths (learning curves).
* **Test:** each final policy runs for 100,000 periods on ten held-out demand paths. The optimal policy and the
  heuristics face exactly the same demand.
* **Metrics:** average cost per period, gap to the optimal cost, fill rate, steps until within 5% of optimal,
  training time, spread across seeds, share of seeds within 5%, and collapses (validation cost rising more than
  10% above a run's best). IQM across seeds with 95% stratified bootstrap intervals (Agarwal et al., 2021).

## Requirements

* macOS on Apple silicon (tested on a MacBook Pro M1, 8 GB) or Linux. CPU only; no GPU is used or needed.
* Python 3.12 (Miniforge/conda recommended).
* About 2 hours of compute for the full experiment on an M1; each run uses roughly 300 MB of memory.

## Setup

```bash
# 0. Get the code
git clone https://github.com/YOUR-USERNAME/dark-store-inventory-rl.git
cd dark-store-inventory-rl

# 1. Create the environment (skip if you already have one)
conda create -n rlproj python=3.12 -y
conda activate rlproj            # or: source ~/miniforge3/bin/activate rlproj

# 2. Install the packages
pip install -r requirements.txt

# 3. Run the unit tests (about 1 minute)
python -m pytest -q
```

If rliable is missing, everything still runs: `inventory_rl/stats.py` then computes the same IQM and stratified
bootstrap intervals with NumPy, and `results/summary.json` records which one was used. (The original run installed
rliable 1.2.0 from its GitHub repository, the same version as on PyPI.)

## Run the project

```bash
caffeinate -i ./run_experiments.sh   # steps 1-3: baselines, tuning sweep, 60 training runs (~2 hours on an M1)
./make_outputs.sh                    # steps 4-5: statistics, figures, and the demo (~5 minutes)
```

`caffeinate -i` keeps the Mac awake (macOS only; leave it out on Linux). Each step can also be run on its own:

| Step | Command | Output |
|---|---|---|
| 1 | `python scripts/01_baselines.py` | `results/baselines.json`, `results/dp/` |
| 2 | `python scripts/02_tune.py` | `results/tuning.json`, `results/tuning/` |
| 3 | `python scripts/03_train.py` | `results/runs/*.json`, `results/models/*.zip` |
| 4 | `python scripts/04_analyze.py` | `results/summary.json`, `results/tables/`, `results/figures/` |
| 5 | `python scripts/05_demo.py` | `results/demo/` (add `--show` for a live window) |

Finished runs are skipped, so an interrupted run can simply be started again. If time is short,
`./run_experiments.sh --steps 1000000` halves the steps per run but keeps all seeds, the fallback stated in Part 1.

**Quick check of the whole pipeline (about 10 minutes):** `SMOKE_TEST=1 ./run_experiments.sh && SMOKE_TEST=1 ./make_outputs.sh`
uses tiny budgets and writes to `results_smoke/`, which git ignores.

## Results

After the run, the main numbers are in `results/tables/final_results.md`, every run is listed in
`results/tables/every_run.csv`, the hypothesis checks are in `results/summary.json`, and the figures are in
`results/figures/`. `results/run_log.txt` is the full log.

* `results/figures/`: learning curves, final test gaps, performance profile, sample efficiency, training time,
  the learned ordering rule versus the optimal one (`policy_map_L2_p9.png`), and a demo snapshot.
* `results/demo/`: an animation of the trained agent running the store next to the optimal and base-stock policies.
  `python scripts/05_demo.py --show` plays it live in a window (used for the recorded demo).

The committed `results/` folder holds the outputs of the run on my M1, so the results can be read without rerunning
anything.

## Known limitations

* For PPO and A2C, Stable-Baselines3 seeds environment copy k of seed s with s + k, so neighbouring seeds share most
  of their training demand streams (their networks and actions still differ). Seeds are therefore not fully
  independent; DQN's are.
* Learning-curve evaluations start each 5,000-period validation path from the same starting state with no warm-up,
  which adds a small bias (well under 1%) to validation gaps. The final test uses a 1,000-period warm-up.
* All agents use a discount factor of 0.99 while being judged on average cost, a possible source of bias noted in Part 1.
* Demand is stationary Poisson; real dark-store demand shifts over time.

## Reproducibility

Every run is seeded (training seeds 0 to 4; validation, search, and test demand use separate fixed seeds), uses
one CPU thread, and saves its model. `tests/test_inventory.py` checks that two short runs with the same seed give
identical network weights for each algorithm. Results can differ slightly between machines because floating-point arithmetic
differs between processors.

## AI use

I used Claude (Anthropic) to assist with organizing and structuring the code. I independently conducted all experiments on my Apple M1 computer and reviewed and verified every result.


## References

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS, 34*, 29304–29320.
Gijsbrechts, J., Boute, R. N., Van Mieghem, J. A., & Zhang, D. J. (2022). Can deep reinforcement learning improve inventory management? *Manufacturing & Service Operations Management, 24*(3), 1349–1368.
Huh, W. T., Janakiraman, G., Muckstadt, J. A., & Rusmevichientong, P. (2006). *Asymptotic optimality of order-up-to policies in lost sales inventory systems* [Unpublished manuscript].
Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *JMLR, 22*(268), 1–8.
Xin, L. (2021). Understanding the performance of capped base-stock policies in lost-sales inventory models. *Operations Research, 69*(1), 61–70.
Zipkin, P. (2008). Old and new methods for lost-sales inventory systems. *Operations Research, 56*(5), 1256–1263.
