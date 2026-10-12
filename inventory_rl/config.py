"""All settings in one place. Values follow the Part 1 project plan."""
import os
from pathlib import Path

# Inventory model (Zipkin, 2008; Gijsbrechts et al., 2022)
H = 1.0             # holding cost per unit left at the end of a period
LAM = 5.0           # Poisson mean demand per period
QMAX = 20           # largest order quantity; actions are 0..QMAX
S0_CAP = 60         # cap on stock on hand plus the arriving order in the simulator
HORIZON = 1_000     # training episodes are truncated after this many periods
OBS_SCALE = 20.0    # observations are divided by 20
REWARD_SCALE = 10.0 # reward = -(period cost) / 10

INSTANCES = {        # name: (lead time L, lost-sales penalty p)
    "L2_p4": (2, 4.0),
    "L2_p9": (2, 9.0),
    "L4_p4": (4, 4.0),
    "L4_p9": (4, 9.0),
}
# Optimal average cost per period reported by Huh et al. (2006), Table 2 (Poisson, mean 5)
PUBLISHED_OPTIMUM = {"L2_p4": 4.40, "L2_p9": 6.09, "L4_p4": 4.73, "L4_p9": 6.84}

# Dynamic programming (relative value iteration)
DP_S0_MAX = 40
DP_TOL = 1e-6

# Algorithms (Stable-Baselines3 2.9 defaults, except eight parallel environments for PPO and A2C)
ALGOS = ["PPO", "A2C", "DQN"]
DEFAULT_LR = {"PPO": 3e-4, "A2C": 7e-4, "DQN": 1e-4}
N_ENVS = {"PPO": 8, "A2C": 8, "DQN": 1}
GAMMA = 0.99

# Main experiment
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 2_000_000
EVAL_EVERY = 100_000

# Learning-rate sweep (on the hardest instance)
TUNE_INSTANCE = "L4_p9"
TUNE_SEEDS = [0, 1, 2]
TUNE_STEPS = 500_000
LR_SCALES = [0.3, 1.0, 3.0]

# Demand streams. Training uses the environment's own random stream (seeded 0-4);
# validation, policy search, and the final test each use separate seeds that training never sees.
EVAL_PATHS, EVAL_PERIODS, EVAL_SEED_BASE = 5, 5_000, 10_000          # learning curves and tuning
TEST_PATHS, TEST_PERIODS, TEST_SEED_BASE = 10, 100_000, 20_000       # final test
TEST_WARMUP = 1_000                                                  # periods simulated before costs count
SEARCH_PERIODS, SEARCH_SEED = 50_000, 30_000                         # heuristic grid search

SUCCESS_GAP = 5.0       # percent; "within 5% of optimal"
COLLAPSE_RISE = 10.0    # percent rise above a run's best evaluation cost
WORKERS = 4             # runs at a time (one per M1 performance core)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Quick check of the whole pipeline: SMOKE_TEST=1 ./run_experiments.sh && SMOKE_TEST=1 ./make_outputs.sh
# Tiny budgets, results written to results_smoke/ so they never mix with the real results.
SMOKE = os.environ.get("SMOKE_TEST") == "1"
if SMOKE:
    TOTAL_STEPS, EVAL_EVERY, TUNE_STEPS = 50_000, 25_000, 25_000
    TEST_PERIODS, SEARCH_PERIODS = 5_000, 5_000
    RESULTS = ROOT / "results_smoke"
if os.environ.get("INVRL_RESULTS"):          # optional: analyse a results folder somewhere else
    RESULTS = Path(os.environ["INVRL_RESULTS"]).resolve()
RUNS_DIR = RESULTS / "runs"
TUNE_DIR = RESULTS / "tuning"
MODELS_DIR = RESULTS / "models"
FIG_DIR = RESULTS / "figures"
DP_DIR = RESULTS / "dp"
