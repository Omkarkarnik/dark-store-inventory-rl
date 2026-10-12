"""Step 5: demonstration of what the trained agent does, period by period.

The trained agent, the optimal policy, and the best base-stock policy face the same fresh demand path
(seed 99000, never used before). The animation shows what the agent sees (stock now and orders in transit),
the order it places, demand, lost sales, and the running average cost of all three policies.

Usage:
  python scripts/05_demo.py                     # best algorithm on L4_p9: saves a GIF, a snapshot PNG, a summary
  python scripts/05_demo.py --show              # live window for screen recording (close it to finish)
  python scripts/05_demo.py --algo DQN --instance L2_p9 --periods 40
"""
import argparse
import json
import os
import sys

import _paths  # noqa: F401
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--algo", default="best", choices=["best", "PPO", "A2C", "DQN"],
                help="algorithm to show (default: the one with the smallest overall gap in results/summary.json)")
ap.add_argument("--instance", default="L4_p9", choices=["L2_p4", "L2_p9", "L4_p4", "L4_p9"])
ap.add_argument("--seed", type=int, default=None, help="training seed of the model (default: the median seed)")
ap.add_argument("--periods", type=int, default=30)
ap.add_argument("--show", action="store_true", help="open a live window instead of only saving files")
ap.add_argument("--speed", type=float, default=1.5, help="periods per second")
args = ap.parse_args()
if args.show:
    os.environ["INVRL_SHOW"] = "1"

from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from inventory_rl.agents import ALGO_CLASSES  # noqa: E402
from inventory_rl.baselines import base_stock_fn  # noqa: E402
from inventory_rl.config import DP_DIR, FIG_DIR, INSTANCES, MODELS_DIR, RESULTS, RUNS_DIR, TRAIN_SEEDS  # noqa: E402
from inventory_rl.dp import policy_order_fn  # noqa: E402
from inventory_rl.plotstyle import COLORS, INK, INK2, NAMES, plt, save  # noqa: E402
from inventory_rl.simulate import demand_paths, model_order_fn, trajectory  # noqa: E402

if args.algo == "best":
    summ = RESULTS / "summary.json"
    args.algo = (min(json.load(open(summ))["aggregate"].items(), key=lambda kv: kv[1]["gap_iqm"])[0]
                 if summ.exists() else "PPO")
L, p = INSTANCES[args.instance]
base = json.load(open(RESULTS / "baselines.json"))[args.instance]
seed = args.seed
if seed is None:
    done = [s for s in TRAIN_SEEDS if (RUNS_DIR / f"{args.algo}_{args.instance}_s{s}.json").exists()]
    if not done:
        sys.exit(f"No trained {args.algo} model for {args.instance} yet. Run ./run_experiments.sh first.")
    costs = {s: np.mean(json.load(open(RUNS_DIR / f"{args.algo}_{args.instance}_s{s}.json"))["test_cost"]) for s in done}
    seed = sorted(done, key=costs.get)[len(done) // 2]
model = ALGO_CLASSES[args.algo].load(MODELS_DIR / f"{args.algo}_{args.instance}_s{seed}.zip", device="cpu")

d = demand_paths(1, args.periods + 200, 99_000)[0]
pols = {
    f"{args.algo} agent": model_order_fn(model),
    "Optimal (DP)": policy_order_fn(np.load(DP_DIR / f"optimal_policy_{args.instance}.npy"), L),
    f"Best base-stock (S={base['params']['base_stock']['S']})": base_stock_fn(base["params"]["base_stock"]["S"]),
}
# Start every policy from a typical state: run 200 warm-up periods first, then show the next `periods`.
traj = {k: trajectory(fn, d, L, p)[200:] for k, fn in pols.items()}
agent = traj[f"{args.algo} agent"]
names = list(traj)
colors = [COLORS[args.algo], INK, INK2]
styles = ["-", "-", "--"]
T = len(agent)

print(f"Demo: {args.algo} (seed {seed}) on {NAMES[args.instance]}, {T} periods of new demand\n")
print(f"{'period':>6} | {'stock now':>9} | {'in transit':>14} | {'order':>5} | {'demand':>6} | {'lost':>4} | {'cost':>5}")
for r in agent:
    print(f"{r['t'] - 200:>6} | {r['state'][0]:>9.0f} | {str([int(v) for v in r['state'][1:]]):>14} | {r['order']:>5} | "
          f"{r['demand']:>6} | {r['lost']:>4.0f} | {r['cost']:>5.0f}")
for k in names:
    print(f"{k}: average cost {np.mean([r['cost'] for r in traj[k]]):.2f} per period over these {T} periods")

fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], height_ratios=[1, 1], hspace=0.38, wspace=0.22)
ax_state = fig.add_subplot(gs[0, 0])
ax_time = fig.add_subplot(gs[0, 1])
ax_cost = fig.add_subplot(gs[1, :])
labels = ["Stock now"] + [f"Arrives in {k}" for k in range(1, L)] + ["Order now"]


def draw(n):
    for ax in (ax_state, ax_time, ax_cost):
        ax.clear()
    r = agent[n]
    vals = list(r["state"]) + [r["order"]]
    cols = [COLORS[args.algo]] * L + [INK]
    bars = ax_state.bar(range(L + 1), vals, color=cols, width=0.6)
    bars[-1].set_alpha(0.85)
    for b, v in zip(bars, vals):
        ax_state.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}", ha="center", fontsize=11, color=INK)
    ax_state.set_xticks(range(L + 1), labels, fontsize=9, rotation=20)
    ax_state.set_ylim(0, 26)
    ax_state.set_title(f"What the {args.algo} agent sees, period {n + 1}")
    ax_state.grid(axis="x", visible=False)

    ts = np.arange(1, n + 2)
    stock = [a["state"][0] for a in agent[:n + 1]]
    dem = [a["demand"] for a in agent[:n + 1]]
    lost = [a["lost"] for a in agent[:n + 1]]
    ax_time.bar(ts, dem, color="#c9c8c2", width=0.7, label="Demand")
    ax_time.step(ts, stock, where="mid", color=COLORS[args.algo], label="Stock now")
    lt = [t for t, x in zip(ts, lost) if x > 0]
    ax_time.scatter(lt, [s for s, x in zip(stock, lost) if x > 0], marker="x", s=60, color=INK, zorder=3,
                    label="Lost sales")
    ax_time.set_xlim(0.3, T + 0.7)
    ax_time.set_ylim(0, max(16, max(stock) + 2))
    ax_time.set_title(f"Period {n + 1}: demand {r['demand']}, sold {r['sold']:.0f}, lost {r['lost']:.0f}, "
                      f"ordered {r['order']}")
    ax_time.set_xlabel("Period")
    ax_time.legend(loc="upper left", ncol=3, fontsize=9)

    for k, c, ls in zip(names, colors, styles):
        cc = np.cumsum([x["cost"] for x in traj[k][:n + 1]]) / ts
        ax_cost.plot(ts, cc, color=c, ls=ls, label=k)
        ax_cost.text(ts[-1] + 0.3, min(cc[-1], 24.0), f"{cc[-1]:.2f}", color=c, fontsize=10, va="center")
    ax_cost.set_xlim(0.3, T + 2.5)
    ax_cost.set_ylim(0, 25)   # early periods can be off the chart for a poor policy
    ax_cost.set_xlabel("Period")
    ax_cost.set_ylabel("Average cost per period so far")
    ax_cost.set_title("Same demand for all three policies")
    ax_cost.legend(loc="upper right", fontsize=10)
    fig.suptitle(f"{args.algo} agent on {NAMES[args.instance]} (holding cost 1, lost sale {p:g}, lead time {L})",
                 fontsize=13, fontweight="bold")


out = RESULTS / "demo"
out.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
anim = FuncAnimation(fig, draw, frames=T, interval=1000 / args.speed, repeat=False)
if args.show:
    plt.show()
else:
    gif = out / f"demo_{args.algo}_{args.instance}.gif"
    anim.save(gif, writer=PillowWriter(fps=args.speed))
    draw(T - 1)
    save(fig, FIG_DIR / "demo_snapshot.png")
    lost_periods = sum(r["lost"] > 0 for r in agent)
    summary = dict(algo=args.algo, instance=args.instance, seed=int(seed), periods=T, demand_seed=99_000,
                   avg_cost={k: float(np.mean([r["cost"] for r in traj[k]])) for k in names},
                   agent_name=names[0], optimal_name=names[1], base_name=names[2], lost_periods=int(lost_periods),
                   gif=str(gif.name))
    with open(out / "demo_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nSaved {gif}, {FIG_DIR / 'demo_snapshot.png'} and {out / 'demo_summary.json'}")
