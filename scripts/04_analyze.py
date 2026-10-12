"""Step 4: statistics, tables, figures, and the verdict on each hypothesis and success criterion.

Reads results/baselines.json, results/tuning.json, and results/runs/*.json.
Writes results/summary.json, results/tables/*.md and *.csv, and results/figures/*.png.
Usage: python scripts/04_analyze.py
"""
import csv
import json

import _paths  # noqa: F401
import numpy as np

from inventory_rl.baselines import base_stock_fn, capped_base_stock_fn
from inventory_rl.config import (ALGOS, COLLAPSE_RISE, EVAL_PATHS, EVAL_PERIODS, EVAL_SEED_BASE, FIG_DIR, INSTANCES,
                                 MODELS_DIR, RESULTS, RUNS_DIR, SUCCESS_GAP, TRAIN_SEEDS, DP_DIR, TEST_PATHS,
                                 TEST_PERIODS, OBS_SCALE)
from inventory_rl.dp import policy_order_fn
from inventory_rl.plotstyle import COLORS, INK, INK2, MARKERS, NAMES, plt, save
from inventory_rl.runner import log
from inventory_rl.simulate import demand_paths, simulate, visited_states
from inventory_rl import stats


def gap(cost, g):
    return 100.0 * (np.asarray(cost, dtype=float) / g - 1.0)


def load():
    base = json.load(open(RESULTS / "baselines.json"))
    tuning = json.load(open(RESULTS / "tuning.json")) if (RESULTS / "tuning.json").exists() else None
    runs = {}
    for a in ALGOS:
        for i in INSTANCES:
            for s in TRAIN_SEEDS:
                f = RUNS_DIR / f"{a}_{i}_s{s}.json"
                if f.exists():
                    runs[(a, i, s)] = json.load(open(f))
    return base, tuning, runs


def run_metrics(r, g):
    steps = np.array([pt[0] for pt in r["curve"]])
    curve_gap = gap([pt[1] for pt in r["curve"]], g)
    hit = np.where(curve_gap <= SUCCESS_GAP)[0]
    best = int(np.argmin(curve_gap))
    best_cost = r["curve"][best][1]
    later = [pt[1] for pt in r["curve"][best + 1:]]
    rise = 100.0 * (max(later) / best_cost - 1.0) if later else 0.0
    return dict(test_cost=float(np.mean(r["test_cost"])), test_gap=float(gap(np.mean(r["test_cost"]), g)),
                fill_rate=float(np.mean(r["test_fill_rate"])), steps_to_5=int(steps[hit[0]]) if len(hit) else None,
                collapse=bool(rise > COLLAPSE_RISE), max_rise_after_best=float(rise),
                train_minutes=r["train_seconds"] / 60.0, steps=r["steps"], curve_steps=steps.tolist(),
                curve_gap=curve_gap.tolist(), lr=r["lr"])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    logfile = RESULTS / "run_log.txt"
    log("STEP 4/6: analysis, tables, and figures", logfile)
    base, tuning, runs = load()
    expected = len(ALGOS) * len(INSTANCES) * len(TRAIN_SEEDS)
    if len(runs) < expected:
        log(f"WARNING: only {len(runs)} of {expected} runs are finished; the analysis uses what exists.", logfile)

    g = {i: base[i]["optimal_cost"] for i in INSTANCES}
    # Baselines on the validation paths (for the learning-curve reference lines) and on the test paths
    Dval = demand_paths(EVAL_PATHS, EVAL_PERIODS, EVAL_SEED_BASE)
    ref = {}
    for i, (L, p) in INSTANCES.items():
        prm = base[i]["params"]
        bs = simulate(base_stock_fn(prm["base_stock"]["S"]), Dval, L, p)["cost"].mean()
        cb = simulate(capped_base_stock_fn(prm["capped_base_stock"]["S"], prm["capped_base_stock"]["cap"]), Dval, L, p)["cost"].mean()
        t = base[i]["test"]
        ref[i] = dict(
            val_gap_base=float(gap(bs, g[i])), val_gap_capped=float(gap(cb, g[i])),
            test_cost_optimal_sim=float(np.mean(t["optimal"]["cost"])),
            test_cost_base=float(np.mean(t["base_stock"]["cost"])), test_cost_capped=float(np.mean(t["capped_base_stock"]["cost"])),
            test_gap_base=float(gap(np.mean(t["base_stock"]["cost"]), g[i])),
            test_gap_capped=float(gap(np.mean(t["capped_base_stock"]["cost"]), g[i])),
            test_gap_optimal_sim=float(gap(np.mean(t["optimal"]["cost"]), g[i])),
            fill_optimal=float(np.mean(t["optimal"]["fill_rate"])), fill_base=float(np.mean(t["base_stock"]["fill_rate"])),
            fill_capped=float(np.mean(t["capped_base_stock"]["fill_rate"])))

    M = {k: run_metrics(r, g[k[1]]) for k, r in runs.items()}

    # Per algorithm and instance
    cells = {}
    for a in ALGOS:
        for i in INSTANCES:
            seeds_done = [s for s in TRAIN_SEEDS if (a, i, s) in M]
            ms = [M[(a, i, s)] for s in seeds_done]
            if not ms:
                continue
            costs = np.array([m["test_cost"] for m in ms])
            gaps = np.array([m["test_gap"] for m in ms])
            cost_iqm, cost_lo, cost_hi = stats.iqm_ci(costs)
            gap_iqm, gap_lo, gap_hi = stats.iqm_ci(gaps)
            reached = [m["steps_to_5"] for m in ms if m["steps_to_5"] is not None]
            cells[f"{a}|{i}"] = dict(
                algo=a, instance=i, n=len(ms), seeds=seeds_done,
                gaps=gaps.tolist(), costs=costs.tolist(), cost_iqm=cost_iqm, cost_ci=[cost_lo, cost_hi],
                gap_iqm=gap_iqm, gap_ci=[gap_lo, gap_hi], gap_mean=float(gaps.mean()), gap_std=float(gaps.std(ddof=1)) if len(ms) > 1 else 0.0,
                gap_min=float(gaps.min()), gap_max=float(gaps.max()),
                fill_rate=float(np.mean([m["fill_rate"] for m in ms])),
                success_rate=float(np.mean(gaps <= SUCCESS_GAP)),
                steps_to_5_median=float(np.median([m["steps_to_5"] if m["steps_to_5"] is not None else np.inf for m in ms])),
                reached_5=len(reached), collapses=int(sum(m["collapse"] for m in ms)),
                train_minutes=float(np.mean([m["train_minutes"] for m in ms])),
                beats_base=bool(cost_hi < ref[i]["test_cost_base"]), beats_capped=bool(cost_hi < ref[i]["test_cost_capped"]),
                worse_than_base=int(np.sum(costs > ref[i]["test_cost_base"])))

    def cell(a, i):
        return cells.get(f"{a}|{i}")

    # Aggregate over instances (stratified bootstrap over seeds within each instance)
    agg = {}
    for a in ALGOS:
        mat = [[M[(a, i, s)]["test_gap"] for i in INSTANCES] for s in TRAIN_SEEDS if all((a, i, s) in M for i in INSTANCES)]
        if mat:
            v, lo, hi = stats.iqm_ci(np.array(mat))
            agg[a] = dict(gap_iqm=v, gap_ci=[lo, hi], runs=len(mat) * len(INSTANCES),
                          train_minutes=float(np.mean([M[k]["train_minutes"] for k in M if k[0] == a])),
                          seed_std=float(np.mean([cell(a, i)["gap_std"] for i in INSTANCES if cell(a, i)])),
                          success_rate=float(np.mean([M[k]["test_gap"] <= SUCCESS_GAP for k in M if k[0] == a])),
                          collapses=int(sum(M[k]["collapse"] for k in M if k[0] == a)),
                          worse_than_base=int(sum(cells[f"{a}|{i}"]["worse_than_base"] for i in INSTANCES if f"{a}|{i}" in cells)),
                          reached_5_instances=int(sum(np.isfinite(cells[f"{a}|{i}"]["steps_to_5_median"]) for i in INSTANCES
                                                      if f"{a}|{i}" in cells)),
                          early_share=early_share(M, a))

    # Hypotheses (a policy "beats" another when the upper end of its 95% interval for cost is below the other's cost)
    H = {}
    full = len(cells) == len(ALGOS) * len(INSTANCES)
    beat_base = {a: [i for i in INSTANCES if cell(a, i) and cell(a, i)["beats_base"]] for a in ALGOS}
    H["H1"] = dict(statement="All three algorithms will beat the best base-stock policy on every instance.",
                   supported=full and all(len(beat_base[a]) == len(INSTANCES) for a in ALGOS),
                   detail={a: f"{len(beat_base[a])} of {len(INSTANCES)} instances" for a in ALGOS})
    h2 = {}
    for a in ALGOS:
        l2 = [cell(a, i)["gap_iqm"] for i in ("L2_p4", "L2_p9") if cell(a, i)]
        l4 = [cell(a, i)["gap_iqm"] for i in ("L4_p4", "L4_p9") if cell(a, i)]
        if l2 and l4:
            h2[a] = dict(L2=float(np.mean(l2)), L4=float(np.mean(l4)), larger_at_L4=bool(np.mean(l4) > np.mean(l2)))
    H["H2"] = dict(statement="Optimality gaps will be larger at L = 4 than at L = 2.",
                   supported=bool(h2) and all(v["larger_at_L4"] for v in h2.values()), detail=h2)
    if all(a in agg for a in ALGOS):
        h3a = agg["PPO"]["gap_iqm"] < agg["A2C"]["gap_iqm"]
        h3b = agg["PPO"]["seed_std"] < agg["A2C"]["seed_std"]
        h3c = agg["A2C"]["train_minutes"] < min(agg["PPO"]["train_minutes"], agg["DQN"]["train_minutes"])
        H["H3"] = dict(statement="At equal budgets, PPO will reach smaller gaps than A2C and vary less from seed to seed, "
                                 "while A2C will be fastest in wall-clock time.",
                       supported=bool(h3a and h3b and h3c), parts=dict(smaller_gap=bool(h3a), less_variance=bool(h3b),
                                                                     a2c_fastest=bool(h3c)))
    h4 = {}
    for i in ("L2_p4", "L2_p9"):
        if all(cell(a, i) for a in ALGOS):
            med = {a: cell(a, i)["steps_to_5_median"] for a in ALGOS}
            h4[i] = dict(median_steps=med, dqn_fastest=bool(np.isfinite(med["DQN"]) and med["DQN"] < min(med["PPO"], med["A2C"])),
                         tie=bool(np.isfinite(med["DQN"]) and med["DQN"] == min(med["PPO"], med["A2C"])))
    H["H4"] = dict(statement="DQN will reach the 5% threshold in fewer environment steps than PPO and A2C on the L = 2 instances.",
                   supported=bool(h4) and all(v["dqn_fastest"] for v in h4.values()), detail=h4)
    beat_capped = {a: [i for i in INSTANCES if cell(a, i) and cell(a, i)["beats_capped"]] for a in ALGOS}
    H["H5"] = dict(statement="No algorithm will beat capped base-stock on every instance.",
                   supported=full and not any(len(beat_capped[a]) == len(INSTANCES) for a in ALGOS),
                   detail={a: beat_capped[a] for a in ALGOS})

    # Success criteria from Part 1
    c2 = [a for a in ALGOS if len(beat_base[a]) == len(INSTANCES)]
    c3 = [a for a in c2 if all(cell(a, i)["gap_iqm"] <= SUCCESS_GAP for i in INSTANCES)]
    success = dict(
        all_runs_finished=dict(met=len(runs) == expected, detail=f"{len(runs)} of {expected} runs"),
        beats_base_stock_everywhere=dict(met=bool(c2), algorithms=c2),
        within_5_percent_everywhere=dict(met=bool(c3), algorithms=c3),
        stretch_beats_capped_somewhere=dict(met=any(beat_capped[a] for a in ALGOS), detail=beat_capped))

    agreement = policy_agreement(runs, base)

    summary = dict(policy_agreement=agreement, optimal={i: dict(dp=g[i], published=base[i]["published"], matches=base[i]["matches_published"],
                                    dp_seconds=base[i]["dp_seconds"], params=base[i]["params"]) for i in INSTANCES},
                   reference=ref, cells=cells, aggregate=agg, hypotheses=H, success=success, tuning=tuning,
                   n_runs=len(runs), steps_per_run=int(np.median([r["steps"] for r in runs.values()])) if runs else 0,
                   rliable=stats.HAVE_RLIABLE, test=dict(paths=TEST_PATHS, periods=TEST_PERIODS),
                   total_train_hours=float(sum(r["train_seconds"] + r.get("eval_seconds", 0) for r in runs.values()) / 3600))
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)

    write_tables(summary)
    figures(M, cells, agg, ref, g, base)
    log("Step 4 finished -> results/summary.json, results/tables/, results/figures/", logfile)
    for h, v in H.items():
        log(f"{h}: {'supported' if v['supported'] else 'not supported'}", logfile)


def early_share(M, a):
    """Share of the total improvement in validation gap (median over seeds, averaged over instances) that is
    reached in the first quarter of training."""
    shares = []
    for i in INSTANCES:
        curves = [M[(a, i, s)] for s in TRAIN_SEEDS if (a, i, s) in M]
        if not curves:
            continue
        n = min(len(c["curve_gap"]) for c in curves)
        med = np.median([c["curve_gap"][:n] for c in curves], axis=0)
        q = max(1, (n - 1) // 4)
        total = med[0] - med[-1]
        if total > 0:
            shares.append(float((med[0] - med[q]) / total))
    return float(np.mean(shares)) if shares else float("nan")


def fmt_ci(v, ci):
    return f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def write_tables(S):
    rows = []
    for i in INSTANCES:
        ref = S["reference"][i]
        rows.append([NAMES[i], "Optimal (DP)", f"{S['optimal'][i]['dp']:.3f}", "0.00", f"{ref['fill_optimal']:.3f}", "", ""])
        rows.append([NAMES[i], "Capped base-stock", f"{ref['test_cost_capped']:.3f}", f"{ref['test_gap_capped']:.2f}", f"{ref['fill_capped']:.3f}", "", ""])
        rows.append([NAMES[i], "Base-stock", f"{ref['test_cost_base']:.3f}", f"{ref['test_gap_base']:.2f}", f"{ref['fill_base']:.3f}", "", ""])
        for a in ALGOS:
            c = S["cells"].get(f"{a}|{i}")
            if c:
                rows.append([NAMES[i], a, fmt_ci(c["cost_iqm"], c["cost_ci"]), fmt_ci(c["gap_iqm"], c["gap_ci"]),
                             f"{c['fill_rate']:.3f}", f"{c['success_rate'] * 100:.0f}%", "yes" if c["beats_base"] else "no"])
    head = ["Instance", "Policy", "Cost per period (IQM [95% CI])", "Gap to optimal % (IQM [95% CI])", "Fill rate",
            "Seeds within 5%", "Beats base-stock"]
    with open(RESULTS / "tables" / "final_results.csv", "w", newline="") as f:
        csv.writer(f).writerows([head] + rows)
    with open(RESULTS / "tables" / "final_results.md", "w") as f:
        f.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    # Every seed, as promised in Part 1
    with open(RESULTS / "tables" / "every_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "instance", "seed", "learning_rate", "steps", "train_minutes", "test_cost", "gap_percent",
                    "fill_rate", "steps_to_within_5_percent", "collapsed"])
        for k in sorted(RUNS_DIR.glob("*.json")):
            r = json.load(open(k))
            g = S["optimal"][r["instance"]]["dp"]
            m = run_metrics(r, g)
            w.writerow([r["algo"], r["instance"], r["seed"], f"{r['lr']:.2e}", r["steps"], f"{m['train_minutes']:.2f}",
                        f"{m['test_cost']:.4f}", f"{m['test_gap']:.3f}", f"{m['fill_rate']:.4f}", m["steps_to_5"] or "not reached",
                        m["collapse"]])


def figures(M, cells, agg, ref, g, base):
    insts = list(INSTANCES)
    # 1. Learning curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, i in zip(axes.ravel(), insts):
        for a in ALGOS:
            curves = [M[(a, i, s)] for s in TRAIN_SEEDS if (a, i, s) in M]
            if not curves:
                continue
            n = min(len(c["curve_gap"]) for c in curves)
            x = np.array(curves[0]["curve_steps"][:n]) / 1e6
            Y = np.array([c["curve_gap"][:n] for c in curves])
            ax.fill_between(x, Y.min(0), Y.max(0), color=COLORS[a], alpha=0.15, linewidth=0)
            ax.plot(x, np.median(Y, 0), color=COLORS[a], label=a)
        ax.axhline(ref[i]["test_gap_base"], color=INK2, ls="--", lw=1.5, label="Best base-stock (test)")
        ax.axhline(ref[i]["test_gap_capped"], color=INK2, ls=":", lw=1.8, label="Capped base-stock (test)")
        ax.axhline(0, color=INK, lw=1)
        ax.set_ylim(-1, 30)
        ax.set_title(NAMES[i])
    for ax in axes[1]:
        ax.set_xlabel("Environment steps (millions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Gap to optimal cost (%)")
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=10)
    fig.suptitle("Learning curves on the validation paths: median of five seeds, band = best to worst seed "
                 "(gaps above 30% are cut off)", fontsize=12)
    save(fig, FIG_DIR / "learning_curves.png")

    # 2. Final test gaps, every seed plus IQM with 95% CI
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=False)
    for ax, i in zip(axes, insts):
        # y range: baselines and every 95% interval, plus seeds unless one is far out (those get an arrow and a label)
        his = [cells[f"{a}|{i}"]["gap_ci"][1] for a in ALGOS if f"{a}|{i}" in cells]
        level = max([ref[i]["test_gap_base"]] + [cells[f"{a}|{i}"]["gap_iqm"] for a in ALGOS if f"{a}|{i}" in cells])
        core = max([level] + [h_ for h_ in his if h_ <= 2.5 * level])
        seeds_all = [g_ for a in ALGOS if f"{a}|{i}" in cells for g_ in cells[f"{a}|{i}"]["gaps"]]
        top = max([core] + [v for v in seeds_all if v <= 2.5 * core]) * 1.15 + 0.3
        for k, a in enumerate(ALGOS):
            c = cells.get(f"{a}|{i}")
            if not c:
                continue
            xs = np.full(len(c["gaps"]), k) + np.linspace(-0.12, 0.12, len(c["gaps"]))
            gs = np.array(c["gaps"])
            ax.scatter(xs[gs <= top], gs[gs <= top], s=22, color=COLORS[a], alpha=0.45, zorder=2, edgecolor="none")
            for x_, g_ in zip(xs[gs > top], gs[gs > top]):
                ax.scatter([x_], [top * 0.97], marker="^", s=40, color=COLORS[a], zorder=3)
                ax.text(x_ + 0.08, top * 0.97, f"{g_:.0f}%", fontsize=8, color=INK2, va="center")
            ax.errorbar([k + 0.3], [c["gap_iqm"]], yerr=[[c["gap_iqm"] - c["gap_ci"][0]], [c["gap_ci"][1] - c["gap_iqm"]]],
                        fmt=MARKERS[a], color=COLORS[a], ms=8, capsize=4, lw=2, zorder=3)
        ax.axhline(ref[i]["test_gap_base"], color=INK2, ls="--", lw=1.5)
        ax.axhline(ref[i]["test_gap_capped"], color=INK2, ls=":", lw=1.8)
        ax.axhline(0, color=INK, lw=1)
        ax.text(2.45, ref[i]["test_gap_base"], " base-stock", va="center", fontsize=9, color=INK2)
        ax.text(2.45, ref[i]["test_gap_capped"], " capped", va="center", fontsize=9, color=INK2)
        ax.set_xticks(range(len(ALGOS)), ALGOS)
        ax.set_xlim(-0.5, 3.2)
        ax.set_ylim(min(-0.5, min(seeds_all + [0]) - 0.3), top)
        ax.set_title(NAMES[i])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Gap to optimal cost (%)")
    fig.suptitle("Final test (10 held-out demand paths x 100,000 periods): dots = seeds, marker = IQM with 95% CI, "
                 "triangles = seeds off the top of the chart", fontsize=12)
    save(fig, FIG_DIR / "final_gaps.png")

    # 3. Performance profile
    gaps_by_algo = {}
    for a in ALGOS:
        mat = [[M[(a, i, s)]["test_gap"] for i in insts] for s in TRAIN_SEEDS if all((a, i, s) in M for i in insts)]
        if mat:
            gaps_by_algo[a] = np.array(mat)
    if gaps_by_algo:
        taus = np.linspace(0, 15, 61)
        prof = stats.performance_profile(gaps_by_algo, taus)
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for a, (pt, lo, hi) in prof.items():
            ax.fill_between(taus, lo, hi, color=COLORS[a], alpha=0.15, linewidth=0)
            ax.plot(taus, pt, color=COLORS[a], label=a)
        ax.axvline(SUCCESS_GAP, color=INK2, ls="--", lw=1.2)
        ax.text(SUCCESS_GAP + 0.2, 0.05, "5% target", color=INK2, fontsize=10)
        ax.set_xlabel("Gap to optimal cost, t (%)")
        ax.set_ylabel("Share of runs with gap below t")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
        ax.set_title("Performance profile over all 20 runs per algorithm")
        save(fig, FIG_DIR / "performance_profile.png")

    # 4. Learned ordering rules versus the optimal policy (L = 2 instance with p = 9)
    policy_maps(M, base, "L2_p9")

    # 5. Sample efficiency
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.25
    top = 0
    for k, a in enumerate(ALGOS):
        for j, i in enumerate(insts):
            vals = [M[(a, i, s)]["steps_to_5"] for s in TRAIN_SEEDS if (a, i, s) in M]
            reached = [v / 1e6 for v in vals if v is not None]
            x = j + (k - 1) * width
            if reached:
                ax.scatter(np.full(len(reached), x), reached, color=COLORS[a], s=26, alpha=0.6, edgecolor="none")
                ax.plot([x - 0.08, x + 0.08], [np.median(reached)] * 2, color=COLORS[a], lw=3)
                top = max(top, max(reached))
            missed = len(vals) - len(reached)
            if missed:
                ax.text(x, -0.12, f"{missed}x", ha="center", va="top", fontsize=8, color=COLORS[a])
    for a in ALGOS:
        ax.plot([], [], color=COLORS[a], lw=3, label=a)
    ax.set_xticks(range(len(insts)), [NAMES[i] for i in insts])
    ax.set_ylabel("Steps until within 5% (millions)")
    ax.set_ylim(-0.3, max(top * 1.1, 0.5))
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Sample efficiency: dots = seeds, bar = median (\"nx\" under a group = seeds that never got within 5%)")
    save(fig, FIG_DIR / "sample_efficiency.png")

    # 6. Wall-clock training time per run
    if agg:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        vals = [agg[a]["train_minutes"] for a in ALGOS if a in agg]
        bars = ax.bar([a for a in ALGOS if a in agg], vals, color=[COLORS[a] for a in ALGOS if a in agg], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} min", ha="center", va="bottom", fontsize=11, color=INK)
        ax.set_ylabel("Training minutes per run (mean,\nnot counting evaluation)")
        ax.set_title("Training time per run on the M1 (one CPU core each)")
        ax.grid(axis="x", visible=False)
        save(fig, FIG_DIR / "training_time.png")


def wcorr(x, y, w):
    """Weighted correlation; negative means larger orders when less stock is on hand or on the way."""
    mx, my = np.sum(w * x), np.sum(w * y)
    vx, vy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def policy_agreement(runs, base):
    """Share of the states the optimal policy actually visits (weighted by how often) where each trained
    policy orders within one unit of the optimal order. Averaged over seeds."""
    from inventory_rl.agents import ALGO_CLASSES
    out = {}
    for inst, (L, p) in INSTANCES.items():
        opt_fn = policy_order_fn(np.load(DP_DIR / f"optimal_policy_{inst}.npy"), L)
        X, w = visited_states(opt_fn, demand_paths(1, 21_000, 40_000)[0], L, p)
        a_opt = opt_fn(X)
        out[inst] = {}
        for a in ALGOS:
            shares, corrs = [], []
            for s in TRAIN_SEEDS:
                f = MODELS_DIR / f"{a}_{inst}_s{s}.zip"
                if (a, inst, s) in runs and f.exists():
                    model = ALGO_CLASSES[a].load(f, device="cpu")
                    acts, _ = model.predict((X / OBS_SCALE).astype(np.float32), deterministic=True)
                    shares.append(float(np.sum(w * (np.abs(np.asarray(acts) - a_opt) <= 1))))
                    corrs.append(wcorr(np.asarray(acts, dtype=float), X.sum(axis=1), w))
            if shares:
                out[inst][a] = dict(mean=float(np.mean(shares)), per_seed=shares,
                                    order_vs_position_corr=float(np.nanmean(corrs)))
        out[inst]["optimal_order_vs_position_corr"] = wcorr(np.asarray(a_opt, dtype=float), X.sum(axis=1), w)
    return out


def policy_maps(M, base, inst):
    from inventory_rl.agents import ALGO_CLASSES
    L, p = INSTANCES[inst]
    s0 = np.arange(0, 31)
    s1 = np.arange(0, 21)
    S0, S1 = np.meshgrid(s0, s1, indexing="ij")
    X = np.stack([S0.ravel(), S1.ravel()], 1).astype(float)
    panels = []
    opt = np.load(DP_DIR / f"optimal_policy_{inst}.npy")
    panels.append(("Optimal (DP)\n", policy_order_fn(opt, L)(X)))
    prm = base[inst]["params"]["capped_base_stock"]
    panels.append((f"Capped base-stock\n(S={prm['S']}, cap={prm['cap']})", capped_base_stock_fn(prm["S"], prm["cap"])(X)))
    for a in ALGOS:
        seeds = [s for s in TRAIN_SEEDS if (a, inst, s) in M and (MODELS_DIR / f"{a}_{inst}_s{s}.zip").exists()]
        if not seeds:
            continue
        med = sorted(seeds, key=lambda s: M[(a, inst, s)]["test_cost"])[len(seeds) // 2]   # median seed
        model = ALGO_CLASSES[a].load(MODELS_DIR / f"{a}_{inst}_s{med}.zip", device="cpu")
        acts, _ = model.predict((X / OBS_SCALE).astype(np.float32), deterministic=True)
        panels.append((f"{a}\n(median seed {med})", acts))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.9), sharey=True)
    vmax = max(8, max(int(np.max(acts)) for _, acts in panels))      # same color scale for every panel
    for ax, (title, acts) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(np.asarray(acts).reshape(len(s0), len(s1)).T, origin="lower", cmap="Blues", vmin=0, vmax=vmax,
                       extent=[-0.5, 30.5, -0.5, 20.5], aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Stock now, s0")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Order arriving next period, s1")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Order quantity")
    fig.suptitle(f"Ordering rule by state, {NAMES[inst]} (darker = larger order)", fontsize=12, y=1.08)
    save(fig, FIG_DIR / f"policy_map_{inst}.png")


if __name__ == "__main__":
    main()
