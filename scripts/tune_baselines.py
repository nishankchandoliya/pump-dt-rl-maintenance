import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evaluation import evaluate, paired_difference, summarize
from src.policies import condition_based, fixed_interval, run_to_failure

# a value of None means the action is never planned (the part is only renewed by a failure or by the other action)
FIXED_GRID = {"minor_weeks": list(range(30, 95, 5)), "major_weeks": [None] + list(range(100, 300, 20))}
CONDITION_GRID = {"vib_limit": [round(float(v), 2) for v in np.arange(1.0, 7.01, 0.25)],
                  "eff_limit": [None] + [round(float(e), 3) for e in np.arange(0.785, 0.7299, -0.005)]}


def search(make_policy, grid, seeds):
    names = list(grid)
    keys = list(itertools.product(*grid.values()))
    runs, rows = [], []
    for key in keys:
        params = dict(zip(names, key))
        result = evaluate(make_policy(**params), seeds)
        runs.append(result)
        rows.append({**params, **summarize(result)})
    table = pd.DataFrame(rows)
    best = int(table["cost"].idxmin())
    diffs = [paired_difference(r, runs[best]) for r in runs]
    table["diff_vs_best"] = [d[0] for d in diffs]
    table["diff_ci95"] = [d[1] for d in diffs]
    # not clearly worse than the best: the 95% interval of the paired difference reaches zero
    table["tied_with_best"] = table["diff_vs_best"] - table["diff_ci95"] <= 1e-9
    return table, dict(zip(names, keys[best])), runs[best]


def heatmap(ax, table, grid, title, labels, vmin, vmax):
    names = list(grid)
    cost = table["cost"].to_numpy().reshape(len(grid[names[0]]), len(grid[names[1]]))
    tied = table["tied_with_best"].to_numpy().reshape(cost.shape)
    image = ax.imshow(cost, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    rows, cols = np.nonzero(tied)
    ax.scatter(cols, rows, s=10, color="white", edgecolor="black", linewidth=0.5, label="tied with the best of its panel")
    r, c = np.unravel_index(cost.argmin(), cost.shape)
    ax.scatter([c], [r], marker="*", s=200, color="#eb6834", edgecolor="black", linewidth=0.6, label="best of its panel")
    ax.text(c, r - 0.7, f"{cost.min():.1f}", ha="center", va="bottom", fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none"})
    ax.set_xticks(range(cost.shape[1]), ["never" if v is None else v for v in grid[names[1]]], rotation=90)
    ax.set_yticks(range(cost.shape[0]), grid[names[0]])
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[0])
    ax.set_title(title, loc="left")
    ax.grid(False)
    return image


def plot_tuning(fixed, cond):
    # both panels share one colour scale so their colours can be compared
    vmin = min(fixed["cost"].min(), cond["cost"].min())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    heatmap(axes[0], fixed, FIXED_GRID, "Fixed interval", ["minor every (weeks)", "major every (weeks)"], vmin, vmin + 10)
    image = heatmap(axes[1], cond, CONDITION_GRID, "Condition based",
                    ["minor when excess vibration reaches (mm/s)", "major when 8 week efficiency falls to"], vmin, vmin + 10)
    fig.colorbar(image, ax=axes, label="mean 10 year cost over 50 validation seeds (units)", shrink=0.85)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.45, -0.11))
    fig.savefig(config.FIGURES_DIR / "baseline_tuning.png", dpi=150, bbox_inches="tight")


def main():
    seeds = config.VAL_SEEDS
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fixed, fixed_best, fixed_runs = search(fixed_interval, FIXED_GRID, seeds)
    fixed.to_csv(config.TABLES_DIR / "baseline_tuning_fixed.csv", index=False)
    print(f"fixed interval: best {fixed_best}, {fixed['tied_with_best'].sum()} of {len(fixed)} settings tied with it")

    cond, cond_best, cond_runs = search(condition_based, CONDITION_GRID, seeds)
    cond.to_csv(config.TABLES_DIR / "baseline_tuning_condition.csv", index=False)
    print(f"condition based: best {cond_best}, {cond['tied_with_best'].sum()} of {len(cond)} settings tied with it")

    policies = {"run to failure": evaluate(run_to_failure(), seeds), "fixed interval": fixed_runs, "condition based": cond_runs}
    best_name = min(policies, key=lambda name: policies[name]["cost"].mean())
    rows = []
    for name, runs in policies.items():
        saving, saving_ci = paired_difference(policies["run to failure"], runs)
        diff, diff_ci = paired_difference(runs, policies[best_name])
        rows.append({"policy": name, **summarize(runs), "saving_vs_run_to_failure": saving, "saving_ci95": saving_ci,
                     "diff_vs_best": diff, "diff_ci95": diff_ci})
    results = pd.DataFrame(rows)
    results.to_csv(config.TABLES_DIR / "baseline_results_val.csv", index=False)
    print(f"\nvalidation seeds ({len(seeds)}), 10 year cost in units:")
    print(results.round(3).T.to_string(header=False))

    plot_tuning(fixed, cond)


if __name__ == "__main__":
    main()
