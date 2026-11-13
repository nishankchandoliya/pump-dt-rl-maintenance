import os
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config
from src.dqn import SETTINGS, load_network, network_policy, train
from src.evaluation import evaluate, paired_difference, summarize
from src.policies import condition_based, run_to_failure
from src.pump_env import PumpEnv
from src.tabular import table_policy

EPISODES = 800
EVAL_EVERY = 50
RUN_SEEDS = [0, 1, 2]
VARIANTS = {"DQN": False, "Double DQN": True}
TRAIN_SEEDS = [config.TRAIN_SEED_START + k for k in range(EPISODES)]
COLORS = {"DQN": "#2a78d6", "Double DQN": "#eb6834"}


def val_cost(net):
    return summarize(evaluate(network_policy(net), config.VAL_SEEDS))["cost"]


def train_run(variant, run):
    # one training run. Runs are independent, so they can go to separate processes
    settings = {**SETTINGS, "double": VARIANTS[variant]}
    agent, curve = train(PumpEnv(), EPISODES, config.GAMMA, TRAIN_SEEDS, settings, seed=run,
                         eval_every=EVAL_EVERY, evaluate=val_cost)
    print(f"{variant} run {run}: last curve point {curve[-1][2]:.2f}", flush=True)
    return {"variant": variant, "run": run, "settings": settings, "curve": curve, "state": agent.online.state_dict()}


def value_check(net, seeds):
    # the network's own value of a new pump against the discounted cost really seen when the pump follows the
    # network. The end of the episode is closed with the network's value, which counts only 0.995^520 = 7%
    env = PumpEnv()
    predicted, observed = [], []
    with torch.no_grad():
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            predicted.append(net(torch.as_tensor(obs)).max().item())
            total, discount, truncated = 0.0, 1.0, False
            while not truncated:
                obs, reward, _, truncated, info = env.step(int(net(torch.as_tensor(obs)).argmax()))
                total += discount * reward
                discount *= config.GAMMA ** info["weeks"]
            observed.append(total + discount * net(torch.as_tensor(obs)).max().item())
    return np.mean(predicted), np.mean(observed)


def main():
    torch.set_num_threads(1)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(variant, run) for variant in VARIANTS for run in RUN_SEEDS]
    with Pool(min(len(jobs), os.cpu_count())) as pool:
        trained = pool.starmap(train_run, jobs)

    seeds = config.VAL_SEEDS
    b = config.BASELINES
    reference = {"run to failure": run_to_failure(), "condition based": condition_based(b["vib_limit"], b["eff_limit"]),
                 "Q-learning": table_policy(np.load(config.MODELS_DIR / "tabular_q.npz")["table"])}
    runs = {name: evaluate(policy, seeds) for name, policy in reference.items()}
    curve_rows, run_rows = [], []
    for t in trained:
        name = f"{t['variant']} run {t['run']}"
        path = config.MODELS_DIR / f"dqn_{'double_' if t['settings']['double'] else ''}run{t['run']}.pt"
        torch.save({"state": t["state"], "settings": t["settings"], "gamma": config.GAMMA, "episodes": EPISODES}, path)
        net = load_network(path)
        runs[name] = evaluate(network_policy(net), seeds)
        predicted, observed = value_check(net, seeds)
        curve_rows += [{"variant": t["variant"], "run": t["run"], "episodes": e, "steps": s, "cost": c}
                       for e, s, c in t["curve"]]
        run_rows.append({"variant": t["variant"], "run": t["run"], "model": path.name, "cost": runs[name]["cost"].mean(),
                         "predicted_value": predicted, "observed_value": observed,
                         "overestimate": predicted - observed})

    rows = []
    for name, r in runs.items():
        diff, ci = paired_difference(r, runs["condition based"])
        rows.append({"policy": name, **summarize(r), "diff_vs_condition_based": diff, "diff_ci95": ci})
    results = pd.DataFrame(rows)
    results.to_csv(config.TABLES_DIR / "dqn_results_val.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(config.TABLES_DIR / "dqn_learning_curve.csv", index=False)
    print(results[["policy", "cost", "cost_ci95", "minor", "major", "bearing_failures", "impeller_failures",
                   "diff_vs_condition_based", "diff_ci95"]].round(3).to_string(index=False))

    # the network used in Phases 8 and 9: the run with the lowest validation cost
    run_table = pd.DataFrame(run_rows)
    run_table["chosen"] = run_table.index == run_table["cost"].idxmin()
    run_table.to_csv(config.TABLES_DIR / "dqn_runs.csv", index=False)
    print(run_table.round(3).to_string(index=False))
    chosen = run_table[run_table["chosen"]].iloc[0]
    torch.save(torch.load(config.MODELS_DIR / chosen["model"]), config.MODELS_DIR / "dqn.pt")
    print(f"chosen: {chosen['variant']} run {chosen['run']} ({chosen['model']}), cost {chosen['cost']:.2f}")

    levels = {name: summarize(runs[name])["cost"] for name in ["run to failure", "condition based", "Q-learning"]}
    plot_curves(curves, levels)


def plot_curves(curves, levels):
    # top row: the whole training. Bottom row: the same curves from episode 300 with a finer scale
    fig, axes = plt.subplots(2, len(VARIANTS), figsize=(12, 7.5))
    for col, variant in enumerate(VARIANTS):
        for row in range(2):
            ax = axes[row, col]
            for run, part in curves[curves["variant"] == variant].groupby("run"):
                ax.plot(part["episodes"], part["cost"], color=COLORS[variant], lw=1.5, alpha=1 - 0.3 * run,
                        marker="o", ms=3, label=f"run {run}")
            for (name, level), color, style in zip(levels.items(), ["#52514e", "#1baf7a", "#8a5cd6"], ["-.", "--", ":"]):
                ax.axhline(level, color=color, ls=style, lw=1.5, label=f"{name}: {level:.1f}")
        axes[0, col].set_title(variant, loc="left")
        axes[0, col].legend(frameon=False, fontsize=8, loc="center right")
        axes[1, col].set_ylim(19.4, 21)
        axes[1, col].set_xlim(290, EPISODES + 10)
        axes[1, col].set_xlabel("training episodes (10 years each)")
    axes[0, 0].set_ylabel(f"mean 10 year cost, {len(config.VAL_SEEDS)} validation seeds")
    axes[1, 0].set_ylabel("same, from episode 300")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "dqn_learning_curves.png", dpi=150)


if __name__ == "__main__":
    main()
