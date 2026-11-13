import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src import config
from src.evaluation import evaluate, paired_difference, summarize
from src.policies import condition_based, fixed_interval, run_to_failure
from src.pump_env import PumpEnv
from src.tabular import (SHAPE, VIB_SCALE, collect, discretise, estimate_model, greedy_table, q_learning, table_policy,
                         value_iteration)

MODEL_EPISODES = 5000
Q_EPISODES = 20_000
GAMMAS = [0.99, 0.995, 0.998]
MIN_SAMPLES = 5
EVAL_EVERY = 1000
CURVE_SEEDS = config.VAL_SEEDS[:10]
TRAIN_SEEDS = [config.TRAIN_SEED_START + k for k in range(max(MODEL_EPISODES, Q_EPISODES))]
COLORS = {0: "#e4e3df", 1: "#2a78d6", 2: "#eb6834"}
NAMES = {0: "continue", 1: "minor", 2: "major"}


def make_behaviour(rng):
    # exploring policy for the model data: a condition based rule with random limits in every episode, plus
    # random actions (2% of weeks on a healthy pump, 10% once excess vibration passes 2 mm/s)
    def behaviour():
        base = condition_based(rng.uniform(1, 7.5), None if rng.random() < 0.2 else rng.uniform(0.72, 0.79))

        def policy(obs):
            explore = 0.10 if obs[0] * VIB_SCALE >= 2 else 0.02
            return int(rng.integers(3)) if rng.random() < explore else base(obs)
        return policy
    return behaviour


def visit_counts(table, seeds):
    env = PumpEnv()
    counts = np.zeros(len(table))
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        truncated = False
        while not truncated:
            s = discretise(obs)
            counts[s] += 1
            obs, _, _, truncated, _ = env.step(int(table[s]))
    return counts


def policy_map(ax, table, known, title):
    grid = table.reshape(SHAPE)[:, :, 1]
    has_data = known.any(axis=1).reshape(SHAPE)[:, :, 1]
    for v in range(SHAPE[0]):
        for e in range(SHAPE[1]):
            if has_data[v, e]:
                ax.add_patch(plt.Rectangle((e, v), 1, 1, facecolor=COLORS[grid[v, e]], edgecolor="white", linewidth=2))
            else:
                ax.add_patch(plt.Rectangle((e + 0.04, v + 0.04), 0.92, 0.92, facecolor="white", edgecolor="#9a9994",
                                           linewidth=0.8, hatch="//"))
    ax.set_xlim(0, SHAPE[1])
    ax.set_ylim(0, SHAPE[0])
    ax.set_xticks(np.arange(SHAPE[1]) + 0.5, ["> 0.79", "0.78 to 0.79", "0.77 to 0.78", "0.76 to 0.77", "0.75 to 0.76",
                                               "0.74 to 0.75", "≤ 0.74"], rotation=35, ha="right")
    ax.set_yticks(np.arange(SHAPE[0]) + 0.5, ["< 1", "1 to 2", "2 to 3", "3 to 4", "4 to 5", "5 to 6", "≥ 6"])
    ax.set_xlabel("8 week efficiency")
    ax.set_ylabel("excess vibration (mm/s)")
    ax.set_title(title, loc="left")
    ax.grid(False)


def main():
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = config.VAL_SEEDS
    b = config.BASELINES
    policies = {"run to failure": run_to_failure(), "fixed interval": fixed_interval(b["minor_weeks"], b["major_weeks"]),
                "condition based": condition_based(b["vib_limit"], b["eff_limit"])}

    data = collect(PumpEnv(), TRAIN_SEEDS[:MODEL_EPISODES], make_behaviour(np.random.default_rng(0)))
    print(f"model data: {len(data['s'])} steps from {MODEL_EPISODES} episodes")

    gamma_rows, vi = [], {}
    for gamma in GAMMAS:
        R, M, counts = estimate_model(data, gamma)
        known = counts >= MIN_SAMPLES
        V, Q, iterations = value_iteration(R, M, known)
        table = greedy_table(Q, known)
        vi[gamma] = (V, Q, table, known)
        policies[f"value iteration, gamma {gamma}"] = table_policy(table)
        gamma_rows.append({"gamma": gamma, "iterations": iterations, "known_pairs": int(known.sum()),
                           "states_with_data": int(known.any(axis=1).sum())})
        print(f"value iteration, gamma {gamma}: {iterations} sweeps")

    gamma = config.GAMMA
    V, Q_vi, vi_table, known = vi[gamma]
    np.savez(config.MODELS_DIR / "tabular_vi.npz", V=V, Q=Q_vi, table=vi_table, known=known, gamma=gamma)

    def curve_point(Q, visits):
        return summarize(evaluate(table_policy(greedy_table(Q, visits > 0)), CURVE_SEEDS))["cost"]

    Q, visits, curve = q_learning(PumpEnv(), Q_EPISODES, gamma, TRAIN_SEEDS, eval_every=EVAL_EVERY, evaluate=curve_point)
    q_table = greedy_table(Q, visits > 0)
    np.savez(config.MODELS_DIR / "tabular_q.npz", Q=Q, visits=visits, table=q_table, gamma=gamma)
    policies["Q-learning"] = table_policy(q_table)
    curve = pd.DataFrame(curve, columns=["episodes", "cost"])
    curve.to_csv(config.TABLES_DIR / "tabular_learning_curve.csv", index=False)
    print(f"Q-learning: {Q_EPISODES} episodes, last curve point {curve['cost'].iloc[-1]:.2f}")

    runs = {name: evaluate(policy, seeds) for name, policy in policies.items()}
    result_rows = []
    for name, r in runs.items():
        diff, ci = paired_difference(r, runs["condition based"])
        result_rows.append({"policy": name, **summarize(r), "diff_vs_condition_based": diff, "diff_ci95": ci})
    results = pd.DataFrame(result_rows)
    results.to_csv(config.TABLES_DIR / "tabular_results_val.csv", index=False)
    for row in gamma_rows:
        r = results[results["policy"] == f"value iteration, gamma {row['gamma']}"].iloc[0]
        row.update({k: r[k] for k in ["cost", "cost_ci95", "major", "impeller_failures", "diff_vs_condition_based", "diff_ci95"]})
    pd.DataFrame(gamma_rows).to_csv(config.TABLES_DIR / "tabular_gamma.csv", index=False)
    print(results[["policy", "cost", "cost_ci95", "minor", "major", "bearing_failures", "impeller_failures",
                   "diff_vs_condition_based", "diff_ci95"]].round(3).to_string(index=False))

    # how often the two tabular policies agree, weighted by how often the value iteration policy visits each state
    visits_vi = visit_counts(vi_table, seeds)
    both = known.any(axis=1) & (visits > 0).any(axis=1)
    same = vi_table == q_table
    busy = both & (visits_vi >= 20)
    rare = both & (visits_vi > 0) & (visits_vi < 20)
    agreement = {"states_visited": int((visits_vi > 0).sum()), "busy_states": int(busy.sum()),
                 "agree_busy_states": float(same[busy].mean()),
                 "agree_weighted_by_visits": float((same * visits_vi)[both].sum() / visits_vi[both].sum()),
                 "rare_states": int(rare.sum()), "agree_rare_states": float(same[rare].mean()) if rare.any() else np.nan}
    pd.Series(agreement).to_csv(config.TABLES_DIR / "tabular_agreement.csv", header=["value"])
    print(pd.Series(agreement).round(3).to_string())
    disagree = np.nonzero(both & (visits_vi > 0) & ~same)[0]
    bands = np.unravel_index(disagree, SHAPE)
    disagree_rows = [{"state": int(s), "vib_band": int(v), "eff_band": int(e), "load_band": int(l),
                      "visits": int(visits_vi[s]), "value_iteration": NAMES[int(vi_table[s])], "q_learning": NAMES[int(q_table[s])]}
                     for s, v, e, l in zip(disagree, *bands)]
    pd.DataFrame(disagree_rows).to_csv(config.TABLES_DIR / "tabular_disagreements.csv", index=False)

    levels = {name: summarize(evaluate(policies[name], CURVE_SEEDS))["cost"]
              for name in ["run to failure", "condition based", f"value iteration, gamma {gamma}"]}
    plot_curve(curve, levels)
    plot_maps(vi_table, known, q_table, visits, gamma)


def plot_curve(curve, levels):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve["episodes"], curve["cost"], marker="o", color="#2a78d6", lw=2, label="Q-learning (greedy policy)")
    for (name, level), color, style in zip(levels.items(), ["#52514e", "#eb6834", "#1baf7a"], ["-.", "--", ":"]):
        ax.axhline(level, color=color, ls=style, lw=1.5, label=f"{name}: {level:.1f}")
    ax.set_xlabel("training episodes (10 years each)")
    ax.set_ylabel("mean 10 year cost (units)")
    ax.set_title(f"Q-learning learning curve on {len(CURVE_SEEDS)} validation seeds", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "tabular_learning_curve.png", dpi=150)


def plot_maps(vi_table, known, q_table, visits, gamma):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    policy_map(axes[0], vi_table, known, f"Value iteration (gamma {gamma})")
    policy_map(axes[1], q_table, visits > 0, "Q-learning")
    handles = [Patch(facecolor=COLORS[a], label=NAMES[a]) for a in COLORS]
    handles.append(Patch(facecolor="white", edgecolor="#9a9994", hatch="//", label="no data"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Chosen action at load 0.75", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(config.FIGURES_DIR / "tabular_policy_map.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
