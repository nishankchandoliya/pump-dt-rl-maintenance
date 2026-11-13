import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.dqn import load_network, network_policy
from src.evaluation import COST_PARTS, evaluate, paired_difference, summarize
from src.policies import condition_based, fixed_interval, run_to_failure
from src.tabular import table_policy

# the first and only use of the test seeds. Every policy is frozen: nothing below is tuned or retrained
SEEDS = config.TEST_SEEDS
REFERENCE = "condition based"
DQN_RUNS = ["dqn_run0", "dqn_run1", "dqn_run2", "dqn_double_run0", "dqn_double_run1", "dqn_double_run2"]
COLORS = {"energy": "#8a5cd6", "minor": "#2a78d6", "major": "#eb6834", "failure": "#d03b3b", "downtime": "#9a9994"}


def scale(d, keys, factor):
    return {k: v * factor if k in keys else v for k, v in d.items()}


# sensitivity: the same frozen policies scored with other cost values. Robustness: a pump that wears 25% faster
SCENARIOS = {
    "base": {},
    "failure cost x0.5": {"costs": scale(config.COSTS, ["bearing_failure", "impeller_failure"], 0.5)},
    "failure cost x2": {"costs": scale(config.COSTS, ["bearing_failure", "impeller_failure"], 2.0)},
    "downtime cost x0.5": {"costs": scale(config.COSTS, ["downtime_week"], 0.5)},
    "downtime cost x2": {"costs": scale(config.COSTS, ["downtime_week"], 2.0)},
    "wear 25% faster": {"params": scale(config.PUMP, ["bearing_rate", "impeller_rate"], 1.25)},
}


def make_policies():
    b = config.BASELINES
    return {
        "run to failure": run_to_failure(),
        "fixed interval": fixed_interval(b["minor_weeks"], b["major_weeks"]),
        "condition based": condition_based(b["vib_limit"], b["eff_limit"]),
        "value iteration": table_policy(np.load(config.MODELS_DIR / "tabular_vi.npz")["table"]),
        "Q-learning": table_policy(np.load(config.MODELS_DIR / "tabular_q.npz")["table"]),
        "DQN": network_policy(load_network(config.MODELS_DIR / "dqn.pt")),
    }


def compare(runs):
    rows = []
    for name, r in runs.items():
        diff, ci = paired_difference(r, runs[REFERENCE])
        rows.append({"policy": name, **summarize(r), "diff_vs_condition_based": diff, "diff_ci95": ci})
    return pd.DataFrame(rows)


def main():
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    policies = make_policies()

    tables = []
    for scenario, kwargs in SCENARIOS.items():
        runs = {name: evaluate(policy, SEEDS, **kwargs) for name, policy in policies.items()}
        table = compare(runs).assign(scenario=scenario)
        tables.append(table)
        if scenario == "base":
            base_runs = runs
        print(f"\n{scenario}")
        print(table[["policy", "cost", "cost_ci95", "bearing_failures", "impeller_failures", "availability",
                     "diff_vs_condition_based", "diff_ci95"]].round(3).to_string(index=False))
    results = pd.concat(tables)
    results.to_csv(config.TABLES_DIR / "final_results_test.csv", index=False)

    # DQN against tabular Q-learning, both on the base costs
    diff, ci = paired_difference(base_runs["DQN"], base_runs["Q-learning"])
    print(f"\nDQN minus Q-learning (base): {diff:.3f} ± {ci:.3f}")

    # every trained DQN network, to show how much the result depends on the training run
    rows = []
    for name in DQN_RUNS:
        r = evaluate(network_policy(load_network(config.MODELS_DIR / f"{name}.pt")), SEEDS)
        diff, ci = paired_difference(r, base_runs[REFERENCE])
        rows.append({"network": name, **summarize(r), "diff_vs_condition_based": diff, "diff_ci95": ci})
    nets = pd.DataFrame(rows)
    nets.to_csv(config.TABLES_DIR / "final_dqn_networks_test.csv", index=False)
    print(nets[["network", "cost", "cost_ci95", "bearing_failures", "impeller_failures", "diff_vs_condition_based",
                "diff_ci95"]].round(3).to_string(index=False))

    base = results[results["scenario"] == "base"].set_index("policy")
    plot_breakdown(base)
    plot_availability(base)
    plot_sensitivity(results)


def plot_breakdown(base):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    left = np.zeros(len(base))
    for part in COST_PARTS:
        values = base[f"cost_{part}"].to_numpy()
        ax.barh(base.index, values, left=left, color=COLORS[part], edgecolor="white", label=part)
        left += values
    for y, (cost, ci) in enumerate(zip(base["cost"], base["cost_ci95"])):
        ax.text(cost + 0.4, y, f"{cost:.1f} ± {ci:.1f}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, base["cost"].max() * 1.15)
    ax.set_xlabel(f"mean 10 year cost (units), {len(SEEDS)} test seeds")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9)
    ax.set_title("Where the money goes", loc="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "final_cost_breakdown.png", dpi=150)


def plot_availability(base):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    failures = base["bearing_failures"] + base["impeller_failures"]
    axes[0].barh(base.index, failures, color="#d03b3b")
    axes[1].barh(base.index, base["availability"] * 100, color="#1baf7a")
    for y, (f, a) in enumerate(zip(failures, base["availability"] * 100)):
        axes[0].text(f + 0.05, y, f"{f:.2f}", va="center", fontsize=9)
        axes[1].text(a + 0.05, y, f"{a:.1f}%", va="center", fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("failures per 10 years")
    axes[1].set_xlabel("availability (%)")
    axes[1].set_xlim(94, 100)
    axes[0].set_xlim(0, failures.max() * 1.2)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "final_availability_failures.png", dpi=150)


def plot_sensitivity(results):
    # fixed interval is 3 to 6 units worse in every case (see the table), so it is left out to keep the scale readable
    shown = ["value iteration", "Q-learning", "DQN"]
    colors = {"value iteration": "#1baf7a", "Q-learning": "#8a5cd6", "DQN": "#2a78d6"}
    scenarios = list(SCENARIOS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, name in enumerate(shown):
        part = results[results["policy"] == name].set_index("scenario").loc[scenarios]
        y = np.arange(len(scenarios)) + (k - 1) * 0.22
        ax.errorbar(part["diff_vs_condition_based"], y, xerr=part["diff_ci95"], fmt="o", color=colors[name],
                    capsize=3, label=name)
    ax.axvline(0, color="#1baf7a", ls="--", lw=1)
    ax.set_yticks(np.arange(len(scenarios)), scenarios)
    ax.invert_yaxis()
    ax.set_xlabel("cost minus condition based on the same seeds (units, 95% CI). Left of 0 is better")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Sensitivity and robustness on the test seeds", loc="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "final_sensitivity.png", dpi=150)


if __name__ == "__main__":
    main()
