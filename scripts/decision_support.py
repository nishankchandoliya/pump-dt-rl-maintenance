import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src import config
from src.advisor import ACTIONS, recommend
from src.dqn import load_network, network_policy
from src.policies import VIB_SCALE, condition_based, run_to_failure
from src.pump_env import CONTINUE, MAJOR, MINOR, PumpEnv
from src.pump_twin import sensor_values
from src.tabular import SHAPE, discretise

EXAMPLE_SEED = config.VAL_SEEDS[0]
COLORS = {CONTINUE: "#e4e3df", MINOR: "#2a78d6", MAJOR: "#eb6834"}
# hidden wear of the scripted cases. Readings come from the twin without noise and the ages are close to
# the median age at that wear level for a pump at load 0.75 (shorter at load 0.90)
CASES = [
    ("healthy pump in mid life", 0.75, 0.1, 0.2, 54, 44),
    ("bearing half worn", 0.75, 0.5, 0.3, 86, 66),
    ("bearing close to failure", 0.75, 0.9, 0.3, 97, 66),
    ("impeller worn, bearing healthy", 0.75, 0.1, 0.8, 54, 175),
    ("bearing and impeller worn", 0.75, 0.7, 0.7, 92, 152),
    ("high load, bearing worn", 0.90, 0.75, 0.3, 65, 46),
]


def trace(policy, seed):
    # the observation at each decision, the action taken and whether the step ended in a failure
    env = PumpEnv()
    obs, _ = env.reset(seed=seed)
    rows = []
    truncated = False
    while not truncated:
        action = policy(obs)
        week = env.twin.week
        next_obs, _, _, truncated, info = env.step(action)
        rows.append({"week": week, "vibration": obs[0] * VIB_SCALE,
                     "efficiency": config.PUMP["eff_new"] - config.PUMP["eff_loss"] * obs[2], "action": action,
                     "failure": info["failure"], "state": discretise(obs), "obs": obs})
        obs = next_obs
    return pd.DataFrame(rows)


def trigger_levels(traces):
    # where in the readings each policy starts maintenance
    rows = []
    for name, t in traces.items():
        minor, major = t[t["action"] == MINOR], t[t["action"] == MAJOR]
        rows.append({"policy": name, "minor_per_episode": len(minor) / t["seed"].nunique(),
                     "major_per_episode": len(major) / t["seed"].nunique(),
                     "vibration_at_minor_p10": minor["vibration"].quantile(0.1),
                     "vibration_at_minor_median": minor["vibration"].median(),
                     "vibration_at_minor_p90": minor["vibration"].quantile(0.9),
                     "efficiency_at_major_p10": major["efficiency"].quantile(0.1),
                     "efficiency_at_major_median": major["efficiency"].median(),
                     "efficiency_at_major_p90": major["efficiency"].quantile(0.9),
                     "vibration_at_major_median": major["vibration"].median(),
                     "share_of_majors_above_2mm_s": (major["vibration"] >= 2).mean()})
    return pd.DataFrame(rows)


def decision_map(net, seeds):
    # the network's action on every week of run to failure episodes, which pass through all wear levels; for each
    # band of the Phase 6 table the most common action is shown
    policy = network_policy(net)
    t = pd.concat([trace(run_to_failure(), seed) for seed in seeds])
    t["dqn_action"] = [policy(obs) for obs in t["obs"]]
    counts = np.zeros((np.prod(SHAPE), 3))
    np.add.at(counts, (t["state"].to_numpy(), t["dqn_action"].to_numpy()), 1)
    return counts


def scripted_cases(net):
    rows = []
    for name, load, b, w, bearing_weeks, impeller_weeks in CASES:
        vibration, temperature, efficiency = sensor_values(config.PUMP, load, b, w)
        r = recommend(net, vibration, temperature, efficiency, load, bearing_weeks, impeller_weeks)
        rows.append({"case": name, "load": load, "bearing_wear": b, "impeller_wear": w, "vibration": vibration,
                     "temperature": temperature, "efficiency": efficiency, "bearing_weeks": bearing_weeks,
                     "impeller_weeks": impeller_weeks, "zone": r["vibration_zone"],
                     "temperature_status": r["temperature_status"], "advice": r["name"],
                     **{f"cost_{ACTIONS[a].split()[0]}": r["cost"][a] for a in ACTIONS}, "margin": r["margin"]})
    return pd.DataFrame(rows)


def main():
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    net = load_network(config.MODELS_DIR / "dqn.pt")
    b = config.BASELINES
    policies = {"DQN": network_policy(net), "condition based": condition_based(b["vib_limit"], b["eff_limit"])}

    traces = {}
    for name, policy in policies.items():
        traces[name] = pd.concat([trace(policy, seed).assign(seed=seed) for seed in config.VAL_SEEDS])
    triggers = trigger_levels(traces)
    triggers.to_csv(config.TABLES_DIR / "decision_triggers.csv", index=False)
    print(triggers.round(3).T.to_string())

    cases = scripted_cases(net)
    cases.to_csv(config.TABLES_DIR / "advisor_cases.csv", index=False)
    print(cases[["case", "vibration", "temperature", "efficiency", "zone", "advice", "cost_continue", "cost_minor",
                 "cost_major", "margin"]].round(2).to_string(index=False))

    counts = decision_map(net, config.VAL_SEEDS)
    plot_example({name: t[t["seed"] == EXAMPLE_SEED] for name, t in traces.items()})
    plot_triggers(traces)
    plot_map(counts)


def plot_example(traces):
    dqn, cb = traces["DQN"], traces["condition based"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.4]})
    axes[0].plot(dqn["week"], dqn["vibration"], color="#52514e", lw=1)
    axes[0].axhline(config.BASELINES["vib_limit"], color="#1baf7a", ls="--", lw=1.2)
    axes[0].set_ylabel("excess vibration (mm/s)")
    axes[1].plot(dqn["week"], dqn["efficiency"], color="#52514e", lw=1)
    axes[1].axhline(config.BASELINES["eff_limit"], color="#1baf7a", ls="--", lw=1.2)
    axes[1].set_ylabel("8 week efficiency")
    for ax, column in zip(axes[:2], ["vibration", "efficiency"]):
        for action, marker in [(MINOR, "o"), (MAJOR, "s")]:
            rows = dqn[dqn["action"] == action]
            ax.scatter(rows["week"], rows[column], color=COLORS[action], marker=marker, s=40, zorder=3,
                       label=f"DQN: {ACTIONS[action]}")
    handles = [plt.Line2D([], [], color="#1baf7a", ls="--", label="condition based limits"),
               plt.Line2D([], [], color=COLORS[MINOR], marker="o", ls="", label=f"DQN: {ACTIONS[MINOR]}"),
               plt.Line2D([], [], color=COLORS[MAJOR], marker="s", ls="", label=f"DQN: {ACTIONS[MAJOR]}")]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    for y, (name, t) in enumerate([("condition based", cb), ("DQN", dqn)]):
        for action, marker in [(MINOR, "o"), (MAJOR, "s")]:
            rows = t[t["action"] == action]
            axes[2].scatter(rows["week"], np.full(len(rows), y), color=COLORS[action], marker=marker, s=40)
        fails = t[t["failure"].notna()]
        axes[2].scatter(fails["week"], np.full(len(fails), y), color="#d03b3b", marker="x", s=60)
    handles = [plt.Line2D([], [], color=COLORS[MINOR], marker="o", ls="", label=ACTIONS[MINOR]),
               plt.Line2D([], [], color=COLORS[MAJOR], marker="s", ls="", label=ACTIONS[MAJOR]),
               plt.Line2D([], [], color="#d03b3b", marker="x", ls="", label="failure")]
    axes[2].legend(handles=handles, frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.35))
    axes[2].set_yticks([0, 1], ["condition based", "DQN"])
    axes[2].set_ylim(-0.6, 1.6)
    axes[2].set_xlabel("week")
    fig.suptitle(f"One 10 year episode with the DQN policy (validation seed {EXAMPLE_SEED})", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "decision_example.png", dpi=150)


def plot_triggers(traces):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"DQN": "#2a78d6", "condition based": "#1baf7a"}
    for name, t in traces.items():
        axes[0].hist(t.loc[t["action"] == MINOR, "vibration"], bins=np.arange(0, 7.01, 0.25), alpha=0.6,
                     color=colors[name], label=name)
        axes[1].hist(t.loc[t["action"] == MAJOR, "efficiency"], bins=np.arange(0.715, 0.7701, 0.0025), alpha=0.6,
                     color=colors[name], label=name)
    axes[0].set_xlabel("excess vibration when minor maintenance is done (mm/s)")
    axes[1].set_xlabel("8 week efficiency when the overhaul is done")
    for ax in axes:
        ax.set_ylabel(f"count over {len(config.VAL_SEEDS)} validation episodes")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "decision_triggers.png", dpi=150)


def plot_map(counts):
    table = counts.argmax(axis=1).reshape(SHAPE)
    seen = counts.sum(axis=1).reshape(SHAPE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for k, ax in enumerate(axes):
        for v in range(SHAPE[0]):
            for e in range(SHAPE[1]):
                if seen[v, e, k] > 0:
                    ax.add_patch(plt.Rectangle((e, v), 1, 1, facecolor=COLORS[table[v, e, k]], edgecolor="white", lw=2))
                else:
                    ax.add_patch(plt.Rectangle((e + 0.04, v + 0.04), 0.92, 0.92, facecolor="white",
                                               edgecolor="#9a9994", lw=0.8, hatch="//"))
        ax.set_xlim(0, SHAPE[1])
        ax.set_ylim(0, SHAPE[0])
        ax.set_xticks(np.arange(SHAPE[1]) + 0.5, ["> 0.79", "0.78 to 0.79", "0.77 to 0.78", "0.76 to 0.77",
                                                   "0.75 to 0.76", "0.74 to 0.75", "≤ 0.74"], rotation=35, ha="right")
        ax.set_title(f"load {config.LOAD_LEVELS[k]:.2f}", loc="left")
        ax.set_xlabel("8 week efficiency")
    axes[0].set_yticks(np.arange(SHAPE[0]) + 0.5, ["< 1", "1 to 2", "2 to 3", "3 to 4", "4 to 5", "5 to 6", "≥ 6"])
    axes[0].set_ylabel("excess vibration (mm/s)")
    handles = [Patch(facecolor=COLORS[a], label=ACTIONS[a]) for a in ACTIONS]
    handles.append(Patch(facecolor="white", edgecolor="#9a9994", hatch="//", label="never reached"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Most common DQN action in each band, over the weeks of run to failure episodes", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(config.FIGURES_DIR / "decision_map.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
