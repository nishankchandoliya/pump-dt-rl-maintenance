import numpy as np
import pandas as pd

from src.pump_env import MAJOR, MINOR, PumpEnv

COST_PARTS = ["energy", "minor", "major", "failure", "downtime"]


def run_episode(env, seed, policy, history=None):
    obs, _ = env.reset(seed=seed)
    result = {"seed": seed, "cost": 0.0, **{f"cost_{k}": 0.0 for k in COST_PARTS},
              "minor": 0, "major": 0, "bearing_failures": 0, "impeller_failures": 0, "running_days": 0}
    truncated = False
    while not truncated:
        action = policy(obs)
        week = env.twin.week
        obs, reward, _, truncated, info = env.step(action)
        result["cost"] -= reward
        for k in COST_PARTS:
            result[f"cost_{k}"] += info["cost"][k]
        result["minor"] += action == MINOR
        result["major"] += action == MAJOR
        result["running_days"] += info["running_days"]
        if info["failure"]:
            result[f"{info['failure']}_failures"] += 1
        if history is not None:
            history.append({"week": week, "action": action, "reward": reward, "weeks": info["weeks"],
                            "failure": info["failure"], "obs": obs})
    result["availability"] = result["running_days"] / (7 * env.twin.weeks)
    return result


def evaluate(policy, seeds, params=None, costs=None):
    env = PumpEnv(params, costs)
    return pd.DataFrame([run_episode(env, seed, policy) for seed in seeds])


def ci95(values):
    return 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))


def summarize(runs):
    failures = runs["bearing_failures"].sum() + runs["impeller_failures"].sum()
    running_weeks = runs["running_days"].sum() / 7
    counts = ["minor", "major", "bearing_failures", "impeller_failures", "availability"]
    return {
        "cost": runs["cost"].mean(),
        "cost_ci95": ci95(runs["cost"]),
        **{f"cost_{k}": runs[f"cost_{k}"].mean() for k in COST_PARTS},
        **{k: runs[k].mean() for k in counts},
        # running weeks per failure, pooled over all episodes (infinite when nothing failed)
        "mtbf_weeks": running_weeks / failures if failures else np.inf,
    }


def paired_difference(a, b, column="cost"):
    # mean and 95% CI of (a minus b) over the same seeds. Common random numbers make this much tighter
    # than comparing the two means
    if sorted(a["seed"]) != sorted(b["seed"]):
        raise ValueError("both runs must use the same seeds")
    merged = a[["seed", column]].merge(b[["seed", column]], on="seed", suffixes=("_a", "_b"))
    diff = merged[f"{column}_a"] - merged[f"{column}_b"]
    return diff.mean(), ci95(diff)
