import numpy as np
import pandas as pd

from src import config
from src.pump_env import CONTINUE, MAJOR, MINOR, PumpEnv

NAMES = {CONTINUE: "continue", MINOR: "minor", MAJOR: "major"}


def run_episode(env, seed, choose):
    obs, _ = env.reset(seed=seed)
    result = {"cost": 0.0, "bearing_failures": 0, "impeller_failures": 0, "minor": 0, "major": 0, "running_days": 0}
    truncated = False
    while not truncated:
        action = choose(obs)
        obs, reward, _, truncated, info = env.step(action)
        result["cost"] -= reward
        result["running_days"] += info["running_days"]
        result["minor"] += action == MINOR
        result["major"] += action == MAJOR
        if info["failure"]:
            result[f"{info['failure']}_failures"] += 1
    return result


def fixed_interval(minor_weeks, major_weeks=None):
    def choose(obs):
        if major_weeks and obs[5] * config.DESIGN_LIFE["impeller"] >= major_weeks - 1e-6:
            return MAJOR
        if minor_weeks and obs[4] * config.DESIGN_LIFE["bearing"] >= minor_weeks - 1e-6:
            return MINOR
        return CONTINUE
    return choose


def evaluate(name, choose, seeds):
    env = PumpEnv()
    runs = pd.DataFrame([run_episode(env, seed, choose) for seed in seeds])
    ci = 1.96 * runs["cost"].std() / np.sqrt(len(runs))
    return {"policy": name, "cost": runs["cost"].mean(), "ci95": ci,
            "bearing_failures": runs["bearing_failures"].mean(), "impeller_failures": runs["impeller_failures"].mean(),
            "minor": runs["minor"].mean(), "major": runs["major"].mean(),
            "availability": runs["running_days"].mean() / (7 * config.EPISODE_WEEKS)}


def worked_example(seed=7, weeks_before=60):
    env = PumpEnv()
    env.reset(seed=seed)
    for _ in range(weeks_before):
        env.step(CONTINUE)
    print(f"Worked example: seed {seed}, after {weeks_before} weeks of running")
    for action in (CONTINUE, MINOR, MAJOR):
        week, w = env.twin.week, env.twin.impeller_wear
        obs, reward, _, _, info = env.step(action)
        parts = ", ".join(f"{k} {v:.4f}" for k, v in info["cost"].items() if v)
        print(f"  week {week}: {NAMES[action]}, impeller wear before {w:.3f}, load {env.twin.load_path[week]:.2f}, "
              f"weeks {info['weeks']}, costs: {parts}, reward {reward:.4f}")
        print(f"    observation: {[round(float(x), 4) for x in obs]}")


def main():
    worked_example()
    seeds = config.VAL_SEEDS
    rng = np.random.default_rng(0)
    rows = [evaluate("always continue", lambda obs: CONTINUE, seeds),
            evaluate("random", lambda obs: int(rng.integers(3)), seeds)]
    for minor_weeks in (40, 50, 60, 70, 80):
        for major_weeks in (None, 120, 160, 200):
            name = f"minor every {minor_weeks}" + (f", major every {major_weeks}" if major_weeks else "")
            rows.append(evaluate(name, fixed_interval(minor_weeks, major_weeks), seeds))
    table = pd.DataFrame(rows)
    print(f"\n10 year episodes, {len(seeds)} validation seeds, costs in units (1 unit = about 23.5 lakh Rs)")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
