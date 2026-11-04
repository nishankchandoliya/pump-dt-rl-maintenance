import pandas as pd

from src import config
from src.evaluation import evaluate, summarize
from src.policies import fixed_interval, random_policy, run_to_failure
from src.pump_env import CONTINUE, MAJOR, MINOR, PumpEnv

NAMES = {CONTINUE: "continue", MINOR: "minor", MAJOR: "major"}
COLUMNS = ["policy", "cost", "ci95", "bearing_failures", "impeller_failures", "minor", "major", "availability"]


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


def row(name, policy, seeds):
    s = summarize(evaluate(policy, seeds))
    return {"policy": name, **s, "ci95": s["cost_ci95"]}


def main():
    worked_example()
    seeds = config.VAL_SEEDS
    rows = [row("always continue", run_to_failure(), seeds), row("random", random_policy(0), seeds)]
    for minor_weeks in (40, 50, 60, 70, 80):
        for major_weeks in (None, 120, 160, 200):
            name = f"minor every {minor_weeks}" + (f", major every {major_weeks}" if major_weeks else "")
            rows.append(row(name, fixed_interval(minor_weeks, major_weeks), seeds))
    table = pd.DataFrame(rows)[COLUMNS]
    print(f"\n10 year episodes, {len(seeds)} validation seeds, costs in units (1 unit = about 23.5 lakh Rs)")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
