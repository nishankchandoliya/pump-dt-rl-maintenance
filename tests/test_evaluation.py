import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import ci95, evaluate, paired_difference, run_episode, summarize
from src.policies import condition_based, fixed_interval, run_to_failure
from src.pump_env import PumpEnv

C = config.COSTS


def scripted(actions):
    queue = iter(actions)
    return lambda obs: next(queue)


def test_short_hand_made_episode():
    actions = [0, 0, 1, 0, 2, 0, 0, 0, 0]
    env = PumpEnv(weeks=10)
    history = []
    result = run_episode(env, 3, scripted(actions), history)

    assert [h["action"] for h in history] == actions
    assert sum(h["weeks"] for h in history) == 10
    assert result["minor"] == 1 and result["major"] == 1
    assert result["bearing_failures"] == 0 and result["impeller_failures"] == 0
    assert result["running_days"] == 7 + 7 + 4 + 7 + 0 + 7 + 7 + 7 + 7
    assert np.isclose(result["availability"], 53 / 70)
    assert np.isclose(result["cost_minor"], C["minor"]) and np.isclose(result["cost_major"], C["major"])
    assert np.isclose(result["cost_downtime"], C["downtime_week"] * (3 / 7 + 2))
    assert np.isclose(result["cost"], -sum(h["reward"] for h in history))
    parts = sum(result[f"cost_{k}"] for k in ["energy", "minor", "major", "failure", "downtime"])
    assert np.isclose(result["cost"], parts)


@pytest.mark.parametrize("mode", ["bearing", "impeller"])
def test_failures_are_counted_and_charged(mode):
    fast = dict(config.PUMP)
    fast[f"{mode}_rate"] *= 10
    env = PumpEnv(fast)
    result = run_episode(env, 5, run_to_failure())
    n = result["bearing_failures"] + result["impeller_failures"]
    assert result[f"{mode}_failures"] >= 10
    expected = result["bearing_failures"] * C["bearing_failure"] + result["impeller_failures"] * C["impeller_failure"]
    assert np.isclose(result["cost_failure"], expected)
    # the last repair can be cut short by the end of the episode
    assert C["downtime_week"] * (5 * n - 5) <= result["cost_downtime"] <= C["downtime_week"] * 5 * n
    s = summarize(pd.DataFrame([result]))
    assert np.isclose(s["mtbf_weeks"], result["running_days"] / 7 / n)


def test_evaluate_is_reproducible():
    seeds = config.VAL_SEEDS[:5]
    a = evaluate(fixed_interval(50, 200), seeds)
    b = evaluate(fixed_interval(50, 200), seeds)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["seed"]) == seeds


def test_summary_statistics_by_hand():
    runs = pd.DataFrame({"seed": [1, 2, 3, 4], "cost": [1.0, 2.0, 3.0, 4.0], "running_days": [700] * 4,
                         "bearing_failures": [1, 0, 1, 0], "impeller_failures": [0, 0, 0, 0],
                         "minor": [2, 2, 2, 2], "major": [0, 1, 0, 1], "availability": [0.9] * 4,
                         **{f"cost_{k}": [0.0] * 4 for k in ["energy", "minor", "major", "failure", "downtime"]}})
    s = summarize(runs)
    assert s["cost"] == 2.5
    assert np.isclose(s["cost_ci95"], 1.96 * np.std([1, 2, 3, 4], ddof=1) / 2)
    assert np.isclose(s["mtbf_weeks"], 400 / 2)
    runs["bearing_failures"] = 0
    assert summarize(runs)["mtbf_weeks"] == np.inf


def test_paired_difference():
    a = pd.DataFrame({"seed": [1, 2, 3], "cost": [10.0, 12.0, 17.0]})
    b = pd.DataFrame({"seed": [3, 1, 2], "cost": [14.0, 9.0, 11.0]})
    mean, ci = paired_difference(a, b)
    assert np.isclose(mean, (1 + 1 + 3) / 3)
    assert np.isclose(ci, ci95(np.array([1.0, 1.0, 3.0])))
    with pytest.raises(ValueError):
        paired_difference(a, b[b["seed"] != 3])


def test_tuned_baselines_beat_run_to_failure():
    seeds = config.VAL_SEEDS[:10]
    b = config.BASELINES
    breakdown = evaluate(run_to_failure(), seeds)
    for policy in (fixed_interval(b["minor_weeks"], b["major_weeks"]), condition_based(b["vib_limit"], b["eff_limit"])):
        saving, ci = paired_difference(breakdown, evaluate(policy, seeds))
        assert saving - ci > 0
