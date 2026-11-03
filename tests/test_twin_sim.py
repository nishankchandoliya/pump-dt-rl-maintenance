import numpy as np
import pandas as pd

from src import config
from src.twin_sim import bearing_warnings, check_targets, mode_lifetimes, run_to_failure, simulate, warning_times


def test_run_to_failure_table():
    run = run_to_failure(4)
    expected = {"week", "load", "vibration", "temperature", "efficiency", "bearing_wear", "impeller_wear", "seed", "failure"}
    assert set(run.columns) == expected
    assert not run.isna().any().any()
    assert list(run["week"]) == list(range(1, len(run) + 1))
    assert run["failure"].iloc[0] in ("bearing", "impeller")
    assert run["bearing_wear"].is_monotonic_increasing and run["impeller_wear"].is_monotonic_increasing


def test_simulate_joins_runs():
    runs = simulate([1, 2, 3])
    assert sorted(runs["seed"].unique()) == [1, 2, 3]
    assert len(runs) == sum(len(run_to_failure(s)) for s in [1, 2, 3])


def test_mode_lifetimes_switch_off_the_other_mode():
    p = dict(config.PUMP)
    p["impeller_rate"] = 0.0
    assert run_to_failure(8, p, weeks=3000)["failure"].iloc[0] == "bearing"
    p = dict(config.PUMP)
    p["bearing_rate"] = 0.0
    assert run_to_failure(8, p, weeks=3000)["failure"].iloc[0] == "impeller"
    lives = mode_lifetimes(range(20), "bearing")
    assert np.all(lives > 0) and not np.isnan(lives).any()


def test_fixed_load_is_used():
    run = run_to_failure(3, load=0.9)
    assert (run["load"] == 0.9).all()


def test_warning_times_on_a_hand_made_run():
    p = config.PUMP
    load = 0.75
    normal = p["temp_base"] + p["temp_load"] * load**2
    run = pd.DataFrame({
        "week": [1, 2, 3, 4, 5, 6],
        "load": [load] * 6,
        "vibration": [2.0, 4.0, 5.5, 6.0, 6.5, 9.6],
        "temperature": [normal, normal + 1, normal + 2, normal + 6, 86.0, 91.0],
    })
    t = warning_times(run)
    assert t["vib_zone_c"] == 3
    assert t["vib_alarm"] == 1
    assert t["temp_rise"] == 2
    assert t["temp_alarm"] == 1
    assert t["temp_at_failure"] == 91.0


def test_warning_time_is_nan_when_never_crossed():
    run = pd.DataFrame({"week": [1, 2], "load": [0.6, 0.6], "vibration": [2.0, 2.1], "temperature": [58.0, 58.5]})
    t = warning_times(run)
    assert np.isnan(t["vib_zone_c"]) and np.isnan(t["temp_alarm"])


def test_bearing_warnings_only_uses_bearing_failures():
    runs = simulate(range(30))
    w = bearing_warnings(runs)
    bearing_seeds = runs.groupby("seed")["failure"].first()
    assert sorted(w["seed"]) == sorted(bearing_seeds[bearing_seeds == "bearing"].index)


def test_calibrated_twin_meets_all_targets():
    metrics, passed = check_targets(config.PUMP, config.EDA_SEEDS)
    failed = [name for name, ok in passed.items() if not ok]
    assert not failed, f"targets not met: {failed}, metrics: {metrics}"
