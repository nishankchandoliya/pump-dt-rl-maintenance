import numpy as np
import pandas as pd

from src import config
from src.pump_twin import PumpTwin, sensor_values


def run_to_failure(seed, params=None, load=None, weeks=config.EPISODE_WEEKS):
    twin = PumpTwin(params, weeks=weeks)
    twin.reset(seed)
    if load is not None:
        twin.load_path[:] = load
    rows = []
    while not twin.failure and twin.week < twin.weeks:
        r = twin.step()
        rows.append({"week": twin.week, **r, "bearing_wear": twin.bearing_wear, "impeller_wear": twin.impeller_wear})
    run = pd.DataFrame(rows)
    run["seed"] = seed
    run["failure"] = twin.failure or "none"
    return run


def simulate(seeds, params=None):
    return pd.concat([run_to_failure(seed, params) for seed in seeds], ignore_index=True)


def mode_lifetimes(seeds, mode, params=None, load=None, weeks=3000):
    p = dict(config.PUMP if params is None else params)
    p["impeller_rate" if mode == "bearing" else "bearing_rate"] = 0.0
    twin = PumpTwin(p, weeks=weeks)
    lives = []
    for seed in seeds:
        twin.reset(seed)
        if load is not None:
            twin.load_path[:] = load
        while not twin.failure and twin.week < weeks:
            twin.step()
        lives.append(twin.week if twin.failure else np.nan)
    return np.array(lives, dtype=float)


def warning_times(run, params=None):
    p = config.PUMP if params is None else params
    normal_temp = sensor_values(p, run["load"], 0.0, 0.0)[1]
    signs = {
        "vib_zone_c": run["vibration"] > config.VIB_ZONES[1],
        "vib_alarm": run["vibration"] > config.VIB_ALARM,
        "temp_rise": run["temperature"] - normal_temp > 5,
        "temp_alarm": run["temperature"] > config.TEMP_ALARM,
    }
    fail_week = run["week"].iloc[-1]
    times = {}
    for name, mask in signs.items():
        weeks = run["week"][mask]
        times[name] = fail_week - weeks.iloc[0] if len(weeks) else np.nan
    times["temp_at_failure"] = run["temperature"].iloc[-1]
    return times


def bearing_warnings(runs, params=None):
    rows = []
    for seed, run in runs.groupby("seed"):
        if run["failure"].iloc[0] == "bearing":
            rows.append({"seed": seed, **warning_times(run, params)})
    return pd.DataFrame(rows)


# targets from the parameter table, fixed before calibration (relaxed temperature rule agreed in Phase 3)
TARGETS = {
    "bearing_life": lambda m: 90 <= m["bearing_life_mean"] <= 120,
    "impeller_life": lambda m: 200 <= m["impeller_life_mean"] <= 260,
    "vib_c_4wk": lambda m: m["vib_c_4wk"] >= 0.9,
    "vib_alarm": lambda m: 4 <= m["vib_alarm_median"] <= 8,
    "temp_flat": lambda m: m["temp_flat"] >= 0.9,
    "temp_alarm": lambda m: 1 <= m["temp_alarm_median"] <= 3,
    "temp_alarm_after_vib": lambda m: m["temp_alarm_after_vib"] >= 0.9,
    "temp_at_failure": lambda m: 88 <= m["temp_at_failure"] <= 98,
}


def warning_summary(params, seeds):
    runs = simulate(seeds, params)
    w = bearing_warnings(runs, params)
    first = runs.groupby("seed")["failure"].first()
    return {
        "bearing_first": (first == "bearing").mean(),
        "vib_c_4wk": (w["vib_zone_c"] >= 4).mean(),
        "vib_c_median": w["vib_zone_c"].median(),
        "vib_alarm_median": w["vib_alarm"].median(),
        "temp_flat": (w["temp_rise"].isna() | (w["temp_rise"] <= w["vib_zone_c"])).mean(),
        "temp_rise_median": w["temp_rise"].median(),
        "temp_alarm_median": w["temp_alarm"].median(),
        "temp_alarm_after_vib": (w["temp_alarm"].isna() | (w["temp_alarm"] <= w["vib_zone_c"])).mean(),
        "no_temp_alarm": w["temp_alarm"].isna().mean(),
        "temp_at_failure": w["temp_at_failure"].mean(),
    }


def check_targets(params, seeds):
    m = warning_summary(params, seeds)
    m["bearing_life_mean"] = np.nanmean(mode_lifetimes(seeds, "bearing", params))
    m["impeller_life_mean"] = np.nanmean(mode_lifetimes(seeds, "impeller", params))
    passed = {name: rule(m) for name, rule in TARGETS.items()}
    return m, passed

