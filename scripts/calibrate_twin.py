import itertools

import numpy as np
import pandas as pd

from src import config
from src.twin_sim import TARGETS, check_targets, mode_lifetimes, warning_summary

BEARING_LIFE = 104
IMPELLER_LIFE = 230
GRID = {
    "bearing_offset": [0.01, 0.02, 0.03, 0.05],
    "temp_power": [3, 4, 5, 6, 8],
    "temp_bearing": [30, 33, 36],
}

# Wear rates the tuning starts from (the Phase 2 values). They are fixed here so that a re-run gives
# the same table even after config.PUMP holds the calibrated values.
START_RATES = {"bearing_rate": 0.0557, "impeller_rate": 0.00827}

WARNING_TARGETS = [name for name in TARGETS if not name.endswith("_life")]
CONFLICT_CHECK = lambda m: 2 <= m["temp_rise_median"] <= 4


def tune_rate(params, key, mode, target, seeds):
    p = dict(params)
    for _ in range(3):
        life = np.nanmean(mode_lifetimes(seeds, mode, p))
        p[key] *= life / target
    return p


# Order of preference: (1) temperature at failure at the 90 °C trip, to the nearest degree, as vibration is at its trip;
# (2) alarm lead times near the middle of their target ranges; (3) the lowest power, the smallest change from Phase 2.
def score(row):
    return (round(abs(row["temp_at_failure"] - 90)),
            abs(row["vib_alarm_median"] - 6) + abs(row["temp_alarm_median"] - 2),
            row["temp_power"])


def main():
    seeds = config.CALIBRATION_SEEDS
    base = tune_rate(dict(config.PUMP, **START_RATES), "impeller_rate", "impeller", IMPELLER_LIFE, seeds)
    rows = []
    for offset in GRID["bearing_offset"]:
        p = dict(base, bearing_offset=offset)
        p = tune_rate(p, "bearing_rate", "bearing", BEARING_LIFE, seeds)
        for power, coef in itertools.product(GRID["temp_power"], GRID["temp_bearing"]):
            q = dict(p, temp_power=power, temp_bearing=coef)
            m = warning_summary(q, seeds)
            passed = {name: TARGETS[name](m) for name in WARNING_TARGETS}
            rows.append({"bearing_offset": offset, "bearing_rate": q["bearing_rate"], "impeller_rate": q["impeller_rate"],
                         "temp_power": power, "temp_bearing": coef, **m,
                         "all_targets": all(passed.values()), "conflict_check": CONFLICT_CHECK(m)})
            print(f"offset {offset} power {power} coef {coef}: vib alarm {m['vib_alarm_median']}, temp rise {m['temp_rise_median']}, "
                  f"temp alarm {m['temp_alarm_median']}, temp at failure {m['temp_at_failure']:.1f}, all targets {all(passed.values())}")
    grid = pd.DataFrame(rows)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(config.TABLES_DIR / "twin_calibration_grid.csv", index=False)

    ok = grid[grid["all_targets"]]
    both = ok[ok["conflict_check"]]
    print(f"\ncombinations meeting all targets: {len(ok)} of {len(grid)}; also meeting the 2 to 4 week temperature rise: {len(both)}")
    if ok.empty:
        print("no combination meets all targets")
        return
    pool = both if len(both) else ok
    best = pool.loc[min(pool.index, key=lambda i: score(pool.loc[i]))]
    print("\nchosen:")
    print(best.to_string())

    p = dict(base, bearing_offset=best["bearing_offset"], bearing_rate=best["bearing_rate"],
             temp_power=int(best["temp_power"]), temp_bearing=best["temp_bearing"])
    check, passed = check_targets(p, config.EDA_SEEDS)
    check["all_targets"] = all(passed.values())
    pd.Series(check).to_csv(config.TABLES_DIR / "twin_calibration_check.csv", header=["value"])
    print("\ncheck on fresh EDA seeds:")
    print(pd.Series(check).to_string())


if __name__ == "__main__":
    main()
