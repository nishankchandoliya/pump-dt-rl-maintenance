import numpy as np
import pytest

from src import config
from src.pump_twin import PumpTwin


def run_to_failure(twin, seed, load=None):
    twin.reset(seed)
    if load is not None:
        twin.load_path[:] = load
    while not twin.failure and twin.week < twin.weeks:
        twin.step()
    return twin.week


def test_load_chain_settings():
    p = np.array(config.LOAD_TRANSITIONS)
    assert np.allclose(p.sum(axis=1), 1)
    stationary = np.linalg.matrix_power(p, 200)[0]
    mean_load = stationary @ np.array(config.LOAD_LEVELS)
    assert 0.72 <= mean_load <= 0.76


def test_load_path_uses_only_load_levels():
    twin = PumpTwin()
    twin.reset(3)
    assert len(twin.load_path) == config.EPISODE_WEEKS
    assert set(np.round(twin.load_path, 2)) <= set(config.LOAD_LEVELS)


def test_same_seed_gives_same_run():
    a, b = PumpTwin(), PumpTwin()
    assert a.reset(7) == b.reset(7)
    while not a.failure:
        assert a.step() == b.step()
    assert a.failure == b.failure and a.week == b.week
    assert a.bearing_wear == b.bearing_wear and a.impeller_wear == b.impeller_wear


def test_different_seeds_give_different_runs():
    a, b = PumpTwin(), PumpTwin()
    a.reset(1)
    b.reset(2)
    assert [a.step() for _ in range(5)] != [b.step() for _ in range(5)]


def test_actions_do_not_change_the_scenario():
    a, b = PumpTwin(), PumpTwin()
    a.reset(11)
    b.reset(11)
    for week in range(60):
        if week == 20:
            b.major_overhaul()
        ra, rb = a.step(), b.step()
        assert ra["load"] == rb["load"]
    assert np.array_equal(a.load_path, b.load_path)


def test_wear_never_decreases_without_maintenance():
    twin = PumpTwin()
    twin.reset(5)
    last_b, last_w = 0.0, 0.0
    while not twin.failure and twin.week < twin.weeks:
        twin.step()
        assert twin.bearing_wear > last_b and twin.impeller_wear > last_w
        last_b, last_w = twin.bearing_wear, twin.impeller_wear


def test_minor_maintenance_renews_bearing_only():
    twin = PumpTwin()
    twin.reset(4, bearing_wear=0.6, impeller_wear=0.4)
    twin.minor_maintenance()
    assert twin.bearing_wear == 0.0
    assert twin.impeller_wear == 0.4
    assert twin.bearings_used == 1 and twin.impellers_used == 0


def test_major_overhaul_renews_both():
    twin = PumpTwin()
    twin.reset(4, bearing_wear=0.6, impeller_wear=0.4)
    twin.major_overhaul()
    assert twin.bearing_wear == 0.0 and twin.impeller_wear == 0.0
    assert twin.bearings_used == 1 and twin.impellers_used == 1


def test_failure_flag_matches_thresholds_every_week():
    twin = PumpTwin()
    for seed in range(20):
        twin.reset(seed, bearing_wear=0.5, impeller_wear=0.5)
        while twin.week < twin.weeks:
            twin.step()
            if twin.bearing_wear >= 1:
                assert twin.failure == "bearing"
            elif twin.impeller_wear >= 1:
                assert twin.failure == "impeller"
            else:
                assert twin.failure is None
            if twin.failure:
                break


def test_failed_pump_must_be_repaired():
    twin = PumpTwin()
    run_to_failure(twin, 1)
    assert twin.failure is not None
    with pytest.raises(RuntimeError):
        twin.step()
    with pytest.raises(RuntimeError):
        twin.minor_maintenance()
    with pytest.raises(RuntimeError):
        twin.major_overhaul()
    twin.repair()
    assert twin.failure is None
    assert twin.bearing_wear == 0.0 and twin.impeller_wear == 0.0
    twin.step()


def test_repair_on_healthy_pump_is_an_error():
    twin = PumpTwin()
    twin.reset(0)
    with pytest.raises(RuntimeError):
        twin.repair()


def test_bad_inputs_are_rejected():
    twin = PumpTwin()
    with pytest.raises(ValueError):
        twin.reset(0, bearing_wear=1.2)
    twin.reset(0)
    with pytest.raises(ValueError):
        twin.step(run_days=0)
    with pytest.raises(ValueError):
        twin.step(run_days=8)
    with pytest.raises(ValueError):
        twin.idle(-1)


def test_run_stops_at_horizon():
    twin = PumpTwin(weeks=10)
    twin.reset(0)
    for _ in range(10):
        twin.step()
    with pytest.raises(RuntimeError):
        twin.step()


def test_idle_has_no_wear_and_stops_at_horizon():
    twin = PumpTwin(weeks=20)
    twin.reset(2, bearing_wear=0.3, impeller_wear=0.2)
    assert twin.idle(5) == 5
    assert twin.week == 5
    assert twin.bearing_wear == 0.3 and twin.impeller_wear == 0.2
    assert twin.idle(30) == 15
    assert twin.week == 20


def test_partial_week_gives_proportional_wear():
    full, part = PumpTwin(), PumpTwin()
    full.reset(9, bearing_wear=0.2, impeller_wear=0.2)
    part.reset(9, bearing_wear=0.2, impeller_wear=0.2)
    full.step(run_days=7)
    part.step(run_days=4)
    assert np.isclose((part.bearing_wear - 0.2) / (full.bearing_wear - 0.2), 4 / 7)
    assert np.isclose((part.impeller_wear - 0.2) / (full.impeller_wear - 0.2), 4 / 7)


def test_new_pump_readings_are_normal_at_every_load():
    twin = PumpTwin()
    for seed in range(100):
        r = twin.reset(seed)
        assert r["vibration"] < config.VIB_ZONES[0]
        assert r["temperature"] < config.TEMP_ALARM
        assert 0.78 < r["efficiency"] < 0.82


def test_readings_stay_physical_until_failure():
    twin = PumpTwin()
    for seed in range(20):
        twin.reset(seed)
        while not twin.failure and twin.week < twin.weeks:
            r = twin.step()
            assert 0 < r["vibration"] < 15
            assert 40 < r["temperature"] < 110
            assert 0.6 < r["efficiency"] < 0.85


def test_bearing_failure_shows_in_vibration_and_temperature():
    twin = PumpTwin()
    for seed in range(50):
        twin.reset(seed)
        while not twin.failure and twin.week < twin.weeks:
            r = twin.step()
        if twin.failure == "bearing":
            assert r["vibration"] > config.VIB_ZONES[2]
            assert r["temperature"] > 80


def test_part_quality_has_mean_one():
    twin = PumpTwin()
    q = []
    for seed in range(300):
        twin.reset(seed)
        q.extend(twin.bearing_quality[:5])
    assert abs(np.mean(q) - 1) < 0.03


def test_higher_load_wears_faster():
    twin = PumpTwin()
    lives = {}
    for load in (0.75, 0.90):
        lives[load] = np.mean([run_to_failure(twin, seed, load) for seed in range(200)])
    ratio = lives[0.90] / lives[0.75]
    assert 0.6 < ratio < 0.8


def test_wear_grows_with_load_squared():
    low, high = PumpTwin(), PumpTwin()
    low.reset(6, bearing_wear=0.3, impeller_wear=0.3)
    high.reset(6, bearing_wear=0.3, impeller_wear=0.3)
    low.load_path[0] = 0.75
    high.load_path[0] = 0.90
    low.step()
    high.step()
    expected = (0.90 / 0.75) ** 2
    assert np.isclose((high.bearing_wear - 0.3) / (low.bearing_wear - 0.3), expected)
    assert np.isclose((high.impeller_wear - 0.3) / (low.impeller_wear - 0.3), expected)
