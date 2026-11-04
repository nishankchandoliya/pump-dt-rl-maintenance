import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src import config
from src.pump_env import CONTINUE, MAJOR, MINOR, PumpEnv
from src.pump_twin import PumpTwin

C = config.COSTS


def energy(load, impeller_wear, run=1.0):
    efficiency = 0.80 - 0.08 * min(impeller_wear, 1.0)
    return run * load**3 * (0.80 / efficiency - 1)


def run_weeks(env, weeks):
    for _ in range(weeks):
        env.step(CONTINUE)


def test_env_passes_gymnasium_checker():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(PumpEnv(), skip_render_check=True)


def test_new_pump_observation():
    env = PumpEnv()
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        assert obs.dtype == np.float32 and obs.shape == (6,)
        assert abs(obs[0]) < 0.2 and abs(obs[1]) < 0.2 and abs(obs[2]) < 0.4
        assert obs[3] in (0.0, 0.5, 1.0)
        assert obs[4] == 0.0 and obs[5] == 0.0


def test_continue_reward_by_hand():
    env, twin = PumpEnv(), PumpTwin()
    env.reset(seed=3)
    twin.reset(3)
    for _ in range(30):
        _, reward, _, _, info = env.step(CONTINUE)
        r = twin.step()
        assert np.isclose(reward, -energy(r["load"], twin.impeller_wear))
        assert info["weeks"] == 1 and info["running_days"] == 7


def test_minor_step_by_hand():
    env, twin = PumpEnv(), PumpTwin()
    env.reset(seed=3)
    twin.reset(3)
    run_weeks(env, 40)
    for _ in range(40):
        twin.step()
    obs, reward, _, _, info = env.step(MINOR)
    twin.minor_maintenance()
    r = twin.step(run_days=4)
    expected = C["minor"] + C["downtime_week"] * 3 / 7 + energy(r["load"], twin.impeller_wear, run=4 / 7)
    assert np.isclose(reward, -expected)
    assert info["weeks"] == 1 and info["running_days"] == 4
    assert np.isclose(obs[4], (4 / 7) / 104)
    assert np.isclose(obs[5], (40 + 4 / 7) / 230)


def test_major_step_by_hand():
    env = PumpEnv()
    env.reset(seed=3)
    run_weeks(env, 40)
    obs, reward, _, _, info = env.step(MAJOR)
    assert np.isclose(reward, -(C["major"] + 2 * C["downtime_week"]))
    assert info["weeks"] == 2 and info["running_days"] == 0
    assert obs[4] == 0.0 and obs[5] == 0.0
    assert env.twin.week == 42 and env.twin.bearing_wear == 0.0 and env.twin.impeller_wear == 0.0


def test_failure_step():
    env = PumpEnv()
    env.reset(seed=1)
    while True:
        week = env.twin.week
        b, w = env.twin.bearing_wear, env.twin.impeller_wear
        obs, reward, _, _, info = env.step(CONTINUE)
        if info["failure"]:
            break
    mode = info["failure"]
    assert mode in ("bearing", "impeller")
    assert info["weeks"] == 1 + config.DOWNTIME["repair_weeks"]
    assert np.isclose(info["cost"]["failure"], C[f"{mode}_failure"])
    assert np.isclose(info["cost"]["downtime"], 5 * C["downtime_week"])
    assert reward < -(C[f"{mode}_failure"] + 5 * C["downtime_week"])
    assert env.twin.week == week + 6 and env.twin.failure is None
    assert obs[4] == 0.0 and obs[5] == 0.0
    assert b < 1 and w < 1


def test_failure_cost_depends_on_the_mode():
    env = PumpEnv()
    for mode in ("bearing", "impeller"):
        env.reset(seed=0)
        setattr(env.twin, f"{mode}_wear", 0.9999)
        _, reward, _, _, info = env.step(CONTINUE)
        assert info["failure"] == mode
        assert np.isclose(info["cost"]["failure"], C[f"{mode}_failure"])
        assert np.isclose(reward, -(C[f"{mode}_failure"] + 5 * C["downtime_week"] + info["cost"]["energy"]))


def test_maintenance_resets_show_in_observation():
    env = PumpEnv()
    env.reset(seed=8)
    run_weeks(env, 60)
    obs, _, _, _, _ = env.step(CONTINUE)
    assert np.isclose(obs[4], 61 / 104) and np.isclose(obs[5], 61 / 230)
    obs, _, _, _, _ = env.step(MINOR)
    assert obs[4] < 0.01 and obs[5] > 0.26
    assert len(env.eff_history) == config.EFF_WINDOW
    obs, _, _, _, _ = env.step(MAJOR)
    assert obs[4] == 0.0 and obs[5] == 0.0
    assert len(env.eff_history) == 1


def test_episode_length_and_truncation():
    env = PumpEnv()
    env.reset(seed=4)
    weeks = 0
    while True:
        _, _, terminated, truncated, info = env.step(CONTINUE)
        weeks += info["weeks"]
        assert not terminated
        if truncated:
            break
    assert weeks == config.EPISODE_WEEKS == env.twin.week
    with pytest.raises(RuntimeError):
        env.step(CONTINUE)


def test_downtime_is_cut_at_the_horizon():
    env = PumpEnv()
    env.reset(seed=2)
    while env.twin.week < config.EPISODE_WEEKS - 1:
        env.step(MINOR)
    _, reward, _, truncated, info = env.step(MAJOR)
    assert info["weeks"] == 1 and truncated
    assert np.isclose(reward, -(C["major"] + C["downtime_week"]))


def test_same_seed_gives_same_episode():
    actions = np.random.default_rng(0).integers(3, size=300)
    runs = []
    for _ in range(2):
        env = PumpEnv()
        obs, _ = env.reset(seed=11)
        history = [obs]
        for a in actions:
            obs, reward, _, truncated, _ = env.step(int(a))
            history.append(np.append(obs, reward))
            if truncated:
                break
        runs.append(np.concatenate(history))
    assert np.array_equal(runs[0], runs[1])
    other = PumpEnv()
    assert not np.array_equal(other.reset(seed=12)[0], PumpEnv().reset(seed=11)[0])


def test_reset_without_seed_continues_the_seeded_sequence():
    a, b = PumpEnv(), PumpEnv()
    a.reset(seed=5)
    b.reset(seed=5)
    assert np.array_equal(a.reset()[0], b.reset()[0])


def test_observations_stay_in_space_and_costs_are_positive():
    env = PumpEnv()
    rng = np.random.default_rng(1)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        truncated = False
        while not truncated:
            action = int(rng.choice(3, p=[0.9, 0.07, 0.03]))
            obs, reward, _, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert all(v >= 0 for v in info["cost"].values())
            assert np.isclose(-reward, sum(info["cost"].values()))


def test_invalid_action_is_rejected():
    env = PumpEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(3)


def test_read_does_not_change_the_twin():
    twin = PumpTwin()
    twin.reset(0)
    twin.step()
    twin.idle(2)
    state = (twin.week, twin.bearing_wear, twin.impeller_wear)
    assert twin.read() == twin.read()
    assert (twin.week, twin.bearing_wear, twin.impeller_wear) == state
    assert twin.read()["load"] == twin.load_path[3]
