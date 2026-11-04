import numpy as np

from src import config
from src.evaluation import run_episode
from src.policies import VIB_SCALE, condition_based, fixed_interval, random_policy, run_to_failure
from src.pump_env import CONTINUE, MAJOR, MINOR, PumpEnv


def make_obs(vib=0.0, eff=0.80, bearing_weeks=0.0, impeller_weeks=0.0):
    return np.array([vib / VIB_SCALE, 0.0, (0.80 - eff) / 0.08, 0.5,
                     bearing_weeks / config.DESIGN_LIFE["bearing"],
                     impeller_weeks / config.DESIGN_LIFE["impeller"]], dtype=np.float32)


def test_run_to_failure_always_continues():
    policy = run_to_failure()
    assert all(policy(make_obs(vib=v, eff=0.72, bearing_weeks=300)) == CONTINUE for v in (0, 5, 9))


def test_fixed_interval_rule():
    policy = fixed_interval(50, 200)
    assert policy(make_obs(bearing_weeks=49, impeller_weeks=100)) == CONTINUE
    assert policy(make_obs(bearing_weeks=50, impeller_weeks=100)) == MINOR
    assert policy(make_obs(bearing_weeks=50, impeller_weeks=200)) == MAJOR
    assert policy(make_obs(bearing_weeks=10, impeller_weeks=200)) == MAJOR
    assert fixed_interval(50)(make_obs(bearing_weeks=10, impeller_weeks=500)) == CONTINUE


def test_condition_based_rule_in_physical_units():
    policy = condition_based(2.0, 0.75)
    assert policy(make_obs(vib=1.99, eff=0.76)) == CONTINUE
    assert policy(make_obs(vib=2.01, eff=0.76)) == MINOR
    assert policy(make_obs(vib=0.5, eff=0.749)) == MAJOR
    assert policy(make_obs(vib=3.0, eff=0.749)) == MAJOR
    assert condition_based(2.0)(make_obs(vib=0.5, eff=0.72)) == CONTINUE


def test_vibration_scale_matches_the_environment():
    assert np.isclose(VIB_SCALE, PumpEnv().vib_scale)
    assert np.isclose(VIB_SCALE, 7.30)


def test_fixed_interval_in_the_environment():
    env = PumpEnv()
    history = []
    run_episode(env, 1000, fixed_interval(30, 90), history)
    obs = env.reset(seed=1000)[0]
    for step in history:
        bearing_weeks = obs[4] * config.DESIGN_LIFE["bearing"]
        impeller_weeks = obs[5] * config.DESIGN_LIFE["impeller"]
        if step["action"] == MAJOR:
            assert impeller_weeks >= 90 - 1e-3
        elif step["action"] == MINOR:
            assert bearing_weeks >= 30 - 1e-3 and impeller_weeks < 90
        else:
            assert bearing_weeks < 30 and impeller_weeks < 90
        obs = step["obs"]


def test_random_policy_is_reproducible():
    a, b = random_policy(3), random_policy(3)
    obs = make_obs()
    assert [a(obs) for _ in range(50)] == [b(obs) for _ in range(50)]
