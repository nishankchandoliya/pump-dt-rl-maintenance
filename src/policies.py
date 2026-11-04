import numpy as np

from src import config
from src.pump_env import CONTINUE, MAJOR, MINOR, obs_scales

VIB_SCALE, _ = obs_scales(config.PUMP)
EPS = 1e-6


def run_to_failure():
    return lambda obs: CONTINUE


def fixed_interval(minor_weeks, major_weeks=None):
    # preventive maintenance by calendar: renew a part after it has run a fixed number of weeks
    bearing_limit = minor_weeks / config.DESIGN_LIFE["bearing"]
    impeller_limit = None if major_weeks is None else major_weeks / config.DESIGN_LIFE["impeller"]

    def policy(obs):
        if impeller_limit is not None and obs[5] >= impeller_limit - EPS:
            return MAJOR
        if obs[4] >= bearing_limit - EPS:
            return MINOR
        return CONTINUE
    return policy


def condition_based(vib_limit, eff_limit=None):
    # vib_limit: excess vibration over a new pump in mm/s. eff_limit: lowest accepted 8 week average efficiency
    vib = vib_limit / VIB_SCALE
    estimate = None if eff_limit is None else (config.PUMP["eff_new"] - eff_limit) / config.PUMP["eff_loss"]

    def policy(obs):
        if estimate is not None and obs[2] >= estimate:
            return MAJOR
        if obs[0] >= vib:
            return MINOR
        return CONTINUE
    return policy


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(3))
