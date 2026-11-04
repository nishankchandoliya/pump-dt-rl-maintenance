from collections import deque

import gymnasium as gym
import numpy as np

from src import config
from src.pump_twin import PumpTwin, sensor_values

CONTINUE, MINOR, MAJOR = 0, 1, 2
FEATURES = ["vibration", "temperature", "impeller_estimate", "load", "bearing_age", "impeller_age"]


def obs_scales(p):
    # scales from D30: trip limit minus a new pump at load 0.75, so 0 means as new and 1 about the trip level
    vib_new, temp_new, _ = sensor_values(p, 0.75, 0.0, 0.0)
    return config.VIB_TRIP - vib_new, config.TEMP_TRIP - temp_new


class PumpEnv(gym.Env):
    def __init__(self, params=None, costs=None, weeks=config.EPISODE_WEEKS):
        self.twin = PumpTwin(params, weeks=weeks)
        self.p = self.twin.p
        self.costs = dict(config.COSTS if costs is None else costs)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-1.0, 5.0, shape=(len(FEATURES),), dtype=np.float32)
        self.vib_scale, self.temp_scale = obs_scales(self.p)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2**31))
        reading = self.twin.reset(seed)
        self.bearing_age = 0.0
        self.impeller_age = 0.0
        self.eff_history = deque(maxlen=config.EFF_WINDOW)
        return self._observe(reading), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        if self.twin.week >= self.twin.weeks:
            raise RuntimeError("the episode is over: call reset()")
        cost = {"energy": 0.0, "minor": 0.0, "major": 0.0, "failure": 0.0, "downtime": 0.0}
        start = self.twin.week
        running_days = 0
        failure = None

        if action == MAJOR:
            self.twin.major_overhaul()
            self._renew(impeller=True)
            cost["major"] = self.costs["major"]
            cost["downtime"] = self.costs["downtime_week"] * self.twin.idle(config.DOWNTIME["major_weeks"])
            reading = self.twin.read()
        else:
            running_days = 7
            if action == MINOR:
                self.twin.minor_maintenance()
                self._renew(impeller=False)
                running_days -= config.DOWNTIME["minor_days"]
                cost["minor"] = self.costs["minor"]
                cost["downtime"] = self.costs["downtime_week"] * config.DOWNTIME["minor_days"] / 7
            reading = self.twin.step(running_days)
            run = running_days / 7
            self.bearing_age += run
            self.impeller_age += run
            # extra energy because of impeller wear. Pump power grows with load cubed (affinity laws)
            cost["energy"] = run * reading["load"] ** 3 * (self.p["eff_new"] / self.twin.efficiency() - 1)

            failure = self.twin.failure
            if failure:
                cost["failure"] = self.costs[f"{failure}_failure"]
                self.twin.repair()
                self._renew(impeller=True)
                cost["downtime"] += self.costs["downtime_week"] * self.twin.idle(config.DOWNTIME["repair_weeks"])
                reading = self.twin.read()

        info = {
            "weeks": self.twin.week - start,
            "running_days": running_days,
            "failure": failure,
            "cost": cost,
            "bearing_wear": self.twin.bearing_wear,
            "impeller_wear": self.twin.impeller_wear,
        }
        truncated = self.twin.week >= self.twin.weeks
        return self._observe(reading), -sum(cost.values()), False, truncated, info

    def _renew(self, impeller):
        self.bearing_age = 0.0
        if impeller:
            self.impeller_age = 0.0
            self.eff_history.clear()

    def _observe(self, reading):
        vib_new, temp_new, _ = sensor_values(self.p, reading["load"], 0.0, 0.0)
        self.eff_history.append(reading["efficiency"])
        efficiency = np.mean(self.eff_history)
        low, high = config.LOAD_LEVELS[0], config.LOAD_LEVELS[-1]
        obs = [
            (reading["vibration"] - vib_new) / self.vib_scale,
            (reading["temperature"] - temp_new) / self.temp_scale,
            (self.p["eff_new"] - efficiency) / self.p["eff_loss"],
            (reading["load"] - low) / (high - low),
            self.bearing_age / config.DESIGN_LIFE["bearing"],
            self.impeller_age / config.DESIGN_LIFE["impeller"],
        ]
        return np.array(obs, dtype=np.float32)
