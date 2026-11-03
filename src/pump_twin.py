import numpy as np

from src import config


def sensor_values(p, load, b, w):
    # Wear can pass 1 during the failure week. Above 1 the part has already failed, so the readings use at most 1.
    b, w = np.minimum(b, 1.0), np.minimum(w, 1.0)
    vibration = p["vib_base"] + p["vib_load"] * load + p["vib_bearing"] * b**2 + p["vib_impeller"] * w
    temperature = p["temp_base"] + p["temp_load"] * load**2 + p["temp_bearing"] * b ** p["temp_power"]
    efficiency = p["eff_new"] - p["eff_loss"] * w
    return vibration, temperature, efficiency


class PumpTwin:
    def __init__(self, params=None, weeks=config.EPISODE_WEEKS):
        self.p = dict(config.PUMP if params is None else params)
        self.weeks = weeks
        self.levels = np.array(config.LOAD_LEVELS)
        self.transitions = np.array(config.LOAD_TRANSITIONS)

    def reset(self, seed, bearing_wear=0.0, impeller_wear=0.0):
        if not (0 <= bearing_wear < 1 and 0 <= impeller_wear < 1):
            raise ValueError("initial wear must be in [0, 1)")

        # All randomness of the run is drawn here, so every policy run with this seed
        # sees the same load path, the same weekly noise and the same spare parts.
        rng = np.random.default_rng(seed)
        self.load_path = self._make_load_path(rng)
        shape = self.p["noise_shape"]
        self.bearing_noise = rng.gamma(shape, 1 / shape, self.weeks)
        self.impeller_noise = rng.gamma(shape, 1 / shape, self.weeks)
        self.sensor_noise = rng.standard_normal((self.weeks + 1, 3))
        self.bearing_quality = self._part_quality(rng)
        self.impeller_quality = self._part_quality(rng)

        self.week = 0
        self.bearings_used = 0
        self.impellers_used = 0
        self.bearing_wear = bearing_wear
        self.impeller_wear = impeller_wear
        self.failure = None
        return self._readings(self.load_path[0], self.sensor_noise[0])

    def step(self, run_days=7):
        if self.failure:
            raise RuntimeError("the pump has failed: call repair() first")
        if self.week >= self.weeks:
            raise RuntimeError("no weeks left in this run: call reset()")
        if not 1 <= run_days <= 7:
            raise ValueError("run_days must be between 1 and 7")

        p = self.p
        k = self.week
        load = self.load_path[k]
        running = run_days / 7

        bearing_speed = p["bearing_rate"] * self.bearing_quality[self.bearings_used]
        impeller_speed = p["impeller_rate"] * self.impeller_quality[self.impellers_used]
        growth = (self.bearing_wear + p["bearing_offset"]) * self.bearing_noise[k]
        self.bearing_wear += running * bearing_speed * load**2 * growth
        self.impeller_wear += running * impeller_speed * load**2 * self.impeller_noise[k]
        self.week += 1

        if self.bearing_wear >= 1:
            self.failure = "bearing"
        elif self.impeller_wear >= 1:
            self.failure = "impeller"
        return self._readings(load, self.sensor_noise[self.week])

    def idle(self, weeks):
        if weeks < 0:
            raise ValueError("weeks must not be negative")
        weeks = min(weeks, self.weeks - self.week)
        self.week += weeks
        return weeks

    def minor_maintenance(self):
        self._check_not_failed()
        self.bearing_wear = 0.0
        self.bearings_used += 1

    def major_overhaul(self):
        self._check_not_failed()
        self._renew_all()

    def repair(self):
        if not self.failure:
            raise RuntimeError("the pump has not failed: nothing to repair")
        self._renew_all()
        self.failure = None

    def efficiency(self):
        return sensor_values(self.p, 0.0, 0.0, self.impeller_wear)[2]

    def _renew_all(self):
        self.bearing_wear = 0.0
        self.impeller_wear = 0.0
        self.bearings_used += 1
        self.impellers_used += 1

    def _check_not_failed(self):
        if self.failure:
            raise RuntimeError("the pump has failed: use repair() instead")

    def _readings(self, load, noise):
        p = self.p
        vibration, temperature, efficiency = sensor_values(p, load, self.bearing_wear, self.impeller_wear)
        return {
            "load": float(load),
            "vibration": float(vibration + p["vib_noise"] * noise[0]),
            "temperature": float(temperature + p["temp_noise"] * noise[1]),
            "efficiency": float(efficiency + p["eff_noise"] * noise[2]),
        }

    def _make_load_path(self, rng):
        n = len(self.levels)
        start = np.linalg.matrix_power(self.transitions, 200)[0]
        state = rng.choice(n, p=start)
        path = np.empty(self.weeks)
        for k in range(self.weeks):
            path[k] = self.levels[state]
            state = rng.choice(n, p=self.transitions[state])
        return path

    def _part_quality(self, rng):
        spread = self.p["quality_spread"]
        return np.exp(spread * rng.standard_normal(self.weeks + 1) - spread**2 / 2)
