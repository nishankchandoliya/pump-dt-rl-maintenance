import numpy as np
import pytest
import torch

from src import config
from src.advisor import recommend, signal_status
from src.dqn import load_network
from src.pump_env import CONTINUE, MAJOR, MINOR, PumpEnv, make_observation
from src.pump_twin import PumpTwin, sensor_values

MODEL = config.MODELS_DIR / "dqn.pt"


class FixedQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, x):
        return self.values


def test_observation_is_built_as_in_the_environment():
    for seed in range(5):
        reading = PumpTwin().reset(seed)
        obs, _ = PumpEnv().reset(seed=seed)
        assert np.array_equal(obs, make_observation(config.PUMP, reading, reading["efficiency"], 0.0, 0.0))


def test_signal_status():
    assert [signal_status(v, 60, 0.8)["vibration_zone"] for v in [3.4, 3.5, 6.0, 7.6, 9.0]] == list("ABCDD")
    temps = [signal_status(3, t, 0.8)["temperature_status"] for t in [84.9, 85.0, 89.9, 90.0]]
    assert temps == ["normal", "alarm", "alarm", "above trip"]
    assert np.isclose(signal_status(3, 60, 0.77)["efficiency_drop"], 0.03)


def test_recommend_uses_the_network_output():
    r = recommend(FixedQ([-5.0, -3.0, -4.5]), 4.0, 62.0, 0.78, 0.75, 50, 100)
    assert r["action"] == MINOR and r["name"] == "minor maintenance"
    assert np.allclose(r["cost"], [5.0, 3.0, 4.5]) and np.isclose(r["margin"], 1.5)
    reading = {"load": 0.75, "vibration": 4.0, "temperature": 62.0}
    assert np.array_equal(r["observation"], make_observation(config.PUMP, reading, 0.78, 50, 100))


def test_bad_inputs_are_rejected():
    net = FixedQ([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        recommend(net, 3.0, 60.0, 0.8, 1.0, 10, 10)
    with pytest.raises(ValueError):
        recommend(net, 3.0, 60.0, 0.8, 0.75, -1, 10)


@pytest.mark.skipif(not MODEL.exists(), reason="needs the trained model from scripts/train_dqn.py")
def test_trained_advisor_makes_sensible_choices():
    # readings of the twin without noise. A new pump keeps running, a bearing at the point of failure
    # (vibration zone D, temperature near the trip) is never left running and an impeller at the end of
    # its life with a healthy bearing gets an overhaul. This holds at every load level
    net = load_network(MODEL)

    def advice(load, b, w, bearing_weeks, impeller_weeks):
        vibration, temperature, efficiency = sensor_values(config.PUMP, load, b, w)
        return recommend(net, vibration, temperature, efficiency, load, bearing_weeks, impeller_weeks)["action"]

    for load in config.LOAD_LEVELS:
        assert advice(load, 0.0, 0.0, 0, 0) == CONTINUE
        assert advice(load, 0.95, 0.2, 97, 44) != CONTINUE
        assert advice(load, 0.1, 0.97, 54, 210) == MAJOR
