import numpy as np
import torch

from src import config
from src.pump_env import CONTINUE, MAJOR, MINOR, make_observation

ACTIONS = {CONTINUE: "continue", MINOR: "minor maintenance", MAJOR: "major overhaul"}
WORK = {CONTINUE: "keep the pump running", MINOR: "renew the bearing, 3 days stopped",
        MAJOR: "renew bearing and impeller, 2 weeks stopped"}


def signal_status(vibration, temperature, efficiency):
    # the condition of each signal in the terms an operator uses: ISO 10816-7 zone, temperature limits and
    # the efficiency drop from a new pump
    zone = "ABCD"[np.searchsorted(config.VIB_ZONES, vibration, side="right")]
    if temperature >= config.TEMP_TRIP:
        temp = "above trip"
    elif temperature >= config.TEMP_ALARM:
        temp = "alarm"
    else:
        temp = "normal"
    return {"vibration_zone": zone, "temperature_status": temp, "efficiency_drop": config.PUMP["eff_new"] - efficiency}


def recommend(net, vibration, temperature, efficiency, load, bearing_weeks, impeller_weeks):
    # vibration in mm/s RMS, temperature in °C, efficiency: average of the last 8 weeks, load: fraction of rated
    # flow, ages: running weeks since the bearing and the impeller were renewed
    low, high = config.LOAD_LEVELS[0], config.LOAD_LEVELS[-1]
    if not low <= load <= high:
        raise ValueError(f"load must be between {low} and {high}: the agent was trained only on this range")
    if bearing_weeks < 0 or impeller_weeks < 0:
        raise ValueError("ages must not be negative")
    reading = {"load": load, "vibration": vibration, "temperature": temperature}
    obs = make_observation(config.PUMP, reading, efficiency, bearing_weeks, impeller_weeks)
    with torch.no_grad():
        q = net(torch.as_tensor(obs)).numpy()
    action = int(q.argmax())
    # a Q value is minus the expected discounted cost from now on, so the best action has the lowest cost
    cost = -q
    return {"action": action, "name": ACTIONS[action], "work": WORK[action], "cost": cost,
            "margin": float(np.sort(cost)[1] - cost[action]), "observation": obs,
            **signal_status(vibration, temperature, efficiency)}
