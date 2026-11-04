from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

# seed ranges are fixed before any experiment so that tuning and final testing never share episodes
# training episode k uses seed TRAIN_SEED_START + k
VAL_SEEDS = list(range(1000, 1050))
TEST_SEEDS = list(range(2000, 2200))
TRAIN_SEED_START = 10_000

# seeds used only to calibrate and study the Digital Twin (never for policies)
CALIBRATION_SEEDS = list(range(5000, 5300))
EDA_SEEDS = list(range(6000, 6500))

# one step = one week. One episode = 10 years
EPISODE_WEEKS = 520

# weekly average unit load as a fraction of rated flow, changing as a Markov chain
LOAD_LEVELS = [0.60, 0.75, 0.90]
LOAD_TRANSITIONS = [
    [0.80, 0.15, 0.05],
    [0.10, 0.80, 0.10],
    [0.05, 0.15, 0.80],
]

# constants of the Digital Twin, calibrated in Phase 3 (scripts/calibrate_twin.py). Wear rates are per week of running
PUMP = {
    "bearing_rate": 0.0889,
    "bearing_offset": 0.01,
    "impeller_rate": 0.00843,
    "noise_shape": 4.0,
    "quality_spread": 0.3,
    "vib_base": 1.6,
    "vib_load": 0.8,
    "vib_bearing": 7.0,
    "vib_impeller": 0.6,
    "vib_noise": 0.2,
    "temp_base": 55.0,
    "temp_load": 10.0,
    "temp_bearing": 30.0,
    "temp_power": 5,
    "temp_noise": 1.0,
    "eff_new": 0.80,
    "eff_loss": 0.08,
    "eff_noise": 0.005,
}

DOWNTIME = {"minor_days": 3, "major_weeks": 2, "repair_weeks": 5}

# vibration limits from ISO 10816-7, Category I pump above 200 kW (mm/s RMS)
VIB_ZONES = [3.5, 5.0, 7.6]
VIB_ALARM = 6.3
VIB_TRIP = 9.5

# journal bearing limits from a BHEL power plant pump specification (°C)
TEMP_ALARM = 85.0
TEMP_TRIP = 90.0

# design lives in weeks: the calibration targets and the scale of the part age features
DESIGN_LIFE = {"bearing": 104, "impeller": 230}

# settings of the RL environment (Phase 4). Costs are in units of the energy cost of one week of running a new pump at full load
# (about 4 MW x 168 h x 3.5 Rs/kWh = 23.5 lakh Rs). Values and sources are in notes/rl_formulation.md
COSTS = {"minor": 0.5, "major": 2.5, "bearing_failure": 5.0, "impeller_failure": 2.5, "downtime_week": 0.5}
GAMMA = 0.99
EFF_WINDOW = 8

# baseline settings tuned in Phase 5 (scripts/tune_baselines.py) on the validation seeds. Frozen for the final test
# fixed interval: minor every 50 running weeks of the bearing, major every 140 of the impeller
# condition based: minor at 5.0 mm/s excess vibration, major when the 8 week efficiency falls to 0.74
BASELINES = {"minor_weeks": 50, "major_weeks": 140, "vib_limit": 5.0, "eff_limit": 0.74}
