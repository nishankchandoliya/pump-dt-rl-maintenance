from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

# Seed ranges are fixed before any experiment so that tuning and final testing never share episodes.
# Training episode k uses seed TRAIN_SEED_START + k.
VAL_SEEDS = list(range(1000, 1050))
TEST_SEEDS = list(range(2000, 2200))
TRAIN_SEED_START = 10_000

# One step = one week. One episode = 10 years.
EPISODE_WEEKS = 520

# Weekly average unit load as a fraction of rated flow, changing as a Markov chain.
LOAD_LEVELS = [0.60, 0.75, 0.90]
LOAD_TRANSITIONS = [
    [0.80, 0.15, 0.05],
    [0.10, 0.80, 0.10],
    [0.05, 0.15, 0.80],
]

# Digital Twin constants. Wear rates are per week of running.
PUMP = {
    "bearing_rate": 0.0557,
    "bearing_offset": 0.05,
    "impeller_rate": 0.00827,
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
    "temp_noise": 1.0,
    "eff_new": 0.80,
    "eff_loss": 0.08,
    "eff_noise": 0.005,
}

DOWNTIME = {"minor_days": 3, "major_weeks": 2, "repair_weeks": 5}

# ISO 10816-7, Category I pump above 200 kW (mm/s RMS)
VIB_ZONES = [3.5, 5.0, 7.6]
VIB_ALARM = 6.3
VIB_TRIP = 9.5

# Journal bearing limits from a BHEL power plant pump specification (°C)
TEMP_ALARM = 85.0
TEMP_TRIP = 90.0
