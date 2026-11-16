"""Project-wide settings: paths, dataset URLs and domain constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
SQL_DIR = ROOT / "sql"
DB_PATH = DATA_PROCESSED / "nk_wells.db"
STATE_PATH = OUTPUTS / "session_state.json"

HF_REPO = "Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
SPLIT_FILES = {"train": "train.csv", "test": "test.csv"}

CHUNKSIZE = 50_000

# Memory-friendly dtypes for the long-format CSV
CSV_DTYPES = {
    "well_id": "category",
    "well_type": "category",
    "parameter": "category",
    "value": "float32",
    "unit": "category",
    "n_obs": "int16",
    "flag": "category",
}

FOUNTAIN = "fountain"
SRP = "sucker_rod_pump"

FOUNTAIN_PARAMS = ["tubing_pressure", "casing_pressure", "line_pressure"]
SRP_PARAMS = ["SPM", "pump_fillage", "min_rod_weight", "max_rod_weight", "dynamometer_area"]

# Primary signal each detector watches (the "is the well running?" signal)
PRIMARY_SIGNAL = {FOUNTAIN: "tubing_pressure", SRP: "SPM"}

# Flags whose values are physically impossible -> value set to NaN before analysis
MASK_FLAGS = ("weight_overflow", "area_extreme")

# Known data issues documented in the dataset card (verified in quality.py)
NK7_WELL = "NK-7"
NK7_REGIME_START = "2026-06-09"          # data-driven: daily SPM median drops to ~1-2 from 9 June
AREA_SHIFT_FACTOR = 5.0                   # daily median / pre-period median that marks the scale shift

# Reference ("silver") label thresholds
ZERO_EPS = 1e-9          # SRP: hourly mean exactly 0 => every raw record was 0
PRESSURE_OFF_ATM = 0.5   # fountain: all pressures below this => well closed / depressurised

# Detector settings
BASELINE_SPM = 0.5          # strokes/min
BASELINE_PRESSURE = 0.5     # atm
RULE_PRIMARY_MAX = 0.5      # primary signal "near zero"
RULE_FILLAGE_MAX = 5.0      # % pump fillage: no real pumping card
RULE_MERGE_GAP_H = 2        # bridge data gaps up to 2 h inside an event
STAT_COMPONENTS = 3         # GMM states: shut-in / reduced / normal operation
STAT_LOW_MEAN = 0.405       # log1p(0.5): component mean below this = shut-in state
STAT_SMOOTH_H = 3           # rolling-median window for GMM posteriors
RANDOM_STATE = 42
