from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Core data
# --------------------
TEST_CSV = DATA_DIR / "dataset_splits" / "test.csv"
METADATA_CSV = DATA_DIR / "metadata" / "sid_to_name.csv"

# --------------------
# Hindcast
# --------------------
HINDCAST_TRACKS_CSV = RESULTS_DIR / "hindcast" / "hindcast_tracks_all_models.csv"
HINDCAST_METRICS_CSV = RESULTS_DIR / "hindcast" / "hindcast_all_models.csv"

# --------------------
# Forecast
# --------------------
FORECAST_TRACKS_CSV = RESULTS_DIR / "forecast" / "forecast_2025331N07081.csv"

# --------------------
# Models
# --------------------
MODELS = {
    "cnn_mlp": {"label": "CNN-MLP", "color": "red"},
    "cnn_gru": {"label": "CNN-GRU", "color": "blue"},
}

DEFAULT_MODEL = "cnn_mlp"

TIME_STEP_HOURS = 6
LEAD_TIMES = [6, 12, 24, 48]

MAP_BOUNDS = {
    "lon_min": 65,
    "lon_max": 95,
    "lat_min": -5,
    "lat_max": 25,
}

APP_TITLE = "Cyclone Track Forecasting Dashboard"
