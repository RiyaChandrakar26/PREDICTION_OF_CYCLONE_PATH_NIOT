import pandas as pd

from utils.constants import (
    TEST_CSV,
    METADATA_CSV,
    HINDCAST_TRACKS_CSV,
    HINDCAST_METRICS_CSV,
    FORECAST_TRACKS_CSV
)

# ==============================
# Core loaders
# ==============================

def load_test_data():
    df = pd.read_csv(TEST_CSV, parse_dates=["time"])
    df = df.rename(columns={"sid": "SID"})
    return df


def load_metadata():
    return pd.read_csv(METADATA_CSV)


# ==============================
# Hindcast
# ==============================

def load_hindcast_tracks():
    df = pd.read_csv(HINDCAST_TRACKS_CSV, parse_dates=["time"])

    # Normalize SID column name
    if "sid" in df.columns and "SID" not in df.columns:
        df = df.rename(columns={"sid": "SID"})

    return df


def load_hindcast_metrics():
    """
    Aggregate MAE table
    """
    return pd.read_csv(HINDCAST_METRICS_CSV)


# ==============================
# Forecast
# ==============================

def load_forecast_tracks():
    df = pd.read_csv(FORECAST_TRACKS_CSV)

    # parse forecast_time properly
    df["forecast_time"] = pd.to_datetime(
        df["forecast_time"], dayfirst=True
    )

    return df



# ==============================
# Helpers
# ==============================

def get_available_storms():
    test_df = load_test_data()
    meta_df = load_metadata()

    return (
        test_df[["SID"]]
        .drop_duplicates()
        .merge(meta_df, on="SID", how="left")
        .dropna(subset=["NAME"])
        .sort_values("NAME")
        .reset_index(drop=True)
    )


def get_observed_track(sid):
    df = load_test_data()
    return df[df["SID"] == sid].sort_values("time")


def get_last_observation():
    df = load_test_data()
    return df.sort_values("time").iloc[-1]
