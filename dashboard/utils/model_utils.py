import pandas as pd
from pathlib import Path

from utils.constants import (
    TEST_CSV,
    SID_NAME_CSV,
    HINDCAST_ALL_CSV,
    FORECAST_ALL_CSV
)


def load_test_data():
    df = pd.read_csv(TEST_CSV, parse_dates=["time"])
    df = df.rename(columns={"sid": "SID"})
    return df


def load_metadata():
    meta = pd.read_csv(SID_NAME_CSV)
    return meta


def load_hindcast_data():
    df = pd.read_csv(HINDCAST_ALL_CSV, parse_dates=["time"])
    return df


def load_forecast_data():
    df = pd.read_csv(FORECAST_ALL_CSV, parse_dates=["time"])
    return df



def get_available_storms():
    test_df = load_test_data()
    meta_df = load_metadata()

    storms = (
        test_df[["SID"]]
        .drop_duplicates()
        .merge(meta_df, on="SID", how="left")
        .dropna(subset=["NAME"])
        .sort_values("NAME")
        .reset_index(drop=True)
    )

    return storms


def get_observed_track(sid):
    df = load_test_data()

    track = (
        df[df["SID"] == sid]
        .sort_values("time")
        .reset_index(drop=True)
    )

    return track


def get_last_observation():
    df = load_test_data()

    last_row = (
        df.sort_values("time")
        .iloc[-1]
    )

    return last_row


def attach_storm_names(df):
    meta = load_metadata()
    return df.merge(meta, on="SID", how="left")
