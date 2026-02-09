import streamlit as st
import matplotlib.pyplot as plt

from utils.data_loader import (
    load_metadata,
    load_test_data,
    load_hindcast_tracks,
    load_hindcast_metrics
)

st.set_page_config(page_title="Cyclone Track Hindcast Analysis", layout="wide")
st.title("Cyclone Track Hindcast Analysis")

meta_df = load_metadata()
test_df = load_test_data()
tracks_df = load_hindcast_tracks()
metrics_df = load_hindcast_metrics()


observed_sids = set(test_df["SID"].unique())
hindcast_sids = set(tracks_df["SID"].unique())
valid_sids = observed_sids.intersection(hindcast_sids)

storm_options = (
    meta_df[meta_df["SID"].isin(valid_sids)][["SID", "NAME"]]
    .dropna()
    .drop_duplicates()
    .sort_values("NAME")
    .reset_index(drop=True)
)

if storm_options.empty:
    st.error("No storms available for hindcast visualization.")
    st.stop()

selected = st.selectbox(
    "Select Storm",
    storm_options.itertuples(index=False),
    format_func=lambda x: f"{x.NAME} ({x.SID})"
)

SID = selected.SID
NAME = selected.NAME

st.markdown("### Selected Storm")
st.write(f"**Storm Name:** {NAME}")
st.write(f"**Storm ID:** `{SID}`")

obs_df = test_df[test_df["SID"] == SID].sort_values("time")

if obs_df.empty:
    st.warning("Observed track not available for this storm.")
    st.stop()

models = sorted(tracks_df["model"].unique())

selected_models = st.multiselect(
    "Select models for comparison",
    models
)

generate = st.button("Generate Hindcast Comparison")

if generate:

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        obs_df["lon"],
        obs_df["lat"],
        "-ok",
        linewidth=2,
        markersize=4,
        label="Observed Track"
    )

    ax.scatter(
        obs_df["lon"].iloc[-1],
        obs_df["lat"].iloc[-1],
        c="black",
        s=120,
        marker="X",
        label="Forecast Start",
        zorder=5
    )

    colors = {
        "cnn_mlp": "red",
        "cnn_gru": "blue",
        "gru_motion": "green",
        "gru_fusion": "purple"
    }

    for m in selected_models:
        dfm = (
            tracks_df[
                (tracks_df["SID"] == SID) &
                (tracks_df["model"] == m)
            ]
            .sort_values("time")
        )

        if dfm.empty:
            continue

        ax.plot(
            dfm["lon_pred"],
            dfm["lat_pred"],
            "--o",
            color=colors.get(m, "gray"),
            linewidth=2,
            markersize=5,
            label=m.upper()
        )

    ax.set_title(f"Hindcast Track Comparison — {NAME}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    st.subheader("Overall Hindcast Performance")
    st.dataframe(metrics_df)