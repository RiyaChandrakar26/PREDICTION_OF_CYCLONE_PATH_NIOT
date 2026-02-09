import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import (
    load_test_data,
    load_forecast_tracks,
    load_metadata,
    get_last_observation
)
from utils.constants import MAP_BOUNDS

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("Cyclone Track Forecast")

# -------------------------------------------------
# Load data
# -------------------------------------------------
forecast_df = load_forecast_tracks()
test_df = load_test_data()
meta_df = load_metadata()

last = get_last_observation()
SID = last["SID"]
init_time = last["time"]

storm_meta = meta_df[meta_df["SID"] == SID].iloc[0]

# -------------------------------------------------
# Header info
# -------------------------------------------------
st.subheader("Latest Observed Storm")

c1, c2, c3 = st.columns(3)
c1.metric("Storm ID", SID)
c2.metric("Storm Name", storm_meta["NAME"])
c3.metric("Initialization Time", init_time.strftime("%Y-%m-%d %H:%M UTC"))

# -------------------------------------------------
# Observed history
# -------------------------------------------------
history = (
    test_df[test_df["SID"] == SID]
    .sort_values("time")
    .reset_index(drop=True)
)

# -------------------------------------------------
# Model selector (NEW)
# -------------------------------------------------
available_models = sorted(forecast_df["model"].unique())

selected_models = st.multiselect(
    "Select Forecast Models",
    available_models,
    default=["cnn_gru"] if "cnn_gru" in available_models else available_models
)

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# -------------------------------------------------
# Map
# -------------------------------------------------
fig = go.Figure()

# Observed track
fig.add_trace(
    go.Scattergeo(
        lon=history["lon"],
        lat=history["lat"],
        mode="lines+markers",
        name="Observed Track",
        line=dict(color="black", width=3),
        hovertext=history["time"].dt.strftime("%Y-%m-%d %H:%M UTC"),
        hoverinfo="text"
    )
)

# Forecast start
fig.add_trace(
    go.Scattergeo(
        lon=[history["lon"].iloc[-1]],
        lat=[history["lat"].iloc[-1]],
        mode="markers",
        marker=dict(size=14, symbol="x", color="black"),
        name="Forecast Start",
        hovertext=init_time.strftime("%Y-%m-%d %H:%M UTC"),
        hoverinfo="text"
    )
)

# -------------------------------------------------
# Forecast tracks (SELECTED MODELS)
# -------------------------------------------------
MODEL_COLORS = {
    "cnn_gru": "blue",
    "cnn_mlp": "red",
    "gru_motion": "green",
    "gru_fusion": "purple"
}

for model in selected_models:
    dfm = (
        forecast_df[
            (forecast_df["model"] == model)
        ]
        .sort_values("forecast_time")
    )

    if dfm.empty:
        continue

    fig.add_trace(
        go.Scattergeo(
            lon=dfm["lon"],
            lat=dfm["lat"],
            mode="lines+markers",
            name=f"{model.upper()} Forecast",
            line=dict(
                color=MODEL_COLORS.get(model, "gray"),
                width=3,
                dash="dash"
            ),
            marker=dict(size=7),
            hovertext=dfm["forecast_time_str"],
            hoverinfo="text"
        )
    )

fig.update_layout(
    geo=dict(
        projection_type="mercator",
        lataxis_range=[MAP_BOUNDS["lat_min"], MAP_BOUNDS["lat_max"]],
        lonaxis_range=[MAP_BOUNDS["lon_min"], MAP_BOUNDS["lon_max"]],
        showcountries=True,
        showland=True,
        landcolor="rgb(240,240,240)",
        countrycolor="gray",
        showcoastlines=True
    ),
    legend=dict(orientation="h", y=0.02),
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToAdd": [
            "zoomInGeo",
            "zoomOutGeo",
            "resetGeo",
            "panGeo"
        ],
    }
)

# -------------------------------------------------
# 📊 TABLE BELOW MAP
# -------------------------------------------------
st.subheader("Forecast Values")

# Model order priority
model_order = ["cnn_gru", "cnn_mlp", "gru_motion", "gru_fusion"]

table_df = forecast_df[
    (forecast_df["model"].isin(selected_models))
].copy()

table_df["model"] = pd.Categorical(
    table_df["model"],
    categories=model_order,
    ordered=True
)

table_df = table_df.sort_values(
    ["model", "forecast_time"]
)

table_df = table_df[
    ["model", "lead_time", "forecast_time_str", "lat", "lon"]
].rename(columns={
    "model": "Model",
    "lead_time": "Lead Time",
    "forecast_time_str": "Forecast Time (UTC)",
    "lat": "Latitude",
    "lon": "Longitude"
})

st.dataframe(table_df, use_container_width=True)