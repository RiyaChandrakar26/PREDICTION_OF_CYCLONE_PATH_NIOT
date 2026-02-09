import pandas as pd
import plotly.graph_objects as go


MODEL_COLORS = {
    "cnn_mlp": "red",
    "cnn_gru": "blue",
}

OBSERVED_COLOR = "black"



def create_base_map(title):
    fig = go.Figure()

    fig.update_layout(
        title=title,
        geo=dict(
            projection_type="mercator",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showcountries=True,
            countrycolor="gray",
            showcoastlines=True,
            coastlinecolor="black",
            lataxis=dict(range=[5, 25]),
            lonaxis=dict(range=[65, 95]),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5
        )
    )

    return fig



def add_observed_track(fig, obs_df):
    fig.add_trace(
        go.Scattergeo(
            lon=obs_df["lon"],
            lat=obs_df["lat"],
            mode="lines+markers",
            line=dict(color=OBSERVED_COLOR, width=3),
            marker=dict(size=6),
            name="Observed Track",
            text=obs_df["time"].dt.strftime("%Y-%m-%d %H:%M"),
            hovertemplate=(
                "<b>Observed</b><br>"
                "Lat: %{lat:.2f}<br>"
                "Lon: %{lon:.2f}<br>"
                "Time: %{text}<extra></extra>"
            )
        )
    )

    last = obs_df.iloc[-1]

    fig.add_trace(
        go.Scattergeo(
            lon=[last["lon"]],
            lat=[last["lat"]],
            mode="markers",
            marker=dict(
                size=14,
                symbol="x",
                color="black"
            ),
            name="Forecast Start",
            hovertemplate=(
                "<b>Forecast Start</b><br>"
                "Lat: %{lat:.2f}<br>"
                "Lon: %{lon:.2f}<br>"
                f"Time: {last['time']}<extra></extra>"
            )
        )
    )

    return fig



def add_hindcast_tracks(fig, hindcast_df, sid, models):
    for model in models:
        df = (
            hindcast_df[
                (hindcast_df["SID"] == sid) &
                (hindcast_df["model"] == model)
            ]
            .sort_values("time")
        )

        if df.empty:
            continue

        fig.add_trace(
            go.Scattergeo(
                lon=df["lon_pred"],
                lat=df["lat_pred"],
                mode="lines+markers",
                line=dict(
                    color=MODEL_COLORS[model],
                    width=2,
                    dash="dash"
                ),
                marker=dict(size=6),
                name=f"{model.upper()} Hindcast",
                text=df["time"].dt.strftime("%Y-%m-%d %H:%M"),
                hovertemplate=(
                    f"<b>{model.upper()} Hindcast</b><br>"
                    "Lat: %{lat:.2f}<br>"
                    "Lon: %{lon:.2f}<br>"
                    "Time: %{text}<extra></extra>"
                )
            )
        )

    return fig



def add_forecast_tracks(fig, forecast_df, sid, models):
    for model in models:
        df = (
            forecast_df[
                (forecast_df["SID"] == sid) &
                (forecast_df["model"] == model)
            ]
            .sort_values("time")
        )

        if df.empty:
            continue

        fig.add_trace(
            go.Scattergeo(
                lon=df["lon_pred"],
                lat=df["lat_pred"],
                mode="lines+markers",
                line=dict(
                    color=MODEL_COLORS[model],
                    width=3
                ),
                marker=dict(size=7),
                name=f"{model.upper()} Forecast",
                text=df["time"].dt.strftime("%Y-%m-%d %H:%M"),
                hovertemplate=(
                    f"<b>{model.upper()} Forecast</b><br>"
                    "Lat: %{lat:.2f}<br>"
                    "Lon: %{lon:.2f}<br>"
                    "Time: %{text}<extra></extra>"
                )
            )
        )

    return fig
