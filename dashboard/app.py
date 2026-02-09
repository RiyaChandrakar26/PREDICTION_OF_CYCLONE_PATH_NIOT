import streamlit as st

st.set_page_config(
    page_title="Cyclone Track Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <h1 style="text-align:center;">
        Cyclone Track Hindcast & Forecast Dashboard
    </h1>
    <hr>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Select Mode",
    options=["Hindcast", "Forecast"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Models Used**
    - CNN-MLP
    - CNN-GRU
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Data Sources**
    - IBTrACS (Best Track)
    - ERA5 Reanalysis
    """
)

if mode == "Hindcast":
    st.switch_page("pages/1_Hindcast.py")

elif mode == "Forecast":
    st.switch_page("pages/2_Forecast.py")
