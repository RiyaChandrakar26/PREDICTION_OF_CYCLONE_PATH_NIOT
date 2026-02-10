# 🌪️ Cyclone Track Forecasting using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Description

This project presents a **data-driven cyclone track prediction system** developed for the **North Indian Ocean (NIO)** region.

The system uses **deep learning models** to learn cyclone motion from historical observations and atmospheric conditions, and provides both:

- **Hindcast analysis** (model evaluation on historical cyclones)  
- **Real-time forecasting** (future cyclone track prediction)

An **interactive Streamlit dashboard** is developed to visualize observed tracks, predicted tracks, model comparison, and forecast values in an intuitive way.

---

## 📊 Dataset Details

### 1️⃣ Cyclone Track Data

- **Source:** IBTrACS (International Best Track Archive for Climate Stewardship)  
- **Region:** North Indian Ocean  
- **Longitude:** 65°E – 95°E  
- **Latitude:** 5°S – 25°N  

**Temporal Coverage**
- 2006 – 2025  
- 6-hourly intervals (00, 06, 12, 18 UTC)

**Variables Used**
- Cyclone ID (SID)  
- Timestamp  
- Latitude  
- Longitude  

**Derived Motion Variables**
- ΔLatitude (6-hour displacement)  
- ΔLongitude (6-hour displacement)  

These displacement values are the **prediction targets** for all models.

---

### 2️⃣ Atmospheric Reanalysis Data

- **Source:** ERA5 (ECMWF)  
- **Spatial Resolution:** 0.25° × 0.25°
- **Temporal Resolution:** 6-hourly (aligned with cyclone tracks)

**Atmospheric Variables Used**
- U850 – Zonal wind at 850 hPa  
- V850 – Meridional wind at 850 hPa  
- U200 – Zonal wind at 200 hPa  
- V200 – Meridional wind at 200 hPa  
- Z500 – Geopotential height at 500 hPa  

These variables capture **environmental steering flow** and **large-scale circulation** influencing cyclone movement.

---

### 3️⃣ Final Dataset Summary

- **Input Tensor Shape:** (5, 80, 80) per time step  
- **Prediction Horizon:** 6 hours (recursive for longer lead times)  
- **Total Samples:** ~3,000+  
- **Train / Validation / Test Split:** Cyclone-wise split to avoid data leakage  

---

## 🧠 Models Used


### 🔹 CNN-MLP
- CNN extracts spatial features from ERA5 atmospheric fields  
- MLP predicts cyclone displacement (Δlat, Δlon)  
- Captures spatial patterns influencing cyclone movement  

---

### 🔹 CNN-GRU
- CNN extracts spatial atmospheric features  
- GRU models temporal evolution of cyclone motion  
- Best suited for sequential and time-dependent behavior  

---

### 🔹 GRU-Fusion
- Combines motion features and atmospheric features  
- Jointly models environmental and kinematic effects  

Model performance is evaluated using **Mean Absolute Error (MAE)**:

- MAE in kilometers (track error)  
- MAE in latitude & longitude degrees  

### 🔢 MAE Comparison Table

| Model        | Lead Time | MAE (km) | 
|--------------|-----------|----------|
| CNN-MLP      | 6h        | 37.83    | 
| CNN-MLP      | 12h       | 75.28    | 
| CNN-MLP      | 24h       | 157.44   |
| CNN-MLP      | 48h       | 327.63   | 
| CNN-GRU      | 6h        | 56.40    | 
| CNN-GRU      | 12h       | 106.50   | 
| CNN-GRU      | 24h       | 200.84   | 
| CNN-GRU      | 48h       | 365.86   | 
| GRU-Fusion   | 6h        | 32.77    | 
| GRU-Fusion   | 12h       | 65.03    | 
| GRU-Fusion   | 24h       | 141.97   | 
| GRU-Fusion   | 48h       | 303.07   |


---

## 🔁 Hindcast Analysis

Hindcasting evaluates how well models reproduce historical cyclone tracks.
| Model   | Lead Time | Samples | MAE (km) |
|-------- |-----------|---------|----------|
| CNN-MLP | 6h        | 477     | 20.32    |
| CNN-MLP | 12h       | 447     | 37.78    |
| CNN-MLP | 24h       | 389     | 69.89    |
| CNN-MLP | 48h       | 300     | 127.51   |
| CNN-GRU | 6h        | 477     | 56.75    |
| CNN-GRU | 12h       | 447     | 106.95   |
| CNN-GRU | 24h       | 389     | 199.90   |
| CNN-GRU | 48h       | 300     | 356.27   |

**Key Observations**
- CNN-MLP achieves lower track error at all lead times compared to CNN-GRU.
- Error increases monotonically with forecast lead time for both models.
- CNN-GRU captures temporal dynamics but exhibits faster error growth at longer lead times.  

**Conclusion (Hindcast)**

> **CNN-MLP is the most reliable and robust model for cyclone track prediction in the North Indian Ocean.**

---

## 🔮 Forecasting Results

Forecasting mode predicts cyclone movement starting from the **latest observed position**.

### Forecast Characteristics
- Uses last observed timestamp as initialization  
- Supports multiple lead times: 6h, 12h, 24h, 48h  
- Recursive prediction strategy  
- Visualized on an interactive map  

### Forecast Output
- Predicted latitude & longitude  
- Forecast timestamp (UTC)  
- Model-wise comparison  
- Tabular output below map for precise values  

### Forecast Table Example

| Model   | Lead Time | Forecast Time (UTC) | Latitude  | Longitude |
|---------|-----------|---------------------|-----------|-----------|
| CNN-MLP | 6h        | 02-12-2025 06:00    | 12.4843   | 80.8896   |
| CNN-MLP | 12h       | 02-12-2025 12:00    | 12.4530   | 81.4687   |
| CNN-MLP | 24h       | 03-12-2025 00:00    | 12.3904   | 82.6270   |
| CNN-MLP | 48h       | 04-12-2025 00:00    | 12.2652   | 84.9436   |
| CNN-GRU | 6h        | 02-12-2025 06:00    | 12.5814   | 80.6417   |
| CNN-GRU | 12h       | 02-12-2025 12:00    | 12.5271   | 80.6139   |
| CNN-GRU | 24h       | 03-12-2025 00:00    | 12.6900   | 80.6974   |
| CNN-GRU | 48h       | 04-12-2025 00:00    | 12.9072   | 80.8087   |

---

## 🖥️ Dashboard Features

- Storm selection (only valid storms)  
- Model selection  
- Interactive zoom & pan map  
- Hindcast vs Observed comparison  
- Forecast visualization  
- Model-wise MAE tables  
- Clean UI built with **Streamlit + Plotly**  

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.8+  
- TensorFlow / PyTorch  
- Streamlit  
- Pandas, NumPy, Scikit-learn, Matplotlib, Plotly  

### Run Dashboard
```bash
streamlit run app.py
