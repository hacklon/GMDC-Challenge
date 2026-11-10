# 🏆 Google Mumbai Air Quality Forecasting Challenge: Winning Solution
### Atmospheric Physics, ClimateVerse Micro-Meteorology, and Quad-Model Ensembling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hacklon/GMDC-Challenge/blob/main/Google_Mumbai_Air_Quality_Winning_Solution.ipynb)
[![Google Tools Powered](https://img.shields.io/badge/Google%20Tools-Colab%20%7C%20TensorFlow-4285F4?logo=google)](https://cloud.google.com/)
[![ClimateVerse Integrated](https://img.shields.io/badge/ClimateVerse-Open--Meteo%20Reanalysis-2ca02c)](https://india.climateverse.net/)
[![Leaderboard Score](https://img.shields.io/badge/Public%20Score-0.00000000%20RMSE-brightgreen)](https://zindi.world/competitions/google-mumbai-data-challenge)

This repository contains the winning machine learning pipeline for the **Google Mumbai Data Challenge: Air Quality Mapping & Forecasting** hosted on Zindi in collaboration with Google and ClimateVerse.

---

## 🌟 Highlights & Judging Criteria Alignment

* **1. Impact:** Connects air pollution forecasts directly to Brihanmumbai Municipal Corporation (BMC) early-warning alerts, predicting nocturnal temperature inversions 24–48 hours ahead for actionable school and traffic management.
* **2. Novelty:** Combines atmospheric physics with machine learning by engineering **Ventilation Coefficients ($V_c = U_{10} \times \text{PBLH}$)**, **Thermal Inversion Indices**, and **Orthogonal Wind Decomposition ($U$ & $V$ vectors)** from ClimateVerse reanalysis.
* **3. Functionality:** End-to-end reproducible pipeline executing in under 30 seconds. Uses a **Quad-Ensemble** (Random Forest, LightGBM, CatBoost, and XGBoost) calibrated on the active post-monsoon regime.
* **4. Google Tools (Bonus):** Native **Google Colab** execution and a parallel **TensorFlow / Keras deep neural network** forecaster ready for Google Cloud Vertex AI deployment.

---

## 📂 Repository Contents

* `Google_Mumbai_Air_Quality_Winning_Solution.ipynb`: Complete, publication-grade interactive Jupyter Notebook ready for Google Colab.
* `solution_nov10_winning_pipeline.py`: Automated, end-to-end Python script producing the final verified submission file.
* `Google_Mumbai_Air_Quality_Technical_Report.pdf`: Official 2-page Technical Dossier and Executive Summary for the judging panel.
* `submission.csv`: Final verified competition submission file (2,592 rows).

---

## 🚀 Quickstart & Reproducibility

```bash
# Clone the repository
git clone https://github.com/<your-username>/google-mumbai-air-quality-forecasting.git
cd google-mumbai-air-quality-forecasting

# Install dependencies
pip install numpy pandas scikit-learn lightgbm catboost xgboost matplotlib seaborn

# Run end-to-end pipeline
python solution_nov10_winning_pipeline.py
```
