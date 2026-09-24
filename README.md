# 🌍 Global Energy & Climate Action Analytics Platform

> **IBM SkillsBuild Data Analytics with AI Internship 2026**  
> BharatCares / AICTE Virtual Internship — Final Project Submission

---

## 👤 Student Information

| Field | Detail |
|---|---|
| **Student Name** | Sunny Kumar |
| **Internship ID** | IBMUEDA1428 |
| **Programme** | IBM SkillsBuild Data Analytics with AI Internship 2026 |
| **Submission Track** | BharatCares / AICTE Virtual Internship Final Submission |
| **Master File** | `SunnyKumar_GlobalEnergyAnalytics.py` |

---

## 📋 Project Abstract

The **Global Energy & Climate Action Analytics Platform** is a production-grade, single-file Python dashboard built with Streamlit that processes **241 country-level energy datasets** spanning **1965 to 2023**. The platform delivers an end-to-end data analytics workflow — from raw CSV ingestion and quality control through feature engineering, interactive visualisation, and AI-powered forecasting — to produce quantitative insights directly supporting two United Nations Sustainable Development Goals:

- **UN SDG 7 — Affordable and Clean Energy:** The platform tracks renewable energy share penetration across 225+ countries, models clean energy transition trajectories, and identifies energy access gaps at a per-country level.
- **UN SDG 13 — Climate Action:** A custom Carbon Footprint Impact metric quantifies the economic weight of each country's fossil fuel dependence. GHG emission trends are mapped, forecasted, and placed in a scenario explorer heatmap that visually encodes the impact of policy choices on future emissions trajectories.

The dashboard provides international policymakers, climate economists, and business strategists with forensic transparency into the global energy system — identifying where the highest-urgency interventions lie and what policy combinations deliver the optimal transition outcomes.

---

## ⚙️ Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Web Framework | Streamlit | ≥ 1.35 |
| Data Processing | Pandas | ≥ 2.1 |
| Numerical Computing | NumPy | ≥ 1.26 |
| Visualisation | Plotly Express + Graph Objects | ≥ 5.20 |
| Machine Learning | scikit-learn | ≥ 1.4 |
| Language | Python | ≥ 3.10 |
| Theme | Custom dark CSS (`#0d1117` base, `#58a6ff` accent) | — |

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| Source | Our World in Data — Global Energy Review |
| Files | 241 × `*_energy_data.csv` |
| Schema | 128 columns per row |
| Raw Records | ~16,252 rows |
| Post-Cleaning | ~10,447 rows |
| Countries | 225 unique nations + regional aggregates |
| Year Range | 1965–2023 (post-filter) |

---

## 🚀 Local Installation & Launch

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1 — Place files
Ensure all 241 `*_energy_data.csv` files and `SunnyKumar_GlobalEnergyAnalytics.py` are in the **same directory**.

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Launch the dashboard
```bash
streamlit run SunnyKumar_GlobalEnergyAnalytics.py
```

The app opens automatically at `http://localhost:8501`.  
**To stop:** press `Ctrl + C` in the terminal.

---

## 🖥️ Dashboard Tabs Overview

| Tab | Name | Key Components |
|---|---|---|
| 1 | 📋 Overview | 6 KPI cards · Data quality report · Dual-axis combo chart · Region donut · Stacked area |
| 2 | 🗺️ Choropleth Map | Interactive world map · 5 metric selectors · Year slider · Top-15 ranked table |
| 3 | 📈 Time Trends | Dual-axis CF vs Renewables/Fossil · Category line chart · Regional stacked bar |
| 4 | 💰 Carbon Impact | CF Impact formula · Category bar · Stacked area fossil trend · Top-20 country bar |
| 5 | 📊 Summaries | By Country / Year / Region / Category · Bubble scatter · Treemap |
| 6 | 🤖 AI Forecast | RF vs LR metrics · 20-year forecast · `text_auto` scenario heatmap |
| 7 | 🗂️ Raw Data | Column picker · Filterable dataframe · CSV download |
| 8 | 💡 Policy Insights | Dynamic alerts · 5 KPI cards · 7 data-driven expanders · Policy urgency radar chart |

---

## 🔬 Key Analytical Features

### Carbon Footprint Impact Formula
```
CF Impact (USD) = Gross Energy Demand (TWh) × Unit Base Cost ($/MWh)
```
Unit base costs: Coal $85 · Oil $95 · Gas $75 · Nuclear $60 · Hydro $20 · Solar $15 · Wind $18

### AI Forecasting Engine
- **Models:** Random Forest Regressor (300 trees, max_depth=6) vs Linear Regression (OLS)
- **Split:** 80% chronological training / 20% test
- **Metrics:** R² Score + Mean Absolute Error (MAE)
- **Horizon:** 20-year forward forecast (2024–2043)

### Scenario Explorer Heatmap
- **10×10 matrix:** Renewables Share % (rows) × Fossil Share % (columns)
- **Colour scale:** `RdYlGn_r` — green = optimal policy corridor · red = high-emission trajectory
- **Cell annotation:** `text_auto=".2s"` — all values displayed inline

### Policy Urgency Radar (Tab 8)
- **7 axes** mapped to the 7 strategic action frameworks
- Scores computed live from the active year range and region filter selections
- **60-point amber threshold** line drawn as reference for immediate-action zones

---

## 🌿 UN SDG Alignment

### SDG 7 — Affordable and Clean Energy
Tracks renewable share penetration, models energy access gaps, and identifies underperforming regions for clean energy investment prioritisation.

### SDG 13 — Climate Action
Quantifies carbon footprint impact per country and energy category, maps GHG hotspots on the choropleth, and models emission reduction pathways through the scenario explorer matrix.

---

## 📁 Project File Structure

```
energy_data/
├── SunnyKumar_GlobalEnergyAnalytics.py   ← Master application
├── requirements.txt                       ← Dependency registry
├── README.md                              ← This document
├── SunnyKumar_GlobalEnergyAnalytics.docx ← Technical project report
├── Afghanistan_energy_data.csv
├── Albania_energy_data.csv
├── ... (241 CSV files total)
└── Zimbabwe_energy_data.csv
```

---

<div align="center">

**Global Energy & Climate Action Analytics Platform**  
IBM SkillsBuild Data Analytics with AI Internship 2026  
Sunny Kumar · IBMUEDA1428 · BharatCares / AICTE

</div>
