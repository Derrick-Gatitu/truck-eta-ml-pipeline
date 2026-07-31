# Truck Delivery Delay Prediction — End-to-End ML Pipeline

An end-to-end machine learning pipeline that predicts truck delivery delays using multi-source trip, route, traffic, and weather data. Built locally (Windows) with a clear path to AWS migration.

## Overview

Truck delivery delays affect logistics costs, customer satisfaction, and route planning. This project builds a classifier that predicts whether a scheduled trip will be delayed, using historical driver, truck, route, traffic, and weather data.

**Result:** A tuned Random Forest model achieves **0.78 ROC-AUC** on held-out test data, correctly identifying **73% of actual delayed trips** at 74% precision.

## Architecture
## Data Sources

Two local databases combined:

| Database | Tables |
|---|---|
| PostgreSQL | `drivers_table`, `trucks_table`, `routes_details`, `routes_weather` |
| MySQL | `traffic_data`, `truck_schedule_table`, `city_weather` |

## Feature Engineering

The core challenge: `truck_schedule_table` gives a trip's start/end time, but `traffic_data` (hourly) and `routes_weather` (every 6 hours) are just snapshots — neither maps cleanly to a variable-length trip window.

**Approach:**
1. Explode each trip into every hour it spans (`pd.date_range` + `.explode()`)
2. Join traffic data on `(route_id, date, hour)`
3. Join weather data on `(route_id, date, hour rounded down to nearest 6)`, since weather snapshots only exist at hours 0, 6, 12, 18
4. Aggregate back to one row per trip: `avg_no_of_vehicles`, `max_accident`, `avg_temp`, `avg_wind_speed`, `avg_precip`, `avg_humidity`

This produces `final_df` — one row per trip, combining driver, truck, route, traffic, and weather features with the delay label.

## Models Compared

| Model | Validation ROC-AUC | Test ROC-AUC | Test Recall (Delayed) | Test Precision (Delayed) |
|---|---|---|---|---|
| Logistic Regression (scaled) | 0.657 | — | — | — |
| Random Forest (tuned) | **0.746** | **0.781** | **0.73** | 0.74 |
| XGBoost (tuned) | 0.749 | 0.781 | 0.68 | 0.74 |

**Random Forest was selected** as the final model — while XGBoost matched it on ROC-AUC, Random Forest caught meaningfully more actual delays (recall) at the same precision, which matters more than raw accuracy for this use case.

### Key findings from EDA
- ~35% of trips are delayed (class imbalance addressed via `class_weight="balanced"` / `scale_pos_weight`)
- **Route distance and average trip hours** are the strongest predictors
- **Traffic volume and weather conditions** (temp, humidity, wind) meaningfully contribute
- **Driver/truck demographics** (gender, driving style, fuel type) show almost no predictive signal

## Tech Stack

- **Languages/Libraries:** Python, pandas, scikit-learn, XGBoost, SQLAlchemy, pymysql
- **Databases:** PostgreSQL, MySQL (local)
- **Feature Store / Model Registry:** Hopsworks
- **Environment:** Local (Windows, conda), designed for straightforward migration to AWS RDS/SageMaker

## Project Structure
## Setup

Database credentials are loaded from a `.env` file (not committed) using `python-dotenv`:
Hopsworks API key is entered interactively via prompt rather than hardcoded.

## Future Work (Part 3 — Deployment & Monitoring)

Planned but not yet implemented:
- Scheduled inference pipeline (e.g. Prefect/Airflow)
- Persisted trip risk scores (Hopsworks/PostgreSQL)
- Data & model drift detection (Evidently AI)
- Automated retraining triggers

This scope was deliberately deferred to focus on a solid, well-validated modeling foundation first.
