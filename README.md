# France Weather Recommender — MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.12-blue)
![Airflow](https://img.shields.io/badge/airflow-2.8-orange)
![MLflow](https://img.shields.io/badge/mlflow-2.8-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**End-to-end ML system for recommending weekend destinations based on weather forecasts with multi-profile support.**

Technologies: Python, PostgreSQL (Neon), Airflow, MLflow, scikit-learn, Docker, Streamlit, Plotly

**🌐 Live Demo:** [France Weather Recommender on HF Spaces](https://huggingface.co/spaces/AgaHei/France_Weather_Recommender)

---

## 🎯 Project Overview

This is a **production-grade MLOps system** that:
- Fetches daily weather data for 20 French cities
- Engineers features using rolling windows
- Supports **5 user profiles** (leisure, surfer, cyclist, stargazer, skier) with customized recommendations
- Trains two ML models weekly (K-Means clustering + Gradient Boosting regression)
- Generates daily recommendations with champion/challenger model promotion
- Provides interactive **Streamlit UI** with maps and profile selection
- Logs all experiments to MLflow
- Deployed to **Hugging Face Spaces** for public access
- Runs entirely on free/open-source tools

**Business goal:** Help users plan weekend trips by recommending destinations with the best weather.

**Learning goal:** Master the complete MLOps lifecycle from data ingestion → training → deployment → monitoring.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DAG 1: fetch_weather (daily, 6am)                              │
│    Open-Meteo API → raw_weather table                           │
│         ↓                                                       │
│  DAG 2: compute_features (triggered by DAG 1)                   │
│    raw_weather → rolling windows → weather_features table       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DAG 3: retrain_models (weekly, Sunday midnight)                │
│    ┌──────────────────┐         ┌──────────────────┐            │
│    │   K-Means Model  │         │ Regression Model │            │
│    │   (clustering)   │         │ (comfort score)  │            │
│    └────────┬─────────┘         └────────┬─────────┘            │
│             │                            │                      │
│             ├────────── MLflow ──────────┤                      │
│             │       (experiment          │                      │
│             │        tracking)           │                      │
│             ↓                            ↓                      │
│    Champion/Challenger Comparison                               │
│      - If new model better → promote                            │
│      - Log to model_runs table                                  │
│      - Save artifacts                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     SERVING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DAG 4: generate_recommendations (daily, 7am)                   │
│    Load champion models                                         │
│         ↓                                                       │
│    Stage 1: K-Means (coarse filter)                             │
│      - Cluster cities by weather profile                        │
│      - Identify "good weather" clusters                         │
│         ↓                                                       │
│    Stage 2: Regression (fine ranking)                           │
│      - Predict comfort scores for each user profile             │
│      - Rank cities within good clusters                         │
│         ↓                                                       │
│    Write to recommendations & profile_scores tables             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        UI/DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Streamlit Web App (deployed to HF Spaces)                      │
│    ┌──────────────────┐         ┌──────────────────┐            │
│    │   Profile        │         │ Interactive Map  │            │
│    │   Selector       │         │  (Plotly/Mapbox) │            │
│    └────────┬─────────┘         └────────┬─────────┘            │
│             │                            │                      │
│             ├────── Neon Database  ──────┤                      │
│             │      (recommendations      │                      │
│             │       + profile_scores)    │                      │
│             ↓                            ↓                      │
│    Detailed Cards & Weather Info                                │
│      - Top 3 recommendations with metrics                       │
│      - Hotel suggestions from OSM                               │
│      - Model performance stats                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Model

### **PostgreSQL (Neon) Tables:**

**raw_weather** — Daily weather data from Open-Meteo  
- Columns: city, date, temp_max, temp_min, temp_mean, precipitation, wind_speed_max, weather_code
- Updated: Daily by DAG 1
- Rows: ~1,800 (20 cities × 90 days rolling window)

**weather_features** — Engineered features with rolling windows  
- Columns: city, feature_date, temp_mean_7d, temp_mean_3d, precip_sum_7d, precip_sum_3d, wind_max_7d, wind_max_3d, comfort_score
- Updated: Daily by DAG 2
- Rows: ~1,800

**cities** — Static city metadata  
- Columns: city, latitude, longitude
- Updated: Once (seed data)
- Rows: 20

**recommendations** — Daily weekend recommendations  
- Columns: recommendation_date, city, cluster_id, comfort_score_pred, rank, profile_name
- Updated: Daily by DAG 4
- Rows: ~500 (10 recommendations × 5 profiles per day, 7-14 day history)

**scoring_profiles** — User profile definitions
- Columns: profile_name, temp_weight, rain_weight, wind_weight, temp_optimal, description
- Updated: Static configuration
- Rows: 5 (leisure, surfer, cyclist, stargazer, skier)

**profile_scores** — Actual comfort scores per profile
- Columns: city, feature_date, profile_name, comfort_score
- Updated: Daily by DAG 2 (feature engineering)
- Rows: ~1,800 (20 cities × 5 profiles × ~18 days rolling window)

**hotels** — Hotel/accommodation data from OpenStreetMap
- Columns: city, hotel_name, latitude, longitude, hotel_type
- Updated: Weekly by DAG 5 (`fetch_hotels`)
- Rows: ~200-300 hotels across 20 cities

**model_runs** — Model training audit log  
- Columns: run_date, model_type, metric_name, metric_value, artifact_path, is_champion
- Updated: Weekly by DAG 3
- Rows: Grows indefinitely (audit trail)

---

## 🧮 The ML Models

### **1. K-Means Clustering (Coarse Filter)**

**Purpose:** Group cities by weather profile

**Features used:**
- `temp_mean_7d` — Average temperature over last 7 days
- `precip_sum_7d` — Total precipitation over last 7 days
- `wind_max_7d` — Max wind speed over last 7 days

**Output:** 4 clusters (typically: Mediterranean, Atlantic, Continental, Extreme)

**Metric:** Silhouette score (cluster quality)

**Use case:** Quickly filter out cities in "bad weather" clusters

---

### **2. Gradient Boosting Regression (Fine Ranking)**

**Purpose:** Predict comfort score for weekend trips

**Features used:**
- `temp_mean_3d` — Average temperature over next 3 days (forecast)
- `precip_sum_3d` — Total precipitation over next 3 days
- `wind_max_3d` — Max wind speed over next 3 days

**Target:** `comfort_score` (0-100 scale)

**Formula:**
```python
comfort_score = 50 * exp(-0.5*((temp-20)/6)²) + 30*exp(-rain/5) + 20*exp(-wind/25)
```

**Metrics:** R² (0.995), RMSE (0.92 points), MAE (0.54 points)

**Use case:** Rank cities within good clusters by predicted comfort

---

## 🎓 Key MLOps Concepts Demonstrated

### **1. Champion/Challenger Pattern**
- Every week, new models are trained
- Compared to current "champion" based on metrics
- Promoted only if they beat champion by a threshold
- Prevents noisy model swaps

### **2. Experiment Tracking (MLflow)**
- Every training run logged with params, metrics, artifacts
- Reproducible: you can reload any past model
- Enables model comparison across time

### **3. Feature Engineering Pipeline**
- Raw data → rolling windows → ML-ready features
- Separated into its own DAG (modularity)
- Versioned by `feature_date`

### **4. Two-Stage Retrieval**
- Stage 1: Cheap clustering (fast coarse filter)
- Stage 2: Expensive regression (accurate fine ranking)
- Same pattern used by Netflix, Spotify, Amazon

### **5. Scheduled Retraining**
- Models retrain weekly on fresh data
- Handles concept drift (seasonal weather changes)
- Fully automated (no manual intervention)

### **6. Batch Inference**
- DAG 4 scores all cities every morning
- Pre-computed recommendations (low latency for users)
- Alternative: real-time API (would score on-demand)

---

## 🚀 Getting Started

### **Prerequisites:**
- Python 3.11+
- Docker + Docker Compose
- Neon PostgreSQL account (free tier)

### **Setup:**

```bash
# 1. Clone/unzip the project
cd france-weather-recommender

# 2. Create .env file
cp .env.example .env
# Edit .env and add your Neon connection string

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python -c "from src.data.db import init_schema, seed_cities; init_schema(); seed_cities()"

# 5. Backfill historical data (one-time, ~2 minutes)
python scripts/backfill_historical_weather.py

# 6. Train initial models (one-time)
python scripts/train_models.py

# 7. Start Airflow
docker-compose up airflow-init   # Wait for completion
docker-compose up -d

# 8. Access UIs
# Airflow: http://localhost:8080 (admin/admin)
# MLflow:  http://localhost:5001

# 9. Run Streamlit app (optional)
cd app
streamlit run streamlit_app.py
# Streamlit: http://localhost:8501
```

**🌐 Alternative:** Use the live deployed version at [HF Spaces](https://huggingface.co/spaces/AgaHei/France_Weather_Recommender)

### **Running DAGs:**

1. **Enable the DAGs** in Airflow UI
2. **Manually trigger** `fetch_weather` to test the full pipeline
3. Watch the cascade: `fetch_weather` → `compute_features` → (wait for Sunday) → `retrain_models` → `generate_recommendations`

---

## 📈 Model Performance

**K-Means Clustering:**
- Silhouette score: 0.367 (moderate separation)
- 4 clusters identified
- Brest forms singleton cluster (extreme Atlantic weather)

**Regression Model:**
- Test R²: 0.995 (near-perfect prediction)
- Test RMSE: 0.92 points (on 0-100 scale)
- CV R²: 0.996 ± 0.001 (very stable)
- Feature importances:
  - Precipitation: 76.7% (rain is the dealbreaker!)
  - Temperature: 19.2%
  - Wind: 4.1%

---

## 🔮 User Profiles (Phase 3 - Completed)

The system supports **5 distinct user profiles** with customized comfort score calculations:

### **🏖️ Leisure (Default)**
- **Focus:** General comfort for sightseeing and relaxation
- **Optimal temp:** 20°C | **Weights:** Temp 50%, Rain 30%, Wind 20%
- **Best for:** City tours, outdoor dining, general tourism

### **🏄 Surfer**  
- **Focus:** Wind-powered activities, tolerates cooler temps
- **Optimal temp:** 18°C | **Weights:** Temp 25%, Rain 25%, **Wind 50%**
- **Best for:** Surfing, windsurfing, kitesurfing

### **🚴 Cyclist**
- **Focus:** Moderate weather, avoids extreme rain/wind  
- **Optimal temp:** 15°C | **Weights:** Temp 35%, **Rain 40%**, Wind 25%
- **Best for:** Road cycling, mountain biking, touring

### **🌌 Stargazer**
- **Focus:** Clear skies (minimal rain), cool temperatures preferred
- **Optimal temp:** 12°C | **Weights:** Temp 30%, **Rain 60%**, Wind 10%  
- **Best for:** Astronomy, night photography, camping

### **⛷️ Skier**
- **Focus:** Cold temperatures, mountain conditions
- **Optimal temp:** 5°C | **Weights:** **Temp 70%**, Rain 20%, Wind 10%
- **Best for:** Skiing, snowboarding, winter sports

**Implementation:** Each profile uses the same Gaussian-exponential comfort formula but with different weights and optimal temperatures, allowing personalized recommendations from the same underlying ML pipeline.

---

## 🌐 Streamlit Web Interface (Deployed)

**Live Demo:** https://huggingface.co/spaces/AgaHei/France_Weather_Recommender

### **Features:**
- 📊 **Profile Selection** - Choose from 5 user profiles with distinct preferences
- 🗺️ **Interactive Map** - Plotly-powered map with hover details and colored markers
- 📅 **Date Selector** - Browse recommendations for different dates
- 📊 **Weather Metrics** - Temperature, precipitation, wind speed for each city
- 🏨 **Hotel Integration** - Nearby accommodations from OpenStreetMap data
- 📈 **Model Stats** - Real-time ML model performance metrics

### **Technical Stack:**
- **Frontend:** Streamlit with Plotly interactive maps
- **Backend:** Direct connection to Neon PostgreSQL
- **Deployment:** Hugging Face Spaces (Git-based CI/CD)
- **Maps:** Plotly + Mapbox for interactive markers with hover tooltips

---

## 🔮 Future Enhancements (Phase 4 & Beyond)

**Phase 4: CI/CD & Advanced Features**
- GitHub Actions workflow
- Automated testing (pytest)
- Docker image builds  
- Real-time weather updates API
- Mobile-responsive design

**Phase 5: Intelligence Layer**
- Explain AI recommendations (SHAP values)
- Historical trend analysis
- Weather pattern clustering
- Seasonal recommendation adjustments

---

## 🎯 Learning Outcomes

By building this project, these notions are reviewed and learned:
- ✅ End-to-end ML pipeline (data → training → deployment)
- ✅ Airflow orchestration (5 DAGs, dependencies, scheduling)
- ✅ MLflow experiment tracking
- ✅ Champion/Challenger pattern
- ✅ Feature engineering (rolling windows)
- ✅ Model serving (batch inference)
- ✅ Multi-profile recommendation systems
- ✅ PostgreSQL data modeling
- ✅ Docker containerization
- ✅ Streamlit web application development
- ✅ Interactive data visualization (Plotly)
- ✅ Cloud deployment (Hugging Face Spaces)
- ✅ Non-linear regression (Gaussian, exponential functions)
- ✅ Unsupervised learning (K-Means)
- ✅ Model evaluation (R², silhouette score, cross-validation)
- ✅ Git-based CI/CD workflows

**Suitable for:** AI Architect exam, ML Engineer interviews, portfolio projects

---

## 📚 Key Files

```
dags/
  dag_fetch_weather.py          # DAG 1: Daily weather ingestion
  dag_compute_features.py       # DAG 2: Feature engineering + profile scores
  dag_retrain_models.py         # DAG 3: Weekly model training
  dag_generate_recommendations.py  # DAG 4: Daily recommendations
  dag_fetch_hotels.py           # DAG 5: Weekly hotel data from OSM

src/
  data/
    cities.py                   # 20 French cities + coordinates
    fetch_weather.py            # Open-Meteo API client
    db.py                       # Neon PostgreSQL connection
  features/
    engineer.py                 # Rolling windows + multi-profile comfort scores
  models/
    clustering.py               # K-Means model
    regression.py               # Gradient Boosting model

app/
  streamlit_app.py              # Interactive web interface
  requirements.txt              # Streamlit dependencies

france-weather-hf/              # HF Spaces deployment
  app.py                        # Streamlit app for cloud deployment
  requirements.txt              # Cloud dependencies
  README.md                     # HF Spaces documentation

scripts/
  backfill_historical_weather.py  # One-time data backfill
  train_models.py                 # Standalone model training

docker-compose.yml              # Airflow + MLflow setup
requirements.txt                # Python dependencies
```

---

## 📖 References

**Data Source:**
- Open-Meteo (weather API): https://open-meteo.com

**Technologies:**
- Airflow: https://airflow.apache.org
- MLflow: https://mlflow.org
- scikit-learn: https://scikit-learn.org
- Neon: https://neon.tech

**Mathematical Background:**
- Gaussian distribution (Carl Gauss, 1809)
- Exponential decay (Isaac Newton, 1700s)
- K-Means clustering (Stuart Lloyd, 1957)
- Gradient Boosting (Jerome Friedman, 1999)

---

Built as a personal exercice project for AI Architect certification at Jedha (Paris).
