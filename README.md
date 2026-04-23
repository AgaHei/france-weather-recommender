# France Weather Recommender — MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.12-blue)
![Airflow](https://img.shields.io/badge/airflow-2.8-orange)
![MLflow](https://img.shields.io/badge/mlflow-2.8-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28-red)
![Docker](https://img.shields.io/badge/docker-compose-blue)
![PostgreSQL](https://img.shields.io/badge/postgresql-neon-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-github--actions-green)
![Tests](https://img.shields.io/badge/tests-44%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**End-to-end ML system for recommending weekend destinations based on weather forecasts with multi-profile support.**

Technologies: Python, PostgreSQL (Neon), Airflow, MLflow, scikit-learn, Docker, Streamlit, Plotly

**🌐 Live Demo:** [France Weather Recommender on HF Spaces](https://huggingface.co/spaces/AgaHei/France_Weather_Recommender)

---

## 🎯 Project Overview

This is a **production-grade MLOps system** that:
- Fetches daily weather data for 20 French cities
- Engineers features using rolling windows
- Supports **4 user profiles** (leisure, wind_sports_enthusiast, cyclist, stargazer) with customized recommendations
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
- Rows: 4 (leisure, wind_sports_enthusiast, cyclist, stargazer)

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

### **🏄 Wind Sports Enthusiast**  
- **Focus:** Wind-powered activities, tolerates cooler temps
- **Optimal temp:** 18°C | **Weights:** Temp 25%, Rain 25%, **Wind 50%**
- **Best for:** Windsurfing, kitesurfing, sailing, skydiving, paragliding, land sailing

### **🚴 Cyclist**
- **Focus:** Moderate weather, avoids extreme rain/wind  
- **Optimal temp:** 15°C | **Weights:** Temp 35%, **Rain 40%**, Wind 25%
- **Best for:** Road cycling, mountain biking, touring

### **🌌 Stargazer**
- **Focus:** Clear skies (minimal rain), cool temperatures preferred
- **Optimal temp:** 12°C | **Weights:** Temp 30%, **Rain 60%**, Wind 10%  
- **Best for:** Astronomy, night photography, camping

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

## ✅ Phase 4: CI/CD Pipeline (Completed)

**Production-Ready Automation with GitHub Actions**

Implemented a comprehensive CI/CD pipeline that ensures code quality and automated deployment:

### **🔄 Automated Workflow (.github/workflows/ci-cd.yml)**

**6-Stage Pipeline:**
1. **🧪 Test** — Runs 44 unit tests with pytest covering all ML components
2. **🔍 Lint** — Code quality checks with flake8 and black formatter
3. **🛡️ Security** — Vulnerability scanning with bandit
4. **🚀 Deploy** — Automated deployment to production environment
5. **📊 Model Check** — ML model validation and performance regression detection
6. **📢 Notify** — Stakeholder notifications on success/failure

### **🎯 Key Features:**
- **Automated Testing:** 44 comprehensive tests covering ML models, feature engineering, and data pipeline
- **Quality Gates:** All jobs must pass before deployment
- **Model Validation:** Prevents regression in ML model performance
- **Fast Feedback:** Complete pipeline runs in ~3 minutes
- **Docker-Free:** Streamlined workflow using docker-compose with official images

### **🧪 Test Coverage:**
- **ML Models:** K-Means clustering, Gradient Boosting regression
- **Feature Engineering:** Rolling windows, comfort score calculations
- **Data Pipeline:** Weather API integration, database operations
- **Multi-Profile System:** All 5 user profiles validated
- **Drift Detection:** Statistical tests, Kolmogorov-Smirnov analysis, performance monitoring

**Business Impact:** Zero-downtime deployments, guaranteed code quality, automated model performance monitoring

---

## ✅ Phase 5: Production Monitoring (Completed)

**Automated ML Model Drift Detection & Alerting**

Implemented comprehensive monitoring system for production ML model health:

### **📊 Drift Detection Framework**

**3 Types of Drift Monitoring:**
1. **Feature Drift** — Weather pattern changes using Kolmogorov-Smirnov tests
2. **Model Performance Drift** — ML metrics degradation over time
3. **Prediction Drift** — Comfort score distribution shifts

### **🔄 Automated Workflow (DAG 6: detect_drift)**

**Schedule:** Weekly on Sunday 11:00 PM (after model retraining)

**Monitoring Pipeline:**
1. **Statistical Analysis** — KS tests comparing 7-day vs 28-day windows
2. **Performance Tracking** — R² and silhouette score trend analysis
3. **Threshold Detection** — Configurable drift severity (minor/major/critical)
4. **Database Logging** — Historical drift metrics in `drift_monitoring` table
5. **Automated Alerting** — Triggers retraining when drift exceeds thresholds
6. **Comprehensive Reports** — JSON reports with statistical details

### **🎯 Key Capabilities:**
- **Seasonal Adaptation:** Detects natural weather pattern changes (winter→spring)
- **Model Health:** Prevents silent model degradation in production
- **Statistical Rigor:** Scipy-powered Kolmogorov-Smirnov and t-tests
- **Weather-Optimized Thresholds:** Domain-specific sensitivity tuning
- **Historical Tracking:** Complete audit trail of model performance
- **Action Triggers:** Automatic model retraining when drift detected

### **🌤️ Weather-Domain Thresholds:**
- **Temperature:** KS threshold 0.35 (gradual seasonal changes)
- **Precipitation:** KS threshold 0.5 (very spiky, high natural variation)  
- **Wind:** KS threshold 0.4 (moderate natural variation)
- **Model Performance:** 10% R² drop threshold (ML model tolerance)
- **Feature Shift:** 2.5 standard deviations (weather volatility accommodation)

### **📈 Statistical Methods:**
- **Kolmogorov-Smirnov Test:** Non-parametric distribution comparison
- **Mean Shift Detection:** Standard deviation-based change points
- **Performance Degradation:** R² drop threshold monitoring
- **Trend Analysis:** Time-series performance tracking

### **🌱 Seasonal Adaptation vs Model Drift**

**Key MLOps Insight:** The system successfully distinguishes between healthy seasonal adaptation and problematic model drift:

**✅ Healthy Seasonal Adaptation (Expected):**
- **Natural weather transitions** (winter → spring → summer) with different comfort patterns
- **Model performance remains excellent** (R² = 0.998, Silhouette = 0.455)
- **Recommendation shifts reflect real seasonal preferences** (cooler cities preferred in summer)
- **Statistical drift detection confirms natural variation** vs model degradation

**❌ Problematic Model Drift (Alerts Required):**
- **Sudden performance drops** (R² degradation, poor clustering)
- **Prediction accuracy deterioration** independent of seasonal patterns
- **Data quality issues** or **API changes** affecting input features
- **Model staleness** requiring immediate retraining

**Production Validation (April 2026):** Analysis confirmed winter→spring transition showed natural seasonal preferences with maintained model accuracy, demonstrating the system correctly adapts to legitimate weather pattern changes while monitoring for true performance degradation.

**Business Impact:** Automated model health monitoring, proactive drift detection, maintained prediction accuracy

---

## 🔮 Future Enhancements (Phase 6 & Beyond)

**Phase 5: Intelligence Layer**
- Explain AI recommendations (SHAP values)
- Historical trend analysis
- Weather pattern clustering
- Seasonal recommendation adjustments
- Real-time weather updates API
- Mobile-responsive design

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
- ✅ **CI/CD automation (GitHub Actions, automated testing, deployment pipelines)**
- ✅ **Production-grade testing (44 unit tests, quality gates, model validation)**
- ✅ **ML monitoring (drift detection, statistical analysis, automated alerting)**
- ✅ **Production monitoring (KS tests, performance tracking, threshold alerts)**

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
  dag_detect_drift.py           # DAG 6: Weekly drift detection & monitoring

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
  check_drift.py                  # Drift detection & monitoring
  add_drift_monitoring_table.py   # Database setup for monitoring

tests/                          # Comprehensive test suite (44 tests)
  test_features.py              # Feature engineering tests
  test_models.py                # ML model tests
  test_drift.py                 # Drift detection tests (13 tests)
  conftest.py                   # Test configuration

.github/workflows/
  ci-cd.yml                     # GitHub Actions CI/CD pipeline

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
