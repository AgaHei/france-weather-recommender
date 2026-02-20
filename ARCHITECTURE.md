# France Weather Recommender — System Architecture

## Overview Diagram

```mermaid
graph TB
    subgraph "External Data Sources"
        API1[Open-Meteo API<br/>Weather Data]
        API2[OpenStreetMap<br/>Overpass API<br/>Hotels]
    end

    subgraph "Data Ingestion Layer"
        DAG1[DAG 1: fetch_weather<br/>Daily 6:00 AM<br/>7-day forecast]
        DAG2[DAG 2: compute_features<br/>Triggered by DAG 1<br/>Rolling windows + scoring]
    end

    subgraph "Database Layer — PostgreSQL Neon"
        DB[(Neon PostgreSQL)]
        T1[raw_weather<br/>1,800+ rows]
        T2[weather_features<br/>Multi-profile scores]
        T3[profile_scores<br/>5 profiles × 20 cities]
        T4[cities<br/>20 French cities]
        T5[recommendations<br/>Daily top picks]
        T6[hotels<br/>30+ POIs]
        T7[model_runs<br/>Training history]
        T8[scoring_profiles<br/>5 user personas]
    end

    subgraph "ML Training Layer"
        DAG3[DAG 3: retrain_models<br/>Weekly Sunday 00:00<br/>Champion/Challenger]
        KMEANS[K-Means Clustering<br/>Silhouette: 0.367]
        REGRESSION[Gradient Boosting<br/>R²: 0.995]
        MLFLOW[MLflow Tracking<br/>Experiment history]
    end

    subgraph "Inference Layer"
        DAG4[DAG 4: generate_recommendations<br/>Daily 7:00 AM<br/>All 5 profiles]
        STAGE1[Stage 1: Clustering<br/>Coarse filter]
        STAGE2[Stage 2: Regression<br/>Fine ranking]
    end

    subgraph "Data Enrichment Layer"
        DAG5[DAG 5: fetch_hotels<br/>Weekly Monday 1:00 AM<br/>Top 3 cities]
    end

    subgraph "Monitoring Layer"
        DAG6[DAG 6: detect_drift<br/>Weekly Sunday 23:00<br/>Post-retraining check]
        DRIFT1[Feature Drift<br/>Temp: 0.35, Rain: 0.5, Wind: 0.4]
        DRIFT2[Model Drift<br/>10% tolerance]
        DRIFT3[Prediction Drift<br/>Distribution shift]
        ALERT[Alert System<br/>Slack/Email]
    end

    subgraph "User Interface Layer"
        UI[Streamlit App<br/>HF Spaces<br/>Interactive Maps]
        MAP[Folium/Plotly Maps<br/>Ranked markers]
    end

    subgraph "CI/CD Layer"
        GHA[GitHub Actions<br/>6 automated jobs]
        TEST[Test Suite<br/>44 comprehensive tests]
        LINT[Code Quality<br/>Black, Flake8]
        SECURITY[Security Scan<br/>Safety]
        PERF[Model Performance<br/>Threshold checks]
    end

    %% Data Flow Connections
    API1 -->|HTTP GET| DAG1
    DAG1 -->|INSERT| T1
    T1 -->|READ| DAG2
    DAG2 -->|COMPUTE| T2
    DAG2 -->|INSERT| T3
    T8 -->|LOAD| DAG2

    T2 -->|READ| DAG3
    T3 -->|READ| DAG3
    DAG3 -->|TRAIN| KMEANS
    DAG3 -->|TRAIN| REGRESSION
    KMEANS -->|LOG| MLFLOW
    REGRESSION -->|LOG| MLFLOW
    KMEANS -->|SAVE| T7
    REGRESSION -->|SAVE| T7

    T2 -->|READ| DAG4
    T3 -->|READ| DAG4
    T7 -->|LOAD CHAMPION| DAG4
    DAG4 -->|CLUSTER| STAGE1
    STAGE1 -->|RANK| STAGE2
    STAGE2 -->|INSERT| T5

    T5 -->|TOP 3 CITIES| DAG5
    API2 -->|HTTP POST| DAG5
    DAG5 -->|INSERT| T6

    T2 -->|READ| DAG6
    T3 -->|READ| DAG6
    T7 -->|READ| DAG6
    DAG6 -->|CHECK| DRIFT1
    DAG6 -->|CHECK| DRIFT2
    DAG6 -->|CHECK| DRIFT3
    DRIFT1 -->|IF DRIFT| ALERT
    DRIFT2 -->|IF DRIFT| ALERT
    DRIFT3 -->|IF DRIFT| ALERT
    ALERT -.->|TRIGGER| DAG3

    T5 -->|QUERY| UI
    T6 -->|QUERY| UI
    T4 -->|COORDINATES| UI
    UI -->|DISPLAY| MAP

    GHA -->|ON PUSH| TEST
    GHA -->|ON PUSH| LINT
    GHA -->|ON PUSH| SECURITY
    GHA -->|ON PUSH| PERF
    TEST -.->|PASS| DAG1
    
    %% Styling
    classDef dagStyle fill:#4A90E2,stroke:#2E5C8A,stroke-width:2px,color:#fff
    classDef dbStyle fill:#50C878,stroke:#2E7D4E,stroke-width:2px,color:#fff
    classDef mlStyle fill:#9B59B6,stroke:#6C3483,stroke-width:2px,color:#fff
    classDef uiStyle fill:#E74C3C,stroke:#A93226,stroke-width:2px,color:#fff
    classDef ciStyle fill:#F39C12,stroke:#B9770E,stroke-width:2px,color:#fff
    
    class DAG1,DAG2,DAG3,DAG4,DAG5,DAG6 dagStyle
    class DB,T1,T2,T3,T4,T5,T6,T7,T8 dbStyle
    class KMEANS,REGRESSION,MLFLOW mlStyle
    class UI,MAP uiStyle
    class GHA,TEST,LINT,SECURITY,PERF ciStyle
```

## Detailed Component Breakdown

### 🔄 Data Pipeline (DAGs 1-2)
**Daily Execution Flow:**
```
06:00 → DAG 1 fetches 7-day forecast for 20 cities
06:05 → DAG 2 triggered automatically
06:05 → Computes rolling windows (7-day, 3-day)
06:06 → Computes comfort scores for ALL 5 profiles
06:07 → Inserts ~100 profile_score records (20 cities × 5 profiles)
```

**Outputs:**
- `weather_features`: temp_mean_7d, temp_mean_3d, precip_sum_7d, precip_sum_3d, wind_max_7d, wind_max_3d
- `profile_scores`: Leisure: 67.0/100, Wind Sports Enthusiast: 83.8/100, Cyclist: 30.5/100, etc.

---

### 🤖 ML Training Layer (DAG 3)
**Weekly Execution Flow:**
```
Sunday 00:00 → Load last 90 days of features
Sunday 00:01 → Train K-Means (k=4)
Sunday 00:02 → Train Gradient Boosting
Sunday 00:03 → Log to MLflow
Sunday 00:04 → Champion/Challenger comparison
Sunday 00:05 → Promote if better (R² +0.01 or Silhouette +0.05)
```

**Models:**
- **K-Means:** 4 clusters, silhouette=0.367
- **Gradient Boosting:** R²=0.995, RMSE=0.92, Feature importance: Precip 77%, Temp 19%, Wind 4%

**Champion/Challenger Logic:**
```
IF new_r2 > champion_r2 + 0.01:
    promote_to_champion()
    demote_old_champion()
ELSE:
    keep_current_champion()
```

---

### 🎯 Inference Layer (DAG 4)
**Two-Stage Retrieval:**
```
Stage 1: K-Means Clustering (Coarse Filter)
├─ Cluster 20 cities into 4 weather profiles
├─ Rank clusters by comfort (Mediterranean > Atlantic > Continental)
└─ Keep cities in top 2 clusters

Stage 2: Regression Scoring (Fine Ranking)
├─ Predict comfort scores for filtered cities
├─ Rank by score (highest first)
└─ Assign ranks 1-N per profile
```

**Output:**
- `recommendations`: 5 profiles × ~8 cities each = ~40 records/day
- Example: (2026-02-19, Nice, leisure, cluster=2, score=67.0, rank=1)

---

### 🏨 Data Enrichment (DAG 5)
**Weekly Execution Flow:**
```
Monday 01:00 → Query recommendations for top 3 cities
Monday 01:01 → Overpass API: Search hotels within 10km radius
Monday 01:02 → Filter to top 10 hotels per city (by stars)
Monday 01:03 → Insert ~30 hotels total
```

**Hotel Filtering:**
- ✅ hotel, guest_house, apartment types
- ✅ Sort by: stars DESC, then alphabetical
- ✅ Limit: 10 per city
- ❌ Exclude: hostels (unless user wants budget)

---

### 📊 Monitoring Layer (DAG 6)
**Drift Detection Strategy:**

**Adjusted Thresholds (Production-Tuned):**
```python
DRIFT_THRESHOLDS = {
    # Feature drift (K-S test)
    'temp_mean_3d':   0.35,  # Seasonal changes expected
    'precip_sum_3d':  0.50,  # Natural spikiness in rain
    'wind_max_3d':    0.40,  # Moderate variation
    
    # Model performance drift
    'performance_drop': 0.10,  # 10% tolerance for noise
    
    # Mean shift (std deviations)
    'feature_shift': 2.0,  # 2 sigma rule
}
```

**Why These Thresholds:**
- **Temperature (0.35):** Winter→Summer can shift 10°C+, need tolerance
- **Precipitation (0.50):** Rain is binary (rainy days vs dry spells)
- **Wind (0.40):** Coastal vs inland variance
- **Performance (10%):** R²: 0.995→0.895 still acceptable, avoid noise alerts

**Weekly Execution Flow:**
```
Sunday 23:00 → Compare current week vs 4-week baseline
Sunday 23:01 → Run K-S tests on features
Sunday 23:02 → Check model performance (R², silhouette)
Sunday 23:03 → Check prediction distributions
Sunday 23:04 → Generate drift report
Sunday 23:05 → Send alerts IF drift detected
Sunday 23:06 → (Optional) Trigger DAG 3 retraining
```

**Drift Decision Tree:**
```
IF ks_statistic > threshold:
    IF mean_shift > 2.0:
        severity = CRITICAL → Retrain immediately
    ELSE:
        severity = MAJOR → Alert, schedule retraining
ELSE IF performance_drop > 0.10:
    severity = MAJOR → Retrain next cycle
ELSE:
    severity = NONE → Continue monitoring
```

---

### 🎨 User Interface (Streamlit on HF Spaces)
**Features:**
- Profile dropdown: 🏖️ Leisure, 🏄 Wind Sports Enthusiast, 🚴 Cyclist, ⭐ Stargazer
- Interactive map: Folium/Plotly with ranked markers
- City cards: Weather metrics + top 5 hotels
- Model metrics: Current champion R² + silhouette
- Date selector: Historical recommendations

**Technology Stack:**
- Streamlit 1.28+ (UI framework)
- Plotly (interactive maps, better HF Spaces compatibility)
- Neon PostgreSQL (live data queries)
- Deployed: Hugging Face Spaces (public)

---

### 🔧 CI/CD Pipeline (GitHub Actions)
**6 Automated Jobs:**

1. **🧪 Test Suite**
   - 44 comprehensive tests (drift, features, models)
   - Complete ML pipeline validation
   - Tests: unit, integration, statistical methods

2. **🎨 Code Quality**
   - Black (formatting)
   - Flake8 (linting)
   - isort (import sorting)

3. **🔒 Security Scan**
   - Safety check for CVEs
   - Dependency vulnerability scanning

4. **📊 Model Performance**
   - Train on synthetic data
   - Verify R² > 0.75
   - Verify silhouette > 0.3

5. **🚀 Deploy**
   - Only on main branch
   - Only if tests pass
   - Automated deployment pipeline

6. **📢 Notify**
   - On failure only
   - Would send Slack/email in production

**Trigger:** Every push to main or PR

**Workflow:**
```
git push → GitHub Actions
  ├─ [Parallel] Test + Lint + Security
  ├─ [Sequential] Model Performance
  ├─ [If Pass] Deploy
  └─ [If Fail] Notify
```

---

## 📈 System Metrics

**Scale:**
- **Cities:** 20 French destinations
- **Profiles:** 5 user personas
- **Daily Records:** ~100 (20 cities × 5 profiles)
- **Weekly Hotels:** ~30 (top 3 cities × 10 hotels)
- **DAG Runs:** 6 DAGs × weekly = ~40 executions/week
- **Test Suite:** 44 comprehensive tests
- **CI/CD Jobs:** 6 per commit

**Performance:**
- **API Latency:** Open-Meteo: <500ms, Overpass: 1-2s
- **DAG Duration:** DAG 1: 30s, DAG 2: 45s, DAG 3: 2min, DAG 4: 30s, DAG 5: 45s, DAG 6: 30s
- **Model Training:** K-Means: 15s, Regression: 45s
- **Batch Inference:** 20 cities × 5 profiles in <30s

**Reliability:**
- **Idempotent Writes:** ON CONFLICT DO UPDATE (no duplicate rows)
- **Error Handling:** Per-city try/catch (one failure doesn't kill DAG)
- **Retries:** Airflow retries=1, retry_delay=5min
- **Monitoring:** Drift detection weekly, alerts on threshold breach

---

## 🎯 MLOps Best Practices Demonstrated

✅ **Orchestration:** Airflow DAGs with dependencies  
✅ **Feature Engineering:** Separate DAG (reproducibility)  
✅ **Experiment Tracking:** MLflow with full lineage  
✅ **Model Versioning:** Champion/Challenger with Git SHA tags  
✅ **Automated Testing:** 44 comprehensive tests in CI/CD  
✅ **Monitoring:** Feature drift + model drift + prediction drift  
✅ **Deployment:** Automated via CI/CD on main branch  
✅ **Infrastructure as Code:** Docker Compose, GitHub Actions YAML  
✅ **Documentation:** README, code comments, architecture diagrams  

---

## 🔮 Future Enhancements (Post-Exam)

**Phase 4: Real-Time API**
- FastAPI endpoint for on-demand recommendations
- Redis caching for low latency (<100ms)
- Rate limiting per user

**Phase 5: Advanced Monitoring**
- Grafana dashboards (drift trends, model performance)
- DataDog APM integration
- PagerDuty alerts for critical drift

**Phase 6: Scale-Up**
- Kubernetes deployment (replace Docker Compose)
- Horizontal pod autoscaling
- 100+ cities across Europe

**Not Needed Yet:**
- Feature store (scale too small)
- Model explainability (SHAP) — nice-to-have
- A/B testing framework (Champion/Challenger sufficient)

---

Built by Aga for AI Architect Certification — February 2026
