"""
dag_retrain_models.py
---------------------
DAG 3: Weekly retraining of K-Means clustering and comfort score regression models.

Schedule: Every Sunday at midnight
Duration: ~30 seconds

What it does:
1. Loads last 90 days of weather features from Neon
2. Trains K-Means clustering model
3. Trains comfort score regression model (Gradient Boosting)
4. Logs experiments to MLflow
5. Compares new models to current champion
6. Promotes new models if they outperform
7. Saves model artifacts and logs to database

MLOps patterns demonstrated:
- Scheduled retraining (weekly cadence)
- Experiment tracking (MLflow)
- Champion/Challenger pattern (A/B model comparison)
- Model versioning (timestamp-based artifacts)
- Automated promotion (metric-based decision)
- Audit logging (model_runs table)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query, execute_write
from src.features.engineer import get_kmeans_matrix, get_regression_matrix
from src.models.clustering import WeatherClusterModel
from src.models.regression import ComfortScoreModel

# MLflow setup
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("weather-models")

# Model save directory
MODELS_DIR = '/opt/airflow/mlflow/models'
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'aga',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 18),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_models',
    default_args=default_args,
    description='Weekly retraining of weather ML models',
    schedule_interval='0 0 * * 0',  # Sunday at midnight
    catchup=False,
    tags=['ml', 'retraining', 'mlops'],
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_training_data(days_back: int = 90) -> pd.DataFrame:
    """
    Load weather features for the last N days from Neon.
    
    Args:
        days_back: Number of days of history to use for training
    
    Returns:
        DataFrame with feature columns
    """
    query = f"""
        SELECT 
            city,
            feature_date,
            temp_mean_7d,
            temp_mean_3d,
            precip_sum_7d,
            precip_sum_3d,
            wind_max_7d,
            wind_max_3d,
            comfort_score
        FROM weather_features
        WHERE feature_date >= CURRENT_DATE - INTERVAL '{days_back} days'
          AND temp_mean_7d IS NOT NULL
          AND temp_mean_3d IS NOT NULL
        ORDER BY feature_date DESC
    """
    
    data = execute_query(query)
    df = pd.DataFrame(data)
    
    print(f"📊 Loaded {len(df)} training samples")
    print(f"   Date range: {df['feature_date'].min()} to {df['feature_date'].max()}")
    print(f"   Cities: {df['city'].nunique()}")
    
    return df


def get_current_champion_metrics(model_type: str) -> dict:
    """
    Get metrics of the current champion model from database.
    
    Args:
        model_type: 'kmeans' or 'regression'
    
    Returns:
        dict with metric_name, metric_value, or None if no champion exists
    """
    query = """
        SELECT metric_name, metric_value
        FROM model_runs
        WHERE model_type = %s
          AND is_champion = TRUE
        ORDER BY created_at DESC
        LIMIT 1
    """
    
    result = execute_query(query, (model_type,))
    
    if result:
        return {result[0]['metric_name']: result[0]['metric_value']}
    return None


def log_model_run(model_type: str, metric_name: str, metric_value: float, 
                  artifact_path: str, is_champion: bool):
    """
    Log a model training run to the model_runs table.
    
    Args:
        model_type: 'kmeans' or 'regression'
        metric_name: e.g., 'silhouette_score' or 'test_r2'
        metric_value: numeric metric value
        artifact_path: path to saved model file
        is_champion: whether this model is promoted to champion
    """
    query = """
        INSERT INTO model_runs (run_date, model_type, metric_name, metric_value, artifact_path, is_champion)
        VALUES (CURRENT_DATE, %s, %s, %s, %s, %s)
    """
    
    execute_write(query, (model_type, metric_name, metric_value, artifact_path, is_champion))


def demote_old_champions(model_type: str):
    """
    Set is_champion=FALSE for all previous champions of this model type.
    """
    query = """
        UPDATE model_runs
        SET is_champion = FALSE
        WHERE model_type = %s
          AND is_champion = TRUE
    """
    
    execute_write(query, (model_type,))


# ---------------------------------------------------------------------------
# Training tasks
# ---------------------------------------------------------------------------

def train_clustering_model(**context):
    """
    Train K-Means clustering model and log to MLflow.
    
    Champion promotion criteria:
    - New silhouette score > current champion silhouette + 0.05
    """
    print("\n" + "="*70)
    print("TRAINING K-MEANS CLUSTERING MODEL")
    print("="*70)
    
    # Load data
    features_df = load_training_data(days_back=90)
    
    # Use most recent data for clustering (current weather profile)
    latest_date = features_df['feature_date'].max()
    recent_df = features_df[features_df['feature_date'] == latest_date]
    
    X, city_names = get_kmeans_matrix(recent_df)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"kmeans_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Train model (using k=4 based on our earlier analysis)
        model = WeatherClusterModel(n_clusters=4)
        metrics = model.train(X, city_names)
        
        # Log to MLflow
        mlflow.log_param("n_clusters", 4)
        mlflow.log_param("n_samples", metrics['n_samples'])
        mlflow.log_metric("silhouette_score", metrics['silhouette_score'])
        mlflow.log_metric("inertia", metrics['inertia'])
        
        # Log cluster stats as text artifact
        cluster_summary = "\n".join([
            f"{name}: size={stats['size']}, temp={stats['avg_temp']}°C, "
            f"precip={stats['avg_precip']}mm, wind={stats['avg_wind']} km/h"
            for name, stats in metrics['cluster_stats'].items()
        ])
        mlflow.log_text(cluster_summary, "cluster_breakdown.txt")
        
        # Save model artifact
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODELS_DIR, f'kmeans_{timestamp}.joblib')
        model.save(model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model.model, "kmeans_model")
        
        print(f"\n✅ K-Means training complete!")
        print(f"   Silhouette score: {metrics['silhouette_score']}")
        
        # Champion/Challenger logic
        current_champion = get_current_champion_metrics('kmeans')
        
        if current_champion is None:
            # No previous champion, this becomes champion by default
            is_champion = True
            print(f"   🏆 Promoted to champion (no previous champion)")
        else:
            current_score = current_champion.get('silhouette_score', 0)
            new_score = metrics['silhouette_score']
            
            # Require 0.05 improvement to promote (avoid noisy promotions)
            if new_score > current_score + 0.05:
                is_champion = True
                print(f"   🏆 Promoted to champion!")
                print(f"      Old score: {current_score:.3f}")
                print(f"      New score: {new_score:.3f}")
                print(f"      Improvement: +{new_score - current_score:.3f}")
                
                # Demote old champion
                demote_old_champions('kmeans')
            else:
                is_champion = False
                print(f"   ⚠️  Not promoted (insufficient improvement)")
                print(f"      Champion score: {current_score:.3f}")
                print(f"      New score: {new_score:.3f}")
                print(f"      Delta: {new_score - current_score:.3f} (need +0.05)")
        
        # Log to database
        log_model_run(
            model_type='kmeans',
            metric_name='silhouette_score',
            metric_value=metrics['silhouette_score'],
            artifact_path=model_path,
            is_champion=is_champion
        )
        
        # If champion, also save as "latest"
        if is_champion:
            latest_path = os.path.join(MODELS_DIR, 'kmeans_champion.joblib')
            model.save(latest_path)
        
        # Push to XCom for downstream tasks
        context['task_instance'].xcom_push(key='kmeans_metrics', value=metrics)
        context['task_instance'].xcom_push(key='kmeans_is_champion', value=is_champion)
        
        return metrics


def train_regression_model(**context):
    """
    Train comfort score regression model and log to MLflow.
    
    Champion promotion criteria:
    - New test R² > current champion R² + 0.01
    """
    print("\n" + "="*70)
    print("TRAINING REGRESSION MODEL")
    print("="*70)
    
    # Load data
    features_df = load_training_data(days_back=90)
    
    X, y, city_names = get_regression_matrix(features_df)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Train model
        model = ComfortScoreModel(model_type='gradient_boosting')
        metrics = model.train(X, y, test_size=0.2)
        
        # Log to MLflow
        mlflow.log_param("model_type", "gradient_boosting")
        mlflow.log_param("n_samples", metrics['n_samples'])
        mlflow.log_param("n_train", metrics['n_train'])
        mlflow.log_param("n_test", metrics['n_test'])
        
        mlflow.log_metric("train_rmse", metrics['train_rmse'])
        mlflow.log_metric("train_r2", metrics['train_r2'])
        mlflow.log_metric("test_rmse", metrics['test_rmse'])
        mlflow.log_metric("test_r2", metrics['test_r2'])
        mlflow.log_metric("cv_r2_mean", metrics['cv_r2_mean'])
        mlflow.log_metric("cv_r2_std", metrics['cv_r2_std'])
        
        # Log feature importances
        if 'feature_importances' in metrics:
            for feat, imp in metrics['feature_importances'].items():
                mlflow.log_metric(f"importance_{feat}", imp)
        
        # Save model artifact
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(MODELS_DIR, f'regression_{timestamp}.joblib')
        model.save(model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model.model, "regression_model")
        
        print(f"\n✅ Regression training complete!")
        print(f"   Test R²: {metrics['test_r2']:.3f}")
        print(f"   Test RMSE: {metrics['test_rmse']:.2f}")
        
        # Champion/Challenger logic
        current_champion = get_current_champion_metrics('regression')
        
        if current_champion is None:
            is_champion = True
            print(f"   🏆 Promoted to champion (no previous champion)")
        else:
            current_r2 = current_champion.get('test_r2', 0)
            new_r2 = metrics['test_r2']
            
            # Require 0.01 improvement (1% better R²)
            if new_r2 > current_r2 + 0.01:
                is_champion = True
                print(f"   🏆 Promoted to champion!")
                print(f"      Old R²: {current_r2:.3f}")
                print(f"      New R²: {new_r2:.3f}")
                print(f"      Improvement: +{new_r2 - current_r2:.3f}")
                
                demote_old_champions('regression')
            else:
                is_champion = False
                print(f"   ⚠️  Not promoted (insufficient improvement)")
                print(f"      Champion R²: {current_r2:.3f}")
                print(f"      New R²: {new_r2:.3f}")
                print(f"      Delta: {new_r2 - current_r2:.3f} (need +0.01)")
        
        # Log to database
        log_model_run(
            model_type='regression',
            metric_name='test_r2',
            metric_value=metrics['test_r2'],
            artifact_path=model_path,
            is_champion=is_champion
        )
        
        # If champion, also save as "latest"
        if is_champion:
            latest_path = os.path.join(MODELS_DIR, 'regression_champion.joblib')
            model.save(latest_path)
        
        context['task_instance'].xcom_push(key='regression_metrics', value=metrics)
        context['task_instance'].xcom_push(key='regression_is_champion', value=is_champion)
        
        return metrics


def summarize_training_run(**context):
    """
    Print a summary of the training run.
    """
    kmeans_metrics = context['task_instance'].xcom_pull(
        task_ids='train_kmeans', key='kmeans_metrics'
    )
    regression_metrics = context['task_instance'].xcom_pull(
        task_ids='train_regression', key='regression_metrics'
    )
    
    kmeans_champion = context['task_instance'].xcom_pull(
        task_ids='train_kmeans', key='kmeans_is_champion'
    )
    regression_champion = context['task_instance'].xcom_pull(
        task_ids='train_regression', key='regression_is_champion'
    )
    
    print("\n" + "="*70)
    print("WEEKLY RETRAINING SUMMARY")
    print("="*70)
    
    print(f"\n📊 K-Means Clustering:")
    print(f"   Silhouette score: {kmeans_metrics['silhouette_score']:.3f}")
    print(f"   Status: {'🏆 CHAMPION' if kmeans_champion else '⚪ Challenger'}")
    
    print(f"\n📊 Regression Model:")
    print(f"   Test R²: {regression_metrics['test_r2']:.3f}")
    print(f"   Test RMSE: {regression_metrics['test_rmse']:.2f}")
    print(f"   CV R²: {regression_metrics['cv_r2_mean']:.3f} ± {regression_metrics['cv_r2_std']:.3f}")
    print(f"   Status: {'🏆 CHAMPION' if regression_champion else '⚪ Challenger'}")
    
    if 'feature_importances' in regression_metrics:
        print(f"\n📊 Feature Importances:")
        for feat, imp in sorted(regression_metrics['feature_importances'].items(), 
                                key=lambda x: x[1], reverse=True):
            print(f"   {feat}: {imp:.3f}")
    
    print("\n" + "="*70)
    print("✅ RETRAINING COMPLETE")
    print("="*70)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

task_train_kmeans = PythonOperator(
    task_id='train_kmeans',
    python_callable=train_clustering_model,
    dag=dag,
)

task_train_regression = PythonOperator(
    task_id='train_regression',
    python_callable=train_regression_model,
    dag=dag,
)

task_summarize = PythonOperator(
    task_id='summarize_training',
    python_callable=summarize_training_run,
    dag=dag,
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# Train both models in parallel, then summarize
[task_train_kmeans, task_train_regression] >> task_summarize
