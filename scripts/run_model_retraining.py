#!/usr/bin/env python3
"""
run_model_retraining.py
-----------------------
Standalone script to test weekly model retraining without Airflow/MLflow.
Implements champion/challenger pattern and model promotion logic.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query, execute_write
from src.features.engineer import get_kmeans_matrix, get_regression_matrix
from src.models.clustering import WeatherClusterModel
from src.models.regression import ComfortScoreModel

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'mlflow', 'models')


def load_training_data(days_back: int = 90):
    """
    Load weather features for the last N days from Neon.
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


def get_current_champion_metrics(model_type: str):
    """
    Get metrics of the current champion model from database.
    """
    query = """
        SELECT metric_name, metric_value, artifact_path
        FROM model_runs
        WHERE model_type = %s 
          AND is_champion = true
        ORDER BY created_at DESC
        LIMIT 1
    """
    
    result = execute_query(query, [model_type])
    
    if result:
        return {
            'metric_name': result[0]['metric_name'],
            'metric_value': result[0]['metric_value'],
            result[0]['metric_name']: result[0]['metric_value']
        }
    else:
        return None


def log_model_run(model_type: str, metric_name: str, metric_value: float, 
                  artifact_path: str, is_champion: bool = False):
    """
    Log model run to database.
    """
    query = """
        INSERT INTO model_runs 
        (run_date, model_type, metric_name, metric_value, artifact_path, is_champion)
        VALUES (CURRENT_DATE, %s, %s, %s, %s, %s)
    """
    
    execute_write(query, [model_type, metric_name, metric_value, artifact_path, is_champion])


def demote_old_champions(model_type: str):
    """
    Set all previous champions to is_champion=false.
    """
    query = "UPDATE model_runs SET is_champion = false WHERE model_type = %s"
    execute_write(query, [model_type])


def train_clustering_model():
    """
    Train K-Means clustering model with champion/challenger logic.
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
    
    # Train model (using k=4 based on our earlier analysis)
    model = WeatherClusterModel(n_clusters=4)
    metrics = model.train(X, city_names)
    
    # Save model artifact
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODELS_DIR, f'kmeans_{timestamp}.joblib')
    model.save(model_path)
    
    print(f"\n✅ K-Means training complete!")
    print(f"   Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"   Clusters: {model.n_clusters}")
    print(f"   Inertia: {metrics['inertia']:.1f}")
    
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
    
    # If champion, also save as champion
    if is_champion:
        champion_path = os.path.join(MODELS_DIR, 'kmeans_champion.joblib')
        model.save(champion_path)
        print(f"   💾 Saved as champion: kmeans_champion.joblib")
    
    return metrics, is_champion


def train_regression_model():
    """
    Train comfort score regression model with champion/challenger logic.
    """
    print("\n" + "="*70)
    print("TRAINING REGRESSION MODEL")
    print("="*70)
    
    # Load data
    features_df = load_training_data(days_back=90)
    
    X, y, city_names = get_regression_matrix(features_df)
    
    # Train model
    model = ComfortScoreModel(model_type='gradient_boosting')
    metrics = model.train(X, y, test_size=0.2)
    
    # Save model artifact
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODELS_DIR, f'regression_{timestamp}.joblib')
    model.save(model_path)
    
    print(f"\n✅ Regression training complete!")
    print(f"   Test R²: {metrics['test_r2']:.3f}")
    print(f"   Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"   CV R²: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
    
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
    
    # If champion, also save as champion
    if is_champion:
        champion_path = os.path.join(MODELS_DIR, 'regression_champion.joblib')
        model.save(champion_path)
        print(f"   💾 Saved as champion: regression_champion.joblib")
    
    return metrics, is_champion


def summarize_training_run(kmeans_metrics, kmeans_champion, regression_metrics, regression_champion):
    """
    Print a summary of the training run.
    """
    print("\n" + "="*70)
    print("WEEKLY RETRAINING SUMMARY")
    print("="*70)
    
    print(f"\\n📊 K-Means Clustering:")
    print(f"   Silhouette score: {kmeans_metrics['silhouette_score']:.3f}")
    print(f"   Inertia: {kmeans_metrics['inertia']:.1f}")
    print(f"   Status: {'🏆 CHAMPION' if kmeans_champion else '⚪ Challenger'}")
    
    print(f"\\n📊 Regression Model:")
    print(f"   Test R²: {regression_metrics['test_r2']:.3f}")
    print(f"   Test RMSE: {regression_metrics['test_rmse']:.2f}")
    print(f"   CV R²: {regression_metrics['cv_r2_mean']:.3f} ± {regression_metrics['cv_r2_std']:.3f}")
    print(f"   Status: {'🏆 CHAMPION' if regression_champion else '⚪ Challenger'}")
    
    if 'feature_importances' in regression_metrics:
        print(f"\\n📊 Feature Importances:")
        for feat, imp in sorted(regression_metrics['feature_importances'].items(), 
                                key=lambda x: x[1], reverse=True):
            print(f"   {feat}: {imp:.3f}")
    
    print("\\n" + "="*70)
    print("✅ RETRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    print(f"🚀 Starting weekly model retraining at {datetime.now()}")
    
    try:
        # Train both models
        print("\\n🤖 Training models in sequence...")
        
        # Train K-Means clustering
        kmeans_metrics, kmeans_champion = train_clustering_model()
        
        # Train regression model  
        regression_metrics, regression_champion = train_regression_model()
        
        # Summarize results
        summarize_training_run(kmeans_metrics, kmeans_champion, regression_metrics, regression_champion)
        
        print(f"\\n🎉 Weekly retraining completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        sys.exit(1)