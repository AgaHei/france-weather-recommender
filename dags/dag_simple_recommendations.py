"""
dag_simple_recommendations.py
------------------------------
Simplified recommendations DAG that works without trained ML models.
Uses basic weather scoring logic as backup when retrain_models fails.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.db import execute_query, bulk_insert

# ---------------------------------------------------------------------------
# DAG configuration
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'aga',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 18),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'simple_recommendations',
    default_args=default_args,
    description='Generate weather recommendations using simple scoring rules',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['backup', 'recommendations', 'simple'],
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def load_weather_features(**context):
    """Load today's weather features."""
    query = """
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
        WHERE feature_date = CURRENT_DATE
          AND temp_mean_7d IS NOT NULL
          AND temp_mean_3d IS NOT NULL
        ORDER BY city
    """
    
    data = execute_query(query, fetch=True)
    
    if not data:
        raise ValueError("No features found for today. Run compute_features first.")
    
    df = pd.DataFrame(data, columns=[
        'city', 'feature_date', 'temp_mean_7d', 'temp_mean_3d',
        'precip_sum_7d', 'precip_sum_3d', 'wind_max_7d', 'wind_max_3d', 'comfort_score'
    ])
    
    print(f"📊 Loaded features for {len(df)} cities on {df['feature_date'].iloc[0]}")
    
    # Push to XCom for next task
    context['task_instance'].xcom_push(key='weather_features', value=df.to_dict('records'))
    
    return df.to_dict('records')


def simple_scoring_recommendations(**context):
    """Generate recommendations using simple weather rules."""
    print("\n" + "="*70)
    print("GENERATING SIMPLE WEATHER RECOMMENDATIONS")
    print("="*70)
    
    # Get features from previous task
    features_data = context['task_instance'].xcom_pull(
        task_ids='load_features',
        key='weather_features'
    )
    
    if not features_data:
        raise ValueError("No features data received from previous task")
    
    df = pd.DataFrame(features_data)
    
    print(f"\n🌤️ Scoring {len(df)} cities using simple rules...")
    
    # Simple scoring rules (0-100 scale)
    scores = []
    
    for _, row in df.iterrows():
        city = row['city']
        
        # Temperature score (ideal: 18-25°C)
        temp = row['temp_mean_3d']
        if 18 <= temp <= 25:
            temp_score = 100
        elif 15 <= temp < 18 or 25 < temp <= 28:
            temp_score = 80
        elif 10 <= temp < 15 or 28 < temp <= 32:
            temp_score = 60
        else:
            temp_score = 30
        
        # Precipitation score (less is better)
        precip = row['precip_sum_3d']
        if precip <= 1:
            precip_score = 100
        elif precip <= 5:
            precip_score = 80
        elif precip <= 15:
            precip_score = 50
        else:
            precip_score = 20
        
        # Wind score (calmer is better)
        wind = row['wind_max_3d']
        if wind <= 15:
            wind_score = 100
        elif wind <= 25:
            wind_score = 80
        elif wind <= 35:
            wind_score = 60
        else:
            wind_score = 30
        
        # Overall score (weighted average)
        overall_score = (temp_score * 0.5 + precip_score * 0.3 + wind_score * 0.2)
        
        scores.append({
            'city': city,
            'score': round(overall_score, 1),
            'temperature': temp,
            'precipitation': precip,
            'wind_speed': wind,
            'generated_at': datetime.utcnow().isoformat()
        })
    
    # Sort by score (highest first)
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Show top recommendations
    print(f"\n🏆 Top 10 Recommendations:")
    for i, rec in enumerate(scores[:10], 1):
        print(f"  {i:2d}. {rec['city']:<20} "
              f"Score: {rec['score']:5.1f} "
              f"(T:{rec['temperature']:4.1f}°C, "
              f"P:{rec['precipitation']:4.1f}mm, "
              f"W:{rec['wind_speed']:4.1f}km/h)")
    
    # Store recommendations in database
    print(f"\n💾 Storing {len(scores)} recommendations in database...")
    
    # Clear old recommendations first
    execute_query("DELETE FROM recommendations WHERE generated_at < NOW() - INTERVAL '7 days'")
    
    # Insert new recommendations
    bulk_insert(
        table='recommendations',
        rows=scores,
        conflict_action='(city, DATE(generated_at::timestamp)) DO UPDATE SET '
                       'score = EXCLUDED.score, '
                       'generated_at = EXCLUDED.generated_at'
    )
    
    print(f"✅ Recommendations stored successfully!")
    
    # Push summary to XCom
    summary = {
        'total_cities': len(scores),
        'top_city': scores[0]['city'],
        'top_score': scores[0]['score'],
        'generated_at': scores[0]['generated_at']
    }
    
    context['task_instance'].xcom_push(key='recommendations_summary', value=summary)
    
    print(f"\n🎯 Best recommendation: {summary['top_city']} ({summary['top_score']}/100)")
    print("="*70)
    
    return summary


# Define tasks
load_features_task = PythonOperator(
    task_id='load_features',
    python_callable=load_weather_features,
    dag=dag,
)

generate_recommendations_task = PythonOperator(
    task_id='generate_simple_recommendations',
    python_callable=simple_scoring_recommendations,
    dag=dag,
)

# Set task dependencies
load_features_task >> generate_recommendations_task