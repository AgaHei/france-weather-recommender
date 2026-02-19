"""
dag_detect_drift.py
-------------------
DAG 6: Automated drift detection and alerting.

Schedule: Weekly on Sunday at 11:00 PM (after weekly retraining)
Duration: ~30 seconds

What it does:
1. Checks feature drift (weather patterns changing)
2. Checks model performance drift (metrics degrading)
3. Checks prediction drift (scores shifting)
4. Generates drift report
5. Triggers retraining if drift detected

MLOps patterns demonstrated:
- Automated monitoring
- Statistical drift detection
- Alerting and remediation
- Model health checks
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path so we can import from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import drift detection functions
from scripts.check_drift import (
    detect_feature_drift,
    detect_model_performance_drift,
    detect_prediction_drift,
    generate_drift_report
)

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
    'detect_drift',
    default_args=default_args,
    description='Weekly drift detection and monitoring',
    schedule_interval='0 23 * * 0',  # Sunday at 11:00 PM (after retraining)
    catchup=False,
    tags=['monitoring', 'drift-detection', 'mlops'],
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def run_drift_detection(**context):
    """
    Run comprehensive drift detection across all dimensions.
    """
    print("\n" + "="*70)
    print("WEEKLY DRIFT DETECTION")
    print("="*70)
    
    # Generate full drift report
    report = generate_drift_report()
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='drift_report', value=report)
    context['task_instance'].xcom_push(key='drift_detected', value=report['overall_drift_detected'])
    
    return report


def analyze_drift_results(**context):
    """
    Analyze drift results and determine actions.
    """
    report = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drift_report'
    )
    
    print("\n" + "="*70)
    print("DRIFT ANALYSIS")
    print("="*70)
    
    if report['overall_drift_detected']:
        print("\n⚠️  DRIFT DETECTED - Analysis:")
        
        # Feature drift analysis
        feature_drifts = [
            f for f, result in report['feature_drift'].items()
            if result.get('drift_detected', False)
        ]
        
        if feature_drifts:
            print(f"\n📊 Feature drift in: {', '.join(feature_drifts)}")
            print("   Possible causes:")
            print("   • Seasonal weather pattern changes")
            print("   • Climate anomalies")
            print("   • Data quality issues")
        
        # Model performance drift
        model_drifts = [
            m for m, result in report['model_drift'].items()
            if result.get('drift_detected', False)
        ]
        
        if model_drifts:
            print(f"\n📉 Model performance degradation in: {', '.join(model_drifts)}")
            print("   Possible causes:")
            print("   • Concept drift (user preferences changing)")
            print("   • Feature drift affecting model")
            print("   • Model staleness")
        
        # Prediction drift
        if report['prediction_drift'].get('leisure', {}).get('drift_detected'):
            print(f"\n🎯 Prediction distribution shifted")
            print("   Possible causes:")
            print("   • Seasonal effects")
            print("   • Model degradation")
            print("   • New weather patterns")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Decide if we need to trigger retraining
        needs_retraining = any([
            feature_drifts,
            model_drifts,
        ])
        
        context['task_instance'].xcom_push(key='needs_retraining', value=needs_retraining)
        
        if needs_retraining:
            print(f"\n🚨 ACTION: Triggering model retraining DAG")
        
    else:
        print("\n✅ NO DRIFT DETECTED")
        print("   All systems operating within normal parameters")
        print("   No action required")
        
        context['task_instance'].xcom_push(key='needs_retraining', value=False)


def log_drift_metrics(**context):
    """
    Log drift metrics to database for historical tracking.
    
    In production, you'd store these in a monitoring table:
    - drift_monitoring (timestamp, metric_name, metric_value, drift_detected)
    """
    report = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drift_report'
    )
    
    print("\n📊 Drift Metrics Summary:")
    
    # Feature drift metrics
    for feature, result in report['feature_drift'].items():
        if 'ks_statistic' in result:
            print(f"   {feature}: KS={result['ks_statistic']:.3f}, "
                  f"drift={'YES' if result['drift_detected'] else 'NO'}")
    
    # Model performance metrics
    for model, result in report['model_drift'].items():
        if 'champion_value' in result:
            print(f"   {model}: metric={result['champion_value']:.3f}, "
                  f"drift={'YES' if result['drift_detected'] else 'NO'}")
    
    # In production, you'd insert these into a monitoring table
    # This creates a historical record of drift over time
    # Useful for: trend analysis, alerting thresholds, compliance


def send_alert_if_drift(**context):
    """
    Send alert if drift detected (Slack, email, PagerDuty, etc.)
    
    In production, integrate with your alerting system.
    """
    drift_detected = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drift_detected'
    )
    
    needs_retraining = context['task_instance'].xcom_pull(
        task_ids='analyze_results',
        key='needs_retraining'
    )
    
    if drift_detected:
        # In production, send actual alerts
        alert_message = f"""
        🚨 ML DRIFT ALERT
        
        Drift detected in weather recommendation system.
        
        Automatic retraining: {'Triggered' if needs_retraining else 'Not needed'}
        
        View details: [Airflow Logs]
        Drift report: [S3 bucket / logging system]
        """
        
        print("\n" + "="*70)
        print("ALERT NOTIFICATION")
        print("="*70)
        print(alert_message)
        
        # Production integrations:
        # - Slack: requests.post(webhook_url, json={'text': alert_message})
        # - Email: smtplib.send_mail(...)
        # - PagerDuty: pypd.Incident.create(...)
        # - DataDog: statsd.increment('ml.drift_detected')
        
        print("📧 Alert would be sent to:")
        print("   • Slack: #ml-monitoring")
        print("   • Email: ml-team@company.com")
        print("   • PagerDuty: ML On-call")
    else:
        print("\n✅ No alerts needed - system healthy")


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

task_detect = PythonOperator(
    task_id='detect_drift',
    python_callable=run_drift_detection,
    dag=dag,
)

task_analyze = PythonOperator(
    task_id='analyze_results',
    python_callable=analyze_drift_results,
    dag=dag,
)

task_log = PythonOperator(
    task_id='log_metrics',
    python_callable=log_drift_metrics,
    dag=dag,
)

task_alert = PythonOperator(
    task_id='send_alert',
    python_callable=send_alert_if_drift,
    dag=dag,
)

# Conditional retraining trigger (only if drift detected)
# Note: This is commented out to avoid auto-triggering in demo
# In production, you'd enable this
# 
# task_retrain = TriggerDagRunOperator(
#     task_id='trigger_retraining',
#     trigger_dag_id='retrain_models',
#     dag=dag,
#     trigger_rule='all_done',  # Run even if upstream fails
# )

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

task_detect >> task_analyze >> task_log >> task_alert
# task_alert >> task_retrain  # Uncomment to enable auto-retraining
