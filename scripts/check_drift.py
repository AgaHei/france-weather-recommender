"""
check_drift.py
--------------
Drift detection for production ML monitoring.

Checks:
1. Feature drift - Compare current weather distributions vs baseline
2. Model performance drift - Track metrics over time
3. Prediction drift - Monitor comfort score distributions

Run manually:
    python scripts/check_drift.py

Run in CI/CD:
    pytest tests/test_drift.py

Run in Airflow:
    Add to weekly schedule as DAG 6
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from src.data.db import execute_query, execute_write
import json


# ============================================================================
# Configuration
# ============================================================================

DRIFT_THRESHOLDS = {
    # Weather data has natural seasonal variations - use higher thresholds
    'ks_statistic': 0.4,       # Kolmogorov-Smirnov test threshold (was 0.2)
    'performance_drop': 0.10,   # 10% R² drop triggers alert (was 0.05) 
    'feature_shift': 2.5,       # 2.5 standard deviations from mean (was 2.0)
    
    # Feature-specific thresholds for weather domain
    'temp_ks_threshold': 0.35,   # Temperature: gradual seasonal changes
    'precip_ks_threshold': 0.5,  # Precipitation: very spiky, high natural variation
    'wind_ks_threshold': 0.4,    # Wind: moderate natural variation
}

BASELINE_WINDOW_DAYS = 28  # 4 weeks of baseline data
CURRENT_WINDOW_DAYS = 7    # Compare last 7 days


def log_drift_to_database(check_type, metric_name, metric_value, baseline_value, 
                          drift_score, drift_detected, threshold, metadata=None):
    """
    Log drift metrics to database for historical tracking.
    
    Args:
        check_type: 'feature', 'model', or 'prediction'
        metric_name: Name of metric (e.g., 'temp_mean_3d')
        metric_value: Current value
        baseline_value: Baseline for comparison
        drift_score: KS statistic or % change
        drift_detected: Boolean
        threshold: Threshold used
        metadata: Optional dict with additional context
    """
    # Determine severity
    if not drift_detected:
        severity = 'none'
    elif drift_score < threshold * 1.5:
        severity = 'minor'
    elif drift_score < threshold * 2.0:
        severity = 'major'
    else:
        severity = 'critical'
    
    # Insert into database
    query = """
        INSERT INTO drift_monitoring 
        (check_type, metric_name, metric_value, baseline_value, drift_score, 
         drift_detected, threshold_used, drift_severity, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    metadata_json = json.dumps(metadata) if metadata else None
    
    # Convert numpy types to Python native types for PostgreSQL
    def convert_numpy(value):
        """Convert numpy types to Python native types."""
        if hasattr(value, 'item'):  # numpy scalars
            return value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    # Convert all numeric values
    metric_value = convert_numpy(metric_value) if metric_value is not None else None
    baseline_value = convert_numpy(baseline_value) if baseline_value is not None else None
    drift_score = convert_numpy(drift_score) if drift_score is not None else None
    threshold = convert_numpy(threshold) if threshold is not None else None
    
    try:
        execute_write(query, (
            check_type, metric_name, metric_value, baseline_value, 
            drift_score, drift_detected, threshold, severity, metadata_json
        ))
    except Exception as e:
        # Don't fail drift detection if logging fails
        print(f"⚠️  Warning: Could not log to database: {e}")


# ============================================================================
# Feature Drift Detection
# ============================================================================

def detect_feature_drift(feature_name='temp_mean_3d', profile_name='leisure', log_to_db=True):
    """
    Detect drift in weather features using Kolmogorov-Smirnov test.
    
    Compares current week's distribution to baseline (last 4 weeks).
    
    Args:
        feature_name: Feature to monitor (temp_mean_3d, precip_sum_3d, wind_max_3d)
        profile_name: Profile to check (affects scoring but not weather features)
    
    Returns:
        dict with drift metrics
    """
    print(f"\n{'='*70}")
    print(f"FEATURE DRIFT DETECTION: {feature_name}")
    print(f"{'='*70}")
    
    # Get baseline data (4 weeks ago)
    baseline_start = datetime.now().date() - timedelta(days=BASELINE_WINDOW_DAYS + CURRENT_WINDOW_DAYS)
    baseline_end = datetime.now().date() - timedelta(days=CURRENT_WINDOW_DAYS)
    
    baseline_query = f"""
        SELECT {feature_name}
        FROM weather_features
        WHERE feature_date BETWEEN %s AND %s
          AND {feature_name} IS NOT NULL
    """
    
    baseline_data = execute_query(baseline_query, (baseline_start, baseline_end))
    
    if not baseline_data:
        print("⚠️  No baseline data available")
        return {'drift_detected': False, 'reason': 'no_baseline_data'}
    
    baseline = [row[feature_name] for row in baseline_data]
    
    # Get current data (last 7 days)
    current_start = datetime.now().date() - timedelta(days=CURRENT_WINDOW_DAYS)
    
    current_query = f"""
        SELECT {feature_name}
        FROM weather_features
        WHERE feature_date >= %s
          AND {feature_name} IS NOT NULL
    """
    
    current_data = execute_query(current_query, (current_start,))
    
    if not current_data:
        print("⚠️  No current data available")
        return {'drift_detected': False, 'reason': 'no_current_data'}
    
    current = [row[feature_name] for row in current_data]
    
    # Statistical comparison
    print(f"\nBaseline: {len(baseline)} samples ({baseline_start} to {baseline_end})")
    print(f"  Mean: {np.mean(baseline):.2f}")
    print(f"  Std:  {np.std(baseline):.2f}")
    print(f"  Min:  {np.min(baseline):.2f}")
    print(f"  Max:  {np.max(baseline):.2f}")
    
    print(f"\nCurrent: {len(current)} samples ({current_start} to {datetime.now().date()})")
    print(f"  Mean: {np.mean(current):.2f}")
    print(f"  Std:  {np.std(current):.2f}")
    print(f"  Min:  {np.min(current):.2f}")
    print(f"  Max:  {np.max(current):.2f}")
    
    # Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(baseline, current)
    
    print(f"\n📊 Kolmogorov-Smirnov Test:")
    print(f"   KS Statistic: {ks_statistic:.4f}")
    print(f"   P-value: {p_value:.4f}")
    
    # Mean shift test (in standard deviations)
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    current_mean = np.mean(current)
    
    if baseline_std > 0:
        mean_shift = abs(current_mean - baseline_mean) / baseline_std
    else:
        mean_shift = 0
    
    print(f"   Mean shift: {mean_shift:.2f} standard deviations")
    
    # Select appropriate threshold based on feature type
    if 'temp' in feature_name:
        ks_threshold = DRIFT_THRESHOLDS['temp_ks_threshold']
    elif 'precip' in feature_name:
        ks_threshold = DRIFT_THRESHOLDS['precip_ks_threshold'] 
    elif 'wind' in feature_name:
        ks_threshold = DRIFT_THRESHOLDS['wind_ks_threshold']
    else:
        ks_threshold = DRIFT_THRESHOLDS['ks_statistic']  # default
    
    # Drift detection
    drift_detected = False
    reasons = []
    
    if ks_statistic > ks_threshold:
        drift_detected = True
        reasons.append(f"KS statistic {ks_statistic:.3f} > threshold {ks_threshold}")
    
    if mean_shift > DRIFT_THRESHOLDS['feature_shift']:
        drift_detected = True
        reasons.append(f"Mean shifted {mean_shift:.2f} std devs (threshold: {DRIFT_THRESHOLDS['feature_shift']})")
    
    if drift_detected:
        print(f"\n⚠️  DRIFT DETECTED!")
        for reason in reasons:
            print(f"   • {reason}")
    else:
        print(f"\n✅ No significant drift detected (threshold: {ks_threshold:.2f})")
    
    # Log to database if enabled
    if log_to_db:
        log_drift_to_database(
            check_type='feature',
            metric_name=feature_name,
            metric_value=current_mean,
            baseline_value=baseline_mean,
            drift_score=ks_statistic,
            drift_detected=drift_detected,
            threshold=ks_threshold,
            metadata={
                'p_value': p_value,
                'mean_shift_std': mean_shift,
                'baseline_samples': len(baseline),
                'current_samples': len(current),
                'threshold_used': ks_threshold,
                'feature_type': feature_name
            }
        )
    
    return {
        'drift_detected': drift_detected,
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'mean_shift_std': mean_shift,
        'baseline_mean': baseline_mean,
        'current_mean': current_mean,
        'reasons': reasons
    }


# ============================================================================
# Model Performance Drift Detection
# ============================================================================

def detect_model_performance_drift(model_type='regression', metric='test_r2'):
    """
    Track model performance over time using MLflow data.
    
    Args:
        model_type: 'regression' or 'kmeans'
        metric: Metric to track ('test_r2', 'silhouette_score')
    
    Returns:
        dict with performance drift metrics
    """
    print(f"\n{'='*70}")
    print(f"MODEL PERFORMANCE DRIFT: {model_type}")
    print(f"{'='*70}")
    
    # Get performance history from model_runs table
    query = """
        SELECT created_at, metric_value, is_champion
        FROM model_runs
        WHERE model_type = %s
          AND metric_name = %s
        ORDER BY created_at DESC
        LIMIT 10
    """
    
    history = execute_query(query, (model_type, metric))
    
    if len(history) < 2:
        print("⚠️  Not enough historical data (need at least 2 runs)")
        return {'drift_detected': False, 'reason': 'insufficient_history'}
    
    df = pd.DataFrame(history)
    
    # Current champion performance
    champion = df[df['is_champion'] == True].iloc[0] if any(df['is_champion']) else df.iloc[0]
    champion_value = champion['metric_value']
    
    # Historical average (last 5-10 runs)
    historical_mean = df['metric_value'].mean()
    historical_std = df['metric_value'].std()
    
    print(f"\nPerformance History ({len(df)} runs):")
    print(f"  Current champion: {champion_value:.4f}")
    print(f"  Historical mean:  {historical_mean:.4f}")
    print(f"  Historical std:   {historical_std:.4f}")
    
    # Check for degradation
    performance_drop = (historical_mean - champion_value) / historical_mean
    
    print(f"\n📊 Performance Analysis:")
    print(f"   Drop from mean: {performance_drop:.2%}")
    
    drift_detected = False
    reason = None
    
    if performance_drop > DRIFT_THRESHOLDS['performance_drop']:
        drift_detected = True
        reason = f"Performance dropped {performance_drop:.2%} (threshold: {DRIFT_THRESHOLDS['performance_drop']:.2%})"
        print(f"\n⚠️  PERFORMANCE DEGRADATION DETECTED!")
        print(f"   {reason}")
        print(f"\n💡 Recommendation: Trigger model retraining")
    else:
        print(f"\n✅ Model performance stable")
    
    return {
        'drift_detected': drift_detected,
        'champion_value': champion_value,
        'historical_mean': historical_mean,
        'performance_drop': performance_drop,
        'reason': reason
    }


# ============================================================================
# Prediction Drift Detection
# ============================================================================

def detect_prediction_drift(profile_name='leisure'):
    """
    Monitor distribution of comfort scores over time.
    
    Args:
        profile_name: Profile to monitor
    
    Returns:
        dict with prediction drift metrics
    """
    print(f"\n{'='*70}")
    print(f"PREDICTION DRIFT: {profile_name}")
    print(f"{'='*70}")
    
    # Get baseline predictions (4 weeks ago)
    baseline_start = datetime.now().date() - timedelta(days=BASELINE_WINDOW_DAYS + CURRENT_WINDOW_DAYS)
    baseline_end = datetime.now().date() - timedelta(days=CURRENT_WINDOW_DAYS)
    
    baseline_query = """
        SELECT comfort_score
        FROM profile_scores
        WHERE feature_date BETWEEN %s AND %s
          AND profile_name = %s
    """
    
    baseline_data = execute_query(baseline_query, (baseline_start, baseline_end, profile_name))
    
    if not baseline_data:
        print("⚠️  No baseline predictions available")
        return {'drift_detected': False, 'reason': 'no_baseline_predictions'}
    
    baseline = [row['comfort_score'] for row in baseline_data]
    
    # Get current predictions (last 7 days)
    current_start = datetime.now().date() - timedelta(days=CURRENT_WINDOW_DAYS)
    
    current_query = """
        SELECT comfort_score
        FROM profile_scores
        WHERE feature_date >= %s
          AND profile_name = %s
    """
    
    current_data = execute_query(current_query, (current_start, profile_name))
    
    if not current_data:
        print("⚠️  No current predictions available")
        return {'drift_detected': False, 'reason': 'no_current_predictions'}
    
    current = [row['comfort_score'] for row in current_data]
    
    # Compare distributions
    print(f"\nBaseline predictions: {len(baseline)} samples")
    print(f"  Mean: {np.mean(baseline):.2f}")
    print(f"  Std:  {np.std(baseline):.2f}")
    
    print(f"\nCurrent predictions: {len(current)} samples")
    print(f"  Mean: {np.mean(current):.2f}")
    print(f"  Std:  {np.std(current):.2f}")
    
    # K-S test
    ks_statistic, p_value = stats.ks_2samp(baseline, current)
    
    print(f"\n📊 Distribution Comparison:")
    print(f"   KS Statistic: {ks_statistic:.4f}")
    print(f"   P-value: {p_value:.4f}")
    
    drift_detected = ks_statistic > DRIFT_THRESHOLDS['ks_statistic']
    
    if drift_detected:
        print(f"\n⚠️  PREDICTION DRIFT DETECTED!")
        print(f"   Scores shifted significantly")
        print(f"\n💡 Possible causes:")
        print(f"   • Seasonal weather changes")
        print(f"   • Model degradation")
        print(f"   • User preference shifts")
    else:
        print(f"\n✅ Predictions stable")
    
    return {
        'drift_detected': drift_detected,
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'baseline_mean': np.mean(baseline),
        'current_mean': np.mean(current)
    }


# ============================================================================
# Main Drift Report
# ============================================================================

def generate_drift_report():
    """
    Generate comprehensive drift report across all dimensions.
    
    Returns:
        dict summarizing all drift checks
    """
    print("=" * 70)
    print("DRIFT DETECTION REPORT")
    print(f"Generated: {datetime.now()}")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'feature_drift': {},
        'model_drift': {},
        'prediction_drift': {},
        'overall_drift_detected': False,
        'recommendations': []
    }
    
    # Check feature drift for each weather feature
    for feature in ['temp_mean_3d', 'precip_sum_3d', 'wind_max_3d']:
        try:
            drift_result = detect_feature_drift(feature)
            results['feature_drift'][feature] = drift_result
            
            if drift_result['drift_detected']:
                results['overall_drift_detected'] = True
                results['recommendations'].append(
                    f"Feature drift detected in {feature} - consider retraining models"
                )
        except Exception as e:
            print(f"❌ Error checking {feature}: {e}")
            results['feature_drift'][feature] = {'error': str(e)}
    
    # Check model performance drift
    for model_type, metric in [('regression', 'test_r2'), ('kmeans', 'silhouette_score')]:
        try:
            perf_result = detect_model_performance_drift(model_type, metric)
            results['model_drift'][model_type] = perf_result
            
            if perf_result['drift_detected']:
                results['overall_drift_detected'] = True
                results['recommendations'].append(
                    f"{model_type.capitalize()} model performance degraded - trigger retraining"
                )
        except Exception as e:
            print(f"❌ Error checking {model_type}: {e}")
            results['model_drift'][model_type] = {'error': str(e)}
    
    # Check prediction drift for leisure profile (main profile)
    try:
        pred_result = detect_prediction_drift('leisure')
        results['prediction_drift']['leisure'] = pred_result
        
        if pred_result['drift_detected']:
            results['overall_drift_detected'] = True
            results['recommendations'].append(
                "Prediction distributions shifted - investigate seasonal effects"
            )
    except Exception as e:
        print(f"❌ Error checking predictions: {e}")
        results['prediction_drift']['leisure'] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*70}")
    print("DRIFT SUMMARY")
    print(f"{'='*70}")
    
    if results['overall_drift_detected']:
        print("\n⚠️  DRIFT DETECTED - ACTION REQUIRED")
        print("\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✅ NO SIGNIFICANT DRIFT DETECTED")
        print("   All systems operating within normal parameters")
    
    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Generate full drift report
    report = generate_drift_report()
    
    # Save to file
    report_path = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Exit with error code if drift detected (for CI/CD)
    if report['overall_drift_detected']:
        print("\n⚠️  Exiting with code 1 (drift detected)")
        sys.exit(1)
    else:
        print("\n✅ Exiting with code 0 (no drift)")
        sys.exit(0)
