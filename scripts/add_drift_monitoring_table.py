"""
add_drift_monitoring_table.py
------------------------------
Database migration: Add drift monitoring table (optional production enhancement).

This table stores historical drift metrics for:
- Compliance auditing
- Trend analysis
- Dynamic threshold tuning
- Alerting history

Run with:
    python scripts/add_drift_monitoring_table.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.db import get_connection

CREATE_DRIFT_TABLE = """
CREATE TABLE IF NOT EXISTS drift_monitoring (
    id                  SERIAL PRIMARY KEY,
    check_timestamp     TIMESTAMPTZ     DEFAULT NOW(),
    check_type          VARCHAR(50)     NOT NULL,  -- 'feature', 'model', 'prediction'
    metric_name         VARCHAR(100)    NOT NULL,  -- 'temp_mean_3d', 'regression_r2', etc.
    metric_value        FLOAT,                     -- Current metric value
    baseline_value      FLOAT,                     -- Baseline for comparison
    drift_score         FLOAT,                     -- KS statistic or % change
    drift_detected      BOOLEAN         DEFAULT FALSE,
    threshold_used      FLOAT,                     -- Threshold that was applied
    drift_severity      VARCHAR(20),               -- 'none', 'minor', 'major', 'critical'
    metadata            JSONB,                     -- Additional context (p-values, distributions, etc.)
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_monitoring(check_timestamp);
CREATE INDEX IF NOT EXISTS idx_drift_type ON drift_monitoring(check_type);
CREATE INDEX IF NOT EXISTS idx_drift_detected ON drift_monitoring(drift_detected);
CREATE INDEX IF NOT EXISTS idx_drift_metric ON drift_monitoring(metric_name);

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_drift_time_series 
    ON drift_monitoring(metric_name, check_timestamp);
"""

def add_drift_monitoring_table():
    """Add drift monitoring table to database."""
    print("=" * 70)
    print("DATABASE MIGRATION: Adding drift_monitoring table")
    print("=" * 70)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_DRIFT_TABLE)
    
    print("\n✅ drift_monitoring table created successfully!")
    print("\nTable structure:")
    print("  - check_timestamp: When drift check ran")
    print("  - check_type: 'feature', 'model', or 'prediction'")
    print("  - metric_name: Which metric was checked")
    print("  - metric_value: Current value")
    print("  - baseline_value: Comparison baseline")
    print("  - drift_score: KS statistic or % change")
    print("  - drift_detected: Boolean flag")
    print("  - metadata: JSON with full details (p-values, etc.)")
    
    print("\n📊 Usage examples:")
    print("\n  # Track feature drift over time")
    print("  SELECT check_timestamp, metric_name, drift_score, drift_detected")
    print("  FROM drift_monitoring")
    print("  WHERE check_type = 'feature'")
    print("  ORDER BY check_timestamp DESC;")
    
    print("\n  # Count drift alerts by metric")
    print("  SELECT metric_name, COUNT(*) as drift_count")
    print("  FROM drift_monitoring")
    print("  WHERE drift_detected = TRUE")
    print("  GROUP BY metric_name;")
    
    print("\n  # Trend analysis for temperature drift")
    print("  SELECT check_timestamp, drift_score")
    print("  FROM drift_monitoring")
    print("  WHERE metric_name = 'temp_mean_3d'")
    print("  ORDER BY check_timestamp;")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    add_drift_monitoring_table()
