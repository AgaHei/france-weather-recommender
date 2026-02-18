"""
run_pipeline_in_order.py
-------------------------
Automated script to run the complete ML pipeline in correct dependency order.

Executes DAGs via Airflow API in the proper sequence:
1. fetch_weather (3-5 minutes)
2. engineer_features (1-2 minutes)  
3. retrain_models (10-15 minutes)
4. generate_recommendations (2-3 minutes)
5. fetch_hotels (5-10 minutes)

Total estimated time: 25-35 minutes
"""

import requests
import time
import sys
from datetime import datetime


# Airflow connection settings
AIRFLOW_URL = "http://localhost:8080"
AUTH = ("admin", "admin")  # Default Airflow credentials


def trigger_dag(dag_id: str) -> bool:
    """
    Trigger a DAG via Airflow REST API.
    
    Args:
        dag_id: Name of the DAG to trigger
        
    Returns:
        True if successful, False otherwise
    """
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    data = {
        "dag_run_id": f"manual_run_{int(time.time())}",
        "execution_date": datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, auth=AUTH)
        response.raise_for_status()
        
        print(f"✅ Successfully triggered {dag_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to trigger {dag_id}: {e}")
        return False


def wait_for_dag_completion(dag_id: str, timeout_minutes: int = 20) -> bool:
    """
    Wait for DAG to complete execution.
    
    Args:
        dag_id: Name of the DAG to monitor
        timeout_minutes: Maximum wait time in minutes
        
    Returns:
        True if completed successfully, False if failed or timeout
    """
    print(f"⏳ Waiting for {dag_id} to complete (timeout: {timeout_minutes} min)...")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while time.time() - start_time < timeout_seconds:
        try:
            # Get latest DAG run
            url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
            response = requests.get(url, auth=AUTH, params={"limit": 1, "order_by": "-execution_date"})
            response.raise_for_status()
            
            dag_runs = response.json().get("dag_runs", [])
            if not dag_runs:
                print(f"   No DAG runs found for {dag_id}")
                time.sleep(10)
                continue
            
            latest_run = dag_runs[0]
            state = latest_run.get("state")
            
            if state == "success":
                print(f"✅ {dag_id} completed successfully!")
                return True
            elif state == "failed":
                print(f"❌ {dag_id} failed!")
                return False
            else:
                print(f"   Status: {state} (elapsed: {int(time.time() - start_time)}s)")
                time.sleep(15)
                
        except Exception as e:
            print(f"   Error checking status: {e}")
            time.sleep(10)
    
    print(f"⏰ Timeout waiting for {dag_id} ({timeout_minutes} minutes)")
    return False


def run_pipeline():
    """Run the complete ML pipeline in correct order."""
    print("🚀" * 35)
    print("AUTOMATED ML PIPELINE EXECUTION")
    print("🚀" * 35)
    
    # Pipeline configuration
    pipeline_steps = [
        ("fetch_weather", 8),      # 8 minutes timeout
        ("compute_features", 5),   # 5 minutes timeout
        ("retrain_models", 20),     # 20 minutes timeout (ML training)
        ("generate_recommendations", 8),  # 8 minutes timeout
        ("fetch_hotels", 15)        # 15 minutes timeout (API calls)
    ]
    
    start_time = time.time()
    
    for i, (dag_id, timeout_min) in enumerate(pipeline_steps, 1):
        print(f"\n{'='*70}")
        print(f"STEP {i}/{len(pipeline_steps)}: {dag_id.upper()}")
        print(f"{'='*70}")
        
        # Trigger DAG
        if not trigger_dag(dag_id):
            print(f"\n❌ PIPELINE FAILED: Could not trigger {dag_id}")
            return False
        
        # Wait for completion
        if not wait_for_dag_completion(dag_id, timeout_min):
            print(f"\n❌ PIPELINE FAILED: {dag_id} did not complete successfully")
            return False
    
    # Pipeline completed successfully
    total_time = int(time.time() - start_time)
    print(f"\n{'🎉'*70}")
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total execution time: {total_time // 60}m {total_time % 60}s")
    print(f"{'🎉'*70}")
    
    print(f"\n📊 Pipeline Results:")
    print(f"• Weather data: Fetched for 20 French cities")
    print(f"• ML models: Trained and logged to MLflow")
    print(f"• Recommendations: Generated for all cities")
    print(f"• Hotels: Fetched for top 3 recommended cities")
    
    print(f"\n🔗 Access Your Results:")
    print(f"• Airflow UI: http://localhost:8080")
    print(f"• MLflow Tracking: http://localhost:5001")
    print(f"• Database: Check your Neon PostgreSQL")
    
    return True


def check_airflow_connection():
    """Check if Airflow is accessible."""
    try:
        response = requests.get(f"{AIRFLOW_URL}/health", auth=AUTH, timeout=10)
        response.raise_for_status()
        print("✅ Airflow connection successful")
        return True
    except Exception as e:
        print(f"❌ Cannot connect to Airflow: {e}")
        print(f"   Make sure Airflow is running at {AIRFLOW_URL}")
        return False


if __name__ == "__main__":
    print("Checking Airflow connection...")
    if not check_airflow_connection():
        sys.exit(1)
    
    print(f"\n⚠️  This will run the complete ML pipeline (~25-35 minutes)")
    print(f"   Press Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print(f"\n❌ Cancelled by user")
        sys.exit(0)
    
    success = run_pipeline()
    sys.exit(0 if success else 1)