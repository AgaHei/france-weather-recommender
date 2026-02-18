"""
db.py
-----
Neon PostgreSQL connection helpers.
Uses psycopg2 with a connection string from environment variables.

Environment variables expected (in .env):
    NEON_DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
"""

import os
import contextlib
import psycopg2
import psycopg2.extras
from psycopg2 import pool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection_string() -> str:
    url = os.getenv("NEON_DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "NEON_DATABASE_URL environment variable is not set. "
            "Add it to your .env file."
        )
    return url


@contextlib.contextmanager
def get_connection():
    """Context manager that yields a psycopg2 connection and auto-commits/rolls back."""
    conn = psycopg2.connect(get_connection_string())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextlib.contextmanager
def get_cursor(cursor_factory=psycopg2.extras.RealDictCursor):
    """Context manager that yields a cursor (returns rows as dicts by default)."""
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Raw weather data (one row per city per day, as fetched from Open-Meteo)
CREATE TABLE IF NOT EXISTS raw_weather (
    id              SERIAL PRIMARY KEY,
    city            VARCHAR(100)    NOT NULL,
    date            DATE            NOT NULL,
    temp_max        FLOAT,          -- °C
    temp_min        FLOAT,          -- °C
    temp_mean       FLOAT,          -- °C (we'll compute this)
    precipitation   FLOAT,          -- mm
    wind_speed_max  FLOAT,          -- km/h
    wind_gusts_max  FLOAT,          -- km/h
    weather_code    INTEGER,        -- WMO weather code
    fetched_at      TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(city, date)              -- no duplicates per city/day
);

-- Engineered features (rolling windows, used for ML)
CREATE TABLE IF NOT EXISTS weather_features (
    id                      SERIAL PRIMARY KEY,
    city                    VARCHAR(100)    NOT NULL,
    feature_date            DATE            NOT NULL,  -- the "as of" date
    temp_mean_7d            FLOAT,          -- rolling 7-day avg temp
    temp_mean_3d            FLOAT,          -- rolling 3-day avg temp (forecast window)
    precip_sum_7d           FLOAT,          -- total rain last 7 days
    precip_sum_3d           FLOAT,
    wind_max_7d             FLOAT,          -- max wind last 7 days
    wind_max_3d             FLOAT,
    comfort_score           FLOAT,          -- computed label (0-100)
    computed_at             TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(city, feature_date)
);

-- City metadata (static, loaded once)
CREATE TABLE IF NOT EXISTS cities (
    id          SERIAL PRIMARY KEY,
    city        VARCHAR(100)    NOT NULL UNIQUE,
    latitude    FLOAT           NOT NULL,
    longitude   FLOAT           NOT NULL
);

-- Daily recommendations output (written by the scoring DAG)
CREATE TABLE IF NOT EXISTS recommendations (
    id                  SERIAL PRIMARY KEY,
    recommendation_date DATE            NOT NULL,
    city                VARCHAR(100)    NOT NULL,
    cluster_id          INTEGER,        -- from K-Means
    comfort_score_pred  FLOAT,          -- from regression model
    rank                INTEGER,        -- 1 = best destination
    created_at          TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(recommendation_date, city)
);

-- Model run log (lightweight alternative to full MLflow for DAG logging)
CREATE TABLE IF NOT EXISTS model_runs (
    id              SERIAL PRIMARY KEY,
    run_date        DATE            NOT NULL,
    model_type      VARCHAR(50)     NOT NULL,   -- 'kmeans' or 'regression'
    metric_name     VARCHAR(100),
    metric_value    FLOAT,
    artifact_path   VARCHAR(500),               -- path to saved model file
    is_champion     BOOLEAN         DEFAULT FALSE,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
"""


def init_schema():
    """Create all tables if they don't exist. Safe to run multiple times."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
    print("Schema initialized successfully.")


def seed_cities():
    """Insert city coordinates into the cities table (idempotent)."""
    from src.data.cities import CITY_COORDINATES

    rows = [
        (city, coords["latitude"], coords["longitude"])
        for city, coords in CITY_COORDINATES.items()
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO cities (city, latitude, longitude)
                VALUES %s
                ON CONFLICT (city) DO UPDATE
                    SET latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude
                """,
                rows,
            )
    print(f"Seeded {len(rows)} cities into the cities table.")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def execute_query(sql: str, params=None) -> list[dict]:
    """Run a SELECT query and return results as a list of dicts."""
    with get_cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def execute_write(sql: str, params=None):
    """Run an INSERT/UPDATE/DELETE."""
    with get_cursor(cursor_factory=None) as cur:
        cur.execute(sql, params)


def bulk_insert(table: str, rows: list[dict], conflict_action: str = "DO NOTHING"):
    """
    Bulk insert a list of dicts into a table.

    Args:
        table: table name
        rows: list of dicts (all dicts must have the same keys)
        conflict_action: SQL ON CONFLICT clause action, e.g. "DO NOTHING"
    """
    if not rows:
        return

    columns = list(rows[0].keys())
    values = [[row[col] for col in columns] for row in rows]

    sql = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES %s
        ON CONFLICT {conflict_action}
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, sql, values)


if __name__ == "__main__":
    # Quick connectivity test
    print("Testing Neon connection...")
    with get_cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"Connected! PostgreSQL version: {version['version']}")

    init_schema()
    seed_cities()
