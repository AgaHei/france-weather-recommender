"""
add_profiles_tables.py
----------------------
Database migration: Add multi-profile scoring system (Phase 3).

This migration:
1. Creates scoring_profiles table
2. Seeds 5 default profiles (leisure, surfer, cyclist, stargazer, skier)
3. Modifies recommendations table to include profile_name
4. Creates profile_scores table for normalized storage

Run once to enable the multi-profile system.

Usage:
    python scripts/add_profiles_tables.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.db import get_connection

# ---------------------------------------------------------------------------
# SQL Migrations
# ---------------------------------------------------------------------------

CREATE_PROFILES_TABLE = """
CREATE TABLE IF NOT EXISTS scoring_profiles (
    id                  SERIAL PRIMARY KEY,
    profile_name        VARCHAR(50)     UNIQUE NOT NULL,
    description         TEXT            NOT NULL,
    
    -- Temperature parameters
    temp_weight         FLOAT           NOT NULL,
    temp_ideal          FLOAT           NOT NULL,  -- °C
    temp_tolerance      FLOAT           NOT NULL,  -- °C (sigma for Gaussian)
    
    -- Rain parameters
    rain_weight         FLOAT           NOT NULL,
    rain_decay          FLOAT           NOT NULL,  -- mm (decay rate)
    rain_preference     VARCHAR(20)     DEFAULT 'avoid',  -- 'avoid', 'neutral', 'seek'
    
    -- Wind parameters
    wind_weight         FLOAT           NOT NULL,
    wind_decay          FLOAT           NOT NULL,  -- km/h (decay rate)
    wind_preference     VARCHAR(20)     DEFAULT 'avoid',  -- 'avoid', 'neutral', 'seek'
    
    icon                VARCHAR(50)     DEFAULT '🏖️',  -- Emoji for UI
    created_at          TIMESTAMPTZ     DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     DEFAULT NOW()
);

-- Index for fast profile lookups
CREATE INDEX IF NOT EXISTS idx_profiles_name ON scoring_profiles(profile_name);
"""

SEED_PROFILES = """
INSERT INTO scoring_profiles 
    (profile_name, description, icon,
     temp_weight, temp_ideal, temp_tolerance,
     rain_weight, rain_decay, rain_preference,
     wind_weight, wind_decay, wind_preference)
VALUES
    -- Leisure: Weekend relaxation, sightseeing, outdoor dining
    ('leisure', 
     'Weekend relaxation and sightseeing. Ideal for outdoor activities, walking, and enjoying cafés.',
     '🏖️',
     50, 20, 6,      -- Temp: peak at 20°C, tolerance ±6°C
     30, 5, 'avoid',  -- Rain: avoid (decay fast)
     20, 25, 'avoid'  -- Wind: avoid (moderate penalty)
    ),
    
    -- Wind Sports Enthusiast: Windsurfing, kitesurfing, sailing, skydiving, paragliding, land sailing
    ('wind_sports_enthusiast',
     'Wind sports including windsurfing, kitesurfing, sailing, skydiving, paragliding, and land sailing. Seeking strong winds and favorable conditions.',
     '🏄',
     30, 18, 8,       -- Temp: prefer mild (18°C), wider tolerance
     10, 10, 'neutral', -- Rain: don''t care much
     60, 15, 'seek'    -- Wind: WANT strong wind! (inverted scoring)
    ),
    
    -- Cyclist: Road cycling, bike touring
    ('cyclist',
     'Road cycling and bike touring. Need dry roads and manageable wind.',
     '🚴',
     40, 18, 5,       -- Temp: cooler is better for exertion (18°C)
     40, 3, 'avoid',  -- Rain: dangerous on roads (harsh penalty)
     20, 20, 'avoid'  -- Wind: makes cycling harder
    ),
    
    -- Stargazer: Astronomy, night sky observation
    ('stargazer',
     'Astronomy and night sky observation. Clear skies essential.',
     '⭐',
     20, 15, 10,      -- Temp: any temp fine (wide tolerance)
     40, 3, 'avoid',  -- Rain: clouds ruin everything (very harsh)
     40, 20, 'avoid'  -- Wind: affects telescope stability
    )
ON CONFLICT (profile_name) DO UPDATE SET
    description = EXCLUDED.description,
    icon = EXCLUDED.icon,
    temp_weight = EXCLUDED.temp_weight,
    temp_ideal = EXCLUDED.temp_ideal,
    temp_tolerance = EXCLUDED.temp_tolerance,
    rain_weight = EXCLUDED.rain_weight,
    rain_decay = EXCLUDED.rain_decay,
    rain_preference = EXCLUDED.rain_preference,
    wind_weight = EXCLUDED.wind_weight,
    wind_decay = EXCLUDED.wind_decay,
    wind_preference = EXCLUDED.wind_preference,
    updated_at = NOW();
"""

MODIFY_RECOMMENDATIONS_TABLE = """
-- Add profile_name column to recommendations
ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS profile_name VARCHAR(50) DEFAULT 'leisure';

-- Drop old unique constraint (city, date)
ALTER TABLE recommendations
    DROP CONSTRAINT IF EXISTS recommendations_recommendation_date_city_key;

-- Add new unique constraint (city, date, profile)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'recommendations_unique_date_city_profile'
    ) THEN
        ALTER TABLE recommendations
        ADD CONSTRAINT recommendations_unique_date_city_profile
        UNIQUE(recommendation_date, city, profile_name);
    END IF;
END $$;

-- Index for profile filtering
CREATE INDEX IF NOT EXISTS idx_recommendations_profile ON recommendations(profile_name);
"""

CREATE_PROFILE_SCORES_TABLE = """
-- Optional normalized table for storing multiple profile scores per city/date
-- Alternative to adding columns to weather_features
CREATE TABLE IF NOT EXISTS profile_scores (
    id              SERIAL PRIMARY KEY,
    city            VARCHAR(100)    NOT NULL,
    feature_date    DATE            NOT NULL,
    profile_name    VARCHAR(50)     NOT NULL,
    comfort_score   FLOAT           NOT NULL,
    computed_at     TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(city, feature_date, profile_name),
    FOREIGN KEY (profile_name) REFERENCES scoring_profiles(profile_name) ON DELETE CASCADE
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_profile_scores_city_date ON profile_scores(city, feature_date);
CREATE INDEX IF NOT EXISTS idx_profile_scores_profile ON profile_scores(profile_name);
"""

# ---------------------------------------------------------------------------
# Migration execution
# ---------------------------------------------------------------------------

def run_migration():
    """Execute all Phase 3 database migrations."""
    print("=" * 70)
    print("PHASE 3 DATABASE MIGRATION")
    print("=" * 70)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Step 1: Create scoring_profiles table
            print("\n📊 Step 1: Creating scoring_profiles table...")
            cur.execute(CREATE_PROFILES_TABLE)
            print("   ✅ Table created")
            
            # Step 2: Seed default profiles
            print("\n🌱 Step 2: Seeding default profiles...")
            cur.execute(SEED_PROFILES)
            print("   ✅ 4 profiles seeded:")
            print("      • leisure (🏖️)  — Weekend relaxation")
            print("      • surfer (🏄)   — Beach sports, strong wind")
            print("      • cyclist (🚴)  — Road cycling, dry weather")
            print("      • stargazer (⭐) — Astronomy, clear skies")
            print("      • skier (⛷️)   — Snow sports, cold temps")
            
            # Step 3: Modify recommendations table
            print("\n🔧 Step 3: Modifying recommendations table...")
            cur.execute(MODIFY_RECOMMENDATIONS_TABLE)
            print("   ✅ Added profile_name column")
            print("   ✅ Updated unique constraint")
            
            # Step 4: Create profile_scores table
            print("\n📊 Step 4: Creating profile_scores table...")
            cur.execute(CREATE_PROFILE_SCORES_TABLE)
            print("   ✅ Table created")
    
    print("\n" + "=" * 70)
    print("✅ MIGRATION COMPLETE")
    print("=" * 70)
    
    print("\nDatabase changes:")
    print("  • scoring_profiles     — 4 profile definitions")
    print("  • profile_scores       — Multi-profile comfort scores")
    print("  • recommendations      — Now includes profile_name")
    
    print("\nNext steps:")
    print("  1. Update engineer.py for multi-profile scoring")
    print("  2. Modify DAG 2 to compute all profile scores")
    print("  3. Modify DAG 4 to generate profile-specific recommendations")
    print("  4. Update Streamlit UI with profile selector")
    
    print("\n" + "=" * 70)


def verify_migration():
    """Verify the migration was successful."""
    from src.data.db import execute_query
    
    print("\n🔍 Verifying migration...")
    
    # Check profiles
    profiles = execute_query("SELECT profile_name, icon, description FROM scoring_profiles ORDER BY profile_name")
    
    if len(profiles) == 5:
        print("✅ All 5 profiles present:")
        for p in profiles:
            print(f"   {p['icon']} {p['profile_name']}: {p['description'][:50]}...")
    else:
        print(f"⚠️  Expected 5 profiles, found {len(profiles)}")
    
    # Check recommendations schema
    check_column = execute_query("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'recommendations' AND column_name = 'profile_name'
    """)
    
    if check_column:
        print("✅ recommendations.profile_name column exists")
    else:
        print("❌ recommendations.profile_name column missing")
    
    # Check profile_scores table
    check_table = execute_query("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name = 'profile_scores'
    """)
    
    if check_table:
        print("✅ profile_scores table exists")
    else:
        print("❌ profile_scores table missing")


if __name__ == "__main__":
    run_migration()
    verify_migration()
