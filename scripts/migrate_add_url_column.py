"""
migrate_add_url_column.py
--------------------------
Database migration: Add website_url column to existing hotels table.

Run this once to add the website_url column to your existing hotels table in Neon.

Usage:
    python scripts/migrate_add_url_column.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.db import get_connection

ADD_URL_COLUMN_SQL = """
-- Add website_url column if it doesn't exist
DO $$ 
BEGIN 
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'hotels' 
        AND column_name = 'website_url'
    ) THEN
        ALTER TABLE hotels ADD COLUMN website_url TEXT;
        PRINT 'Added website_url column to hotels table';
    ELSE
        PRINT 'website_url column already exists';
    END IF;
END $$;
"""


def add_url_column():
    """Add website_url column to existing hotels table."""
    print("=" * 70)
    print("DATABASE MIGRATION: Adding website_url column to hotels table")
    print("=" * 70)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check if column exists first
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'hotels' 
                AND column_name = 'website_url'
            """)
            
            if cur.fetchone():
                print("\n✅ website_url column already exists!")
            else:
                print("\n📝 Adding website_url column...")
                cur.execute("ALTER TABLE hotels ADD COLUMN website_url TEXT;")
                print("✅ website_url column added successfully!")
            
            # Verify the column was added
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'hotels' 
                ORDER BY ordinal_position
            """)
            
            columns = cur.fetchall()
            
    print("\n📋 Current hotels table structure:")
    for col_name, data_type, nullable in columns:
        nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
        print(f"  - {col_name:<15} {data_type:<12} {nullable_str}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    add_url_column()