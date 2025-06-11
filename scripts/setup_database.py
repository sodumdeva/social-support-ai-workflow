#!/usr/bin/env python3

"""
Database setup script for Social Support AI Workflow

This script creates the necessary database tables and initial setup.
"""
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import create_tables, engine
from config import settings
from loguru import logger
from sqlalchemy.sql import text

def setup_database():
    """Initialize the database with all necessary tables"""
    try:
        logger.info("Setting up database...")
        logger.info(f"Database URL: {settings.database_url}")
        
        # Create all tables
        create_tables()
        logger.success("Database tables created successfully!")
        
        # Verify connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.success("Database connection verified!")
        
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def reset_database():
    """Reset database by dropping and recreating all tables"""
    try:
        from src.models.database import Base
        
        logger.warning("Resetting database - dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        
        logger.info("Recreating tables...")
        create_tables()
        
        logger.success("Database reset completed!")
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup database for Social Support AI Workflow")
    parser.add_argument("--reset", action="store_true", help="Reset database (drop and recreate tables)")
    
    args = parser.parse_args()
    
    if args.reset:
        success = reset_database()
    else:
        success = setup_database()
    
    if success:
        logger.info("Database setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("Database setup failed!")
        sys.exit(1) 