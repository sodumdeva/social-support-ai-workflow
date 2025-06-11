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
            result = connection.execute("SELECT 1")
            logger.success("Database connection verified!")
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

def main():
    """Main setup function"""
    logger.info("Starting Social Support AI Database Setup")
    setup_database()
    logger.success("Database setup completed successfully!")

if __name__ == "__main__":
    main() 