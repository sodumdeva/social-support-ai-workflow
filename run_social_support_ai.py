#!/usr/bin/env python3
"""
Main Launcher for Social Support AI Workflow System

This script provides options to run:
1. API Server only
2. Frontend only  
3. Both API and Frontend
4. Setup and training

Aligns with the government social security department requirements:
- Automated application processing (5-20 days → 2-5 minutes)
- AI agent orchestration with specialized agents
- Multimodal document processing (text, images, tabular data)
- Interactive chatbot interface
- Local ML/LLM models
- Real-time decision making
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import and setup logging
from src.utils.logging_config import setup_logging, get_logger
from config import settings

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data/uploads",
        "data/processed", 
        "data/raw",
        "logs",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def run_api_server():
    """Run the API server with enhanced logging"""
    
    logger = setup_logging(log_level="DEBUG", log_to_file=True, log_to_console=True)
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Social Support AI API Server")
    logger.info("=" * 60)
    logger.info(f"API Host: {settings.api_host}")
    logger.info(f"API Port: {settings.api_port}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("=" * 60)
    
    setup_directories()
    
    # Import and run the API
    try:
        import uvicorn
        from src.api.main import app
        
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

def run_frontend():
    """Run the frontend with enhanced logging"""
    
    logger = setup_logging(log_level="DEBUG", log_to_file=True, log_to_console=True)
    
    logger.info("=" * 60)
    logger.info("🎨 Starting Social Support AI Frontend")
    logger.info("=" * 60)
    logger.info(f"Frontend Host: {settings.frontend_host}")
    logger.info(f"Frontend Port: {settings.frontend_port}")
    logger.info(f"API Base URL: http://{settings.api_host}:{settings.api_port}")
    logger.info("=" * 60)
    
    setup_directories()
    
    # Check if frontend file exists
    frontend_file = "src/frontend/main.py"
    if not os.path.exists(frontend_file):
        logger.error(f"Frontend file not found: {frontend_file}")
        sys.exit(1)
    
    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        frontend_file,
        "--server.port", str(settings.frontend_port),
        "--server.address", settings.frontend_host
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start frontend: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Frontend stopped by user")

def run_both():
    """Run both API and Frontend in separate processes"""
    
    print("🚀 Starting Social Support AI - Full System")
    print("=" * 60)
    
    setup_directories()
    
    # Start API in background
    print("Starting API server...")
    api_process = subprocess.Popen([
        sys.executable, __file__, "--api"
    ])
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Start Frontend
    print("Starting Frontend...")
    try:
        frontend_process = subprocess.Popen([
            sys.executable, __file__, "--frontend"
        ])
        
        # Wait for both processes
        api_process.wait()
        frontend_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api_process.terminate()
        frontend_process.terminate()

def run_setup():
    """Run initial setup and model training"""
    
    logger = setup_logging(log_level="INFO", log_to_file=True, log_to_console=True)
    
    logger.info("🔧 Running Social Support AI Setup")
    logger.info("=" * 40)
    
    setup_directories()
    
    # Run setup scripts
    scripts = [
        "scripts/setup_database.py",
        "scripts/train_ml_models.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            logger.info(f"Running {script}...")
            try:
                subprocess.run([sys.executable, script], check=True)
                logger.info(f"✅ {script} completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ {script} failed: {e}")
        else:
            logger.warning(f"⚠️ Script not found: {script}")
    
    logger.info("🎉 Setup completed!")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Social Support AI Workflow System - Government Social Security Department",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_social_support_ai.py --api          # Run API server only
  python run_social_support_ai.py --frontend     # Run frontend only
  python run_social_support_ai.py --both         # Run both API and frontend
  python run_social_support_ai.py --setup        # Run initial setup
        """
    )
    
    parser.add_argument("--api", action="store_true", help="Run API server only")
    parser.add_argument("--frontend", action="store_true", help="Run frontend only")
    parser.add_argument("--both", action="store_true", help="Run both API and frontend")
    parser.add_argument("--setup", action="store_true", help="Run initial setup and training")
    
    args = parser.parse_args()
    
    if args.api:
        run_api_server()
    elif args.frontend:
        run_frontend()
    elif args.both:
        run_both()
    elif args.setup:
        run_setup()
    else:
        # Default: show help and run both
        parser.print_help()
        print("\n" + "=" * 60)
        print("🚀 No specific option selected, starting full system...")
        print("=" * 60)
        run_both()

if __name__ == "__main__":
    main() 