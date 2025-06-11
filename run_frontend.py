#!/usr/bin/env python3
"""
Frontend Launcher for Social Support AI Workflow
"""
import sys
import os
import subprocess

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🎨 Starting Social Support AI Frontend...")
    print("📍 Frontend will be available at: http://localhost:8501")
    print("🔗 Make sure API is running at: http://localhost:8000")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/frontend/main.py", 
        "--server.port", "8501",
        "--server.address", "localhost"
    ]) 