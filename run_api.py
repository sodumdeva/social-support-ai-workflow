#!/usr/bin/env python3
"""
API Server Launcher for Social Support AI Workflow
"""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the FastAPI app
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    print("🚀 Starting Social Support AI API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("✨ Health Check: http://localhost:8000/health")
    print("\n" + "="*50)
    
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        reload=True,
        reload_dirs=[project_root]
    ) 