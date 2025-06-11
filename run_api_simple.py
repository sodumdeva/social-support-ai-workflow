#!/usr/bin/env python3
"""
Simple API Server Launcher for Social Support AI Workflow
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the app
if __name__ == "__main__":
    import uvicorn
    from src.api.main import app
    
    print("🚀 Starting Social Support AI API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        reload=False  # Disable reload to avoid subprocess issues
    ) 