#!/usr/bin/env python3
"""
Simple system startup script for Social Support AI Workflow
Starts only the core components needed for the main conversation workflow.
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def start_api():
    """Start the FastAPI backend"""
    print("🚀 Starting Social Support AI Workflow API...")
    
    # Change to project directory
    os.chdir(project_root)
    
    # Start the API server
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])
    
    return api_process

def start_frontend():
    """Start the React frontend"""
    print("🎨 Starting React frontend...")
    
    frontend_dir = project_root / "src" / "frontend"
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found. Skipping frontend startup.")
        return None
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Start the frontend development server
    frontend_process = subprocess.Popen([
        "npm", "start"
    ])
    
    return frontend_process

def main():
    """Main startup function"""
    print("=" * 60)
    print("🤖 Social Support AI Workflow - Main System Startup")
    print("=" * 60)
    
    processes = []
    
    try:
        # Start API
        api_process = start_api()
        processes.append(("API", api_process))
        time.sleep(3)  # Give API time to start
        
        # Start Frontend
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(("Frontend", frontend_process))
        
        print("\n✅ System startup complete!")
        print("📍 API: http://localhost:8000")
        print("📍 Frontend: http://localhost:3000")
        print("📍 API Docs: http://localhost:8000/docs")
        print("\n🔧 Main workflow components:")
        print("   • LangGraph conversation workflow")
        print("   • ConversationAgent, EligibilityAssessmentAgent, DataExtractionAgent")
        print("   • PostgreSQL database")
        print("   • React frontend")
        print("\n⏹️  Press Ctrl+C to stop all services")
        
        # Wait for processes
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        
        for name, process in processes:
            print(f"   Stopping {name}...")
            process.terminate()
            
        # Wait for graceful shutdown
        time.sleep(2)
        
        # Force kill if needed
        for name, process in processes:
            if process.poll() is None:
                print(f"   Force killing {name}...")
                process.kill()
        
        print("✅ System shutdown complete!")

if __name__ == "__main__":
    main() 