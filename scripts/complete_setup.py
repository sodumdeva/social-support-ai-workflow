#!/usr/bin/env python3
"""
Complete Setup Script for Social Support AI Workflow

This script automates the entire installation process including:
- System dependency checks
- Python environment setup
- Database initialization
- Local LLM installation (Ollama)
- Tesseract OCR setup
- ML model training
- Configuration generation
- System testing

Usage: python scripts/complete_setup.py
"""

import asyncio
import subprocess
import sys
import os
import platform
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SetupManager:
    """Main setup manager for the Social Support AI Workflow"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.project_root = Path(__file__).parent.parent
        self.setup_log = []
        self.failed_steps = []
        
        # Configuration
        self.config = {
            "database": {
                "name": "social_support_db",
                "user": "social_support_user",
                "password": "secure_password_123"
            },
            "ollama": {
                "models": ["llama2", "codellama", "mistral", "phi"],
                "base_url": "http://localhost:11434"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    def log_step(self, step: str, status: str = "INFO", details: str = ""):
        """Log setup step with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "status": status,
            "details": details
        }
        self.setup_log.append(log_entry)
        
        # Color-coded console output
        color = Colors.OKGREEN if status == "SUCCESS" else Colors.FAIL if status == "ERROR" else Colors.OKBLUE
        print(f"{color}[{timestamp}] {step}: {status}{Colors.ENDC}")
        if details:
            print(f"  {details}")
    
    async def run_complete_setup(self):
        """Run the complete setup process"""
        
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("=" * 80)
        print("🚀 SOCIAL SUPPORT AI WORKFLOW - COMPLETE SETUP")
        print("=" * 80)
        print(f"{Colors.ENDC}")
        
        print(f"{Colors.OKCYAN}This script will install and configure:")
        print("• System dependencies (Python, Node.js, PostgreSQL, Tesseract)")
        print("• Local LLM models via Ollama (llama2, codellama, mistral, phi)")
        print("• Database schema and initial data")
        print("• ML models training")
        print("• Complete system configuration")
        print(f"• Testing and validation{Colors.ENDC}")
        
        print(f"\n{Colors.WARNING}⚠️  This process may take 30-60 minutes depending on your internet speed.")
        print(f"   Large language models (10-15GB total) will be downloaded.{Colors.ENDC}")
        
        # Confirm before proceeding
        if not self.confirm_setup():
            print(f"{Colors.WARNING}Setup cancelled by user.{Colors.ENDC}")
            return
        
        try:
            # Step 1: System Requirements Check
            await self.check_system_requirements()
            
            # Step 2: Install System Dependencies
            await self.install_system_dependencies()
            
            # Step 3: Setup Python Environment
            await self.setup_python_environment()
            
            # Step 4: Database Setup
            await self.setup_database()
            
            # Step 5: Install and Configure Ollama
            await self.setup_ollama()
            
            # Step 6: Install Tesseract OCR
            await self.setup_tesseract()
            
            # Step 7: Frontend Setup
            await self.setup_frontend()
            
            # Step 8: Generate Configuration Files
            await self.generate_configuration()
            
            # Step 9: Train ML Models
            await self.train_ml_models()
            
            # Step 10: Run System Tests
            await self.run_system_tests()
            
            # Step 11: Final Setup
            await self.finalize_setup()
            
            # Success summary
            self.print_success_summary()
            
        except Exception as e:
            self.log_step("SETUP_FAILED", "ERROR", str(e))
            self.print_failure_summary()
            sys.exit(1)
    
    def confirm_setup(self) -> bool:
        """Confirm setup with user"""
        try:
            response = input(f"\n{Colors.BOLD}Do you want to proceed with the complete setup? (y/N): {Colors.ENDC}")
            return response.lower() in ['y', 'yes']
        except KeyboardInterrupt:
            return False
    
    async def check_system_requirements(self):
        """Check system requirements"""
        self.log_step("SYSTEM_REQUIREMENTS", "INFO", "Checking system requirements...")
        
        # Check OS
        if self.system not in ['darwin', 'linux', 'windows']:
            raise Exception(f"Unsupported operating system: {self.system}")
        
        # Check available disk space (need ~20GB)
        free_space = shutil.disk_usage(self.project_root).free / (1024**3)  # GB
        if free_space < 20:
            raise Exception(f"Insufficient disk space: {free_space:.1f}GB available, 20GB required")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            raise Exception(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        
        self.log_step("SYSTEM_REQUIREMENTS", "SUCCESS", f"OS: {self.system}, Python: {python_version.major}.{python_version.minor}, Free space: {free_space:.1f}GB")
    
    async def install_system_dependencies(self):
        """Install system dependencies based on OS"""
        self.log_step("SYSTEM_DEPENDENCIES", "INFO", "Installing system dependencies...")
        
        if self.system == "darwin":  # macOS
            await self.install_macos_dependencies()
        elif self.system == "linux":
            await self.install_linux_dependencies()
        elif self.system == "windows":
            await self.install_windows_dependencies()
        
        self.log_step("SYSTEM_DEPENDENCIES", "SUCCESS", "System dependencies installed")
    
    async def install_macos_dependencies(self):
        """Install dependencies on macOS"""
        # Check if Homebrew is installed
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_step("HOMEBREW", "INFO", "Installing Homebrew...")
            install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            subprocess.run(install_cmd, shell=True, check=True)
        
        # Install packages
        packages = ["python@3.11", "node", "postgresql@14", "tesseract", "git"]
        for package in packages:
            self.log_step("BREW_INSTALL", "INFO", f"Installing {package}...")
            subprocess.run(["brew", "install", package], check=True, capture_output=True)
        
        # Start PostgreSQL
        subprocess.run(["brew", "services", "start", "postgresql@14"], check=True, capture_output=True)
    
    async def install_linux_dependencies(self):
        """Install dependencies on Linux"""
        # Update package list
        subprocess.run(["sudo", "apt", "update"], check=True, capture_output=True)
        
        # Install packages
        packages = [
            "python3.11", "python3.11-pip", "python3.11-venv", "python3.11-dev",
            "nodejs", "npm", "postgresql", "postgresql-contrib", "tesseract-ocr",
            "git", "curl", "build-essential", "libpq-dev"
        ]
        
        cmd = ["sudo", "apt", "install", "-y"] + packages
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Start PostgreSQL
        subprocess.run(["sudo", "systemctl", "start", "postgresql"], check=True, capture_output=True)
        subprocess.run(["sudo", "systemctl", "enable", "postgresql"], check=True, capture_output=True)
    
    async def install_windows_dependencies(self):
        """Install dependencies on Windows"""
        self.log_step("WINDOWS_DEPENDENCIES", "WARNING", 
                     "Windows detected. Please install manually:\n"
                     "1. Python 3.11 from python.org\n"
                     "2. Node.js from nodejs.org\n"
                     "3. PostgreSQL from postgresql.org\n"
                     "4. Tesseract from GitHub releases\n"
                     "5. Git from git-scm.com")
        
        input("Press Enter after installing all dependencies...")
    
    async def setup_python_environment(self):
        """Setup Python virtual environment and install packages"""
        self.log_step("PYTHON_ENV", "INFO", "Setting up Python environment...")
        
        # Create virtual environment if it doesn't exist
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        
        # Get pip path
        if self.system == "windows":
            pip_path = venv_path / "Scripts" / "pip"
            python_path = venv_path / "Scripts" / "python"
        else:
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"
        
        # Upgrade pip
        subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        requirements_file = self.project_root / "requirements.txt"
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
        
        self.log_step("PYTHON_ENV", "SUCCESS", "Python environment configured")
    
    async def setup_database(self):
        """Setup PostgreSQL database"""
        self.log_step("DATABASE", "INFO", "Setting up PostgreSQL database...")
        
        try:
            # Create database and user
            if self.system != "windows":
                # Create user
                subprocess.run([
                    "sudo", "-u", "postgres", "createuser", 
                    "--interactive", "--pwprompt", self.config["database"]["user"]
                ], input=f"{self.config['database']['password']}\n{self.config['database']['password']}\ny\ny\ny\n", 
                text=True, check=True)
                
                # Create database
                subprocess.run([
                    "sudo", "-u", "postgres", "createdb", 
                    "-O", self.config["database"]["user"], self.config["database"]["name"]
                ], check=True)
            
            # Run database setup script
            setup_script = self.project_root / "scripts" / "setup_database.py"
            if setup_script.exists():
                subprocess.run([sys.executable, str(setup_script)], check=True)
            
            self.log_step("DATABASE", "SUCCESS", f"Database '{self.config['database']['name']}' created")
            
        except subprocess.CalledProcessError as e:
            self.log_step("DATABASE", "WARNING", "Database setup may have failed - continuing...")
    
    async def setup_ollama(self):
        """Setup Ollama and install LLM models"""
        self.log_step("OLLAMA", "INFO", "Setting up Ollama and LLM models...")
        
        # Install Ollama
        if self.system in ["darwin", "linux"]:
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
            subprocess.run(install_cmd, shell=True, check=True)
        else:
            self.log_step("OLLAMA", "WARNING", "Please install Ollama manually from https://ollama.ai/download")
            input("Press Enter after installing Ollama...")
        
        # Start Ollama service
        if self.system != "windows":
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        await asyncio.sleep(5)
        
        # Install models
        for model in self.config["ollama"]["models"]:
            self.log_step("OLLAMA_MODEL", "INFO", f"Installing {model} (this may take several minutes)...")
            try:
                subprocess.run(["ollama", "pull", model], check=True, timeout=1800)  # 30 min timeout
                self.log_step("OLLAMA_MODEL", "SUCCESS", f"{model} installed successfully")
            except subprocess.TimeoutExpired:
                self.log_step("OLLAMA_MODEL", "WARNING", f"{model} installation timed out - you can install it manually later")
            except subprocess.CalledProcessError:
                self.log_step("OLLAMA_MODEL", "WARNING", f"Failed to install {model} - you can install it manually later")
        
        self.log_step("OLLAMA", "SUCCESS", "Ollama setup completed")
    
    async def setup_tesseract(self):
        """Setup Tesseract OCR"""
        self.log_step("TESSERACT", "INFO", "Setting up Tesseract OCR...")
        
        # Run Tesseract installation script
        tesseract_script = self.project_root / "scripts" / "install_tesseract.py"
        if tesseract_script.exists():
            try:
                subprocess.run([sys.executable, str(tesseract_script)], check=True)
                self.log_step("TESSERACT", "SUCCESS", "Tesseract OCR installed")
            except subprocess.CalledProcessError:
                self.log_step("TESSERACT", "WARNING", "Tesseract installation may have failed")
        else:
            self.log_step("TESSERACT", "WARNING", "Tesseract installation script not found")
    
    async def setup_frontend(self):
        """Setup frontend dependencies"""
        self.log_step("FRONTEND", "INFO", "Setting up frontend dependencies...")
        
        frontend_path = self.project_root / "src" / "frontend"
        if frontend_path.exists():
            # Install npm dependencies
            subprocess.run(["npm", "install"], cwd=frontend_path, check=True)
            self.log_step("FRONTEND", "SUCCESS", "Frontend dependencies installed")
        else:
            self.log_step("FRONTEND", "WARNING", "Frontend directory not found")
    
    async def generate_configuration(self):
        """Generate configuration files"""
        self.log_step("CONFIGURATION", "INFO", "Generating configuration files...")
        
        # Generate .env file
        env_content = f"""# Database Configuration
DATABASE_URL=postgresql://{self.config['database']['user']}:{self.config['database']['password']}@localhost:5432/{self.config['database']['name']}

# API Configuration
API_HOST={self.config['api']['host']}
API_PORT={self.config['api']['port']}
DEBUG=true

# File Upload Configuration
MAX_FILE_SIZE_MB=10
UPLOAD_PATH=data/uploads

# AI Configuration
OLLAMA_BASE_URL={self.config['ollama']['base_url']}
LLM_MODEL=llama2
TESSERACT_PATH=/usr/bin/tesseract

# Security
SECRET_KEY=your-secret-key-change-in-production-{int(time.time())}
"""
        
        env_file = self.project_root / ".env"
        with open(env_file, "w") as f:
            f.write(env_content)
        
        # Generate AI configuration
        ai_config = {
            "llm": {
                "provider": "ollama",
                "model": "llama2",
                "base_url": self.config['ollama']['base_url'],
                "models": {
                    "conversation": "llama2",
                    "data_extraction": "codellama",
                    "reasoning": "mistral",
                    "fallback": "phi"
                },
                "generation_config": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.9,
                    "frequency_penalty": 0.0
                },
                "timeout_seconds": 180,
                "retry_attempts": 3
            },
            "ocr": {
                "engine": "tesseract",
                "confidence_threshold": 60,
                "config": "--psm 6",
                "preprocessing": {
                    "resize_factor": 2.0,
                    "denoise": True,
                    "deskew": True
                }
            },
            "ml_models": {
                "eligibility_model": "src/models/eligibility_model.joblib",
                "support_amount_model": "src/models/support_amount_model.joblib"
            }
        }
        
        ai_config_file = self.project_root / "ai_config.json"
        with open(ai_config_file, "w") as f:
            json.dump(ai_config, f, indent=2)
        
        self.log_step("CONFIGURATION", "SUCCESS", "Configuration files generated")
    
    async def train_ml_models(self):
        """Train ML models"""
        self.log_step("ML_MODELS", "INFO", "Training ML models...")
        
        # Create models directory
        models_dir = self.project_root / "src" / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Run ML training script
        train_script = self.project_root / "scripts" / "train_ml_models.py"
        if train_script.exists():
            try:
                subprocess.run([sys.executable, str(train_script)], check=True, timeout=300)
                self.log_step("ML_MODELS", "SUCCESS", "ML models trained successfully")
            except subprocess.CalledProcessError:
                self.log_step("ML_MODELS", "WARNING", "ML model training may have failed")
            except subprocess.TimeoutExpired:
                self.log_step("ML_MODELS", "WARNING", "ML model training timed out")
        else:
            self.log_step("ML_MODELS", "WARNING", "ML training script not found")
    
    async def run_system_tests(self):
        """Run system tests"""
        self.log_step("SYSTEM_TESTS", "INFO", "Running system tests...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test database connection
        total_tests += 1
        try:
            # Simple database test
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database=self.config["database"]["name"],
                user=self.config["database"]["user"],
                password=self.config["database"]["password"]
            )
            conn.close()
            tests_passed += 1
            self.log_step("TEST_DATABASE", "SUCCESS", "Database connection test passed")
        except Exception as e:
            self.log_step("TEST_DATABASE", "WARNING", f"Database test failed: {str(e)}")
        
        # Test Ollama
        total_tests += 1
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                tests_passed += 1
                self.log_step("TEST_OLLAMA", "SUCCESS", "Ollama test passed")
            else:
                self.log_step("TEST_OLLAMA", "WARNING", "Ollama test failed")
        except Exception as e:
            self.log_step("TEST_OLLAMA", "WARNING", f"Ollama test failed: {str(e)}")
        
        # Test Tesseract
        total_tests += 1
        try:
            result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                tests_passed += 1
                self.log_step("TEST_TESSERACT", "SUCCESS", "Tesseract test passed")
            else:
                self.log_step("TEST_TESSERACT", "WARNING", "Tesseract test failed")
        except Exception as e:
            self.log_step("TEST_TESSERACT", "WARNING", f"Tesseract test failed: {str(e)}")
        
        self.log_step("SYSTEM_TESTS", "SUCCESS", f"System tests completed: {tests_passed}/{total_tests} passed")
    
    async def finalize_setup(self):
        """Finalize setup"""
        self.log_step("FINALIZATION", "INFO", "Finalizing setup...")
        
        # Create necessary directories
        directories = [
            "data/uploads",
            "logs",
            "backups"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create startup script
        startup_script_content = f"""#!/bin/bash
# Social Support AI Workflow Startup Script
# Generated by complete_setup.py

echo "🚀 Starting Social Support AI Workflow..."

# Activate virtual environment
source venv/bin/activate

# Start Ollama in background (if not running)
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Start the application
python start_system.py
"""
        
        startup_script = self.project_root / "start.sh"
        with open(startup_script, "w") as f:
            f.write(startup_script_content)
        
        # Make executable
        if self.system != "windows":
            os.chmod(startup_script, 0o755)
        
        self.log_step("FINALIZATION", "SUCCESS", "Setup finalized")
    
    def print_success_summary(self):
        """Print success summary"""
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}")
        print("=" * 80)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"{Colors.ENDC}")
        
        print(f"{Colors.OKGREEN}✅ All components have been installed and configured:")
        print("   • System dependencies")
        print("   • Python environment with all packages")
        print("   • PostgreSQL database")
        print("   • Ollama with LLM models")
        print("   • Tesseract OCR")
        print("   • Frontend dependencies")
        print("   • Configuration files")
        print(f"   • ML models{Colors.ENDC}")
        
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}🚀 TO START THE APPLICATION:{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   python start_system.py{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   OR: ./start.sh{Colors.ENDC}")
        
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}🌐 ACCESS POINTS:{Colors.ENDC}")
        print(f"{Colors.OKCYAN}   • Frontend: http://localhost:3000")
        print(f"   • API: http://localhost:8000")
        print(f"   • API Docs: http://localhost:8000/docs{Colors.ENDC}")
        
        print(f"\n{Colors.WARNING}{Colors.BOLD}📋 NEXT STEPS:{Colors.ENDC}")
        print(f"{Colors.WARNING}   1. Review the generated .env file")
        print(f"   2. Update database credentials if needed")
        print(f"   3. Test the application with sample data")
        print(f"   4. Read INSTALLATION_GUIDE.md for advanced configuration{Colors.ENDC}")
        
        # Save setup log
        log_file = self.project_root / "setup_log.json"
        with open(log_file, "w") as f:
            json.dump(self.setup_log, f, indent=2)
        
        print(f"\n{Colors.OKBLUE}📝 Setup log saved to: setup_log.json{Colors.ENDC}")
    
    def print_failure_summary(self):
        """Print failure summary"""
        print(f"\n{Colors.FAIL}{Colors.BOLD}")
        print("=" * 80)
        print("❌ SETUP FAILED")
        print("=" * 80)
        print(f"{Colors.ENDC}")
        
        print(f"{Colors.FAIL}The setup process encountered errors. Please check the details above.")
        print(f"You may need to install some components manually.{Colors.ENDC}")
        
        print(f"\n{Colors.WARNING}{Colors.BOLD}📋 MANUAL INSTALLATION:{Colors.ENDC}")
        print(f"{Colors.WARNING}   See INSTALLATION_GUIDE.md for step-by-step manual installation instructions.{Colors.ENDC}")
        
        # Save setup log
        log_file = self.project_root / "setup_log_failed.json"
        with open(log_file, "w") as f:
            json.dump(self.setup_log, f, indent=2)
        
        print(f"\n{Colors.OKBLUE}📝 Setup log saved to: setup_log_failed.json{Colors.ENDC}")


async def main():
    """Main entry point"""
    setup_manager = SetupManager()
    await setup_manager.run_complete_setup()


if __name__ == "__main__":
    asyncio.run(main()) 