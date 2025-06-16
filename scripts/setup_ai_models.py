"""
Setup Script for AI Models and Local LLM Hosting

Installs and configures Ollama with local LLM models for the Social Support AI Workflow.
Implements the locally hosted ML/LLM requirements from the solution specifications.
"""
import asyncio
import subprocess
import sys
import os
import json
import aiohttp
import time
from pathlib import Path


class AIModelSetup:
    """Setup and configuration for AI models and local LLM hosting"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.models_to_install = [
            "llama2",           # Primary conversational model
            "codellama",        # For structured data extraction
            "mistral",          # Alternative model for reasoning
            "phi"               # Lightweight model for quick responses
        ]
        
        # Model configurations
        self.model_configs = {
            "llama2": {
                "description": "Primary conversational model for user interaction",
                "use_case": "conversation, intent analysis, response generation",
                "size": "3.8GB"
            },
            "codellama": {
                "description": "Specialized model for structured data extraction",
                "use_case": "data extraction, JSON parsing, structured output",
                "size": "3.8GB"
            },
            "mistral": {
                "description": "Advanced reasoning model for complex decisions",
                "use_case": "eligibility reasoning, complex analysis",
                "size": "4.1GB"
            },
            "phi": {
                "description": "Lightweight model for quick responses",
                "use_case": "simple queries, fallback responses",
                "size": "1.6GB"
            }
        }
    
    async def setup_complete_ai_environment(self):
        """Complete setup of AI environment"""
        
        print("🚀 Setting up AI Environment for Social Support Workflow")
        print("=" * 60)
        
        try:
            # Step 1: Install Ollama
            await self.install_ollama()
            
            # Step 2: Start Ollama service
            await self.start_ollama_service()
            
            # Step 3: Install LLM models
            await self.install_llm_models()
            
            # Step 4: Setup ML models directory
            await self.setup_ml_models_directory()
            
            # Step 5: Test AI services
            await self.test_ai_services()
            
            # Step 6: Generate configuration
            await self.generate_ai_config()
            
            print("\n✅ AI Environment setup completed successfully!")
            print("\n📋 Next Steps:")
            print("1. Run: python run_api.py")
            print("2. Run: python run_frontend.py")
            print("3. Access the application at http://localhost:8501")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {str(e)}")
            print("Please check the error messages above and try again.")
    
    async def install_ollama(self):
        """Install Ollama for local LLM hosting"""
        
        print("\n📦 Installing Ollama...")
        
        try:
            # Check if Ollama is already installed
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama is already installed")
                print(f"   Version: {result.stdout.strip()}")
                return
        except FileNotFoundError:
            pass
        
        # Install Ollama based on OS
        import platform
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            print("   Installing Ollama for macOS...")
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
        elif system == "linux":
            print("   Installing Ollama for Linux...")
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
        elif system == "windows":
            print("   For Windows, please download Ollama from: https://ollama.ai/download")
            print("   After installation, run this script again.")
            return
        else:
            raise Exception(f"Unsupported operating system: {system}")
        
        # Execute installation
        process = subprocess.Popen(install_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("✅ Ollama installed successfully")
        else:
            raise Exception(f"Ollama installation failed: {stderr.decode()}")
    
    async def start_ollama_service(self):
        """Start Ollama service"""
        
        print("\n🔄 Starting Ollama service...")
        
        try:
            # Check if Ollama is already running
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            print("✅ Ollama service is already running")
                            return
                except:
                    pass
            
            # Start Ollama service
            print("   Starting Ollama service...")
            
            # Start Ollama in background
            if os.name == 'nt':  # Windows
                subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Unix-like
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            print("   Waiting for Ollama service to start...")
            for i in range(30):  # Wait up to 30 seconds
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status == 200:
                                print("✅ Ollama service started successfully")
                                return
                except:
                    pass
                
                await asyncio.sleep(1)
                print(f"   Waiting... ({i+1}/30)")
            
            raise Exception("Ollama service failed to start within 30 seconds")
            
        except Exception as e:
            print(f"❌ Failed to start Ollama service: {str(e)}")
            print("   Please start Ollama manually: ollama serve")
            raise
    
    async def install_llm_models(self):
        """Install required LLM models"""
        
        print("\n📥 Installing LLM models...")
        
        # Check available models
        available_models = await self.get_available_models()
        
        for model_name in self.models_to_install:
            if model_name in available_models:
                print(f"✅ {model_name} is already installed")
                continue
            
            config = self.model_configs.get(model_name, {})
            print(f"\n   Installing {model_name}...")
            print(f"   Description: {config.get('description', 'N/A')}")
            print(f"   Size: {config.get('size', 'Unknown')}")
            print(f"   Use case: {config.get('use_case', 'General')}")
            
            try:
                await self.pull_model(model_name)
                print(f"✅ {model_name} installed successfully")
            except Exception as e:
                print(f"❌ Failed to install {model_name}: {str(e)}")
                print(f"   You can install it manually later: ollama pull {model_name}")
    
    async def get_available_models(self):
        """Get list of available models"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["name"].split(":")[0] for model in data.get("models", [])]
                    else:
                        return []
        except Exception:
            return []
    
    async def pull_model(self, model_name: str):
        """Pull a specific model"""
        
        try:
            async with aiohttp.ClientSession() as session:
                pull_data = {"name": model_name}
                
                async with session.post(
                    f"{self.ollama_url}/api/pull",
                    json=pull_data,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes timeout
                ) as response:
                    
                    if response.status == 200:
                        # Stream the response to show progress
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode())
                                    if "status" in data:
                                        print(f"   {data['status']}", end="\r")
                                except:
                                    pass
                        print()  # New line after progress
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            raise Exception(f"Timeout while downloading {model_name}")
        except Exception as e:
            raise Exception(f"Failed to pull {model_name}: {str(e)}")
    
    async def setup_ml_models_directory(self):
        """Setup ML models directory structure"""
        
        print("\n📁 Setting up ML models directory...")
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "conversation",
            "eligibility", 
            "data_extraction",
            "validation",
            "checkpoints"
        ]
        
        for subdir in subdirs:
            (models_dir / subdir).mkdir(exist_ok=True)
        
        # Create model info file
        model_info = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "ML models for Social Support AI Workflow",
            "structure": {
                "conversation/": "Conversation classification models",
                "eligibility/": "Eligibility assessment models", 
                "data_extraction/": "Data extraction models",
                "validation/": "Data validation models",
                "checkpoints/": "Model checkpoints and backups"
            },
            "models": {
                "eligibility_classifier.pkl": "RandomForest classifier for eligibility",
                "support_amount_regressor.pkl": "Regression model for support amount",
                "risk_classifier.pkl": "Risk assessment classifier",
                "conversation_classifier.pkl": "Intent classification model",
                "data_validator.pkl": "Data validation model",
                "preprocessor.pkl": "Data preprocessing pipeline"
            }
        }
        
        with open(models_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ ML models directory structure created")
    
    async def test_ai_services(self):
        """Test AI services functionality"""
        
        print("\n🧪 Testing AI services...")
        
        # Test Ollama API
        try:
            async with aiohttp.ClientSession() as session:
                # Test basic connectivity
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        installed_models = [model["name"] for model in models_data.get("models", [])]
                        print(f"✅ Ollama API accessible")
                        print(f"   Installed models: {', '.join(installed_models)}")
                    else:
                        raise Exception(f"HTTP {response.status}")
                
                # Test model generation
                if installed_models:
                    test_model = installed_models[0].split(":")[0]
                    test_data = {
                        "model": test_model,
                        "prompt": "Hello, please respond with 'AI test successful'",
                        "stream": False
                    }
                    
                    async with session.post(
                        f"{self.ollama_url}/api/generate",
                        json=test_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"✅ Model generation test successful")
                            print(f"   Test response: {result.get('response', '')[:50]}...")
                        else:
                            print(f"⚠️  Model generation test failed: HTTP {response.status}")
                
        except Exception as e:
            print(f"❌ Ollama API test failed: {str(e)}")
            raise
        
        # Test ML models setup
        try:
            from src.agents.ml_eligibility_agent import MLEligibilityAgent
            agent = MLEligibilityAgent()
            print("✅ ML models initialized successfully")
        except Exception as e:
            print(f"⚠️  ML models test failed: {str(e)}")
            print("   Models will be trained on first use")
    
    async def generate_ai_config(self):
        """Generate AI configuration file"""
        
        print("\n⚙️  Generating AI configuration...")
        
        # Get installed models
        available_models = await self.get_available_models()
        
        config = {
            "ai_environment": {
                "setup_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ollama_url": self.ollama_url,
                "models_directory": "models/",
                "primary_model": "llama2" if "llama2" in available_models else available_models[0] if available_models else None
            },
            "llm_models": {
                "conversation": "llama2",
                "data_extraction": "codellama", 
                "reasoning": "mistral",
                "fallback": "phi"
            },
            "ml_models": {
                "eligibility_classifier": "models/eligibility_classifier.pkl",
                "support_regressor": "models/support_amount_regressor.pkl",
                "risk_classifier": "models/risk_classifier.pkl",
                "conversation_classifier": "models/conversation_classifier.pkl"
            },
            "model_settings": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
                "timeout": 60
            },
            "available_models": available_models,
            "model_info": self.model_configs
        }
        
        with open("ai_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ AI configuration saved to ai_config.json")
    
    async def check_system_requirements(self):
        """Check system requirements for AI models"""
        
        print("\n🔍 Checking system requirements...")
        
        import psutil
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM detected. AI models may run slowly.")
        else:
            print("✅ Sufficient RAM for AI models")
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"   Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 20:
            print("⚠️  Warning: Less than 20GB free space. May not be enough for all models.")
        else:
            print("✅ Sufficient disk space")
        
        # Check Python version
        python_version = sys.version_info
        print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        else:
            print("✅ Python version compatible")
        
        return True


async def main():
    """Main setup function"""
    
    setup = AIModelSetup()
    
    # Check system requirements
    if not await setup.check_system_requirements():
        print("\n❌ System requirements not met. Please upgrade your system.")
        return
    
    # Run complete setup
    await setup.setup_complete_ai_environment()


if __name__ == "__main__":
    asyncio.run(main()) 