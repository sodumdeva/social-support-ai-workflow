# Complete Installation Guide - Social Support AI Workflow

This guide provides step-by-step instructions to set up the complete Social Support AI Workflow system with local LLMs and all dependencies.

## 📋 System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+ (for local LLMs)
- **Storage**: 20GB+ free space (for models and data)
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but recommended for faster LLM inference

### Software Requirements
- **Python**: 3.8 or higher
- **Node.js**: 16.0 or higher
- **PostgreSQL**: 12.0 or higher
- **Git**: Latest version

## 🚀 Quick Installation (Automated)

For a fully automated setup, run:

```bash
# Clone the repository
git clone <repository-url>
cd social-support-ai-workflow

# Run the complete setup script
python scripts/complete_setup.py
```

This will automatically install all dependencies, set up the database, install local LLMs, and configure the system.

## 📝 Manual Installation (Step-by-Step)

### Step 1: System Dependencies

#### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python@3.11 node postgresql tesseract git
brew services start postgresql
```

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.11 python3.11-pip python3.11-venv nodejs npm postgresql postgresql-contrib tesseract-ocr git curl

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### Windows
1. Install Python 3.11 from [python.org](https://python.org)
2. Install Node.js from [nodejs.org](https://nodejs.org)
3. Install PostgreSQL from [postgresql.org](https://postgresql.org)
4. Install Tesseract from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
5. Install Git from [git-scm.com](https://git-scm.com)

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd social-support-ai-workflow
```

### Step 3: Python Environment Setup

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### Step 4: Database Setup

#### Create Database
```bash
# macOS/Linux
sudo -u postgres createdb social_support_db
sudo -u postgres createuser --interactive

# Windows (run in PostgreSQL command prompt)
createdb social_support_db
```

#### Initialize Database Schema
```bash
python scripts/setup_database.py
```

### Step 5: Local LLM Setup (Ollama)

#### Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
```

#### Start Ollama Service
```bash
# Start Ollama (runs in background)
ollama serve
```

#### Install Required Models
```bash
# Install primary models (this will take time - models are 3-4GB each)
ollama pull llama2          # Primary conversational model (3.8GB)
ollama pull codellama       # Data extraction model (3.8GB)
ollama pull mistral         # Advanced reasoning model (4.1GB)
ollama pull phi             # Lightweight model (1.6GB)

# Verify installation
ollama list
```

### Step 6: Tesseract OCR Setup

#### Automated Installation
```bash
python scripts/install_tesseract.py
```

#### Manual Installation (if automated fails)
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows: Download from GitHub releases
```

### Step 7: Frontend Setup

```bash
cd src/frontend
npm install
cd ../..
```

### Step 8: Configuration

#### Create Environment File
```bash
cp .env.example .env
```

#### Edit .env file:
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/social_support_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# File Upload Configuration
MAX_FILE_SIZE_MB=10
UPLOAD_PATH=data/uploads

# AI Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama2
TESSERACT_PATH=/usr/bin/tesseract

# Security
SECRET_KEY=your-secret-key-here
```

#### Create AI Configuration
```bash
python scripts/generate_ai_config.py
```

### Step 9: Train ML Models

```bash
python scripts/train_ml_models.py
```

### Step 10: Test Installation

```bash
# Test all components
python scripts/test_installation.py

# Test individual components
python scripts/test_database.py
python scripts/test_ollama.py
python scripts/test_tesseract.py
```

## 🏃‍♂️ Running the Application

### Option 1: Quick Start (Recommended)
```bash
python start_system.py
```

### Option 2: Manual Start
```bash
# Terminal 1: Start API Server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Frontend
cd src/frontend && npm start
```

### Access Points
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Advanced Configuration

### Local LLM Configuration

#### Model Selection
Edit `ai_config.json` to choose different models:
```json
{
  "llm": {
    "provider": "ollama",
    "models": {
      "conversation": "llama2",
      "data_extraction": "codellama",
      "reasoning": "mistral",
      "fallback": "phi"
    },
    "base_url": "http://localhost:11434"
  }
}
```

#### Performance Tuning
```json
{
  "llm": {
    "generation_config": {
      "temperature": 0.7,
      "max_tokens": 2048,
      "top_p": 0.9,
      "frequency_penalty": 0.0
    },
    "timeout_seconds": 180,
    "retry_attempts": 3
  }
}
```

### Database Configuration

#### Connection Pooling
```python
# In database configuration
SQLALCHEMY_DATABASE_URL = "postgresql://user:pass@localhost/db"
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_pre_ping": True,
    "pool_recycle": 300
}
```

### OCR Configuration

#### Tesseract Settings
```json
{
  "ocr": {
    "engine": "tesseract",
    "config": "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
    "confidence_threshold": 60,
    "preprocessing": {
      "resize_factor": 2.0,
      "denoise": true,
      "deskew": true
    }
  }
}
```

## 🧪 Testing and Validation

### Run Complete Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_workflow.py -v
python -m pytest tests/test_database.py -v
```

### Test Individual Components
```bash
# Test conversation flow
python tests/test_conversation_flow.py

# Test document processing
python tests/test_document_processing.py

# Test ML models
python tests/test_ml_models.py

# Test database operations
python tests/test_database_operations.py
```

### Load Testing
```bash
# Test API performance
python tests/load_test_api.py

# Test concurrent users
python tests/test_concurrent_users.py
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Check available models
ollama list
```

#### 2. Database Connection Error
```bash
# Check PostgreSQL status
sudo systemctl status postgresql  # Linux
brew services list | grep postgres  # macOS

# Test connection
psql -h localhost -U username -d social_support_db
```

#### 3. Tesseract Not Found
```bash
# Check installation
tesseract --version

# Check Python binding
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"

# Reinstall if needed
python scripts/install_tesseract.py
```

#### 4. Frontend Build Issues
```bash
cd src/frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

#### 5. Python Import Errors
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check virtual environment
which python
pip list
```

#### 6. Model Loading Issues
```bash
# Check model files
ls -la src/models/

# Retrain models if needed
python scripts/train_ml_models.py --force
```

### Performance Optimization

#### 1. LLM Performance
- Use GPU acceleration if available
- Adjust model parameters for speed vs quality
- Implement response caching
- Use smaller models for simple tasks

#### 2. Database Performance
- Enable connection pooling
- Add database indexes
- Optimize queries
- Use read replicas for scaling

#### 3. Document Processing
- Implement parallel processing
- Cache OCR results
- Optimize image preprocessing
- Use async processing for large files

## 📊 Monitoring and Maintenance

### Log Files
- **API Logs**: `logs/api.log`
- **Workflow Logs**: `logs/workflow.log`
- **Database Logs**: `logs/database.log`
- **ML Model Logs**: `logs/ml_models.log`

### Health Checks
```bash
# Check system health
curl http://localhost:8000/health

# Check component status
python scripts/health_check.py
```

### Backup and Recovery
```bash
# Backup database
pg_dump social_support_db > backup_$(date +%Y%m%d).sql

# Backup ML models
tar -czf models_backup_$(date +%Y%m%d).tar.gz src/models/

# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz *.json *.env
```

## 🔄 Updates and Upgrades

### Update Dependencies
```bash
# Update Python packages
pip install --upgrade -r requirements.txt

# Update Node.js packages
cd src/frontend && npm update

# Update Ollama models
ollama pull llama2
ollama pull codellama
```

### System Updates
```bash
# Pull latest code
git pull origin main

# Run migration scripts
python scripts/migrate_database.py
python scripts/update_ml_models.py

# Restart services
python start_system.py
```

## 📞 Support and Documentation

### Getting Help
- Check the troubleshooting section above
- Review log files for error details
- Run diagnostic scripts: `python scripts/diagnose_issues.py`
- Check system requirements and dependencies

### Additional Resources
- **API Documentation**: http://localhost:8000/docs
- **Database Schema**: `docs/database_schema.md`
- **Architecture Guide**: `SOLUTION_SUMMARY.md`
- **Development Guide**: `docs/development.md`

---

## ✅ Installation Checklist

- [ ] System dependencies installed (Python, Node.js, PostgreSQL, Tesseract)
- [ ] Repository cloned and virtual environment created
- [ ] Python dependencies installed from requirements.txt
- [ ] Database created and schema initialized
- [ ] Ollama installed and models downloaded
- [ ] Tesseract OCR installed and tested
- [ ] Frontend dependencies installed
- [ ] Environment variables configured
- [ ] ML models trained
- [ ] Installation tested successfully
- [ ] Application running and accessible

**Estimated Total Installation Time**: 30-60 minutes (depending on internet speed for model downloads)

**Total Disk Space Required**: ~15-20GB (including models and dependencies) 