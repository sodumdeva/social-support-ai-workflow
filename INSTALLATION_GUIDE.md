# Complete Installation Guide - Social Support AI Workflow

This guide provides step-by-step instructions to set up the complete Social Support AI Workflow system with local LLMs, ChromaDB vector search, and all dependencies.

## 📋 System Requirements

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+ (for local LLMs)
- **Storage**: 25GB+ free space (for models, data, and ChromaDB)
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but recommended for faster LLM inference

### Software Requirements
- **Python**: 3.8 or higher (3.11 recommended)
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

This will automatically install all dependencies, set up the database, install local LLMs, initialize ChromaDB with sample data, and configure the system.

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

# Install Python dependencies (includes ChromaDB)
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

### Step 6: ChromaDB Setup

#### Initialize ChromaDB with Sample Data
```bash
# This creates training programs and job opportunities collections
python scripts/setup_chromadb_data.py
```

#### Verify ChromaDB Installation
```bash
# Test ChromaDB functionality
python -c "
import chromadb
client = chromadb.PersistentClient(path='data/chroma')
collections = client.list_collections()
print(f'ChromaDB collections: {[c.name for c in collections]}')
"
```

### Step 7: Tesseract OCR Setup

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

### Step 8: Frontend Setup

```bash
cd src/frontend
npm install
cd ../..
```

### Step 9: Configuration

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

# ChromaDB Configuration
CHROMADB_PERSIST_DIRECTORY=data/chroma

# Security
SECRET_KEY=your-secret-key-here
```

#### Create AI Configuration
```bash
python scripts/generate_ai_config.py
```

### Step 10: Train ML Models

```bash
# Train eligibility and support amount models
python scripts/train_models.py
```

### Step 11: System Verification

#### Test All Components
```bash
# Test database connection
python -c "from src.models.database import test_connection; test_connection()"

# Test Ollama connection
python -c "from src.services.llm_service import test_llm; test_llm()"

# Test ChromaDB
python test_chromadb_integration.py

# Test Tesseract
python -c "from src.data.document_processor import test_ocr; test_ocr()"
```

## 🏃‍♂️ Running the System

### Option 1: Quick Start (Recommended)
```bash
python start_system.py
```

### Option 2: Manual Start
```bash
# Terminal 1: Start API Server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Frontend (in new terminal)
cd src/frontend && npm start
```

### Access Points
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🧪 Testing the Installation

### Basic Functionality Test
```bash
# Test conversation endpoint
curl -X POST http://localhost:8000/conversation/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "conversation_state": {}, "conversation_history": []}'
```

### ChromaDB Integration Test
```bash
# Run comprehensive ChromaDB test
python test_chromadb_integration.py
```

### Document Upload Test
```bash
# Test document processing (with a sample image)
curl -X POST http://localhost:8000/conversation/upload-document \
  -F "file=@sample_document.jpg" \
  -F "file_type=emirates_id" \
  -F "conversation_state={}"
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Connection Error
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve

# Test model availability
ollama list
```

#### 2. Database Connection Error
```bash
# Check PostgreSQL status
sudo systemctl status postgresql  # Linux
brew services list | grep postgresql  # macOS

# Create database if missing
createdb social_support_db
```

#### 3. ChromaDB Issues
```bash
# Check ChromaDB directory
ls -la data/chroma/

# Reinitialize ChromaDB
rm -rf data/chroma/
python scripts/setup_chromadb_data.py
```

#### 4. Tesseract Not Found
```bash
# Find Tesseract installation
which tesseract

# Install if missing
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # macOS
```

#### 5. Frontend Build Issues
```bash
cd src/frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

#### 6. Python Package Issues
```bash
# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Performance Optimization

#### For Better LLM Performance
```bash
# Use GPU acceleration (if available)
export OLLAMA_GPU=1

# Increase model context size
export OLLAMA_NUM_CTX=4096
```

#### For Better ChromaDB Performance
```bash
# Increase ChromaDB cache size
export CHROMA_CACHE_SIZE=1000000
```

## 📊 System Monitoring

### Log Files
- **Application Logs**: `logs/app.log`
- **API Logs**: Check terminal output when running uvicorn
- **Frontend Logs**: Check browser console

### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Database health check
python -c "from src.models.database import test_connection; test_connection()"
```

## 🔄 Updates and Maintenance

### Updating Models
```bash
# Update Ollama models
ollama pull llama2
ollama pull mistral
ollama pull phi
ollama pull codellama
```

### Updating ChromaDB Data
```bash
# Add new training programs or job opportunities
python scripts/setup_chromadb_data.py --update
```

### Database Maintenance
```bash
# Backup database
pg_dump social_support_db > backup.sql

# Restore database
psql social_support_db < backup.sql
```

## 🚀 Production Deployment

### Docker Deployment (Recommended)
```bash
# Build Docker image
docker build -t social-support-ai .

# Run with Docker Compose
docker-compose up -d
```

### Environment-Specific Configuration
- **Development**: Use `.env.development`
- **Staging**: Use `.env.staging`
- **Production**: Use `.env.production`

## 📞 Support

If you encounter issues during installation:

1. Check the troubleshooting section above
2. Review log files for error messages
3. Ensure all system requirements are met
4. Verify network connectivity for model downloads

## 🎯 Next Steps

After successful installation:

1. **Test the System**: Run through a complete application flow
2. **Customize Data**: Add your own training programs and job opportunities to ChromaDB
3. **Configure Models**: Adjust AI model parameters in `ai_config.json`
4. **Monitor Performance**: Set up logging and monitoring for production use
5. **Scale as Needed**: Consider horizontal scaling for high-traffic scenarios

The system is now ready for use with full AI-powered conversation flow, document processing, eligibility assessment, and personalized economic enablement recommendations! 