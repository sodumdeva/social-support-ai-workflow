# 🚀 Social Support AI Workflow - Setup & Testing Guide

## 📋 Prerequisites

### 1. Python Dependencies
Dependencies are already installed. If you need to reinstall:
```bash
pip3 install -r requirements.txt
```

### 2. Ollama Installation
**For macOS:**
```bash
# Download and install from https://ollama.com/download
# OR use Homebrew if available:
brew install ollama
```

**For Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Required Models
After installing Ollama, download the required models:
```bash
ollama pull llama2:7b
ollama pull llava:13b
```

## 🔧 System Setup

### Step 1: Environment Configuration
The `.env` file is already created with basic settings. Modify if needed:

```bash
cat .env  # View current settings
```

### Step 2: Database Setup ✅
Database is already configured with SQLite:
```bash
# Database is ready at: social_support_ai.db
# To reset if needed: python3 scripts/setup_database.py --reset
```

### Step 3: Create Required Directories
```bash
mkdir -p data/uploads data/processed data/chroma models logs
```

## 🏃‍♂️ Running the System

### Terminal 1: Start Ollama Service
```bash
ollama serve
```

### Terminal 2: Start API Backend
```bash
cd social-support-ai-workflow
python3 -m uvicorn src.api.main:app --host localhost --port 8000 --reload
```

### Terminal 3: Start Frontend
```bash
cd social-support-ai-workflow
python3 -m streamlit run src/frontend/main.py --server.port 8501
```

## 🧪 Testing the System

### Quick Health Check
```bash
# Test API
curl http://localhost:8000/health

# Test Ollama
curl http://localhost:11434/api/tags
```

### Manual Testing Steps

1. **Access Frontend**: http://localhost:8501
2. **Submit Application**: Use the "Submit Application" page
3. **Upload Documents**: Test document processing
4. **Try Demo Mode**: Use "Quick Demo" for synthetic data testing
5. **Check Analytics**: View processing results

### Test Scenarios

#### Scenario 1: High-Income Applicant (Should be Declined)
- Monthly Income: 15,000 AED
- Family Size: 2
- Employment: Full-time (24+ months)
- Expected: Declined (income too high)

#### Scenario 2: Low-Income Family (Should be Approved)
- Monthly Income: 2,000 AED
- Family Size: 5
- Dependents: 3
- Employment: Part-time
- Expected: Approved with high support amount

#### Scenario 3: Student (Should be Approved)
- Monthly Income: 0 AED
- Employment: Student
- Family Size: 1
- Expected: Approved with education support

## 📊 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /applications/submit` - Submit application
- `POST /applications/{id}/documents` - Upload documents
- `POST /applications/{id}/process` - Process application
- `GET /applications/{id}/status` - Check status
- `GET /applications/{id}/results` - Get results

### Testing Endpoints
- `GET /testing/generate-synthetic-data` - Generate test data
- `POST /applications/process-with-data` - Process with synthetic data

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Not Found**
   ```bash
   # Install Ollama manually from https://ollama.com
   # Add to PATH if needed
   ```

2. **Models Not Downloaded**
   ```bash
   ollama list  # Check available models
   ollama pull llama2:7b  # Download required model
   ```

3. **Port Already in Use**
   ```bash
   # Change ports in .env file
   API_PORT=8001
   FRONTEND_PORT=8502
   ```

4. **Database Issues**
   ```bash
   python3 scripts/setup_database.py --reset
   ```

5. **Import Errors**
   ```bash
   # Ensure you're in the project directory
   cd social-support-ai-workflow
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

## 📝 Development Notes

### File Structure
```
social-support-ai-workflow/
├── src/
│   ├── api/           # FastAPI backend
│   ├── frontend/      # Streamlit interface
│   ├── agents/        # AI agents
│   ├── data/          # Data processing
│   └── models/        # Database models
├── scripts/           # Setup scripts
├── data/              # Runtime data
├── models/            # ML models
└── tests/             # Test files
```

### Key Components
- **Master Orchestrator**: Coordinates AI workflow
- **Data Extraction Agent**: Processes documents
- **Eligibility Agent**: Assesses applications
- **Document Processor**: Handles multimodal input
- **Synthetic Data Generator**: Creates test scenarios

## 🎯 Performance Expectations

- **Processing Time**: < 2 minutes per application
- **Document Types**: PDF, Images, Excel/CSV
- **Accuracy**: 90%+ eligibility assessment
- **Throughput**: 100+ applications/hour

## 📈 Next Steps

1. **Production Deployment**: Configure PostgreSQL, Redis
2. **Security Hardening**: JWT authentication, rate limiting
3. **Monitoring**: Add Prometheus metrics, logging
4. **Scalability**: Kubernetes deployment, load balancing
5. **CI/CD**: Automated testing, deployment pipelines 