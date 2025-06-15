# Social Support AI Workflow

An AI-powered social support application processing system that uses conversational AI, document processing, and machine learning to assess eligibility for financial assistance and provide economic enablement recommendations.

## 🎯 Overview

This system provides an intelligent, conversational interface for citizens to apply for social support benefits. It combines natural language processing, computer vision for document analysis, and machine learning models to automate the eligibility assessment process while providing personalized economic enablement recommendations.

## 🏗️ Architecture

### High-Level System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend │    │   PostgreSQL DB │
│                 │◄──►│                  │◄──►│                 │
│ • Chat Interface│    │ • LangGraph      │    │ • Applications  │
│ • File Upload   │    │   Workflow       │    │ • Documents     │
│ • Status Check  │    │ • REST APIs      │    │ • ML Predictions│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   AI Agents      │
                       │                  │
                       │ • Conversation   │
                       │ • Data Extract   │
                       │ • Eligibility    │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   External AI    │
                       │                  │
                       │ • Ollama LLM     │
                       │ • Tesseract OCR  │
                       │ • ML Models      │
                       └──────────────────┘
```

### Data Flow

1. **User Interaction**: Citizens interact through a conversational chat interface
2. **Document Processing**: Users upload documents (Emirates ID, bank statements, etc.)
3. **Data Extraction**: OCR + LLM extract structured data from documents
4. **Eligibility Assessment**: ML models + rule-based logic determine eligibility
5. **Economic Recommendations**: LLM generates personalized improvement suggestions
6. **Database Storage**: All data and decisions are stored for audit and tracking

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8+ (3.11 recommended)
- **Node.js**: 16.0+ 
- **PostgreSQL**: 12.0+
- **Git**: Latest version
- **System Requirements**: 8GB+ RAM, 20GB+ free disk space

### 🎯 One-Command Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd social-support-ai-workflow

# Run complete automated setup (installs everything)
python scripts/complete_setup.py
```

This automated script will:
- ✅ Install all system dependencies (Python, Node.js, PostgreSQL, Tesseract)
- ✅ Set up Python virtual environment with all packages
- ✅ Create and initialize PostgreSQL database
- ✅ Install Ollama and download LLM models (llama2, codellama, mistral, phi)
- ✅ Configure Tesseract OCR
- ✅ Install frontend dependencies
- ✅ Generate configuration files
- ✅ Train ML models
- ✅ Run system tests

**Estimated time**: 30-60 minutes (depending on internet speed for model downloads)

### 📝 Manual Installation (Step-by-Step)

If you prefer manual installation or the automated script fails, see the comprehensive [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed step-by-step instructions.

#### Quick Manual Setup:

1. **Install System Dependencies**
   ```bash
   # macOS
   brew install python@3.11 node postgresql tesseract git
   
   # Ubuntu/Debian
   sudo apt install python3.11 python3.11-pip nodejs npm postgresql tesseract-ocr git
   ```

2. **Setup Python Environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Setup Database**
   ```bash
   # Create database
   sudo -u postgres createdb social_support_db
   python scripts/setup_database.py
   ```

4. **Install Local LLMs (Ollama)**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve &
   
   # Install models (this takes time - models are 3-4GB each)
   ollama pull llama2      # Primary conversational model
   ollama pull codellama   # Data extraction model  
   ollama pull mistral     # Advanced reasoning model
   ollama pull phi         # Lightweight fallback model
   ```

5. **Setup OCR**
   ```bash
   python scripts/install_tesseract.py
   ```

6. **Frontend Setup**
   ```bash
   cd src/frontend
   npm install
   cd ../..
   ```

7. **Configuration**
   ```bash
   # Copy and edit environment file
   cp .env.example .env
   # Edit .env with your database credentials and settings
   ```

### Running the Application

**Option 1: Quick Start (Recommended)**
```bash
python start_system.py
```

**Option 2: Manual Start**
```bash
# Terminal 1: Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Frontend  
cd src/frontend && npm start
```

### Access Points

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

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

# AI Configuration - Local LLM Setup
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama2
TESSERACT_PATH=/usr/bin/tesseract

# Security
SECRET_KEY=your-secret-key-here
```

### AI Configuration

The system uses `ai_config.json` for AI model settings:

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
    "base_url": "http://localhost:11434",
    "generation_config": {
      "temperature": 0.7,
      "max_tokens": 2048,
      "timeout_seconds": 180
    }
  },
  "ocr": {
    "engine": "tesseract",
    "confidence_threshold": 60,
    "preprocessing": {
      "resize_factor": 2.0,
      "denoise": true,
      "deskew": true
    }
  }
}
```

## 📋 Usage

### 1. Starting a New Application

1. Open the frontend at http://localhost:3000
2. Click "Start New Application"
3. Follow the conversational flow:
   - Provide your full name
   - Enter Emirates ID number
   - Specify employment status
   - Enter monthly income
   - Provide family size
   - Describe housing situation

### 2. Document Upload

During the conversation, you can upload supporting documents:
- Emirates ID
- Bank statements
- Resume/CV
- Credit reports
- Assets/liabilities spreadsheets

### 3. Getting Results

After completing the conversation:
- Receive eligibility decision
- Get support amount (if eligible)
- Access economic enablement recommendations
- Ask follow-up questions about programs

### 4. Checking Application Status

Use the application lookup feature with:
- Reference number
- Emirates ID
- Name + phone number

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_agents.py

# Generate test data
curl http://localhost:8000/testing/generate-synthetic-data
```

### Sample Test Flow

```bash
# Test conversation endpoint
curl -X POST http://localhost:8000/conversation/message \
  -H "Content-Type: application/json" \
  -d '{"message": "John Smith", "conversation_state": {"current_step": "name_collection"}}'

# Test document upload
curl -X POST http://localhost:8000/conversation/upload-document \
  -F "file=@sample_document.pdf" \
  -F "file_type=bank_statement" \
  -F "conversation_state={}"
```

## 🔍 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/conversation/message` | POST | Process conversation messages |
| `/conversation/upload-document` | POST | Upload documents during conversation |
| `/applications/{id}/status` | GET | Get application status |
| `/applications/lookup` | POST | Lookup applications by various criteria |
| `/health` | GET | Health check |

### Example API Usage

```python
import requests

# Send a conversation message
response = requests.post("http://localhost:8000/conversation/message", json={
    "message": "I want to apply for financial support",
    "conversation_state": {},
    "conversation_history": []
})

# Upload a document
with open("emirates_id.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/conversation/upload-document", 
        files={"file": f},
        data={
            "file_type": "emirates_id",
            "conversation_state": "{}"
        }
    )
```

## 🏗️ System Components

### Core Agents

1. **ConversationAgent** (`src/agents/conversation_agent.py`)
   - Manages conversational flow
   - Handles user corrections and navigation
   - Generates contextual responses

2. **EligibilityAssessmentAgent** (`src/agents/eligibility_agent.py`)
   - Runs ML-based eligibility assessment
   - Generates economic enablement recommendations
   - Performs data validation and fraud detection

3. **DataExtractionAgent** (`src/agents/data_extraction_agent.py`)
   - Processes uploaded documents
   - Combines OCR and LLM for data extraction
   - Handles multiple document types

### Workflow Engine

**LangGraph Workflow** (`src/workflows/langgraph_workflow.py`)
- State-based conversation management
- Conditional routing between steps
- Loop prevention and error recovery
- Document processing integration

### Database Models

**PostgreSQL Schema** (`src/models/database.py`)
- Applications table with comprehensive fields
- Document storage and tracking
- ML prediction logging
- Application reviews and audit trail

### Machine Learning

**ML Models** (`src/models/ml_models.py`)
- Scikit-learn based eligibility classifier
- Support amount predictor
- Feature engineering and preprocessing
- Model persistence and loading

## 🔒 Security Considerations

- Input validation and sanitization
- File upload restrictions and scanning
- Database query parameterization
- API rate limiting (configurable)
- Audit logging for all decisions
- Data encryption in transit and at rest

## 📊 Monitoring and Logging

The system includes comprehensive logging:

```python
# Logs are structured and include:
- User interactions and conversation flow
- Document processing results
- ML model predictions and confidence scores
- Error tracking and debugging information
- Performance metrics and timing
```

Log files are stored in the `logs/` directory with rotation.

## 🚧 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Start Ollama service
   ollama serve
   
   # Pull required model
   ollama pull llama2
   ```

2. **Database Connection Error**
   ```bash
   # Check PostgreSQL service
   sudo systemctl status postgresql
   
   # Create database
   createdb social_support_db
   ```

3. **Tesseract Not Found**
   ```bash
   # Install Tesseract
   sudo apt-get install tesseract-ocr  # Ubuntu/Debian
   brew install tesseract              # macOS
   ```

4. **Frontend Build Issues**
   ```bash
   cd src/frontend
   rm -rf node_modules package-lock.json
   npm install
   ```