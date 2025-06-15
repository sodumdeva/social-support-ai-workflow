# Project Structure

## Clean, Production-Ready Codebase

```
social-support-ai-workflow/
├── 📁 src/                          # Main source code
│   ├── 📁 agents/                   # AI Agents
│   │   ├── conversation_agent.py       # Conversation flow management
│   │   ├── data_extraction_agent.py    # Document processing & OCR
│   │   ├── eligibility_agent.py        # ML-based eligibility assessment
│   │   └── base_agent.py               # Base agent class
│   │
│   ├── 📁 api/                      # FastAPI Backend
│   │   └── main.py                     # REST API endpoints
│   │
│   ├── 📁 frontend/                 # React Frontend
│   │   ├── public/                     # Static assets
│   │   ├── src/                        # React components
│   │   ├── package.json                # Dependencies
│   │   └── README.md                   # Frontend setup
│   │
│   ├── 📁 workflows/                # Workflow Orchestration
│   │   └── langgraph_workflow.py       # LangGraph state management
│   │
│   ├── 📁 models/                   # Data & ML Models
│   │   ├── database.py                 # PostgreSQL models
│   │   └── ml_models.py                # Scikit-learn models
│   │
│   ├── 📁 data/                     # Data Processing
│   │   ├── document_processor.py       # Document utilities
│   │   └── synthetic_data.py           # Test data generation
│   │
│   ├── 📁 services/                 # Core Services
│   │   └── llm_service.py              # LLM integration (Ollama)
│   │
│   └── 📁 utils/                    # Utilities
│       └── logging_config.py           # Logging setup
│
├── 📁 scripts/                      # Setup & Deployment Scripts
│   ├── setup_database.py              # Database initialization
│   ├── setup_ai_models.py             # AI model setup
│   └── install_tesseract.py           # OCR setup
│
├── 📁 tests/                        # Test Suite
│   ├── test_agents.py                  # Agent testing
│   ├── test_api.py                     # API testing
│   └── test_workflow.py               # Workflow testing
│
├── 📁 data/                         # Data Storage
│   ├── uploads/                        # Document uploads
│   └── models/                         # Trained ML models
│
├── 📁 logs/                         # Application Logs
│   └── app.log                         # Main log file
│
├── 📄 start_system.py               # System Launcher
├── 📄 config.py                     # Configuration
├── 📄 requirements.txt              # Python dependencies
├── 📄 ai_config.json               # AI model configuration
├── 📄 README.md                    # Setup instructions
├── 📄 SOLUTION_SUMMARY.md          # Technical documentation
└── 📄 .gitignore                   # Git ignore rules
```

## Core Components

### 🤖 AI Agents (`src/agents/`)
- **ConversationAgent**: Manages chat flow and user interactions
- **DataExtractionAgent**: Processes documents with OCR + LLM
- **EligibilityAssessmentAgent**: ML-powered eligibility decisions
- **BaseAgent**: Common functionality for all agents

### 🌐 API Layer (`src/api/`)
- **main.py**: FastAPI application with REST endpoints
- Handles conversation messages, document uploads, status checks
- Automatic API documentation at `/docs`

### ⚛️ Frontend (`src/frontend/`)
- React-based chat interface
- Real-time conversation flow
- Document upload with drag-and-drop
- Application status tracking

### 🔄 Workflow Engine (`src/workflows/`)
- **langgraph_workflow.py**: State-based conversation management
- Handles complex routing and decision logic
- Loop prevention and error recovery

### 🗄️ Data Layer (`src/models/`)
- **database.py**: PostgreSQL models and operations
- **ml_models.py**: Scikit-learn model implementations
- Handles applications, documents, and predictions

### 🛠️ Services (`src/services/`)
- **llm_service.py**: Ollama LLM integration
- Local AI model hosting for privacy

### 📊 Data Processing (`src/data/`)
- **document_processor.py**: Multi-format document handling
- **synthetic_data.py**: Test data generation

## Key Features

✅ **Clean Architecture**: Modular, maintainable code structure  
✅ **Production Ready**: Comprehensive error handling and logging  
✅ **Well Documented**: Clear code comments and documentation  
✅ **Test Coverage**: Unit tests for all major components  
✅ **Configuration Driven**: Easy deployment and customization  
✅ **Security Focused**: Input validation and secure practices  
✅ **Scalable Design**: Async processing and database optimization  

## Technology Stack

- **Backend**: Python 3.8+, FastAPI, PostgreSQL
- **Frontend**: React, TypeScript, Modern UI components
- **AI/ML**: LangGraph, Scikit-learn, Ollama, Tesseract OCR
- **Deployment**: Docker-ready, Kubernetes compatible
- **Testing**: Pytest, comprehensive test suite 