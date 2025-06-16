# Project Structure

## Clean, Production-Ready Codebase

```
social-support-ai-workflow/
├── 📁 src/                          # Main source code
│   ├── 📁 agents/                   # AI Agents
│   │   ├── conversation_agent.py       # Conversation flow + ChromaDB integration
│   │   ├── data_extraction_agent.py    # Document processing & OCR
│   │   ├── eligibility_agent.py        # ML-based eligibility + ChromaDB recommendations
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
│   │   ├── llm_service.py              # LLM integration (Ollama)
│   │   └── vector_store.py             # ChromaDB vector search service
│   │
│   └── 📁 utils/                    # Utilities
│       └── logging_config.py           # Logging setup
│
├── 📁 scripts/                      # Setup & Deployment Scripts
│   ├── setup_database.py              # Database initialization
│   ├── setup_ai_models.py             # AI model setup
│   ├── setup_chromadb_data.py         # ChromaDB initialization with sample data
│   └── install_tesseract.py           # OCR setup
│
├── 📁 tests/                        # Test Suite
│   ├── test_agents.py                  # Agent testing
│   ├── test_api.py                     # API testing
│   └── test_workflow.py               # Workflow testing
│
├── 📁 data/                         # Data Storage
│   ├── uploads/                        # Document uploads
│   ├── models/                         # Trained ML models
│   └── chroma/                         # ChromaDB persistent storage
│
├── 📁 logs/                         # Application Logs
│   └── app.log                         # Main log file
│
├── 📄 test_chromadb_integration.py  # ChromaDB integration testing
├── 📄 start_system.py               # System Launcher
├── 📄 config.py                     # Configuration
├── 📄 requirements.txt              # Python dependencies
├── 📄 ai_config.json               # AI model configuration
├── 📄 README.md                    # Setup instructions
├── 📄 SOLUTION_SUMMARY.md          # Technical documentation
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 INSTALLATION_GUIDE.md        # Detailed installation guide
└── 📄 .gitignore                   # Git ignore rules
```

## Core Components

### 🤖 AI Agents (`src/agents/`)
- **ConversationAgent**: Manages chat flow, user interactions, and ChromaDB-enhanced recommendations
- **DataExtractionAgent**: Processes documents with OCR + LLM
- **EligibilityAssessmentAgent**: ML-powered eligibility decisions with ChromaDB integration
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
- **llm_service.py**: Ollama LLM integration for local AI processing
- **vector_store.py**: ChromaDB vector search service for personalized recommendations

### 📊 Data Processing (`src/data/`)
- **document_processor.py**: Multi-format document handling
- **synthetic_data.py**: Test data generation

### 🎯 Vector Search (`data/chroma/`)
- **Training Programs Collection**: Semantic search for skill development programs
- **Job Opportunities Collection**: Job matching database for recommendations
- **Persistent Storage**: Local ChromaDB storage for privacy

## Key Features

✅ **Clean Architecture**: Modular, maintainable code structure  
✅ **Production Ready**: Comprehensive error handling and logging  
✅ **Well Documented**: Clear code comments and documentation  
✅ **Test Coverage**: Unit tests for all major components  
✅ **Configuration Driven**: Easy deployment and customization  
✅ **Privacy Focused**: All AI processing happens locally  
✅ **Scalable Design**: Async processing and database optimization  
✅ **ChromaDB Integration**: Personalized recommendations via vector search  
✅ **Fallback Systems**: Multiple layers of error recovery  

## Technology Stack

### Backend Technologies
- **Python 3.8+**: Core programming language
- **FastAPI**: Modern, fast web framework with automatic API docs
- **PostgreSQL**: ACID-compliant database for applications
- **ChromaDB**: Vector database for semantic search

### AI/ML Technologies
- **LangGraph**: Workflow orchestration for complex AI flows
- **Scikit-learn**: Machine learning models for eligibility assessment
- **Ollama**: Local LLM hosting (llama2, mistral, phi, codellama)
- **Tesseract OCR**: Document text extraction
- **Sentence Transformers**: Text embeddings for ChromaDB

### Frontend Technologies
- **React**: Component-based UI framework
- **TypeScript**: Type-safe JavaScript
- **Modern UI Components**: Responsive, accessible interface

### Development & Deployment
- **Docker-ready**: Containerization support
- **Pytest**: Comprehensive test suite
- **Structured Logging**: Detailed application monitoring
- **Environment Configuration**: Easy deployment customization

## Recent Enhancements

### ChromaDB Integration (Latest)
- ✅ **Vector Search Service**: Semantic similarity matching for recommendations
- ✅ **Training Programs Database**: 6 sample programs with detailed metadata
- ✅ **Job Opportunities Database**: 8 sample job postings with requirements
- ✅ **User Profile Mapping**: Intelligent mapping of user data to searchable profiles
- ✅ **LLM Enhancement**: Combines ChromaDB data with natural language generation
- ✅ **Fallback Mechanisms**: Ensures recommendations are always available

### System Improvements
- ✅ **Enhanced Error Handling**: Graceful degradation at all levels
- ✅ **Improved Logging**: Structured logging with performance metrics
- ✅ **Database Optimizations**: Better indexing and query performance
- ✅ **Document Processing**: Multi-format support with quality validation
- ✅ **ML Model Optimization**: Faster predictions with better accuracy

## Testing Infrastructure

### Test Files
- **test_chromadb_integration.py**: Comprehensive ChromaDB testing
- **tests/test_agents.py**: AI agent functionality testing
- **tests/test_api.py**: REST API endpoint testing
- **tests/test_workflow.py**: LangGraph workflow testing

### Test Coverage
- ✅ **Unit Tests**: Individual component testing
- ✅ **Integration Tests**: End-to-end workflow testing
- ✅ **API Tests**: REST endpoint validation
- ✅ **ChromaDB Tests**: Vector search functionality
- ✅ **ML Model Tests**: Prediction accuracy validation

## Configuration Management

### Configuration Files
- **ai_config.json**: AI model settings and parameters
- **.env**: Environment variables and secrets
- **requirements.txt**: Python package dependencies
- **package.json**: Frontend dependencies

### Setup Scripts
- **setup_database.py**: PostgreSQL initialization
- **setup_ai_models.py**: Ollama model installation
- **setup_chromadb_data.py**: Vector database population
- **install_tesseract.py**: OCR system setup

This structure ensures maintainability, scalability, and ease of deployment while providing comprehensive functionality for social support application processing with AI-powered recommendations. 