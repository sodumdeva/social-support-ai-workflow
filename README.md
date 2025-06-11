# Social Support AI Workflow Automation

## Overview
An AI-powered workflow automation system for government social security departments to streamline application processing from 5-20 days to real-time decision making.

## Problem Statement
- **Current Issues**: Manual data gathering, inconsistent validation, subjective decision-making
- **Solution**: 100% automated application processing with AI agents and multimodal data processing
- **Target**: Real-time approval/decline decisions with economic enablement recommendations

## Features
- 🤖 Agentic AI orchestration with specialized agents
- 📄 Multimodal data processing (text, images, tabular data)
- 💬 Interactive chatbot interface
- 🏠 Locally hosted ML/LLM models
- 🔍 Comprehensive eligibility assessment
- 📊 Economic enablement recommendations

## Architecture Components
- **Data Ingestion**: Forms, bank statements, Emirates ID, resumes, credit reports
- **AI Agents**: Master orchestrator, data extraction, validation, eligibility check, decision recommendation
- **Local LLM**: Ollama with observability via LangSmith/Langfuse
- **Frontend**: Streamlit interactive interface
- **Backend**: FastAPI with PostgreSQL and ChromaDB

## Tech Stack
- **Language**: Python 3.9+
- **Data Pipeline**: Pandas, LlamaIndex, ChromaDB, PostgreSQL
- **AI/ML**: Scikit-learn, Transformers, OpenCV
- **Agent Framework**: LangGraph for orchestration
- **LLM Hosting**: Ollama
- **Observability**: LangSmith
- **API**: FastAPI
- **Frontend**: Streamlit
- **Database**: PostgreSQL + ChromaDB

## Quick Start

### Prerequisites
```bash
# Install Python 3.9+
# Install PostgreSQL
# Install Ollama
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd social-support-ai-workflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database
python scripts/setup_database.py

# Pull local LLM models
ollama pull llama2:7b
ollama pull llava:7b
```

### Running the Application
```bash
# Start backend API
uvicorn src.api.main:app --reload --port 8000

# Start Streamlit frontend
streamlit run src/frontend/app.py --server.port 8501
```

## Project Structure
```
social-support-ai-workflow/
├── src/
│   ├── agents/          # AI agent implementations
│   ├── api/            # FastAPI backend
│   ├── data/           # Data processing modules
│   ├── frontend/       # Streamlit UI
│   ├── models/         # ML model definitions
│   └── utils/          # Utility functions
├── data/
│   ├── raw/           # Raw input data
│   ├── processed/     # Processed data
│   └── synthetic/     # Generated test data
├── models/            # Trained model artifacts
├── tests/             # Unit and integration tests
├── scripts/           # Setup and utility scripts
├── docs/              # Documentation
└── requirements.txt   # Python dependencies
```

## Development Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Project setup and structure
- [x] Basic requirements and dependencies
- [ ] Database schema and connections
- [ ] Local LLM setup with Ollama

### Phase 2: Data Processing
- [ ] Document ingestion pipeline
- [ ] Multimodal data extraction
- [ ] Data validation framework
- [ ] Synthetic data generation

### Phase 3: AI Agents
- [ ] Master orchestrator agent
- [ ] Data extraction agent
- [ ] Validation agent
- [ ] Eligibility assessment agent
- [ ] Decision recommendation agent

### Phase 4: Integration
- [ ] Agent orchestration with LangGraph
- [ ] API development
- [ ] Frontend development
- [ ] End-to-end testing

### Phase 5: Observability & Deployment
- [ ] LangSmith integration
- [ ] Performance monitoring
- [ ] Documentation
- [ ] Final testing and optimization

## Contributing
Please follow the development guidelines and ensure all commits are well-documented for the evaluation process.

## License
This project is part of an AI assessment and follows standard development practices. 