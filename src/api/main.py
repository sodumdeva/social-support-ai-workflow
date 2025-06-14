"""
FastAPI main application for Social Support AI Workflow

Provides REST API endpoints for:
- Application submission
- Document upload
- Application processing
- Status checking
- Results retrieval
- Machine Learning model operations
"""
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import shutil
import asyncio
import json
from pathlib import Path
import numpy as np

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import logging configuration first
from src.utils.logging_config import get_logger

# Setup logging
logger = get_logger("api")

from config import settings, get_upload_path
from src.models.database import get_db, Application, Document
from src.agents.master_orchestrator import MasterOrchestrator
from src.data.synthetic_data import SyntheticDataGenerator
from src.agents.conversation_agent import ConversationAgent
from src.workflows.langgraph_workflow import create_conversation_workflow, ConversationState

# Import ML endpoints
try:
    from .ml_endpoints import router as ml_router
    ML_ENDPOINTS_AVAILABLE = True
except ImportError:
    ML_ENDPOINTS_AVAILABLE = False
    print("Warning: ML endpoints not available")

# Pydantic models for API
class ApplicationData(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    emirates_id: Optional[str] = None
    monthly_income: Optional[float] = None
    employment_status: Optional[str] = None
    family_size: int = 1
    # Removed fields that don't exist in database model


class ProcessApplicationRequest(BaseModel):
    application_data: ApplicationData
    use_synthetic_data: bool = False


class ApplicationStatus(BaseModel):
    application_id: str
    status: str
    submitted_at: datetime
    processed_at: Optional[datetime] = None


class ProcessingResult(BaseModel):
    application_id: str
    status: str
    final_decision: dict
    processing_summary: dict
    workflow_state: dict


# Initialize FastAPI app
app = FastAPI(
    title="Social Support AI Workflow API",
    description="AI-powered social support application processing system with ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ML endpoints router if available
if ML_ENDPOINTS_AVAILABLE:
    app.include_router(ml_router)

# Initialize agents
orchestrator = MasterOrchestrator()
synthetic_generator = SyntheticDataGenerator()

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.get("/")
async def root():
    """Root endpoint with API information"""
    
    endpoints = {
        "submit_application": "/applications/submit",
        "upload_documents": "/applications/{application_id}/documents",
        "process_application": "/applications/{application_id}/process",
        "get_status": "/applications/{application_id}/status",
        "get_results": "/applications/{application_id}/results",
        "generate_synthetic": "/testing/generate-synthetic-data"
    }
    
    # Add ML endpoints if available
    if ML_ENDPOINTS_AVAILABLE:
        endpoints.update({
            "ml_status": "/ml/status",
            "ml_train": "/ml/train", 
            "ml_predict_eligibility": "/ml/predict/eligibility",
            "ml_predict_risk": "/ml/predict/risk",
            "ml_predict_support": "/ml/predict/support-amount",
            "ml_detect_fraud": "/ml/predict/fraud",
            "ml_match_programs": "/ml/predict/programs",
            "ml_comprehensive": "/ml/predict/comprehensive",
            "ml_models": "/ml/models",
            "ml_evaluate": "/ml/evaluate"
        })
    
    return {
        "message": "Social Support AI Workflow API with ML Models",
        "version": "1.0.0",
        "status": "running",
        "ml_models_available": ML_ENDPOINTS_AVAILABLE,
        "endpoints": endpoints,
        "features": [
            "Document processing and data extraction",
            "Multi-modal AI agent orchestration", 
            "Real-time eligibility assessment",
            "Economic enablement recommendations",
            "Scikit-learn ML classification models",
            "Fraud detection and risk assessment",
            "Interactive chat interface support"
        ]
    }


@app.post("/applications/submit", response_model=dict)
async def submit_application(
    application_data: ApplicationData,
    db: Session = Depends(get_db)
):
    """Submit a new social support application"""
    try:
        # Generate unique application ID
        application_id = f"APP-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        # Create application record in database
        db_application = Application(
            application_id=application_id,
            first_name=application_data.first_name,
            last_name=application_data.last_name,
            email=application_data.email,
            phone=application_data.phone,
            emirates_id=application_data.emirates_id,
            monthly_income=application_data.monthly_income,
            employment_status=application_data.employment_status,
            family_size=application_data.family_size,
            status="submitted"
        )
        
        db.add(db_application)
        db.commit()
        db.refresh(db_application)
        
        return {
            "application_id": application_id,
            "status": "submitted",
            "message": "Application submitted successfully",
            "next_steps": [
                "Upload required documents using the /applications/{application_id}/documents endpoint",
                "Process application using the /applications/{application_id}/process endpoint"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit application: {str(e)}")


@app.post("/applications/{application_id}/documents")
async def upload_documents(
    application_id: str,
    files: List[UploadFile] = File(...),
    document_types: List[str] = Form(...),
    db: Session = Depends(get_db)
):
    """Upload documents for an application"""
    try:
        # Check if application exists
        application = db.query(Application).filter(Application.application_id == application_id).first()
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Validate number of files matches document types
        if len(files) != len(document_types):
            raise HTTPException(status_code=400, detail="Number of files must match number of document types")
        
        # Create upload directory for this application
        upload_dir = os.path.join(get_upload_path(), application_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_documents = []
        
        for file, doc_type in zip(files, document_types):
            # Validate file size
            if file.size > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} exceeds maximum size of {settings.max_file_size_mb}MB"
                )
            
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{doc_type}_{uuid.uuid4().hex[:8]}{file_extension}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Save file to disk
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Create document record in database
            db_document = Document(
                application_id=application_id,
                document_type=doc_type,
                original_filename=file.filename,
                file_path=file_path,
                file_size_bytes=file.size,
                mime_type=file.content_type,
                processing_status="uploaded"
            )
            
            db.add(db_document)
            uploaded_documents.append({
                "document_type": doc_type,
                "filename": file.filename,
                "file_path": file_path,
                "size_bytes": file.size
            })
        
        db.commit()
        
        return {
            "application_id": application_id,
            "uploaded_documents": uploaded_documents,
            "message": f"Successfully uploaded {len(files)} documents",
            "next_steps": [
                "Process application using the /applications/{application_id}/process endpoint"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload documents: {str(e)}")


@app.post("/applications/{application_id}/process", response_model=ProcessingResult)
async def process_application(
    application_id: str,
    db: Session = Depends(get_db)
):
    """Process an application through the AI workflow"""
    try:
        # Check if application exists
        application = db.query(Application).filter(Application.application_id == application_id).first()
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Get application documents
        documents = db.query(Document).filter(Document.application_id == application_id).all()
        
        # Prepare application data for processing
        application_data = {
            "monthly_income": application.monthly_income,
            "employment_status": application.employment_status,
            "family_size": application.family_size
        }
        
        # Prepare document data for processing
        document_data = [
            {
                "file_path": doc.file_path,
                "document_type": doc.document_type
            }
            for doc in documents
        ]
        
        # Prepare input for orchestrator
        orchestrator_input = {
            "application_data": application_data,
            "documents": document_data,
            "application_id": application_id
        }
        
        # Update application status
        application.status = "processing"
        db.commit()
        
        # Process through orchestrator
        result = await orchestrator.process(orchestrator_input)
        
        # Update application with results
        if result["status"] == "success":
            final_decision = result["final_decision"]
            application.status = "completed"
            application.processed_at = datetime.utcnow()
            application.eligibility_score = final_decision.get("eligibility_score", 0)
            application.is_eligible = final_decision.get("decision") == "approved"
            application.support_amount = final_decision.get("support_amount", 0)
            application.assessment_data = result["workflow_state"]
        else:
            application.status = "failed"
            application.processed_at = datetime.utcnow()
        
        db.commit()
        
        return ProcessingResult(
            application_id=application_id,
            status=result["status"],
            final_decision=result.get("final_decision", {}),
            processing_summary=result.get("processing_summary", {}),
            workflow_state=result.get("workflow_state", {})
        )
        
    except Exception as e:
        # Update application status to failed
        application = db.query(Application).filter(Application.application_id == application_id).first()
        if application:
            application.status = "failed"
            application.processed_at = datetime.utcnow()
            db.commit()
        
        raise HTTPException(status_code=500, detail=f"Failed to process application: {str(e)}")


@app.get("/applications/{application_id}/status", response_model=ApplicationStatus)
async def get_application_status(
    application_id: str,
    db: Session = Depends(get_db)
):
    """Get application status"""
    application = db.query(Application).filter(Application.application_id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    return ApplicationStatus(
        application_id=application.application_id,
        status=application.status,
        submitted_at=application.submitted_at,
        processed_at=application.processed_at
    )


@app.get("/applications/{application_id}/results")
async def get_application_results(
    application_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed application results"""
    application = db.query(Application).filter(Application.application_id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    
    return {
        "application_id": application.application_id,
        "status": application.status,
        "eligibility_score": application.eligibility_score,
        "is_eligible": application.is_eligible,
        "support_amount": application.support_amount,
        "submitted_at": application.submitted_at,
        "processed_at": application.processed_at,
        "assessment_data": application.assessment_data
    }


@app.post("/applications/process-with-data")
async def process_application_with_data(
    request: ProcessApplicationRequest,
    db: Session = Depends(get_db)
):
    """Process application with provided data (for testing/demo)"""
    try:
        # Generate unique application ID
        application_id = f"APP-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        # Create application record
        db_application = Application(
            application_id=application_id,
            first_name=request.application_data.first_name,
            last_name=request.application_data.last_name,
            email=request.application_data.email,
            phone=request.application_data.phone,
            emirates_id=request.application_data.emirates_id,
            monthly_income=request.application_data.monthly_income,
            employment_status=request.application_data.employment_status,
            family_size=request.application_data.family_size,
            status="processing"
        )
        
        db.add(db_application)
        db.commit()
        
        # Convert pydantic model to dict
        application_data = request.application_data.dict()
        
        # Process without documents for demo
        orchestrator_input = {
            "application_data": application_data,
            "documents": [],  # No documents for this demo endpoint
            "application_id": application_id
        }
        
        # Process through orchestrator
        result = await orchestrator.process(orchestrator_input)
        
        # Update application with results
        if result["status"] == "success":
            final_decision = result["final_decision"]
            db_application.status = "completed"
            db_application.processed_at = datetime.utcnow()
            db_application.eligibility_score = final_decision.get("eligibility_score", 0)
            db_application.is_eligible = final_decision.get("decision") == "approved"
            db_application.support_amount = final_decision.get("support_amount", 0)
            db_application.assessment_data = result["workflow_state"]
        else:
            db_application.status = "failed"
            db_application.processed_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "application_id": application_id,
            "status": result["status"],
            "final_decision": result.get("final_decision", {}),
            "processing_summary": result.get("processing_summary", {}),
            "workflow_state": result.get("workflow_state", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process application: {str(e)}")


@app.get("/testing/generate-synthetic-data")
async def generate_synthetic_data():
    """Generate synthetic test data for development and testing"""
    try:
        # Generate synthetic application
        synthetic_app = synthetic_generator.generate_application()
        
        # Generate synthetic documents data
        synthetic_docs = {
            "bank_statement": synthetic_generator.generate_bank_statement_data(),
            "emirates_id": synthetic_generator.generate_emirates_id_data(),
            "resume": synthetic_generator.generate_resume_data(),
            "credit_report": synthetic_generator.generate_credit_report_data(),
            "assets": synthetic_generator.generate_assets_liabilities_data()
        }
        
        return {
            "application_data": synthetic_app,
            "document_data": synthetic_docs,
            "usage": "Use this data to test the application processing workflow"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate synthetic data: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Add conversation endpoints
@app.post("/conversation/message")
async def process_conversation_message(request: Dict[str, Any]):
    """Process a conversation message through the AI agent"""
    
    try:
        user_message = request.get("message", "")
        conversation_history = request.get("conversation_history", [])
        conversation_state = request.get("conversation_state", {})
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Initialize conversation agent
        conversation_agent = ConversationAgent()
        
        # Process the message
        response = await conversation_agent.process_message(
            user_message,
            conversation_history,
            conversation_state
        )
        
        # Convert numpy types to Python native types
        clean_response = convert_numpy_types(response)
        
        return {
            "status": "success",
            **clean_response
        }
        
    except Exception as e:
        logger.error(f"Error processing conversation message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.post("/conversation/upload-document")
async def process_conversation_document(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    conversation_state: str = Form("{}")
):
    """Process uploaded document during conversation"""
    
    try:
        logger.info(f"Document upload started - File: {file.filename}, Type: {file_type}, Size: {file.size}")
        
        # Parse conversation state
        try:
            state = json.loads(conversation_state) if conversation_state else {}
            logger.info(f"Conversation state parsed successfully. Current step: {state.get('current_step', 'unknown')}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse conversation state: {str(e)}")
            state = {}
        
        # Save uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        logger.info(f"Saving file to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            logger.info(f"File saved successfully. Size: {len(content)} bytes")
        
        # Initialize conversation agent
        logger.info("Initializing conversation agent")
        conversation_agent = ConversationAgent()
        
        # Process document
        logger.info(f"Starting document processing with conversation agent")
        response = await conversation_agent.process_document_upload(
            file_path,
            file_type,
            state
        )
        logger.info(f"Document processing completed. Response keys: {list(response.keys())}")
        
        # Convert numpy types to Python native types
        clean_response = convert_numpy_types(response)
        
        result = {
            "status": "success",
            "file_path": file_path,
            **clean_response
        }
        
        logger.info(f"Document upload endpoint completed successfully for: {file.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing document upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/conversation/workflow/start")
async def start_conversation_workflow(request: Dict[str, Any]):
    """Start a new conversation workflow using LangGraph"""
    
    try:
        # Create workflow
        workflow = create_conversation_workflow()
        
        # Initialize state
        initial_state: ConversationState = {
            "messages": [],
            "collected_data": {},
            "current_step": "greeting",
            "eligibility_result": None,
            "final_decision": None,
            "uploaded_documents": [],
            "workflow_history": [],
            "application_id": None,
            "processing_status": "initializing",
            "error_messages": []
        }
        
        # Run initialization
        result = await workflow.ainvoke(initial_state)
        
        return {
            "status": "success",
            "workflow_state": result,
            "application_id": result.get("application_id")
        }
        
    except Exception as e:
        logger.error(f"Error starting conversation workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting workflow: {str(e)}")


@app.post("/conversation/workflow/continue")
async def continue_conversation_workflow(request: Dict[str, Any]):
    """Continue an existing conversation workflow"""
    
    try:
        workflow_state = request.get("workflow_state", {})
        user_message = request.get("user_message", "")
        
        if not workflow_state:
            raise HTTPException(status_code=400, detail="Workflow state is required")
        
        # Add user message to state
        if user_message:
            workflow_state["messages"].append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
        
        # Create workflow
        workflow = create_conversation_workflow()
        
        # Continue workflow
        result = await workflow.ainvoke(workflow_state)
        
        return {
            "status": "success",
            "workflow_state": result,
            "is_complete": result.get("processing_status") == "completed"
        }
        
    except Exception as e:
        logger.error(f"Error continuing conversation workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error continuing workflow: {str(e)}")


@app.post("/conversation/workflow/upload")
async def upload_to_workflow(
    file: UploadFile = File(...),
    workflow_state: str = Form(...),
    file_type: str = Form(...)
):
    """Upload document to existing workflow"""
    
    try:
        # Parse workflow state
        state = json.loads(workflow_state)
        
        # Save uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Add file to workflow state
        if "uploaded_documents" not in state:
            state["uploaded_documents"] = []
        state["uploaded_documents"].append(file_path)
        
        # Create workflow and continue processing
        workflow = create_conversation_workflow()
        result = await workflow.ainvoke(state)
        
        return {
            "status": "success",
            "workflow_state": result,
            "file_path": file_path
        }
        
    except Exception as e:
        logger.error(f"Error uploading to workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/conversation/session/{session_id}")
async def get_conversation_session(session_id: str):
    """Get conversation session data"""
    
    try:
        # In a real implementation, this would fetch from database
        # For now, return mock data
        
        return {
            "status": "success",
            "session_id": session_id,
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your Social Support AI Assistant.",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "state": {
                "current_step": "greeting",
                "collected_data": {},
                "application_id": f"APP-{session_id}"
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching conversation session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")


@app.post("/conversation/session/{session_id}/save")
async def save_conversation_session(session_id: str, request: Dict[str, Any]):
    """Save conversation session data"""
    
    try:
        session_data = request.get("session_data", {})
        
        # In a real implementation, this would save to database
        # For now, just return success
        
        return {
            "status": "success",
            "session_id": session_id,
            "saved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error saving conversation session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving session: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    ) 