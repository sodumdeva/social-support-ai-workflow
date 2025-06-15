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
from src.data.synthetic_data import SyntheticDataGenerator
from src.agents.conversation_agent import ConversationAgent
from src.agents.data_extraction_agent import DataExtractionAgent
from src.agents.eligibility_agent import EligibilityAssessmentAgent
from src.workflows.langgraph_workflow import create_conversation_workflow, ConversationState
from src.models.database import DatabaseManager

# Import ML endpoints
try:
    from src.api.ml_endpoints import router as ml_router
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


class ApplicationStatus(BaseModel):
    application_id: str
    status: str
    submitted_at: datetime
    processed_at: Optional[datetime] = None


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
        "get_status": "/applications/{application_id}/status",
        "get_results": "/applications/{application_id}/results",
        "conversation_message": "/conversation/message",
        "conversation_upload": "/conversation/upload-document",
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
        "message": "Social Support AI Workflow API with LangGraph",
        "version": "2.0.0",
        "status": "running",
        "ml_models_available": ML_ENDPOINTS_AVAILABLE,
        "workflow_engine": "LangGraph",
        "endpoints": endpoints,
        "features": [
            "LangGraph-powered conversation workflow",
            "Interactive chat-based application processing",
            "Document processing and data extraction",
            "Real-time eligibility assessment with ML models",
            "LLM-generated economic enablement recommendations",
            "Scikit-learn ML classification models",
            "Fraud detection and risk assessment",
            "State-based conversation management"
        ],
        "deprecated_endpoints": [
            "/applications/{application_id}/process - Use /conversation/message instead",
            "/applications/process-with-data - Use /conversation/message instead"
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


@app.post("/applications/{application_id}/process")
async def process_application(
    application_id: str,
    db: Session = Depends(get_db)
):
    """Process an application through the AI workflow - DEPRECATED: Use conversation endpoints instead"""
    
    raise HTTPException(
        status_code=410, 
        detail="This endpoint is deprecated. Use /conversation/message for interactive processing with LangGraph workflow."
    )


@app.get("/applications/{application_id}/status")
async def get_application_status(application_id: str, db: Session = Depends(get_db)):
    """Get application status by application ID"""
    
    try:
        application = DatabaseManager.get_application_by_id(db, application_id)
        
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Update status check tracking
        DatabaseManager.update_status_check(db, application_id)
        
        return {
            "application_id": application.application_id,
            "reference_number": application.reference_number,
            "status": application.status.value,
            "full_name": application.full_name,
            "submitted_at": application.created_at.isoformat() if application.created_at else None,
            "processed_at": application.decision_date.isoformat() if application.decision_date else None,
            "is_eligible": application.is_eligible,
            "support_amount": application.recommended_support_amount,
            "last_updated": application.updated_at.isoformat() if application.updated_at else None
        }
        
    except Exception as e:
        logger.error(f"Error getting application status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/applications/lookup")
async def lookup_application(lookup_data: dict, db: Session = Depends(get_db)):
    """Flexible application lookup using multiple methods"""
    
    try:
        application = None
        lookup_method = None
        
        # Method 1: Reference Number
        if lookup_data.get("reference_number"):
            application = DatabaseManager.get_application_by_reference(db, lookup_data["reference_number"])
            lookup_method = "reference_number"
        
        # Method 2: Emirates ID
        elif lookup_data.get("emirates_id"):
            application = DatabaseManager.get_application_by_emirates_id(db, lookup_data["emirates_id"])
            lookup_method = "emirates_id"
        
        # Method 3: Name + Phone Last 4
        elif lookup_data.get("name") and lookup_data.get("phone_last4"):
            application = DatabaseManager.get_application_by_phone_and_name(
                db, lookup_data["phone_last4"], lookup_data["name"]
            )
            lookup_method = "name_phone"
        
        # Method 4: Full Application ID (fallback)
        elif lookup_data.get("application_id"):
            application = DatabaseManager.get_application_by_id(db, lookup_data["application_id"])
            lookup_method = "application_id"
        
        if not application:
            return {
                "found": False,
                "message": "Application not found. Please check your information and try again.",
                "lookup_method": lookup_method
            }
        
        # Update status check tracking
        DatabaseManager.update_status_check(db, application.application_id)
        
        return {
            "found": True,
            "lookup_method": lookup_method,
            "application": {
                "application_id": application.application_id,
                "reference_number": application.reference_number,
                "full_name": application.full_name,
                "status": application.status.value,
                "submitted_at": application.created_at.isoformat() if application.created_at else None,
                "decision_date": application.decision_date.isoformat() if application.decision_date else None,
                "is_eligible": application.is_eligible,
                "support_amount": application.recommended_support_amount,
                "eligibility_reason": application.eligibility_reason,
                "last_status_check": application.last_status_check.isoformat() if application.last_status_check else None,
                "status_check_count": application.status_check_count or 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in application lookup: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Process application with provided data - DEPRECATED: Use conversation endpoints instead"""
    
    raise HTTPException(
        status_code=410, 
        detail="This endpoint is deprecated. Use /conversation/message for interactive processing with LangGraph workflow."
    )


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
    """Process a conversation message through the LangGraph workflow"""
    
    try:
        user_message = request.get("message", "")
        conversation_history = request.get("conversation_history", [])
        conversation_state = request.get("conversation_state", {})
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"Processing conversation message via LangGraph: '{user_message}'")
        
        # Create LangGraph workflow
        workflow = create_conversation_workflow()
        
        # Convert frontend state to LangGraph state
        langgraph_state = convert_frontend_state_to_langgraph(
            conversation_state, 
            conversation_history, 
            user_message
        )
        
        # Process through LangGraph workflow
        result_state = await workflow.ainvoke(
            langgraph_state, 
            config={"recursion_limit": 100}  # Increased from default 25 to 100
        )
        
        # Convert LangGraph state back to frontend format
        frontend_response = convert_langgraph_state_to_frontend(result_state)
        
        # Convert numpy types to Python native types
        clean_response = convert_numpy_types(frontend_response)
        
        logger.info(f"LangGraph workflow completed. Status: {result_state.get('processing_status')}")
        
        return {
            "status": "success",
            **clean_response
        }
        
    except Exception as e:
        logger.error(f"Error processing conversation message via LangGraph: {str(e)}")
        
        # Fallback to direct conversation agent if LangGraph fails
        try:
            logger.info("Falling back to direct conversation agent")
            conversation_agent = ConversationAgent()
            
            response = await conversation_agent.process_message(
                user_message,
                conversation_history,
                conversation_state
            )
            
            clean_response = convert_numpy_types(response)
            
            return {
                "status": "success",
                "fallback_used": True,
                **clean_response
            }
            
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.post("/conversation/upload-document")
async def process_conversation_document(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    conversation_state: str = Form("{}")
):
    """Process uploaded document during conversation using LangGraph workflow"""
    
    try:
        logger.info(f"Document upload started via LangGraph - File: {file.filename}, Type: {file_type}, Size: {file.size}")
        
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
        
        # Create LangGraph workflow
        workflow = create_conversation_workflow()
        
        # Convert frontend state to LangGraph state and add document
        langgraph_state = convert_frontend_state_to_langgraph(state, [], "")
        
        # Add uploaded document to state
        if "uploaded_documents" not in langgraph_state:
            langgraph_state["uploaded_documents"] = []
        langgraph_state["uploaded_documents"].append(file_path)
        
        # Process through LangGraph workflow (will trigger document processing)
        result_state = await workflow.ainvoke(
            langgraph_state,
            config={"recursion_limit": 100}  # Increased from default 25 to 100
        )
        
        # Convert LangGraph state back to frontend format
        frontend_response = convert_langgraph_state_to_frontend(result_state)
        
        # Convert numpy types to Python native types
        clean_response = convert_numpy_types(frontend_response)
        
        result = {
            "status": "success",
            "file_path": file_path,
            **clean_response
        }
        
        logger.info(f"Document upload via LangGraph completed successfully for: {file.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing document upload via LangGraph: {str(e)}", exc_info=True)
        
        # Fallback to direct conversation agent
        try:
            logger.info("Falling back to direct conversation agent for document processing")
            conversation_agent = ConversationAgent()
            
            response = await conversation_agent.process_document_upload(
                file_path,
                file_type,
                state
            )
            
            clean_response = convert_numpy_types(response)
            
            return {
                "status": "success",
                "file_path": file_path,
                "fallback_used": True,
                **clean_response
            }
            
        except Exception as fallback_error:
            logger.error(f"Document processing fallback also failed: {str(fallback_error)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


def convert_frontend_state_to_langgraph(conversation_state: Dict, conversation_history: List[Dict], user_input: str) -> Dict:
    """Convert frontend conversation state to LangGraph state format"""
    
    # Start with conversation history
    messages = conversation_history.copy() if conversation_history else []
    
    # Add user input as latest message if provided
    if user_input:
        messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
    
    # CRITICAL FIX: Determine current step more intelligently
    current_step = conversation_state.get("current_step")
    processing_status = conversation_state.get("processing_status")
    
    # If we have an eligibility result or final decision, we should be in completion
    if (conversation_state.get("eligibility_result") or 
        conversation_state.get("final_decision") or 
        processing_status in ["completed", "completion_chat"]):
        current_step = "completion"
        logger.info(f"🎯 State conversion: Detected completion state, setting current_step to 'completion'")
    elif not current_step:
        # Only default to name_collection if we truly have no state
        current_step = "name_collection"
        logger.info(f"🎯 State conversion: No current_step found, defaulting to 'name_collection'")
    else:
        logger.info(f"🎯 State conversion: Using provided current_step: '{current_step}'")
    
    # Create LangGraph state
    langgraph_state = {
        "messages": messages,
        "collected_data": conversation_state.get("collected_data", {}),
        "current_step": current_step,
        "eligibility_result": conversation_state.get("eligibility_result"),
        "final_decision": conversation_state.get("final_decision"),
        "uploaded_documents": conversation_state.get("uploaded_documents", []),
        "workflow_history": conversation_state.get("workflow_history", []),
        "application_id": conversation_state.get("application_id"),
        "processing_status": processing_status or "in_progress",
        "error_messages": conversation_state.get("error_messages", []),
        "user_input": user_input if user_input else None,
        "last_agent_response": None
    }
    
    logger.info(f"🎯 State conversion complete: step='{current_step}', status='{processing_status}', has_eligibility={bool(conversation_state.get('eligibility_result'))}")
    
    return langgraph_state


def convert_langgraph_state_to_frontend(langgraph_state: Dict) -> Dict:
    """Convert LangGraph state back to frontend response format"""
    
    # CRITICAL FIX: Check for last_agent_response first (used by completion chat)
    latest_message = langgraph_state.get("last_agent_response")
    
    # If no last_agent_response, get the latest assistant message from messages
    if not latest_message:
        assistant_messages = [msg for msg in langgraph_state.get("messages", []) if msg["role"] == "assistant"]
        latest_message = assistant_messages[-1]["content"] if assistant_messages else ""
    
    # Add debugging to track which response source is used
    response_source = "last_agent_response" if langgraph_state.get("last_agent_response") else "messages_array"
    logger.info(f"🎯 Frontend conversion: Using response from '{response_source}', length: {len(latest_message)} chars")
    
    # Prepare state update for frontend
    state_update = {
        "current_step": langgraph_state.get("current_step"),
        "collected_data": langgraph_state.get("collected_data", {}),
        "uploaded_documents": langgraph_state.get("uploaded_documents", []),
        "eligibility_result": langgraph_state.get("eligibility_result"),
        "final_decision": langgraph_state.get("final_decision"),
        "application_id": langgraph_state.get("application_id"),
        "processing_status": langgraph_state.get("processing_status")
    }
    
    # Check if application is complete
    application_complete = (
        langgraph_state.get("processing_status") == "completed" or 
        langgraph_state.get("current_step") == "completion"
    )
    
    response = {
        "message": latest_message,
        "state_update": state_update,
        "application_complete": application_complete,
        "workflow_history": langgraph_state.get("workflow_history", []),
        "error_messages": langgraph_state.get("error_messages", [])
    }
    
    if application_complete:
        response["final_decision"] = langgraph_state.get("final_decision")
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    ) 