"""
FastAPI main application for Social Support AI Workflow

Provides REST API endpoints for:
- Application submission
- Document upload
- Application processing
- Status checking
- Results retrieval
"""
import os
import uuid
from typing import List, Optional
from datetime import datetime
import shutil

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings, get_upload_path
from src.models.database import get_db, Application, Document
from src.agents.master_orchestrator import MasterOrchestrator
from src.data.synthetic_data import SyntheticDataGenerator


# Pydantic models for API
class ApplicationData(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    emirates_id: Optional[str] = None
    monthly_income: Optional[float] = None
    employment_status: Optional[str] = None
    employment_duration_months: Optional[int] = None
    family_size: int = 1
    number_of_dependents: int = 0
    housing_type: Optional[str] = None
    monthly_rent: Optional[float] = None
    education_level: Optional[str] = None
    has_medical_conditions: bool = False
    has_criminal_record: bool = False
    previous_applications: int = 0


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
    description="AI-powered social support application processing system",
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

# Initialize agents
orchestrator = MasterOrchestrator()
synthetic_generator = SyntheticDataGenerator()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Social Support AI Workflow API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "submit_application": "/applications/submit",
            "upload_documents": "/applications/{application_id}/documents",
            "process_application": "/applications/{application_id}/process",
            "get_status": "/applications/{application_id}/status",
            "get_results": "/applications/{application_id}/results",
            "generate_synthetic": "/testing/generate-synthetic-data"
        }
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
            employment_duration_months=application_data.employment_duration_months,
            family_size=application_data.family_size,
            number_of_dependents=application_data.number_of_dependents,
            housing_type=application_data.housing_type,
            monthly_rent=application_data.monthly_rent,
            education_level=application_data.education_level,
            has_medical_conditions=application_data.has_medical_conditions,
            has_criminal_record=application_data.has_criminal_record,
            previous_applications=application_data.previous_applications,
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
            "employment_duration_months": application.employment_duration_months,
            "family_size": application.family_size,
            "number_of_dependents": application.number_of_dependents,
            "housing_type": application.housing_type,
            "monthly_rent": application.monthly_rent,
            "education_level": application.education_level,
            "has_medical_conditions": application.has_medical_conditions,
            "has_criminal_record": application.has_criminal_record,
            "previous_applications": application.previous_applications
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
            employment_duration_months=request.application_data.employment_duration_months,
            family_size=request.application_data.family_size,
            number_of_dependents=request.application_data.number_of_dependents,
            housing_type=request.application_data.housing_type,
            monthly_rent=request.application_data.monthly_rent,
            education_level=request.application_data.education_level,
            has_medical_conditions=request.application_data.has_medical_conditions,
            has_criminal_record=request.application_data.has_criminal_record,
            previous_applications=request.application_data.previous_applications,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    ) 