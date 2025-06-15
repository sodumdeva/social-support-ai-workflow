"""
Database models and configuration for Social Support AI Workflow

SQLAlchemy models for:
- Application: Main application records
- Document: Uploaded document records
- Database session management
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.types import TypeDecorator, VARCHAR

# Import configuration
from config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()


class JSONType(TypeDecorator):
    """Custom JSON type for SQLAlchemy that handles serialization"""
    impl = VARCHAR
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        return value


class Application(Base):
    """Application model for social support applications"""
    
    __tablename__ = "applications"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(50), unique=True, index=True, nullable=False)
    
    # Personal information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    emirates_id = Column(String(20), nullable=True)
    
    # Financial information
    monthly_income = Column(Float, nullable=True)
    employment_status = Column(String(50), nullable=True)
    family_size = Column(Integer, default=1)
    
    # Application status and processing
    status = Column(String(50), default="submitted")  # submitted, processing, completed, approved, declined
    submitted_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Assessment results
    eligibility_score = Column(Float, nullable=True)
    is_eligible = Column(Boolean, nullable=True)
    support_amount = Column(Float, nullable=True)
    
    # Store complex assessment data as JSON
    assessment_data = Column(JSONType, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="application", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Application(id={self.application_id}, name={self.first_name} {self.last_name}, status={self.status})>"


class Document(Base):
    """Document model for uploaded application documents"""
    
    __tablename__ = "documents"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to application
    application_id = Column(String(50), ForeignKey("applications.application_id"), nullable=False)
    
    # Document information
    document_type = Column(String(50), nullable=False)  # bank_statement, emirates_id, resume, etc.
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # Processing status
    processing_status = Column(String(50), default="uploaded")  # uploaded, processing, processed, error
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    
    # Extraction results stored as JSON
    extraction_results = Column(JSONType, nullable=True)
    
    # Relationships
    application = relationship("Application", back_populates="documents")
    
    def __repr__(self):
        return f"<Document(id={self.id}, type={self.document_type}, app_id={self.application_id})>"


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """
    Dependency function to get database session
    Used by FastAPI endpoints with Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize database with tables"""
    try:
        create_tables()
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {str(e)}")
        return False


# Database utility functions
def get_application_by_id(db: Session, application_id: str) -> Optional[Application]:
    """Get application by application_id"""
    return db.query(Application).filter(Application.application_id == application_id).first()


def get_applications_by_status(db: Session, status: str, limit: int = 100) -> list[Application]:
    """Get applications by status"""
    return db.query(Application).filter(Application.status == status).limit(limit).all()


def update_application_status(db: Session, application_id: str, status: str, assessment_data: Dict[str, Any] = None) -> bool:
    """Update application status and assessment data"""
    try:
        application = get_application_by_id(db, application_id)
        if not application:
            return False
        
        application.status = status
        application.processed_at = datetime.utcnow()
        
        if assessment_data:
            application.assessment_data = assessment_data
            application.eligibility_score = assessment_data.get("eligibility_score")
            application.is_eligible = assessment_data.get("is_eligible")
            application.support_amount = assessment_data.get("support_amount")
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error updating application status: {str(e)}")
        return False


def get_documents_by_application(db: Session, application_id: str) -> list[Document]:
    """Get all documents for an application"""
    return db.query(Document).filter(Document.application_id == application_id).all()


def update_document_processing_status(db: Session, document_id: int, status: str, extraction_results: Dict[str, Any] = None) -> bool:
    """Update document processing status and results"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return False
        
        document.processing_status = status
        document.processed_at = datetime.utcnow()
        
        if extraction_results:
            document.extraction_results = extraction_results
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error updating document status: {str(e)}")
        return False


# Initialize database on import
if __name__ == "__main__":
    init_database() 