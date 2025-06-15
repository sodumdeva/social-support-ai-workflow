"""
Database Models for Social Support AI Workflow

PostgreSQL database schema for social support applications including:
- Applications: Core application data with user information and assessment results
- Documents: Document storage with processing status and extracted data  
- ApplicationReviews: Decision audit trail with reviewer information
- MLPredictions: Machine learning model predictions and confidence scores

Features comprehensive audit trails, ML prediction storage, and document management.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.types import TypeDecorator, VARCHAR
from sqlalchemy.pool import QueuePool
import enum

# Enhanced database URL for PostgreSQL with connection pooling
DATABASE_URL = "postgresql://localhost:5432/social_support_ai"

# Create database engine with connection pooling for production
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

# Enums for better data integrity
class ApplicationStatus(enum.Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"

class DocumentType(enum.Enum):
    EMIRATES_ID = "emirates_id"
    BANK_STATEMENT = "bank_statement"
    SALARY_CERTIFICATE = "salary_certificate"
    RESUME = "resume"
    ASSETS_LIABILITIES = "assets_liabilities"
    CREDIT_REPORT = "credit_report"
    FAMILY_CERTIFICATE = "family_certificate"
    HOUSING_CONTRACT = "housing_contract"

class EmploymentStatus(enum.Enum):
    EMPLOYED = "employed"
    UNEMPLOYED = "unemployed"
    SELF_EMPLOYED = "self_employed"
    RETIRED = "retired"
    STUDENT = "student"

class HousingStatus(enum.Enum):
    OWNED = "owned"
    RENTED = "rented"
    FAMILY_OWNED = "family_owned"
    GOVERNMENT_HOUSING = "government_housing"


class Application(Base):
    """
    Application Model for Social Support System
    
    Stores complete application data including personal information, employment details,
    family information, financial data, and assessment results. Integrates with
    LangGraph workflow and ML model predictions.
    """
    __tablename__ = "applications"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(50), unique=True, index=True, nullable=False)
    
    # Personal Information
    full_name = Column(String(200), nullable=False)
    emirates_id = Column(String(20), unique=True, index=True)
    phone_number = Column(String(20))
    email = Column(String(100))
    date_of_birth = Column(DateTime)
    nationality = Column(String(50))
    
    # Employment Information
    employment_status = Column(Enum(EmploymentStatus), nullable=False)
    employer_name = Column(String(200))
    job_title = Column(String(100))
    monthly_income = Column(Float, nullable=False, default=0.0)
    employment_duration_months = Column(Integer)
    
    # Family Information
    family_size = Column(Integer, nullable=False, default=1)
    dependents_count = Column(Integer, default=0)
    spouse_employment_status = Column(Enum(EmploymentStatus))
    spouse_monthly_income = Column(Float, default=0.0)
    
    # Housing Information
    housing_status = Column(Enum(HousingStatus), nullable=False)
    monthly_rent = Column(Float, default=0.0)
    housing_allowance = Column(Float, default=0.0)
    
    # Financial Information
    total_assets = Column(Float, default=0.0)
    total_liabilities = Column(Float, default=0.0)
    monthly_expenses = Column(Float, default=0.0)
    savings_amount = Column(Float, default=0.0)
    credit_score = Column(Integer)
    
    # Application Status and Processing
    status = Column(Enum(ApplicationStatus), default=ApplicationStatus.DRAFT)
    submission_date = Column(DateTime)
    review_date = Column(DateTime)
    decision_date = Column(DateTime)
    
    # Eligibility Results
    is_eligible = Column(Boolean)
    eligibility_score = Column(Float)
    recommended_support_amount = Column(Float)
    eligibility_reason = Column(Text)
    
    # ML Model Results
    ml_eligibility_prediction = Column(Boolean)
    ml_support_amount_prediction = Column(Float)
    ml_model_confidence = Column(Float)
    ml_features_used = Column(JSON)
    
    # Economic Enablement
    recommended_programs = Column(JSON)  # List of recommended training/job programs
    enablement_recommendations = Column(Text)
    
    # Audit Trail
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    updated_by = Column(String(100))
    
    # Additional metadata
    application_metadata = Column(JSON)  # Flexible field for additional data
    conversation_history = Column(JSON)  # Store chat conversation
    
    # User-friendly reference system
    reference_number = Column(String(20), unique=True, index=True)  # Short, memorable reference
    phone_reference = Column(String(15), index=True)  # Last 4 digits of phone for lookup
    sms_notification_sent = Column(Boolean, default=False)
    email_notification_sent = Column(Boolean, default=False)
    
    # Status tracking enhancements
    last_status_check = Column(DateTime)
    status_check_count = Column(Integer, default=0)
    
    # Relationships
    documents = relationship("Document", back_populates="application", cascade="all, delete-orphan")
    reviews = relationship("ApplicationReview", back_populates="application", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Application(id={self.id}, name={self.full_name}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert application to dictionary for API responses"""
        return {
            "id": self.id,
            "application_id": self.application_id,
            "full_name": self.full_name,
            "employment_status": self.employment_status.value if self.employment_status else None,
            "monthly_income": self.monthly_income,
            "family_size": self.family_size,
            "housing_status": self.housing_status.value if self.housing_status else None,
            "status": self.status.value if self.status else None,
            "is_eligible": self.is_eligible,
            "recommended_support_amount": self.recommended_support_amount,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Document(Base):
    """Enhanced Document model with metadata and processing status"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(50), unique=True, index=True, nullable=False)
    
    # Document Information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    document_type = Column(Enum(DocumentType), nullable=False)
    file_path = Column(String(500))
    file_size = Column(Integer)
    mime_type = Column(String(100))
    
    # Processing Information
    is_processed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")
    processing_error = Column(Text)
    extracted_text = Column(Text)
    extracted_data = Column(JSON)  # Structured data extracted from document
    
    # OCR and Analysis Results
    ocr_confidence = Column(Float)
    document_quality_score = Column(Float)
    validation_results = Column(JSON)
    
    # Relationships
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False)
    application = relationship("Application", back_populates="documents")
    
    # Audit Trail
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    uploaded_by = Column(String(100))
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, type={self.document_type})>"


class ApplicationReview(Base):
    """Application review and decision audit trail"""
    __tablename__ = "application_reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Review Information
    reviewer_id = Column(String(100))
    reviewer_name = Column(String(200))
    review_type = Column(String(50))  # "automated", "manual", "appeal"
    
    # Decision Information
    decision = Column(String(50))  # "approved", "rejected", "needs_more_info"
    decision_reason = Column(Text)
    support_amount_recommended = Column(Float)
    conditions = Column(JSON)  # Any conditions attached to approval
    
    # Review Details
    review_notes = Column(Text)
    risk_assessment = Column(JSON)
    compliance_check = Column(JSON)
    
    # Relationships
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False)
    application = relationship("Application", back_populates="reviews")
    
    # Audit Trail
    reviewed_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ApplicationReview(id={self.id}, decision={self.decision}, reviewer={self.reviewer_name})>"


class MLModelPerformance(Base):
    """Track ML model performance and predictions"""
    __tablename__ = "ml_model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model Information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    prediction_type = Column(String(50))  # "eligibility", "support_amount"
    
    # Prediction Details
    input_features = Column(JSON)
    prediction_result = Column(JSON)
    confidence_score = Column(Float)
    processing_time_ms = Column(Float)
    
    # Actual vs Predicted (for model evaluation)
    actual_result = Column(JSON)
    prediction_accuracy = Column(Float)
    
    # Reference to application
    application_id = Column(Integer, ForeignKey("applications.id"))
    
    # Audit Trail
    predicted_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MLModelPerformance(model={self.model_name}, confidence={self.confidence_score})>"


# Database session dependency
def get_db() -> Session:
    """Get database session with proper cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database initialization
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)


# Utility functions for database operations
class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def create_application(db: Session, application_data: Dict[str, Any]) -> Application:
        """Create new application record with proper enum conversion"""
        
        # Generate unique application ID
        application_id = f"APP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{application_data.get('emirates_id', 'UNKNOWN')[-4:]}"
        
        # Generate user-friendly reference number (6-digit format: YYMMDD)
        reference_number = f"SS{datetime.now().strftime('%y%m%d')}"
        
        # Ensure reference number is unique by adding counter if needed
        counter = 1
        base_reference = reference_number
        while db.query(Application).filter(Application.reference_number == reference_number).first():
            reference_number = f"{base_reference}{counter:02d}"
            counter += 1
        
        # Extract phone reference (last 4 digits)
        phone_reference = None
        phone_field = application_data.get("phone_number") or application_data.get("phone")
        if phone_field:
            phone_clean = ''.join(filter(str.isdigit, phone_field))
            if len(phone_clean) >= 4:
                phone_reference = phone_clean[-4:]
        
        # Convert string values to enums
        processed_data = application_data.copy()
        processed_data["application_id"] = application_id
        processed_data["reference_number"] = reference_number
        processed_data["phone_reference"] = phone_reference
        
        # Convert employment_status to enum
        if "employment_status" in processed_data:
            emp_status = processed_data["employment_status"]
            if isinstance(emp_status, str):
                processed_data["employment_status"] = _string_to_employment_status(emp_status)
        
        # Convert spouse_employment_status to enum
        if "spouse_employment_status" in processed_data:
            spouse_emp_status = processed_data["spouse_employment_status"]
            if isinstance(spouse_emp_status, str):
                processed_data["spouse_employment_status"] = _string_to_employment_status(spouse_emp_status)
        
        # Convert housing_status to enum
        if "housing_status" in processed_data:
            housing_status = processed_data["housing_status"]
            if isinstance(housing_status, str):
                processed_data["housing_status"] = _string_to_housing_status(housing_status)
        
        application = Application(**processed_data)
        
        db.add(application)
        db.commit()
        db.refresh(application)
        
        return application
    
    @staticmethod
    def get_application_by_id(db: Session, application_id: str) -> Optional[Application]:
        """Get application by application_id"""
        return db.query(Application).filter(Application.application_id == application_id).first()
    
    @staticmethod
    def get_application_by_reference(db: Session, reference_number: str) -> Optional[Application]:
        """Get application by user-friendly reference number"""
        return db.query(Application).filter(Application.reference_number == reference_number.upper()).first()
    
    @staticmethod
    def get_application_by_phone_and_name(db: Session, phone_last4: str, name: str) -> Optional[Application]:
        """Get application by phone last 4 digits and name (partial match)"""
        return db.query(Application).filter(
            Application.phone_reference == phone_last4,
            Application.full_name.ilike(f"%{name}%")
        ).first()
    
    @staticmethod
    def get_application_by_emirates_id(db: Session, emirates_id: str) -> Optional[Application]:
        """Get application by Emirates ID"""
        return db.query(Application).filter(Application.emirates_id == emirates_id).first()
    
    @staticmethod
    def update_status_check(db: Session, application_id: str) -> bool:
        """Update status check tracking"""
        application = DatabaseManager.get_application_by_id(db, application_id)
        if application:
            application.last_status_check = datetime.utcnow()
            application.status_check_count = (application.status_check_count or 0) + 1
            db.commit()
            return True
        return False
    
    @staticmethod
    def update_application_status(db: Session, application_id: str, status: ApplicationStatus) -> bool:
        """Update application status"""
        application = DatabaseManager.get_application_by_id(db, application_id)
        if application:
            application.status = status
            application.updated_at = datetime.utcnow()
            db.commit()
            return True
        return False
    
    @staticmethod
    def store_ml_prediction(db: Session, application_id: str, model_name: str, 
                          prediction_data: Dict[str, Any]) -> MLModelPerformance:
        """Store ML model prediction results"""
        
        application = DatabaseManager.get_application_by_id(db, application_id)
        
        ml_record = MLModelPerformance(
            model_name=model_name,
            application_id=application.id if application else None,
            **prediction_data
        )
        
        db.add(ml_record)
        db.commit()
        db.refresh(ml_record)
        
        return ml_record
    
    @staticmethod
    def get_applications_by_criteria(db: Session, criteria: Dict[str, Any]) -> List[Application]:
        """Get applications matching specific criteria"""
        query = db.query(Application)
        
        if "employment_status" in criteria:
            # Convert string to enum for proper querying
            emp_status = criteria["employment_status"]
            if isinstance(emp_status, str):
                emp_status = _string_to_employment_status(emp_status)
            query = query.filter(Application.employment_status == emp_status)
        
        if "income_range" in criteria:
            min_income, max_income = criteria["income_range"]
            query = query.filter(Application.monthly_income.between(min_income, max_income))
        
        if "family_size" in criteria:
            query = query.filter(Application.family_size == criteria["family_size"])
        
        if "status" in criteria:
            status = criteria["status"]
            if isinstance(status, str):
                # Convert string to enum if needed
                for status_enum in ApplicationStatus:
                    if status_enum.value == status:
                        status = status_enum
                        break
            query = query.filter(Application.status == status)
        
        return query.all()


def _string_to_employment_status(status_str: str) -> EmploymentStatus:
    """Convert string to EmploymentStatus enum"""
    if not status_str:
        return EmploymentStatus.UNEMPLOYED
    
    status_lower = status_str.lower()
    if "employ" in status_lower and "un" not in status_lower:
        return EmploymentStatus.EMPLOYED
    elif "self" in status_lower:
        return EmploymentStatus.SELF_EMPLOYED
    elif "retire" in status_lower:
        return EmploymentStatus.RETIRED
    elif "student" in status_lower:
        return EmploymentStatus.STUDENT
    else:
        return EmploymentStatus.UNEMPLOYED


def _string_to_housing_status(status_str: str) -> HousingStatus:
    """Convert string to HousingStatus enum"""
    if not status_str:
        return HousingStatus.RENTED
    
    status_lower = status_str.lower()
    if "own" in status_lower:
        return HousingStatus.OWNED
    elif "family" in status_lower:
        return HousingStatus.FAMILY_OWNED
    elif "government" in status_lower:
        return HousingStatus.GOVERNMENT_HOUSING
    else:
        return HousingStatus.RENTED


# Initialize database on import
if __name__ == "__main__":
    print("Creating database tables...")
    create_tables()
    print("Database tables created successfully!") 