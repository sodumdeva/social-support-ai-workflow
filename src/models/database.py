"""
Database models for Social Support AI Workflow
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import settings

Base = declarative_base()


class Application(Base):
    """Main application table"""
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(50), unique=True, index=True)
    
    # Applicant Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(20))
    emirates_id = Column(String(50))
    
    # Address Information
    address = Column(Text)
    city = Column(String(100))
    emirate = Column(String(50))
    
    # Financial Information
    monthly_income = Column(Float)
    employment_status = Column(String(50))
    family_size = Column(Integer)
    
    # Application Status
    status = Column(String(50), default="pending")  # pending, approved, declined, under_review
    priority_score = Column(Float)
    eligibility_score = Column(Float)
    
    # AI Processing Results
    ai_decision = Column(String(50))  # approve, decline, review_required
    ai_confidence = Column(Float)
    ai_reasoning = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    documents = relationship("Document", back_populates="application")
    assessments = relationship("Assessment", back_populates="application")


class Document(Base):
    """Uploaded documents table"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    
    # Document Information
    document_type = Column(String(50))  # bank_statement, emirates_id, resume, credit_report, assets
    filename = Column(String(255))
    file_path = Column(String(500))
    file_size = Column(Integer)
    mime_type = Column(String(100))
    
    # Processing Status
    processing_status = Column(String(50), default="uploaded")  # uploaded, processing, processed, failed
    extracted_data = Column(JSON)  # Extracted information from document
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Relationships
    application = relationship("Application", back_populates="documents")


class Assessment(Base):
    """AI assessment results table"""
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    
    # Assessment Details
    assessment_type = Column(String(50))  # eligibility, risk, recommendation
    agent_name = Column(String(100))  # Which AI agent performed the assessment
    
    # Scores and Results
    score = Column(Float)
    confidence = Column(Float)
    result = Column(String(50))
    reasoning = Column(Text)
    recommendations = Column(JSON)
    
    # Processing Information
    processing_time_ms = Column(Integer)
    model_used = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    application = relationship("Application", back_populates="assessments")


class ChatSession(Base):
    """Chat interaction sessions"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=True)
    
    # Session Information
    user_type = Column(String(50))  # applicant, staff, admin
    status = Column(String(50), default="active")  # active, completed, abandoned
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(Base):
    """Individual chat messages"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    
    # Message Content
    role = Column(String(20))  # user, assistant, system
    content = Column(Text)
    message_type = Column(String(50))  # text, file_upload, system_notification
    
    # AI Processing
    agent_used = Column(String(100))
    processing_time_ms = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


# Database setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 