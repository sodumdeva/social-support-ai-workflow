"""
Logging Configuration for Social Support AI Workflow

Advanced logging setup with structured logging, workflow tracking, and demo-friendly
output formatting. Provides standard loggers and specialized WorkflowLogger for
monitoring application pipeline with color-coded console output.
"""
import logging
import sys
import os
from datetime import datetime
from pathlib import Path


class DemoFormatter(logging.Formatter):
    """Custom formatter for demo-friendly output"""
    
    def format(self, record):
        # Add colors and emojis for different log levels
        level_colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green  
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m'  # Magenta
        }
        
        level_emojis = {
            'DEBUG': '🔍',
            'INFO': '✅', 
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        reset_color = '\033[0m'
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Get color and emoji
        color = level_colors.get(record.levelname, '')
        emoji = level_emojis.get(record.levelname, '')
        
        # Create formatted message
        if hasattr(record, 'demo_step'):
            # Special formatting for demo steps
            formatted = f"{color}🎯 [{timestamp}] {record.demo_step}: {record.getMessage()}{reset_color}"
        elif hasattr(record, 'workflow_step'):
            # Workflow step formatting
            formatted = f"{color}🔄 [{timestamp}] {record.workflow_step} - {record.getMessage()}{reset_color}"
        else:
            # Standard formatting
            formatted = f"{color}{emoji} [{timestamp}] {record.getMessage()}{reset_color}"
        
        return formatted


def setup_logging(log_level: str = "INFO", log_to_file: bool = True, log_to_console: bool = True):
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Demo formatter for console
    demo_formatter = DemoFormatter()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with demo formatting
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(demo_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with detailed formatting
    if log_to_file:
        log_filename = f"logs/social_support_ai_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)  # Reduce uvicorn noise
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    
    # Create application-specific loggers
    app_logger = logging.getLogger("social_support_ai")
    app_logger.setLevel(getattr(logging, log_level.upper()))
    
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(f"social_support_ai.{name}")


def get_demo_logger(name: str) -> logging.Logger:
    """Get a demo-specific logger with enhanced formatting"""
    logger = logging.getLogger(f"demo.{name}")
    return logger


class WorkflowLogger:
    """Enhanced logger for workflow demonstrations"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.demo_logger = get_demo_logger(name)
    
    def log_step(self, step_name: str, message: str, level: str = "INFO"):
        """Log a workflow step with special formatting"""
        record = logging.LogRecord(
            name=self.demo_logger.name,
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.demo_step = step_name
        self.demo_logger.handle(record)
    
    def log_workflow(self, workflow_step: str, message: str, level: str = "INFO"):
        """Log a workflow action with special formatting"""
        record = logging.LogRecord(
            name=self.demo_logger.name,
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.workflow_step = workflow_step
        self.demo_logger.handle(record)
    
    def log_llm_call(self, model: str, prompt_type: str, processing_time: float = None):
        """Log LLM model calls"""
        time_str = f" ({processing_time:.1f}ms)" if processing_time else ""
        self.log_step("LLM_CALL", f"🤖 Calling {model} for {prompt_type}{time_str}")
    
    def log_document_processing(self, doc_type: str, filename: str, status: str):
        """Log document processing steps"""
        self.log_step("DOCUMENT_PROCESSING", f"📄 Processing {doc_type}: {filename} - {status}")
    
    def log_data_verification(self, doc_type: str, matches: int, mismatches: int, score: float):
        """Log data verification results"""
        self.log_step("DATA_VERIFICATION", f"🔍 {doc_type} verification: {matches} matches, {mismatches} mismatches (Score: {score:.2f})")
    
    def log_eligibility_assessment(self, eligible: bool, amount: float, confidence: float):
        """Log eligibility assessment results"""
        status = "ELIGIBLE" if eligible else "NOT ELIGIBLE"
        self.log_step("ELIGIBILITY", f"⚖️ Assessment: {status} - Amount: {amount} AED (Confidence: {confidence:.3f})")
    
    def log_database_operation(self, operation: str, status: str, details: str = ""):
        """Log database operations"""
        self.log_step("DATABASE", f"💾 {operation}: {status} {details}")


# Setup default logging
setup_logging(log_level="INFO", log_to_file=True, log_to_console=True) 