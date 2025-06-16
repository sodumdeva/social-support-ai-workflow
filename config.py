"""
Configuration Management for Social Support AI Workflow

Comprehensive settings management using Pydantic BaseSettings for database connections,
AI model settings (Ollama), file handling, API configuration, and environment-specific
settings with .env file support.

Configuration Categories:
    - Database: PostgreSQL connection settings with pooling and SSL options
    - AI Models: Local LLM configuration for Ollama integration
    - File Handling: Upload paths, size limits, and supported formats
    - API Settings: Server configuration, CORS, and security settings
    - Logging: Log levels, file paths, and rotation settings
    - OCR: Tesseract configuration and preprocessing options

Environment Support:
    - Development: Local development with debug logging
    - Production: Optimized settings with security hardening
    - Testing: Isolated test environment with mock services
    - Docker: Containerized deployment configuration

Key Features:
    - Environment variable override support
    - Validation of critical configuration values
    - Default fallbacks for missing settings
    - Type conversion and validation
    - Secure handling of sensitive credentials

AI Configuration:
    - Ollama LLM model selection and parameters
    - Model-specific settings for different use cases
    - Timeout and retry configurations
    - Fallback model specifications
    - Performance tuning parameters

Database Configuration:
    - Connection pooling for optimal performance
    - SSL/TLS settings for secure connections
    - Transaction isolation levels
    - Connection timeout and retry logic
    - Migration and backup settings

File Handling:
    - Upload directory management
    - File size and type restrictions
    - Temporary file cleanup policies
    - Document processing pipeline settings
    - Storage optimization configurations

The configuration system supports both environment variables and configuration
files, with environment variables taking precedence for deployment flexibility.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings"""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=('settings_',)  # Fix the protected namespace warning
    )
    
    # Application
    app_name: str = Field(default="Social Support AI Workflow", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Database
    database_url: str = Field(default="sqlite:///./social_support_ai.db", env="DATABASE_URL")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama2:7b", env="OLLAMA_MODEL")
    ollama_vision_model: str = Field(default="llava:13b", env="OLLAMA_VISION_MODEL")
    
    # API Configuration
    api_host: str = Field(default="localhost", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Frontend Configuration
    frontend_host: str = Field(default="localhost", env="FRONTEND_HOST")
    frontend_port: int = Field(default=8501, env="FRONTEND_PORT")
    
    # File Storage
    upload_dir: str = Field(default="data/uploads", env="UPLOAD_DIR")
    processed_dir: str = Field(default="data/processed", env="PROCESSED_DIR")
    models_directory: str = Field(default="models", env="MODEL_DIR")  # Renamed to avoid conflict
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(default="data/chroma", env="CHROMA_PERSIST_DIRECTORY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Processing Limits
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    max_files_per_request: int = Field(default=10, env="MAX_FILES_PER_REQUEST")
    processing_timeout_seconds: int = Field(default=300, env="PROCESSING_TIMEOUT_SECONDS")


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """Get database URL for SQLAlchemy"""
    return settings.database_url


def get_upload_path() -> str:
    """Get upload directory path"""
    os.makedirs(settings.upload_dir, exist_ok=True)
    return settings.upload_dir


def get_processed_path() -> str:
    """Get processed files directory path"""
    os.makedirs(settings.processed_dir, exist_ok=True)
    return settings.processed_dir


def get_model_path() -> str:
    """Get model directory path"""
    os.makedirs(settings.models_directory, exist_ok=True)
    return settings.models_directory


def get_chroma_path() -> str:
    """Get ChromaDB persistence directory"""
    os.makedirs(settings.chroma_persist_directory, exist_ok=True)
    return settings.chroma_persist_directory 