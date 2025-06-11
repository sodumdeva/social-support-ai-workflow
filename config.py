"""
Configuration management for Social Support AI Workflow
"""
import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = Field(default="Social Support AI Workflow", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Database
    database_url: str = Field(
        default="postgresql://social_support_user:password@localhost:5432/social_support_db",
        env="DATABASE_URL"
    )
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama2:7b", env="OLLAMA_MODEL")
    ollama_vision_model: str = Field(default="llava:7b", env="OLLAMA_VISION_MODEL")
    
    # ChromaDB
    chromadb_path: str = Field(default="./data/chromadb", env="CHROMADB_PATH")
    
    # File Upload
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    max_upload_size: int = Field(default=10485760, env="MAX_UPLOAD_SIZE")  # 10MB
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    secret_key: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    # LangSmith (Optional)
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="social-support-ai", env="LANGCHAIN_PROJECT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
os.makedirs(settings.chromadb_path, exist_ok=True) 