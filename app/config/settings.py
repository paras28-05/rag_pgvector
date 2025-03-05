import logging
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configure logging
logging.basicConfig(filename="rag_system.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 3

class GeminiSettings(LLMSettings):
    """Gemini-specific settings."""
    api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    default_model: str = Field(default="gemini-2.0-flash")

class DatabaseSettings(BaseModel):
    """Database connection settings."""
    db_name: str = Field(default_factory=lambda: os.getenv("DB_NAME"))
    db_user: str = Field(default_factory=lambda: os.getenv("DB_USER"))
    db_password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD"))
    db_host: str = Field(default_factory=lambda: os.getenv("DB_HOST"))
    db_port: str = Field(default_factory=lambda: os.getenv("DB_PORT"))

class Settings(BaseModel):
    """Main settings class combining all sub-settings."""
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

def get_settings() -> Settings:
    """Create and return an instance of Settings."""
    settings = Settings()
    logging.info("✅ Application settings loaded successfully.")
    return settings
