"""
Application configuration
Reads settings from environment variables
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Service Recommendation System"
    DEBUG: bool = False
    PORT: int = 8080
    ENABLE_CRON: bool = True
    
    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "root"
    DB_NAME: str = "compositions"
    
    # SSL
    SSL_ENABLED: bool = False
    SSL_KEY_PATH: Optional[str] = "/certs/privkey.pem"
    SSL_CERT_PATH: Optional[str] = "/certs/fullchain.pem"
    
    # External API
    CRIS_BASE_URL: str = "http://cris.icc.ru"
    API_TIMEOUT: int = 90
    
    # Paths
    CSV_FILE_PATH: str = "app/static/calls.csv"
    RECOMMENDATIONS_FILE_PATH: str = "app/static/recomendations.json"
    KNN_SCRIPT_PATH: str = "app/static/knn.py"
    
    # Recommendations
    RECOMMENDATION_CACHE_TTL: int = 3600  # 1 hour
    RECOMMENDATION_MODEL_REFRESH_INTERVAL: int = 86400  # 24 hours
    RECOMMENDATION_DEFAULT_ALGORITHM: str = "knn"
    RECOMMENDATION_FALLBACK_ALGORITHM: str = "popularity"
    RECOMMENDATION_MIN_USER_CALLS: int = 3  # Minimum calls for KNN
    KNN_N_NEIGHBORS: int = 4
    KNN_METRIC: str = "cosine"
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL"""
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Construct sync database URL (for Alembic migrations)"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()

