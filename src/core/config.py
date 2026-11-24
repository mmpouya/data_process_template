from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str = "your-api-key"
    OPENAI_BASE_URL: str = "http://localhost:8000/v1"
    OPENAI_MODEL_NAME: str = "gemma-4"

    ENCODER_API_KEY: str = "your-api-key"
    ENCODER_BASE_URL: str = "http://localhost:8001/v1"
    ENCODER_MODEL_NAME: str = "bge-m3"
    ENCODER_MODEL_DIM: int = 1024
    ENCODER_MAX_TOKEN: int = 8192

    DECISION_MODEL_PROVIDER: str = "h2o"
    DECISION_BASE_URL: str = "http://127.0.0.1:8741/v1"
    DECISION_MODEL_NAME: str = "h2oai/h2o-lightning-4b"
    DECISION_API_KEY: str = "none"
    DECISION_GROUNDING_ACCEPT_THRESHOLD: float = 0.85
    DECISION_GROUNDING_REJECT_THRESHOLD: float = 0.35
    DECISION_RELATION_GUARD_THRESHOLD: float = 0.90
    DECISION_RPM: int = 60

    QDRANT_URL: str | None = None
    SESSION_TTL: int = 3000
    QUESTION_TIMEOUT: int = 900
    TOOL_TIMEOUT: int = 120
    DELEGATE_TIMEOUT: int = 600
    request_priority: int | None = None
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
