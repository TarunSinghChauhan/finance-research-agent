from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openrouter_api_key: str = ""
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "finance-research-agent"

    database_url: str = "postgresql+asyncpg://finance_user:finance_pass@localhost:5432/finance_db"
    redis_url: str = "redis://localhost:6379/0"

    max_cost_per_query_usd: float = 0.50
    agent_timeout_seconds: int = 120
    cache_ttl_seconds: int = 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()
