from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional
import uuid

from src.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, pool_size=10)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


class ResearchReport(SQLModel, table=True):
    """Stored research report with full audit trail."""
    __tablename__ = "research_reports"

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], primary_key=True)
    company: str
    query: str
    status: str = "pending"
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Structured reasoning chain stored as JSON
    market_context: Optional[str] = None
    financial_analysis: Optional[str] = None
    risk_assessment: Optional[str] = None
    synthesis: Optional[str] = None
    final_report: Optional[str] = None

    # Tool calls audit trail
    tool_calls: list = Field(default_factory=list, sa_column=Column(JSON))
    reproducibility_hash: Optional[str] = None
