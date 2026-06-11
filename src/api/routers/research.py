import json
import hashlib
import math
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import redis.asyncio as aioredis

from src.agents.research import FinancialResearchAgent
from src.core.config import get_settings
from src.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

_reports: dict[str, dict] = {}
_report_status: dict[str, str] = {}


def clean_nans(obj):
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj


class ResearchRequest(BaseModel):
    company: str
    symbol: str
    query: str = "Provide a comprehensive investment analysis"


SAMPLE_COMPANIES = [
    {"company": "Apple Inc", "symbol": "AAPL", "query": "Comprehensive investment analysis for Apple"},
    {"company": "Microsoft Corporation", "symbol": "MSFT", "query": "Investment thesis for Microsoft"},
    {"company": "NVIDIA Corporation", "symbol": "NVDA", "query": "AI growth analysis for NVIDIA"},
    {"company": "Tesla Inc", "symbol": "TSLA", "query": "Risk and opportunity analysis for Tesla"},
]


async def _run_research(report_id: str, request: ResearchRequest):
    agent = FinancialResearchAgent()
    try:
        _report_status[report_id] = "running"
        result = await agent.research(
            company=request.company,
            symbol=request.symbol,
            query=request.query,
        )
        result["report_id"] = report_id
        _reports[report_id] = clean_nans(result)
        _report_status[report_id] = "completed"
        logger.info("report_completed", report_id=report_id, company=request.company)
    except Exception as e:
        import traceback
        logger.error("report_failed", report_id=report_id, error=str(e), traceback=traceback.format_exc())
        _report_status[report_id] = f"failed: {str(e)}"


@router.post("/analyze")
async def analyze_company(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a financial research analysis for a company."""
    import uuid
    report_id = str(uuid.uuid4())[:8]
    _report_status[report_id] = "pending"
    background_tasks.add_task(_run_research, report_id, request)
    return {
        "report_id": report_id,
        "status": "pending",
        "company": request.company,
        "symbol": request.symbol,
        "message": f"Research started. Poll /research/status/{report_id} for updates.",
    }


@router.get("/status/{report_id}")
async def get_status(report_id: str):
    if report_id not in _report_status:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report_id": report_id, "status": _report_status[report_id]}


@router.get("/results/{report_id}")
async def get_results(report_id: str):
    if report_id not in _report_status:
        raise HTTPException(status_code=404, detail="Report not found")
    if _report_status[report_id] != "completed":
        raise HTTPException(status_code=202, detail=f"Status: {_report_status[report_id]}")
    return JSONResponse(content=_reports[report_id])


@router.get("/audit/{report_id}")
async def get_audit_trail(report_id: str):
    """Get the full tool call audit trail for a report."""
    if report_id not in _reports:
        raise HTTPException(status_code=404, detail="Report not found")
    report = _reports[report_id]
    return {
        "report_id": report_id,
        "company": report.get("company"),
        "reproducibility_hash": report.get("reproducibility_hash"),
        "total_cost_usd": report.get("total_cost_usd"),
        "total_tokens": report.get("total_tokens"),
        "tool_calls": report.get("tool_calls", []),
        "within_budget": report.get("within_budget"),
        "completed_at": report.get("completed_at"),
    }


@router.get("/reports")
async def list_reports():
    return {"reports": [{"report_id": k, "status": v} for k, v in _report_status.items()]}


@router.get("/companies/sample")
async def sample_companies():
    """Get sample companies to research."""
    return {"companies": SAMPLE_COMPANIES}


@router.post("/analyze/sample")
async def analyze_sample(background_tasks: BackgroundTasks):
    """Run a sample analysis on Apple Inc."""
    import uuid
    report_id = str(uuid.uuid4())[:8]
    request = ResearchRequest(
        company="Apple Inc",
        symbol="AAPL",
        query="Comprehensive investment analysis including growth prospects and risks",
    )
    _report_status[report_id] = "pending"
    background_tasks.add_task(_run_research, report_id, request)
    return {
        "report_id": report_id,
        "status": "pending",
        "company": "Apple Inc",
        "symbol": "AAPL",
        "message": f"Sample research started. Poll /research/status/{report_id}",
    }
