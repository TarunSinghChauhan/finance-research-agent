import math
import pytest
from fastapi import HTTPException

from src.api.routers import research as research_module
from src.api.routers.research import clean_nans, get_audit_trail, get_status, get_results


def test_clean_nans_replaces_nan():
    assert clean_nans(float("nan")) == 0.0


def test_clean_nans_replaces_inf():
    assert clean_nans(float("inf")) == 0.0


def test_clean_nans_leaves_normal_values():
    assert clean_nans({"cost": 0.0042, "company": "Apple"}) == {"cost": 0.0042, "company": "Apple"}


def test_clean_nans_handles_nested_structures():
    data = {"metrics": [{"score": float("nan")}, {"score": 8.0}]}
    result = clean_nans(data)
    assert result["metrics"][0]["score"] == 0.0
    assert result["metrics"][1]["score"] == 8.0


@pytest.mark.asyncio
async def test_get_status_raises_404_for_unknown_report(monkeypatch):
    monkeypatch.setattr(research_module, "_report_status", {})
    with pytest.raises(HTTPException) as exc_info:
        await get_status("nonexistent-id")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_status_returns_current_status(monkeypatch):
    monkeypatch.setattr(research_module, "_report_status", {"abc123": "running"})
    result = await get_status("abc123")
    assert result == {"report_id": "abc123", "status": "running"}


@pytest.mark.asyncio
async def test_get_results_raises_202_when_not_completed(monkeypatch):
    monkeypatch.setattr(research_module, "_report_status", {"abc123": "running"})
    with pytest.raises(HTTPException) as exc_info:
        await get_results("abc123")
    assert exc_info.value.status_code == 202


@pytest.mark.asyncio
async def test_get_results_raises_404_for_unknown_report(monkeypatch):
    monkeypatch.setattr(research_module, "_report_status", {})
    with pytest.raises(HTTPException) as exc_info:
        await get_results("nonexistent-id")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_audit_trail_raises_404_for_unknown_report(monkeypatch):
    monkeypatch.setattr(research_module, "_reports", {})
    with pytest.raises(HTTPException) as exc_info:
        await get_audit_trail("nonexistent-id")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_audit_trail_extracts_expected_fields(monkeypatch):
    fake_report = {
        "company": "Apple Inc",
        "reproducibility_hash": "abc123def456",
        "total_cost_usd": 0.0042,
        "total_tokens": 1200,
        "tool_calls": [{"tool": "get_stock_quote"}],
        "within_budget": True,
        "completed_at": "2026-08-18T00:00:00",
    }
    monkeypatch.setattr(research_module, "_reports", {"rep1": fake_report})
    result = await get_audit_trail("rep1")
    assert result["reproducibility_hash"] == "abc123def456"
    assert result["total_cost_usd"] == 0.0042
    assert result["tool_calls"] == [{"tool": "get_stock_quote"}]


@pytest.mark.asyncio
async def test_get_audit_trail_handles_missing_optional_fields(monkeypatch):
    minimal_report = {"company": "Tesla Inc"}
    monkeypatch.setattr(research_module, "_reports", {"rep2": minimal_report})
    result = await get_audit_trail("rep2")
    assert result["reproducibility_hash"] is None
    assert result["tool_calls"] == []


import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.responses import JSONResponse

from src.api.routers.research import (
    _run_research,
    _reports,
    _report_status,
    get_results,
    analyze_sample,
    ResearchRequest,
)


@pytest.mark.asyncio
async def test_run_research_success_stores_cleaned_report_and_marks_completed():
    report_id = str(uuid.uuid4())[:8]
    request = ResearchRequest(company="Apple Inc", symbol="AAPL", query="test")
    fake_result = {"company": "Apple Inc", "score": float("nan")}
    fake_agent = MagicMock()
    fake_agent.research = AsyncMock(return_value=fake_result)
    with patch("src.api.routers.research.FinancialResearchAgent", return_value=fake_agent):
        await _run_research(report_id, request)
    assert _report_status[report_id] == "completed"
    assert _reports[report_id]["report_id"] == report_id
    assert _reports[report_id]["score"] == 0.0  # NaN cleaned


@pytest.mark.asyncio
async def test_run_research_failure_sets_failed_status_with_error_message():
    report_id = str(uuid.uuid4())[:8]
    request = ResearchRequest(company="Apple Inc", symbol="AAPL", query="test")
    fake_agent = MagicMock()
    fake_agent.research = AsyncMock(side_effect=RuntimeError("model timeout"))
    with patch("src.api.routers.research.FinancialResearchAgent", return_value=fake_agent):
        await _run_research(report_id, request)
    assert _report_status[report_id] == "failed: model timeout"
    assert report_id not in _reports


@pytest.mark.asyncio
async def test_get_results_returns_json_response_when_completed():
    report_id = str(uuid.uuid4())[:8]
    _report_status[report_id] = "completed"
    _reports[report_id] = {"company": "Apple Inc", "report_id": report_id}
    result = await get_results(report_id)
    assert isinstance(result, JSONResponse)
    assert json.loads(result.body)["company"] == "Apple Inc"


@pytest.mark.asyncio
async def test_analyze_sample_schedules_apple_research_task():
    fake_bg_tasks = MagicMock()
    result = await analyze_sample(fake_bg_tasks)
    assert result["company"] == "Apple Inc"
    assert result["symbol"] == "AAPL"
    assert result["status"] == "pending"
    assert "report_id" in result
    fake_bg_tasks.add_task.assert_called_once()
    args = fake_bg_tasks.add_task.call_args[0]
    assert args[0] is _run_research
    scheduled_request = args[2]
    assert scheduled_request.company == "Apple Inc"
    assert scheduled_request.symbol == "AAPL"


from src.api.routers.research import analyze_company, list_reports, sample_companies


@pytest.mark.asyncio
async def test_analyze_company_schedules_task_with_given_request():
    fake_bg_tasks = MagicMock()
    request = ResearchRequest(company="Tesla Inc", symbol="TSLA", query="growth analysis")
    result = await analyze_company(request, fake_bg_tasks)
    assert result["company"] == "Tesla Inc"
    assert result["symbol"] == "TSLA"
    assert result["status"] == "pending"
    assert "report_id" in result
    fake_bg_tasks.add_task.assert_called_once()
    args = fake_bg_tasks.add_task.call_args[0]
    assert args[0] is _run_research
    assert args[2] is request


@pytest.mark.asyncio
async def test_list_reports_returns_all_tracked_statuses():
    report_id = str(uuid.uuid4())[:8]
    _report_status[report_id] = "running"
    result = await list_reports()
    matching = [r for r in result["reports"] if r["report_id"] == report_id]
    assert len(matching) == 1
    assert matching[0]["status"] == "running"


@pytest.mark.asyncio
async def test_sample_companies_returns_four_predefined_companies():
    result = await sample_companies()
    assert len(result["companies"]) == 4
    symbols = {c["symbol"] for c in result["companies"]}
    assert symbols == {"AAPL", "MSFT", "NVDA", "TSLA"}
