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
