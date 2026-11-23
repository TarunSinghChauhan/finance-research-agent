"""Request validation, failed-report status codes and the bounded in-memory report store."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.routers import research as research_module
from src.api.routers.research import ResearchRequest, get_results


def req(**kw):
    base = dict(company="Apple Inc", symbol="AAPL", query="outlook")
    base.update(kw)
    return ResearchRequest(**base)


# ---- request validation -----------------------------------------------------------

@pytest.mark.parametrize("symbol", ["AAPL", "brk-b", "^GSPC", "RELIANCE.NS", "EURUSD=X", "A", "A" * 15])
def test_real_tickers_are_accepted(symbol):
    assert req(symbol=symbol).symbol == symbol


@pytest.mark.parametrize("symbol", [
    "AAPL/../x", "AAPL?range=max", "AAPL#frag", "AA PL", "AAPL%2F", "",
    "   ", "A" * 16, "AAPL;DROP", "<script>", "AAPL&x=1",
])
def test_symbols_that_could_change_the_url_are_rejected(symbol):
    with pytest.raises(ValidationError):
        req(symbol=symbol)


@pytest.mark.parametrize("field,value", [
    ("company", ""), ("company", "   "), ("company", "x" * 101),
    ("query", ""), ("query", "x" * 1001),
])
def test_empty_or_oversized_text_fields_are_rejected(field, value):
    with pytest.raises(ValidationError):
        req(**{field: value})


def test_surrounding_whitespace_is_stripped_and_defaults_survive():
    r = ResearchRequest(company="  Apple Inc ", symbol=" AAPL ")
    assert (r.company, r.symbol) == ("Apple Inc", "AAPL")
    assert r.query == "Provide a comprehensive investment analysis"


def test_boundary_lengths_are_accepted():
    assert len(req(company="c" * 100, query="q" * 1000).query) == 1000


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(research_module.router, prefix="/research")
    return TestClient(app)


def test_a_bad_symbol_gets_422_and_starts_no_work(api):
    with patch.object(research_module, "_run_research", new=AsyncMock()) as run:
        resp = api.post("/research/analyze", json={"company": "X", "symbol": "AAPL/../../x"})
    assert resp.status_code == 422
    run.assert_not_called()


def test_a_good_request_still_starts_the_research(api):
    with patch.object(research_module, "_run_research", new=AsyncMock()) as run:
        resp = api.post("/research/analyze", json={"company": "Apple Inc", "symbol": "AAPL"})
    assert resp.status_code == 200 and resp.json()["status"] == "pending"
    run.assert_awaited_once()


def test_the_built_in_sample_companies_pass_validation():
    for c in research_module.SAMPLE_COMPANIES:
        ResearchRequest(**c)


# ---- /results status codes --------------------------------------------------------

@pytest.mark.asyncio
async def test_a_failed_report_is_an_error_not_a_202(monkeypatch):
    monkeypatch.setattr(research_module, "_report_status", {"r1": "failed: model timeout"})
    with pytest.raises(HTTPException) as exc:
        await get_results("r1")
    assert exc.value.status_code == 500
    assert "model timeout" in exc.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "running"])
async def test_an_active_report_is_still_202(monkeypatch, status):
    monkeypatch.setattr(research_module, "_report_status", {"r1": status})
    with pytest.raises(HTTPException) as exc:
        await get_results("r1")
    assert exc.value.status_code == 202


# ---- bounded store ----------------------------------------------------------------

@pytest.fixture
def store(monkeypatch):
    status, reports = {}, {}
    monkeypatch.setattr(research_module, "_report_status", status)
    monkeypatch.setattr(research_module, "_reports", reports)
    monkeypatch.setattr(research_module, "MAX_STORED_REPORTS", 3)
    return status, reports


def test_nothing_is_evicted_while_under_the_limit(store):
    status, reports = store
    status.update(a="completed", b="completed", c="failed: x")
    research_module._evict_finished_reports()
    assert list(status) == ["a", "b", "c"]


def test_the_oldest_finished_reports_go_first_and_take_their_body_with_them(store):
    status, reports = store
    for rid in "abcde":
        status[rid] = "completed"
        reports[rid] = {"report_id": rid}
    research_module._evict_finished_reports()
    assert list(status) == ["c", "d", "e"]
    assert list(reports) == ["c", "d", "e"]


def test_active_reports_are_never_evicted_even_when_oldest(store):
    status, reports = store
    status.update(a="running", b="pending", c="completed", d="completed", e="failed: boom")
    research_module._evict_finished_reports()
    assert "a" in status and "b" in status
    assert len(status) == 3


def test_only_active_reports_over_the_limit_stay(store):
    status, _ = store
    status.update(a="running", b="running", c="pending", d="running")
    research_module._evict_finished_reports()
    assert len(status) == 4


def test_starting_new_analyses_keeps_the_store_bounded(api, store):
    status, reports = store
    with patch.object(research_module, "_run_research", new=AsyncMock()):
        for i in range(10):
            rid = f"old{i}"
            status[rid] = "completed"
            reports[rid] = {}
        api.post("/research/analyze", json={"company": "Apple Inc", "symbol": "AAPL"})
        api.post("/research/analyze/sample")
    assert len(status) == 3
    assert len(reports) <= 3
    assert sum(1 for s in status.values() if s == "pending") == 2
