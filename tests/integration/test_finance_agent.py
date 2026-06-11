import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from src.api.main import app
from src.tools.financial import FinancialTools
from src.agents.research import FinancialResearchAgent


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ─── Sample companies ─────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sample_companies(client):
    resp = await client.get("/research/companies/sample")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["companies"]) >= 4
    assert all("symbol" in c for c in data["companies"])


# ─── Submit research ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_submit_research(client):
    resp = await client.post("/research/analyze", json={
        "company": "Apple Inc",
        "symbol": "AAPL",
        "query": "test query",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "report_id" in data
    assert data["status"] == "pending"
    assert data["company"] == "Apple Inc"


@pytest.mark.asyncio
async def test_unknown_report_404(client):
    resp = await client.get("/research/status/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_reports(client):
    resp = await client.get("/research/reports")
    assert resp.status_code == 200
    assert "reports" in resp.json()


# ─── Financial tools ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_reproducibility_hash():
    tools = FinancialTools()
    tools.tool_calls = [
        {"tool": "get_stock_quote", "params": {}, "result_summary": "", "timestamp": "2026-01-01"},
        {"tool": "get_company_info", "params": {}, "result_summary": "", "timestamp": "2026-01-01"},
    ]
    hash1 = tools.compute_reproducibility_hash("Apple Inc", tools.tool_calls)
    hash2 = tools.compute_reproducibility_hash("Apple Inc", tools.tool_calls)
    assert hash1 == hash2
    assert len(hash1) == 16


def test_reproducibility_hash_different_companies():
    tools = FinancialTools()
    hash1 = tools.compute_reproducibility_hash("Apple Inc", [])
    hash2 = tools.compute_reproducibility_hash("Microsoft", [])
    assert hash1 != hash2


def test_tool_call_logging():
    tools = FinancialTools()
    tools._log_tool_call("test_tool", {"param": "value"}, {"result": "data"})
    assert len(tools.tool_calls) == 1
    assert tools.tool_calls[0]["tool"] == "test_tool"
    assert "timestamp" in tools.tool_calls[0]


# ─── Agent cost tracking ──────────────────────────────────────────────────────
def test_cost_estimation():
    agent = FinancialResearchAgent()
    cost = agent._estimate_cost(1000, 500)
    assert cost > 0
    assert cost < 0.01


def test_cost_budget_initial():
    agent = FinancialResearchAgent()
    assert agent.total_cost == 0.0
    assert agent.total_tokens == 0
