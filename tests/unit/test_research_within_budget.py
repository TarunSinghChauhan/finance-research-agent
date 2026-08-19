from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.research import FinancialResearchAgent


def make_agent(max_cost):
    agent = FinancialResearchAgent.__new__(FinancialResearchAgent)
    agent.llm = AsyncMock()
    agent.tools = MagicMock()
    agent.tools.tool_calls = []
    agent.tools.get_market_overview = AsyncMock(return_value={})
    agent.tools.get_stock_quote = AsyncMock(return_value={})
    agent.tools.get_company_info = AsyncMock(return_value={})
    agent.tools.get_news = AsyncMock(return_value={"articles": []})
    agent.tools.compute_reproducibility_hash = MagicMock(return_value="abc123")
    agent.total_cost = 0.0
    agent.total_tokens = 0
    return agent


@pytest.mark.asyncio
async def test_research_reports_within_budget_true_when_under_limit(monkeypatch):
    from src.agents import research as research_module
    monkeypatch.setattr(research_module.settings, "max_cost_per_query_usd", 1.0)

    agent = make_agent(max_cost=1.0)

    with patch.object(agent, "_llm_call", new=AsyncMock(return_value="analysis text")):
        result = await agent.research("Apple Inc", "AAPL", "test query")

    assert result["within_budget"] is True
    assert "reproducibility_hash" in result
    assert result["company"] == "Apple Inc"


@pytest.mark.asyncio
async def test_research_reports_within_budget_false_when_over_limit(monkeypatch):
    from src.agents import research as research_module
    monkeypatch.setattr(research_module.settings, "max_cost_per_query_usd", 0.0001)

    agent = make_agent(max_cost=0.0001)

    async def fake_llm_call(*args, **kwargs):
        agent.total_cost += 0.001
        return "analysis text"

    with patch.object(agent, "_llm_call", new=fake_llm_call):
        result = await agent.research("Apple Inc", "AAPL", "test query")

    assert result["within_budget"] is False


@pytest.mark.asyncio
async def test_research_final_report_contains_all_sections(monkeypatch):
    from src.agents import research as research_module
    monkeypatch.setattr(research_module.settings, "max_cost_per_query_usd", 1.0)

    agent = make_agent(max_cost=1.0)

    with patch.object(agent, "_llm_call", new=AsyncMock(return_value="section text")):
        result = await agent.research("Tesla Inc", "TSLA", "risk analysis")

    report = result["final_report"]
    assert "Market Context" in report
    assert "Financial Analysis" in report
    assert "Risk Assessment" in report
    assert "Investment Thesis" in report
    assert "Tesla Inc" in report
