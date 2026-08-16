from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.research import FinancialResearchAgent


def test_estimate_cost_matches_gpt4o_mini_pricing():
    agent = FinancialResearchAgent.__new__(FinancialResearchAgent)
    cost = agent._estimate_cost(input_tokens=1000, output_tokens=500)
    expected = (1000 * 0.15 + 500 * 0.60) / 1_000_000
    assert cost == expected


def test_estimate_cost_zero_tokens_is_zero():
    agent = FinancialResearchAgent.__new__(FinancialResearchAgent)
    assert agent._estimate_cost(0, 0) == 0.0


@pytest.mark.asyncio
async def test_llm_call_skips_api_when_budget_already_exceeded(monkeypatch):
    from src.agents import research as research_module
    monkeypatch.setattr(research_module.settings, "max_cost_per_query_usd", 0.01)

    agent = FinancialResearchAgent.__new__(FinancialResearchAgent)
    agent.llm = AsyncMock()
    agent.total_cost = 0.02
    agent.total_tokens = 0

    result = await agent._llm_call("system", "user")

    assert result == "Cost budget exceeded — analysis truncated."
    agent.llm.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_llm_call_proceeds_and_accumulates_cost_when_under_budget(monkeypatch):
    from src.agents import research as research_module
    monkeypatch.setattr(research_module.settings, "max_cost_per_query_usd", 1.0)

    agent = FinancialResearchAgent.__new__(FinancialResearchAgent)
    agent.total_cost = 0.0
    agent.total_tokens = 0

    mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    mock_message = MagicMock(content="Analysis text")
    mock_choice = MagicMock(message=mock_message)
    mock_response = MagicMock(choices=[mock_choice], usage=mock_usage)

    agent.llm = AsyncMock()
    agent.llm.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await agent._llm_call("system", "user")

    assert result == "Analysis text"
    assert agent.total_tokens == 150
    assert agent.total_cost > 0
