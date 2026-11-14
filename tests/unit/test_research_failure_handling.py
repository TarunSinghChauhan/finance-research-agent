"""A failed LLM step must be reported as failed, never written up as analysis."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.agents import research as research_module
from src.agents.research import (
    BUDGET_EXCEEDED_MSG,
    NOT_AVAILABLE,
    UNAVAILABLE_PREFIX,
    FinancialResearchAgent,
    is_unavailable,
)

FAILED = f"{UNAVAILABLE_PREFIX}: the model call failed."


def make_agent(replies):
    """Agent whose network tools are stubbed and whose LLM answers come from `replies`."""
    agent = FinancialResearchAgent()
    agent.tools.get_market_overview = AsyncMock(return_value={"indices": {}})
    agent.tools.get_stock_quote = AsyncMock(return_value={"price": 1.0})
    agent.tools.get_company_info = AsyncMock(return_value={"sector": "Tech"})
    agent.tools.get_news = AsyncMock(return_value={"articles": []})
    agent._llm_call = AsyncMock(side_effect=replies)
    return agent


def prompts(agent):
    return [c.kwargs.get("user") or c.args[1] for c in agent._llm_call.call_args_list]


def test_is_unavailable_recognises_both_failure_notices_only():
    assert is_unavailable(FAILED)
    assert is_unavailable(BUDGET_EXCEEDED_MSG)
    assert not is_unavailable("Apple is doing well.")
    assert not is_unavailable("The analysis unavailable section is fine")  # only a prefix counts


@pytest.mark.asyncio
async def test_healthy_run_is_complete_and_has_no_notice():
    agent = make_agent(["market", "financials", "risks", "thesis"])
    result = await agent.research("Apple Inc", "AAPL", "q")
    assert result["complete"] is True and result["failed_steps"] == []
    assert "Incomplete" not in result["final_report"]
    assert "*Generated:" in result["final_report"].split("## Market Context")[0]


@pytest.mark.asyncio
async def test_a_failed_step_is_listed_and_flagged_in_the_report():
    agent = make_agent(["market", FAILED, "risks", "thesis"])
    result = await agent.research("Apple Inc", "AAPL", "q")
    assert result["complete"] is False
    assert result["failed_steps"] == ["financial_analysis"]
    assert "Incomplete report" in result["final_report"]
    assert "financial_analysis" in result["final_report"].split("## Market Context")[0]


@pytest.mark.asyncio
async def test_failure_text_is_not_fed_to_later_prompts_as_analysis():
    agent = make_agent(["market", FAILED, "risks", "thesis"])
    await agent.research("Apple Inc", "AAPL", "q")
    step3_prompt, step4_prompt = prompts(agent)[2], prompts(agent)[3]
    for p in (step3_prompt, step4_prompt):
        assert UNAVAILABLE_PREFIX not in p
    assert NOT_AVAILABLE[:30] in step3_prompt
    assert NOT_AVAILABLE[:30] in step4_prompt


@pytest.mark.asyncio
async def test_budget_exhaustion_counts_as_a_failed_step():
    agent = make_agent(["market", "financials", BUDGET_EXCEEDED_MSG, BUDGET_EXCEEDED_MSG])
    result = await agent.research("Apple Inc", "AAPL", "q")
    assert result["failed_steps"] == ["risk_assessment", "synthesis"]
    assert result["complete"] is False


@pytest.mark.asyncio
async def test_no_thesis_is_requested_when_nothing_upstream_worked():
    agent = make_agent([FAILED, FAILED, FAILED])
    result = await agent.research("Apple Inc", "AAPL", "q")
    assert agent._llm_call.await_count == 3  # the synthesis call is skipped
    assert result["failed_steps"] == ["market_context", "financial_analysis", "risk_assessment", "synthesis"]
    assert is_unavailable(result["synthesis"])
    assert "Bullish" not in result["final_report"]


@pytest.mark.asyncio
async def test_llm_exception_text_stays_out_of_the_report():
    agent = FinancialResearchAgent()
    agent.llm = AsyncMock()
    agent.llm.chat.completions.create = AsyncMock(side_effect=RuntimeError("secret-detail-123"))
    text = await agent._llm_call("system", "user")
    assert is_unavailable(text)
    assert "secret-detail-123" not in text


@pytest.mark.asyncio
async def test_risk_news_query_uses_the_current_year():
    agent = make_agent(["risks"])
    await agent.step3_risk_assessment("Apple Inc", "AAPL", "analysis")
    query = agent.tools.get_news.call_args.args[0]
    assert str(datetime.now(timezone.utc).year) in query
    assert "2025" not in query or datetime.now(timezone.utc).year == 2025
