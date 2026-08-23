from unittest.mock import AsyncMock

import pytest

from src.agents.research import FinancialResearchAgent


def make_agent():
    agent = FinancialResearchAgent.__new__(FinancialResearchAgent)
    agent.tools = AsyncMock()
    agent.tools.get_news = AsyncMock(return_value={"articles": []})
    return agent


@pytest.mark.asyncio
async def test_step3_truncates_financial_analysis_to_300_chars():
    agent = make_agent()
    agent._llm_call = AsyncMock(return_value="risk analysis result")

    long_analysis = "x" * 1000

    await agent.step3_risk_assessment("Apple Inc", "AAPL", long_analysis)

    call_args = agent._llm_call.call_args
    user_prompt = call_args.kwargs.get("user") or call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["user"]
    included_analysis = user_prompt.split("Financial Analysis Summary:")[1].split("Recent News")[0]
    assert len(included_analysis.strip()) <= 300


@pytest.mark.asyncio
async def test_step4_truncates_each_input_to_200_chars():
    agent = make_agent()
    agent._llm_call = AsyncMock(return_value="synthesis result")

    long_text = "y" * 1000

    await agent.step4_synthesis(
        "Apple Inc", "AAPL",
        market_context=long_text,
        financial_analysis=long_text,
        risk_assessment=long_text,
    )

    call_args = agent._llm_call.call_args
    user_prompt = call_args.kwargs.get("user") or call_args.args[1]
    market_section = user_prompt.split("Market Context:")[1].split("Financial Analysis:")[0]
    assert len(market_section.strip()) <= 200


@pytest.mark.asyncio
async def test_step3_includes_company_and_symbol_in_prompt():
    agent = make_agent()
    agent._llm_call = AsyncMock(return_value="result")

    await agent.step3_risk_assessment("Tesla Inc", "TSLA", "some analysis")

    call_args = agent._llm_call.call_args
    user_prompt = call_args.kwargs.get("user") or call_args.args[1]
    assert "Tesla Inc" in user_prompt
    assert "TSLA" in user_prompt


@pytest.mark.asyncio
async def test_step3_fetches_news_for_risk_context():
    agent = make_agent()
    agent._llm_call = AsyncMock(return_value="result")

    await agent.step3_risk_assessment("Microsoft Corporation", "MSFT", "analysis")

    agent.tools.get_news.assert_called_once()
    call_args = agent.tools.get_news.call_args
    query = call_args.args[0] if call_args.args else call_args.kwargs.get("query", "")
    assert "Microsoft Corporation" in query
