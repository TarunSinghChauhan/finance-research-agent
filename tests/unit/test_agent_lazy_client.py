from unittest.mock import MagicMock, patch

from src.agents import research as research_module
from src.agents.research import FinancialResearchAgent


def test_constructing_agent_does_not_build_client():
    with patch.object(research_module, "AsyncOpenAI") as client_cls:
        agent = FinancialResearchAgent()

    client_cls.assert_not_called()
    assert agent.total_cost == 0.0
    assert agent.total_tokens == 0


def test_llm_is_built_on_first_access_and_cached():
    with patch.object(research_module, "AsyncOpenAI") as client_cls:
        agent = FinancialResearchAgent()
        first = agent.llm
        second = agent.llm

    client_cls.assert_called_once_with(
        api_key=research_module.settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    assert first is second


def test_llm_can_be_overridden_without_building_a_client():
    fake = MagicMock()
    with patch.object(research_module, "AsyncOpenAI") as client_cls:
        agent = FinancialResearchAgent()
        agent.llm = fake
        assert agent.llm is fake

    client_cls.assert_not_called()
