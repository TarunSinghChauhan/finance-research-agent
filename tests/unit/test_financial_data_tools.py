from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.financial import FinancialTools


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class FakeAsyncClient:
    """Fake httpx.AsyncClient supporting `async with` and a mocked .get()."""
    def __init__(self, get_return=None, get_side_effect=None):
        self.get = AsyncMock(return_value=get_return, side_effect=get_side_effect)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


def patch_client(get_return=None, get_side_effect=None):
    fake = FakeAsyncClient(get_return=get_return, get_side_effect=get_side_effect)
    return patch("src.tools.financial.httpx.AsyncClient", return_value=fake), fake


@pytest.mark.asyncio
async def test_get_stock_quote_success_computes_change_and_logs():
    data = {
        "chart": {"result": [{"meta": {
            "regularMarketPrice": 110.0,
            "previousClose": 100.0,
            "currency": "USD",
            "exchangeName": "NMS",
            "marketState": "REGULAR",
            "fiftyTwoWeekHigh": 120.0,
            "fiftyTwoWeekLow": 80.0,
        }}]}
    }
    patcher, fake = patch_client(get_return=FakeResponse(data))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_stock_quote("AAPL")
    assert result["price"] == 110.0
    assert result["change"] == 10.0
    assert result["change_pct"] == 10.0
    assert result["currency"] == "USD"
    assert len(tools.tool_calls) == 1
    assert tools.tool_calls[0]["tool"] == "get_stock_quote"


@pytest.mark.asyncio
async def test_get_stock_quote_returns_error_when_no_result_data():
    data = {"chart": {"result": []}}
    patcher, _ = patch_client(get_return=FakeResponse(data))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_stock_quote("BADSYM")
    assert result == {"error": "No data found for BADSYM"}


@pytest.mark.asyncio
async def test_get_stock_quote_returns_error_dict_on_exception():
    patcher, _ = patch_client(get_side_effect=RuntimeError("network down"))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_stock_quote("AAPL")
    assert result["error"] == "network down"
    assert result["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_company_info_success_extracts_fields():
    data = {
        "quoteSummary": {"result": [{
            "summaryProfile": {"sector": "Tech", "industry": "Software", "fullTimeEmployees": 1000, "longBusinessSummary": "A company."},
            "financialData": {
                "revenueGrowth": {"raw": 0.15},
                "profitMargins": {"raw": 0.2},
                "returnOnEquity": {"raw": 0.3},
                "totalRevenue": {"raw": 1000000},
            },
            "defaultKeyStatistics": {
                "forwardPE": {"raw": 25.0},
                "marketCap": {"raw": 2000000},
                "beta": {"raw": 1.1},
            },
        }]}
    }
    patcher, _ = patch_client(get_return=FakeResponse(data))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_company_info("AAPL")
    assert result["sector"] == "Tech"
    assert result["employees"] == 1000
    assert result["pe_ratio"] == 25.0
    assert result["market_cap"] == 2000000


@pytest.mark.asyncio
async def test_get_company_info_defaults_when_fields_missing():
    data = {"quoteSummary": {"result": [{}]}}
    patcher, _ = patch_client(get_return=FakeResponse(data))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_company_info("AAPL")
    assert result["sector"] == "N/A"
    assert result["employees"] == 0
    assert result["pe_ratio"] == 0


@pytest.mark.asyncio
async def test_get_company_info_returns_error_dict_on_exception():
    patcher, _ = patch_client(get_side_effect=RuntimeError("timeout"))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_company_info("AAPL")
    assert result["error"] == "timeout"
    assert result["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_news_filters_and_truncates_results():
    data = {"RelatedTopics": [
        {"Text": "Headline one", "FirstURL": "https://a.com"},
        {"Text": "Headline two", "FirstURL": "https://b.com"},
        {"NoTextField": True},
        {"Text": "Headline three", "FirstURL": "https://c.com"},
    ]}
    patcher, _ = patch_client(get_return=FakeResponse(data))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_news("Apple earnings", max_results=2)
    assert result["count"] == 2
    assert result["articles"][0]["title"] == "Headline one"


@pytest.mark.asyncio
async def test_get_news_returns_error_dict_with_empty_articles_on_exception():
    patcher, _ = patch_client(get_side_effect=RuntimeError("dns fail"))
    tools = FinancialTools()
    with patcher:
        result = await tools.get_news("Apple")
    assert result["error"] == "dns fail"
    assert result["articles"] == []


@pytest.mark.asyncio
async def test_get_market_overview_aggregates_three_indices():
    tools = FinancialTools()
    quote_side_effects = [
        {"price": 5000.0, "change_pct": 1.2},
        {"price": 40000.0, "change_pct": -0.5},
        {"price": 16000.0, "change_pct": 2.1},
    ]
    with patch.object(tools, "get_stock_quote", AsyncMock(side_effect=quote_side_effects)):
        result = await tools.get_market_overview()
    assert result["indices"]["S&P 500"]["price"] == 5000.0
    assert result["indices"]["Dow Jones"]["change_pct"] == -0.5
    assert result["indices"]["NASDAQ"]["price"] == 16000.0
    assert "timestamp" in result


def test_compute_reproducibility_hash_is_16_char_hex():
    tools = FinancialTools()
    tool_calls = [{"tool": "get_stock_quote"}, {"tool": "get_news"}]
    result = tools.compute_reproducibility_hash("Apple Inc", tool_calls)
    assert isinstance(result, str)
    assert len(result) == 16
    int(result, 16)  # raises ValueError if not valid hex


def test_compute_reproducibility_hash_differs_for_different_company():
    tools = FinancialTools()
    tool_calls = [{"tool": "get_stock_quote"}]
    hash_a = tools.compute_reproducibility_hash("Apple Inc", tool_calls)
    hash_b = tools.compute_reproducibility_hash("Microsoft", tool_calls)
    assert hash_a != hash_b
