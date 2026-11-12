"""Retries must actually happen, and a failed fetch must never look like real data."""
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.tools.financial import FinancialTools

GOOD_QUOTE = {"chart": {"result": [{"meta": {"regularMarketPrice": 110.0, "previousClose": 100.0}}]}}
GOOD_INFO = {"quoteSummary": {"result": [{"summaryProfile": {"sector": "Tech"}}]}}
GOOD_NEWS = {"RelatedTopics": [{"Text": "Headline", "FirstURL": "https://a.example"}]}


class Resp:
    def __init__(self, data=None, status_code=200):
        self._data = {} if data is None else data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("GET", "https://example.test")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=req, response=httpx.Response(self.status_code, request=req)
            )

    def json(self):
        return self._data


class Client:
    """Fake httpx.AsyncClient whose .get() walks through a list of responses/exceptions."""

    def __init__(self, outcomes):
        self.get = AsyncMock(side_effect=outcomes)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


def fake(outcomes):
    client = Client(outcomes)
    return patch("src.tools.financial.httpx.AsyncClient", return_value=client), client


CALLS = [
    ("get_stock_quote", ("AAPL",), GOOD_QUOTE, "price", 110.0),
    ("get_company_info", ("AAPL",), GOOD_INFO, "sector", "Tech"),
    ("get_news", ("apple",), GOOD_NEWS, "count", 1),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args,body,key,expected", CALLS)
@pytest.mark.parametrize("failure", [Resp(status_code=503), Resp(status_code=429), httpx.ConnectTimeout("slow")])
async def test_a_transient_failure_is_retried_and_the_second_attempt_wins(method, args, body, key, expected, failure):
    patcher, client = fake([failure, Resp(body)])
    with patcher:
        result = await getattr(FinancialTools(), method)(*args)
    assert result[key] == expected
    assert "error" not in result
    assert client.get.await_count == 2


@pytest.mark.asyncio
async def test_retrying_is_bounded_and_the_last_failure_is_reported():
    patcher, client = fake([Resp(status_code=429), Resp(status_code=429), Resp(GOOD_QUOTE)])
    with patcher:
        result = await FinancialTools().get_stock_quote("AAPL")
    assert client.get.await_count == 2  # two attempts, never a third
    assert result["symbol"] == "AAPL"
    assert "429" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [Resp(status_code=404), Resp(status_code=401), RuntimeError("boom")])
async def test_permanent_errors_are_not_retried(failure):
    patcher, client = fake([failure, Resp(GOOD_QUOTE)])
    with patcher:
        result = await FinancialTools().get_stock_quote("AAPL")
    assert client.get.await_count == 1
    assert "error" in result


@pytest.mark.asyncio
async def test_a_quote_without_a_price_is_an_error_not_a_price_of_zero():
    body = {"chart": {"result": [{"meta": {"currency": "USD"}}]}}
    patcher, _ = fake([Resp(body)])
    tools = FinancialTools()
    with patcher:
        result = await tools.get_stock_quote("AAPL")
    assert result == {"error": "No price available for AAPL", "symbol": "AAPL"}
    assert tools.tool_calls == []  # nothing is logged as a successful quote


@pytest.mark.asyncio
async def test_a_genuine_zero_price_is_still_a_quote():
    body = {"chart": {"result": [{"meta": {"regularMarketPrice": 0.0, "previousClose": 0.0}}]}}
    patcher, _ = fake([Resp(body)])
    with patcher:
        result = await FinancialTools().get_stock_quote("DEAD")
    assert result["price"] == 0.0 and "error" not in result


@pytest.mark.asyncio
async def test_market_overview_marks_a_failed_index_as_unavailable_instead_of_zero():
    tools = FinancialTools()
    quotes = [
        {"price": 5000.0, "change_pct": 1.2},
        {"error": "boom", "symbol": "^DJI"},
        {"price": 16000.0, "change_pct": -0.4},
    ]
    with patch.object(tools, "get_stock_quote", AsyncMock(side_effect=quotes)):
        result = await tools.get_market_overview()
    assert result["indices"]["Dow Jones"] == {"price": None, "change_pct": None, "error": "boom"}
    assert result["indices"]["S&P 500"] == {"price": 5000.0, "change_pct": 1.2}
    assert result["indices"]["NASDAQ"]["price"] == 16000.0
    assert result["unavailable"] == ["Dow Jones"]


@pytest.mark.asyncio
async def test_market_overview_when_everything_is_down_reports_no_numbers():
    tools = FinancialTools()
    with patch.object(tools, "get_stock_quote", AsyncMock(return_value={"error": "down"})):
        result = await tools.get_market_overview()
    assert result["unavailable"] == ["S&P 500", "Dow Jones", "NASDAQ"]
    assert all(v["price"] is None and v["change_pct"] is None for v in result["indices"].values())


@pytest.mark.asyncio
async def test_a_healthy_market_overview_has_nothing_unavailable():
    tools = FinancialTools()
    with patch.object(tools, "get_stock_quote", AsyncMock(return_value={"price": 1.0, "change_pct": 0.1})):
        result = await tools.get_market_overview()
    assert result["unavailable"] == []
