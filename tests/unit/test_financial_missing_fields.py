"""Missing data is unknown (None), never 0, and an unknown symbol is an error."""
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.financial import FinancialTools

FETCH = "src.tools.financial._fetch_json"
COMPANY_FIELDS = ["employees", "revenue_growth", "profit_margins", "return_on_equity",
                  "total_revenue", "pe_ratio", "market_cap", "beta"]


async def quote(meta):
    with patch(FETCH, new=AsyncMock(return_value={"chart": {"result": [{"meta": meta}]}})):
        return await FinancialTools().get_stock_quote("AAPL")


async def company(payload):
    with patch(FETCH, new=AsyncMock(return_value=payload)):
        return await FinancialTools().get_company_info("AAPL")


# ---- quote ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_previous_close_means_unknown_change_not_change_equal_to_price():
    r = await quote({"regularMarketPrice": 190.0})
    assert r["price"] == 190.0
    assert r["previous_close"] is None
    assert r["change"] is None and r["change_pct"] is None


@pytest.mark.asyncio
async def test_a_zero_previous_close_is_treated_as_missing():
    r = await quote({"regularMarketPrice": 190.0, "previousClose": 0})
    assert r["change"] is None and r["change_pct"] is None


@pytest.mark.asyncio
async def test_chart_previous_close_is_used_when_previous_close_is_absent():
    r = await quote({"regularMarketPrice": 110.0, "chartPreviousClose": 100.0})
    assert r["previous_close"] == 100.0
    assert r["change"] == 10.0 and r["change_pct"] == 10.0


@pytest.mark.asyncio
async def test_previous_close_wins_over_chart_previous_close():
    r = await quote({"regularMarketPrice": 110.0, "previousClose": 105.0, "chartPreviousClose": 100.0})
    assert r["previous_close"] == 105.0 and r["change"] == 5.0


@pytest.mark.asyncio
async def test_missing_52_week_range_is_none_not_zero():
    r = await quote({"regularMarketPrice": 110.0, "previousClose": 100.0})
    assert r["fifty_two_week_high"] is None and r["fifty_two_week_low"] is None


@pytest.mark.asyncio
async def test_a_complete_quote_is_unchanged():
    r = await quote({"regularMarketPrice": 110.0, "previousClose": 100.0, "fiftyTwoWeekHigh": 120.0,
                     "fiftyTwoWeekLow": 80.0, "currency": "USD"})
    assert (r["change"], r["change_pct"], r["fifty_two_week_high"], r["fifty_two_week_low"]) == (10.0, 10.0, 120.0, 80.0)


@pytest.mark.asyncio
async def test_a_real_zero_change_is_still_zero():
    r = await quote({"regularMarketPrice": 100.0, "previousClose": 100.0})
    assert r["change"] == 0.0 and r["change_pct"] == 0.0


@pytest.mark.asyncio
async def test_market_overview_reports_unknown_change_as_none():
    tools = FinancialTools()
    with patch(FETCH, new=AsyncMock(return_value={"chart": {"result": [{"meta": {"regularMarketPrice": 5000.0}}]}})):
        out = await tools.get_market_overview()
    assert out["indices"]["S&P 500"] == {"price": 5000.0, "change_pct": None}
    assert out["unavailable"] == []


# ---- company info -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_fundamentals_are_none_not_zero():
    r = await company({"quoteSummary": {"result": [{}]}})
    assert r["sector"] == "N/A" and r["description"] == ""
    assert all(r[f] is None for f in COMPANY_FIELDS)


@pytest.mark.asyncio
async def test_empty_wrappers_and_null_sections_are_unknown_too():
    r = await company({"quoteSummary": {"result": [{
        "summaryProfile": None,
        "financialData": {"revenueGrowth": {}, "profitMargins": None},
        "defaultKeyStatistics": {"forwardPE": {}},
    }]}})
    assert r["revenue_growth"] is None and r["profit_margins"] is None and r["pe_ratio"] is None


@pytest.mark.asyncio
async def test_a_genuine_zero_is_kept():
    r = await company({"quoteSummary": {"result": [{
        "financialData": {"revenueGrowth": {"raw": 0}, "profitMargins": {"raw": -0.1}},
    }]}})
    assert r["revenue_growth"] == 0 and r["profit_margins"] == -0.1


@pytest.mark.asyncio
async def test_a_null_description_does_not_crash():
    r = await company({"quoteSummary": {"result": [{"summaryProfile": {"longBusinessSummary": None}}]}})
    assert r["description"] == ""


@pytest.mark.asyncio
async def test_present_values_are_extracted_as_before():
    r = await company({"quoteSummary": {"result": [{
        "summaryProfile": {"sector": "Tech", "fullTimeEmployees": 1000, "longBusinessSummary": "x" * 600},
        "financialData": {"totalRevenue": {"raw": 5}},
        "defaultKeyStatistics": {"forwardPE": {"raw": 25.0}, "beta": {"raw": 1.1}},
    }]}})
    assert (r["sector"], r["employees"], r["total_revenue"], r["pe_ratio"], r["beta"]) == ("Tech", 1000, 5, 25.0, 1.1)
    assert len(r["description"]) == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"quoteSummary": {"result": None, "error": {"code": "Not Found", "description": "Quote not found for ticker symbol: ZZZZ"}}},
    {"quoteSummary": {"result": []}},
    {"quoteSummary": None},
    {},
])
async def test_an_unknown_symbol_is_a_clear_error(payload):
    r = await company(payload)
    assert r["error"].startswith("No company data found for AAPL")
    assert r["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_the_yahoo_error_description_is_passed_on():
    r = await company({"quoteSummary": {"result": None, "error": {"description": "Quote not found"}}})
    assert r["error"] == "No company data found for AAPL: Quote not found"


@pytest.mark.asyncio
async def test_unknown_symbol_is_not_logged_as_a_successful_tool_call():
    tools = FinancialTools()
    with patch(FETCH, new=AsyncMock(return_value={"quoteSummary": {"result": None}})):
        await tools.get_company_info("ZZZZ")
    assert tools.tool_calls == []
