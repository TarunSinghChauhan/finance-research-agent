import httpx
import json
import hashlib
from datetime import datetime, timezone
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.core.logging import get_logger


def calculate_price_change(price: float, previous_close: float) -> tuple[float, float]:
    """Returns (change, change_pct). change_pct is 0 if previous_close is falsy."""
    change = price - previous_close
    change_pct = (change / previous_close * 100) if previous_close else 0
    return round(change, 2), round(change_pct, 2)

logger = get_logger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """Network trouble and rate limiting / server errors are worth another attempt."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=1, max=5),
    reraise=True,
)
async def _fetch_json(url: str, params: dict | None = None, headers: dict | None = None):
    """GET url and return the decoded JSON body.

    The tool methods below catch every exception and turn it into an error dict, so a
    @retry on them never saw a failure and never retried. The retry lives here instead,
    where a timeout, a 429 or a 5xx still raises; other errors (a 404, bad JSON) are
    not retried.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


class FinancialTools:
    """
    Financial data tools using free public APIs.
    All tools return structured data with audit trail metadata.
    """

    def __init__(self):
        self.tool_calls: list[dict] = []

    def _log_tool_call(self, tool: str, params: dict, result: dict):
        self.tool_calls.append({
            "tool": tool,
            "params": params,
            "result_summary": str(result)[:200],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def get_stock_quote(self, symbol: str) -> dict:
        """Get real-time stock quote from Yahoo Finance (free, no key needed)."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}
            data = await _fetch_json(url, headers=headers)

            result_data = data.get("chart", {}).get("result", [])
            if not result_data:
                return {"error": f"No data found for {symbol}"}

            meta = result_data[0].get("meta", {})
            if meta.get("regularMarketPrice") is None:
                # Do not report a missing price as a real quote of 0.
                return {"error": f"No price available for {symbol}", "symbol": symbol}
            previous_close = meta.get("previousClose")
            if previous_close is None:
                previous_close = meta.get("chartPreviousClose")
            result = {
                "symbol": symbol,
                "price": meta.get("regularMarketPrice"),
                "previous_close": previous_close,
                "currency": meta.get("currency", "USD"),
                "exchange": meta.get("exchangeName", ""),
                "market_state": meta.get("marketState", ""),
                # Missing is None, not 0: a 52-week low of 0 reads as a real (and absurd) value.
                "fifty_two_week_high": meta.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": meta.get("fiftyTwoWeekLow"),
            }
            if previous_close:
                result["change"], result["change_pct"] = calculate_price_change(
                    result["price"], previous_close
                )
            else:
                # With no previous close, price - 0 used to be reported as the day's change
                # (a stock at 190 "up 190"). Say it is unknown instead.
                result["change"], result["change_pct"] = None, None

            self._log_tool_call("get_stock_quote", {"symbol": symbol}, result)
            logger.info("tool_stock_quote", symbol=symbol, price=result["price"])
            return result
        except Exception as e:
            logger.error("tool_stock_quote_failed", symbol=symbol, error=str(e))
            return {"error": str(e), "symbol": symbol}

    async def get_company_info(self, symbol: str) -> dict:
        """Get company fundamentals from Yahoo Finance."""
        try:
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
            params = {"modules": "summaryProfile,financialData,defaultKeyStatistics"}
            headers = {"User-Agent": "Mozilla/5.0"}
            data = await _fetch_json(url, params=params, headers=headers)

            quote_summary = data.get("quoteSummary") or {}
            results = quote_summary.get("result") or []
            if not results:
                # Yahoo answers an unknown symbol with "result": null plus an error object;
                # indexing that raised a TypeError that surfaced as a confusing message.
                detail = (quote_summary.get("error") or {}).get("description")
                message = f"No company data found for {symbol}"
                return {"error": f"{message}: {detail}" if detail else message, "symbol": symbol}
            summary = results[0] or {}
            profile = summary.get("summaryProfile") or {}
            financial = summary.get("financialData") or {}
            stats = summary.get("defaultKeyStatistics") or {}

            def raw(section: dict, key: str):
                """Yahoo wraps numbers as {"raw": ..., "fmt": ...}; absent or empty means unknown."""
                value = section.get(key)
                return value.get("raw") if isinstance(value, dict) else None

            # Unknown fundamentals are None (null in the prompt). They used to default to 0,
            # which told the analyst a company had zero revenue growth, zero margins, a P/E of
            # 0 and a market cap of 0 when the data simply was not there.
            result = {
                "symbol": symbol,
                "sector": profile.get("sector", "N/A"),
                "industry": profile.get("industry", "N/A"),
                "employees": profile.get("fullTimeEmployees"),
                "description": (profile.get("longBusinessSummary") or "")[:500],
                "revenue_growth": raw(financial, "revenueGrowth"),
                "profit_margins": raw(financial, "profitMargins"),
                "return_on_equity": raw(financial, "returnOnEquity"),
                "total_revenue": raw(financial, "totalRevenue"),
                "pe_ratio": raw(stats, "forwardPE"),
                "market_cap": raw(stats, "marketCap"),
                "beta": raw(stats, "beta"),
            }
            self._log_tool_call("get_company_info", {"symbol": symbol}, result)
            logger.info("tool_company_info", symbol=symbol, sector=result["sector"])
            return result
        except Exception as e:
            logger.error("tool_company_info_failed", symbol=symbol, error=str(e))
            return {"error": str(e), "symbol": symbol}

    async def get_news(self, query: str, max_results: int = 5) -> dict:
        """Get recent financial news using DuckDuckGo (free, no key needed)."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
            data = await _fetch_json(url, params=params)

            results = []
            for item in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(item, dict) and "Text" in item:
                    results.append({
                        "title": item.get("Text", "")[:100],
                        "url": item.get("FirstURL", ""),
                    })

            result = {"query": query, "articles": results, "count": len(results)}
            self._log_tool_call("get_news", {"query": query}, result)
            logger.info("tool_news", query=query, count=len(results))
            return result
        except Exception as e:
            logger.error("tool_news_failed", query=query, error=str(e))
            return {"error": str(e), "query": query, "articles": []}

    async def get_market_overview(self) -> dict:
        """Get major market indices."""
        indices = {}
        unavailable = []
        for symbol, name in [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "NASDAQ")]:
            quote = await self.get_stock_quote(symbol)
            if "error" in quote:
                # The analyst prompt is built from this dict. A failed fetch must not
                # look like "the index is at 0, change 0%".
                indices[name] = {"price": None, "change_pct": None, "error": quote["error"]}
                unavailable.append(name)
                continue
            indices[name] = {
                "price": quote.get("price"),
                "change_pct": quote.get("change_pct"),
            }
        result = {
            "indices": indices,
            "unavailable": unavailable,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log_tool_call("get_market_overview", {}, result)
        return result

    def compute_reproducibility_hash(self, company: str, tool_calls: list) -> str:
        """Generate a hash for report reproducibility verification."""
        payload = json.dumps({
            "company": company,
            "tool_calls": [t["tool"] for t in tool_calls],
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
