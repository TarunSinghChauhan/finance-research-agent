import httpx
import json
import hashlib
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.logging import get_logger

logger = get_logger(__name__)


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
            "timestamp": datetime.utcnow().isoformat(),
        })

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def get_stock_quote(self, symbol: str) -> dict:
        """Get real-time stock quote from Yahoo Finance (free, no key needed)."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = await client.get(url, headers=headers)
                data = resp.json()

                result_data = data.get("chart", {}).get("result", [])
                if not result_data:
                    return {"error": f"No data found for {symbol}"}

                meta = result_data[0].get("meta", {})
                result = {
                    "symbol": symbol,
                    "price": meta.get("regularMarketPrice", 0),
                    "previous_close": meta.get("previousClose", 0),
                    "currency": meta.get("currency", "USD"),
                    "exchange": meta.get("exchangeName", ""),
                    "market_state": meta.get("marketState", ""),
                    "fifty_two_week_high": meta.get("fiftyTwoWeekHigh", 0),
                    "fifty_two_week_low": meta.get("fiftyTwoWeekLow", 0),
                }
                change = result["price"] - result["previous_close"]
                change_pct = (change / result["previous_close"] * 100) if result["previous_close"] else 0
                result["change"] = round(change, 2)
                result["change_pct"] = round(change_pct, 2)

                self._log_tool_call("get_stock_quote", {"symbol": symbol}, result)
                logger.info("tool_stock_quote", symbol=symbol, price=result["price"])
                return result
        except Exception as e:
            logger.error("tool_stock_quote_failed", symbol=symbol, error=str(e))
            return {"error": str(e), "symbol": symbol}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def get_company_info(self, symbol: str) -> dict:
        """Get company fundamentals from Yahoo Finance."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
                params = {"modules": "summaryProfile,financialData,defaultKeyStatistics"}
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = await client.get(url, params=params, headers=headers)
                data = resp.json()

                summary = data.get("quoteSummary", {}).get("result", [{}])[0]
                profile = summary.get("summaryProfile", {})
                financial = summary.get("financialData", {})
                stats = summary.get("defaultKeyStatistics", {})

                result = {
                    "symbol": symbol,
                    "sector": profile.get("sector", "N/A"),
                    "industry": profile.get("industry", "N/A"),
                    "employees": profile.get("fullTimeEmployees", 0),
                    "description": profile.get("longBusinessSummary", "")[:500],
                    "revenue_growth": financial.get("revenueGrowth", {}).get("raw", 0),
                    "profit_margins": financial.get("profitMargins", {}).get("raw", 0),
                    "return_on_equity": financial.get("returnOnEquity", {}).get("raw", 0),
                    "total_revenue": financial.get("totalRevenue", {}).get("raw", 0),
                    "pe_ratio": stats.get("forwardPE", {}).get("raw", 0),
                    "market_cap": stats.get("marketCap", {}).get("raw", 0),
                    "beta": stats.get("beta", {}).get("raw", 0),
                }
                self._log_tool_call("get_company_info", {"symbol": symbol}, result)
                logger.info("tool_company_info", symbol=symbol, sector=result["sector"])
                return result
        except Exception as e:
            logger.error("tool_company_info_failed", symbol=symbol, error=str(e))
            return {"error": str(e), "symbol": symbol}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def get_news(self, query: str, max_results: int = 5) -> dict:
        """Get recent financial news using DuckDuckGo (free, no key needed)."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                url = "https://api.duckduckgo.com/"
                params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
                resp = await client.get(url, params=params)
                data = resp.json()

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
        for symbol, name in [("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "NASDAQ")]:
            quote = await self.get_stock_quote(symbol)
            indices[name] = {
                "price": quote.get("price", 0),
                "change_pct": quote.get("change_pct", 0),
            }
        result = {"indices": indices, "timestamp": datetime.utcnow().isoformat()}
        self._log_tool_call("get_market_overview", {}, result)
        return result

    def compute_reproducibility_hash(self, company: str, tool_calls: list) -> str:
        """Generate a hash for report reproducibility verification."""
        payload = json.dumps({
            "company": company,
            "tool_calls": [t["tool"] for t in tool_calls],
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
