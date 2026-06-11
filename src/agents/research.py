import json
import time
from datetime import datetime
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.tools.financial import FinancialTools
from src.core.config import get_settings
from src.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


class FinancialResearchAgent:
    """
    Autonomous financial research agent with 4-step structured reasoning:
    1. Market Context — macro environment and market conditions
    2. Financial Analysis — company fundamentals and metrics
    3. Risk Assessment — key risks and concerns
    4. Synthesis — final investment thesis

    Features:
    - Per-query cost budget enforcement
    - Full audit trail of all tool calls
    - Reproducibility hash for each report
    - Redis caching for repeated queries
    """

    def __init__(self):
        self.llm = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.tools = FinancialTools()
        self.total_cost = 0.0
        self.total_tokens = 0

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _llm_call(self, system: str, user: str, max_tokens: int = 500) -> str:
        if self.total_cost >= settings.max_cost_per_query_usd:
            logger.warning("cost_budget_exceeded", cost=self.total_cost)
            return "Cost budget exceeded — analysis truncated."

        try:
            response = await self.llm.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
                extra_headers={
                    "HTTP-Referer": "https://github.com/finance-research-agent",
                    "X-Title": "Finance Research Agent",
                },
            )
            text = response.choices[0].message.content or ""
            if response.usage:
                cost = self._estimate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                self.total_cost += cost
                self.total_tokens += response.usage.total_tokens
            return text
        except Exception as e:
            logger.error("llm_call_failed", error=str(e))
            return f"Analysis unavailable: {str(e)}"

    async def step1_market_context(self, company: str, symbol: str) -> str:
        """Step 1: Gather and analyze macro market context."""
        logger.info("step1_market_context", company=company)

        market_data = await self.tools.get_market_overview()
        quote = await self.tools.get_stock_quote(symbol)

        market_summary = json.dumps(market_data, indent=2)
        stock_summary = json.dumps(quote, indent=2)

        return await self._llm_call(
            system="You are a financial analyst. Provide concise, factual market context analysis.",
            user=f"""Analyze the market context for {company} ({symbol}).

Market Overview:
{market_summary}

Stock Quote:
{stock_summary}

Provide a 3-4 sentence market context analysis covering:
1. Current market conditions
2. How the stock is performing vs market
3. Key market factors relevant to this company""",
            max_tokens=300,
        )

    async def step2_financial_analysis(self, company: str, symbol: str) -> str:
        """Step 2: Deep dive into company financials."""
        logger.info("step2_financial_analysis", company=company)

        company_info = await self.tools.get_company_info(symbol)
        info_summary = json.dumps(company_info, indent=2)

        return await self._llm_call(
            system="You are a financial analyst specializing in fundamental analysis.",
            user=f"""Analyze the financial fundamentals of {company} ({symbol}).

Company Data:
{info_summary}

Provide analysis covering:
1. Revenue and profitability metrics
2. Valuation (P/E ratio, market cap)
3. Growth trajectory
4. Key financial strengths and weaknesses

Keep it factual and data-driven. 4-5 sentences.""",
            max_tokens=350,
        )

    async def step3_risk_assessment(self, company: str, symbol: str, financial_analysis: str) -> str:
        """Step 3: Identify and assess key risks."""
        logger.info("step3_risk_assessment", company=company)

        news = await self.tools.get_news(f"{company} stock risks 2025")

        return await self._llm_call(
            system="You are a risk analyst. Be objective and highlight genuine concerns.",
            user=f"""Assess the key risks for {company} ({symbol}).

Financial Analysis Summary:
{financial_analysis[:300]}

Recent News Context:
{json.dumps(news.get('articles', [])[:3], indent=2)}

Identify 3-4 specific risks covering:
1. Market/competitive risks
2. Financial risks
3. Regulatory or macro risks
4. Company-specific risks

Be specific and evidence-based.""",
            max_tokens=300,
        )

    async def step4_synthesis(
        self,
        company: str,
        symbol: str,
        market_context: str,
        financial_analysis: str,
        risk_assessment: str,
    ) -> str:
        """Step 4: Synthesize all analysis into final thesis."""
        logger.info("step4_synthesis", company=company)

        return await self._llm_call(
            system="You are a senior equity analyst. Synthesize research into a clear, balanced investment thesis.",
            user=f"""Synthesize the research on {company} ({symbol}) into a final analysis.

Market Context:
{market_context[:200]}

Financial Analysis:
{financial_analysis[:200]}

Risk Assessment:
{risk_assessment[:200]}

Provide:
1. Overall assessment (Bullish/Neutral/Bearish with reasoning)
2. Key catalysts to watch
3. Main risks to monitor
4. Summary thesis in 2-3 sentences

Be balanced and professional.""",
            max_tokens=400,
        )

    async def research(self, company: str, symbol: str, query: str) -> dict:
        """
        Run the full 4-step research pipeline.
        Returns structured report with audit trail.
        """
        start_time = time.perf_counter()
        self.total_cost = 0.0
        self.total_tokens = 0
        self.tools.tool_calls = []

        logger.info("research_started", company=company, symbol=symbol)

        # Execute 4-step reasoning chain
        market_context = await self.step1_market_context(company, symbol)
        financial_analysis = await self.step2_financial_analysis(company, symbol)
        risk_assessment = await self.step3_risk_assessment(company, symbol, financial_analysis)
        synthesis = await self.step4_synthesis(company, symbol, market_context, financial_analysis, risk_assessment)

        # Build final report
        final_report = f"""# Financial Research Report: {company} ({symbol})
*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*

## Market Context
{market_context}

## Financial Analysis
{financial_analysis}

## Risk Assessment
{risk_assessment}

## Investment Thesis
{synthesis}

---
*Research Query: {query}*
*Total Cost: ${self.total_cost:.4f} | Tokens: {self.total_tokens}*
"""

        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        reproducibility_hash = self.tools.compute_reproducibility_hash(company, self.tools.tool_calls)

        logger.info(
            "research_completed",
            company=company,
            cost=self.total_cost,
            tokens=self.total_tokens,
            latency_ms=latency_ms,
        )

        return {
            "company": company,
            "symbol": symbol,
            "query": query,
            "market_context": market_context,
            "financial_analysis": financial_analysis,
            "risk_assessment": risk_assessment,
            "synthesis": synthesis,
            "final_report": final_report,
            "tool_calls": self.tools.tool_calls,
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "latency_ms": latency_ms,
            "reproducibility_hash": reproducibility_hash,
            "completed_at": datetime.utcnow().isoformat(),
            "within_budget": self.total_cost <= settings.max_cost_per_query_usd,
        }
