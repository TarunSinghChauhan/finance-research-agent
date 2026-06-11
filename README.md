# Autonomous Financial Research Agent

> AI agent that researches any public company using a structured 4-step reasoning chain — Market Context → Financial Analysis → Risk Assessment → Investment Thesis — with full audit trails and cost tracking.

---

## What This Does

Analysts spend 4-6 hours researching a single company. This agent compresses that to **under 60 seconds** by autonomously:

1. Fetching real-time stock data and market context
2. Pulling company fundamentals (revenue, margins, P/E, market cap)
3. Assessing risks using recent news
4. Synthesizing an investment thesis

Every report includes a **reproducibility hash** and full **tool call audit trail** — making outputs defensible and verifiable.

---

## Architecture

```
User Query (company + symbol)
         │
         ▼
┌─────────────────────────────────┐
│     Research Agent              │
│   Cost budget: $0.50/query      │
└──────────────┬──────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
Yahoo Finance  Yahoo Finance  DuckDuckGo
Stock Quote    Fundamentals   News Search
    │          │              │
    └──────────┼──────────────┘
               │
    ┌──────────▼──────────────────┐
    │  Step 1: Market Context     │
    │  Step 2: Financial Analysis │
    │  Step 3: Risk Assessment    │
    │  Step 4: Investment Thesis  │
    └──────────┬──────────────────┘
               │
    ┌──────────▼──────────────────┐
    │  Structured Report          │
    │  + Audit Trail              │
    │  + Reproducibility Hash     │
    │  + Cost Breakdown           │
    └─────────────────────────────┘
```

---

## The Dashboard

A dark finance terminal UI at `http://localhost:8003/` — completely different from P1/P2/P3.

Features:
- Company search with quick-select chips (AAPL, MSFT, NVDA, TSLA...)
- Live 4-step pipeline progress visualization
- Structured report with section-by-section breakdown
- Tool call audit trail
- Cost and token tracking per report

---

## Quickstart

```bash
git clone https://github.com/TarunSinghChauhan/finance-research-agent
cd finance-research-agent
cp .env.example .env
# Add OPENROUTER_API_KEY

docker compose up --build
```

Open: **http://localhost:8003/**

---

## Demo

```bash
# Sample analysis (Apple Inc)
curl -X POST http://localhost:8003/research/analyze/sample

# Custom company
curl -X POST http://localhost:8003/research/analyze \
  -H "Content-Type: application/json" \
  -d '{"company": "NVIDIA", "symbol": "NVDA", "query": "AI growth analysis"}'

# Get results
curl http://localhost:8003/research/results/{report_id}

# Get audit trail
curl http://localhost:8003/research/audit/{report_id}
```

---

## Sample Output

```json
{
  "company": "Apple Inc",
  "symbol": "AAPL",
  "market_context": "Apple is trading at $189.50, down 0.8% vs the S&P 500...",
  "financial_analysis": "Apple shows strong fundamentals with $383B revenue...",
  "risk_assessment": "Key risks include iPhone demand saturation...",
  "synthesis": "NEUTRAL — Strong balance sheet offset by slowing growth...",
  "total_cost_usd": 0.0023,
  "total_tokens": 1847,
  "latency_ms": 8420,
  "reproducibility_hash": "a3f9c2b1d4e5f6a7",
  "within_budget": true,
  "tool_calls": [
    {"tool": "get_market_overview", "timestamp": "2026-06-10T12:00:00"},
    {"tool": "get_stock_quote", "timestamp": "2026-06-10T12:00:01"},
    {"tool": "get_company_info", "timestamp": "2026-06-10T12:00:02"},
    {"tool": "get_news", "timestamp": "2026-06-10T12:00:03"}
  ]
}
```

---

## Tech Stack

`Python 3.12` · `FastAPI` · `OpenRouter API` · `Yahoo Finance API` · `LangSmith` · `PostgreSQL` · `Redis` · `Docker`

---

## Project Structure

```
p4-finance-agent/
├── src/
│   ├── agents/research.py     # 4-step reasoning chain
│   ├── tools/financial.py     # Yahoo Finance + news tools
│   ├── api/
│   │   ├── dashboard.html     # Finance terminal UI
│   │   └── routers/research.py
│   └── core/                  # Config, DB, logging
├── tests/
└── docker-compose.yml
```
