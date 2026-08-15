from src.tools.financial import FinancialTools


def test_reproducibility_hash_is_16_chars():
    tools = FinancialTools()
    h = tools.compute_reproducibility_hash("AAPL", [{"tool": "get_stock_quote"}])
    assert len(h) == 16


def test_reproducibility_hash_is_deterministic_same_day():
    tools = FinancialTools()
    calls = [{"tool": "get_stock_quote"}, {"tool": "get_news"}]
    h1 = tools.compute_reproducibility_hash("AAPL", calls)
    h2 = tools.compute_reproducibility_hash("AAPL", calls)
    assert h1 == h2


def test_reproducibility_hash_differs_for_different_company():
    tools = FinancialTools()
    calls = [{"tool": "get_stock_quote"}]
    h1 = tools.compute_reproducibility_hash("AAPL", calls)
    h2 = tools.compute_reproducibility_hash("MSFT", calls)
    assert h1 != h2


def test_reproducibility_hash_differs_for_different_tool_calls():
    tools = FinancialTools()
    h1 = tools.compute_reproducibility_hash("AAPL", [{"tool": "get_stock_quote"}])
    h2 = tools.compute_reproducibility_hash("AAPL", [{"tool": "get_news"}])
    assert h1 != h2


def test_reproducibility_hash_ignores_tool_call_order_details_beyond_tool_name():
    tools = FinancialTools()
    calls_a = [{"tool": "get_stock_quote", "params": {"symbol": "AAPL"}}]
    calls_b = [{"tool": "get_stock_quote", "params": {"symbol": "DIFFERENT"}}]
    h1 = tools.compute_reproducibility_hash("AAPL", calls_a)
    h2 = tools.compute_reproducibility_hash("AAPL", calls_b)
    assert h1 == h2


def test_log_tool_call_appends_entry():
    tools = FinancialTools()
    assert tools.tool_calls == []
    tools._log_tool_call("get_stock_quote", {"symbol": "AAPL"}, {"price": 150})
    assert len(tools.tool_calls) == 1
    assert tools.tool_calls[0]["tool"] == "get_stock_quote"


def test_log_tool_call_truncates_long_result_summary():
    tools = FinancialTools()
    long_result = {"data": "x" * 500}
    tools._log_tool_call("get_company_info", {}, long_result)
    assert len(tools.tool_calls[0]["result_summary"]) <= 200


def test_log_tool_call_records_timestamp():
    tools = FinancialTools()
    tools._log_tool_call("get_news", {"query": "AAPL"}, {"count": 3})
    assert "timestamp" in tools.tool_calls[0]
