import pytest
from tenacity import wait_none

import src.tools.financial as financial


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    """Retries still happen in tests, but without sleeping 1-5 seconds between attempts."""
    monkeypatch.setattr(financial._fetch_json.retry, "wait", wait_none())
