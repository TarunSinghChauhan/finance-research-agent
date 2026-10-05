from datetime import datetime, timedelta, timezone

from src.core.database import ResearchReport, _utcnow


def _now_naive_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def test_utcnow_returns_naive_value_close_to_current_utc():
    value = _utcnow()
    assert value.tzinfo is None
    assert abs(_now_naive_utc() - value) < timedelta(seconds=5)


def test_research_report_created_at_default_is_naive_utc():
    report = ResearchReport(company="ACME", query="outlook")
    assert report.created_at.tzinfo is None
    assert abs(_now_naive_utc() - report.created_at) < timedelta(seconds=5)
