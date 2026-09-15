from unittest.mock import AsyncMock, mock_open, patch

import pytest
from fastapi.testclient import TestClient

import src.api.main as main_module


@pytest.fixture
def client():
    with patch.object(main_module, "create_tables", AsyncMock()):
        with TestClient(main_module.app) as c:
            yield c


def test_health_route_is_registered(client):
    resp = client.get("/health/")
    assert resp.status_code == 200


def test_research_sample_companies_route_is_registered(client):
    resp = client.get("/research/companies/sample")
    assert resp.status_code == 200


def test_dashboard_returns_html_file_contents(client):
    fake_html = "<html><body>Dashboard</body></html>"
    with patch("builtins.open", mock_open(read_data=fake_html)):
        resp = client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


def test_lifespan_calls_create_tables_on_startup():
    with patch.object(main_module, "create_tables", AsyncMock()) as mock_create:
        with TestClient(main_module.app):
            pass
        mock_create.assert_awaited_once()
