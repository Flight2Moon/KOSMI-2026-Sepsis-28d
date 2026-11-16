from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.monitor import exp123_dashboard


def test_dashboard_routes_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    app = exp123_dashboard.create_app(root / "configs" / "experiment_resource_cpu.json")
    routes = {route.path for route in app.routes}
    assert "/" in routes
    assert "/api/runs" in routes
    assert "/api/system" in routes
    assert "/api/runs/{run_id}/stream" in routes
    assert "/api/runs/{run_id}/diagnosis" in routes


def test_dashboard_rejects_path_traversal(tmp_path: Path) -> None:
    import pytest
    from fastapi import HTTPException

    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(HTTPException):
        exp123_dashboard.safe_run_directory(runs, "..")
