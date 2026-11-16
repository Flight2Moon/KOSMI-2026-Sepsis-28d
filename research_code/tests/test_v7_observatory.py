from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.monitor.exp123_observatory import build_app


def test_v7_observatory_serves_local_workspace(tmp_path: Path) -> None:
    config = json.loads((Path(__file__).parents[1] / "configs" / "experiment_v7.json").read_text())
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    config["paths"]["runs_dir"] = str(runs_dir)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    client = TestClient(build_app(config_path))
    assert client.get("/").status_code == 200
    assert client.get("/api/runs").json() == {"runs": []}
    system = client.get("/api/system")
    assert system.status_code == 200
    assert "cpu_percent" in system.json()
    assert "total_ram_gb" in system.json()
