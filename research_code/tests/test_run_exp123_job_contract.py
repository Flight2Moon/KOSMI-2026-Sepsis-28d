from __future__ import annotations

import sys
from pathlib import Path

import run_exp123_job


def test_model_job_uses_final_freeze_public_contract(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text("{}", encoding="utf-8")
    run_root = tmp_path / "runs"
    config = {"paths": {"runs_dir": str(run_root)}}
    calls: dict[str, object] = {}

    monkeypatch.setattr(run_exp123_job, "load_config", lambda _: config)
    monkeypatch.setattr(
        run_exp123_job,
        "resolve_path",
        lambda _config, raw_path: Path(raw_path),
    )
    monkeypatch.setattr(
        run_exp123_job,
        "configure_torch_runtime",
        lambda _config: {"backend": "cpu"},
    )
    monkeypatch.setattr(run_exp123_job, "configure_progress", lambda **_: None)
    monkeypatch.setattr(run_exp123_job, "emit_progress", lambda *_, **__: None)

    def strict_verify_run_preflight(
        *,
        project_root: Path,
        config_path: Path,
        run_dir: Path,
        verify_dataset_hashes: bool = False,
    ) -> dict[str, object]:
        calls["preflight"] = {
            "project_root": project_root,
            "config_path": config_path,
            "run_dir": run_dir,
            "verify_dataset_hashes": verify_dataset_hashes,
        }
        return {"status": "verified"}

    def strict_assert_discovery_step_allowed(run_dir: Path) -> None:
        calls["discovery"] = run_dir

    monkeypatch.setattr(
        run_exp123_job,
        "verify_run_preflight",
        strict_verify_run_preflight,
    )
    monkeypatch.setattr(
        run_exp123_job,
        "assert_discovery_step_allowed",
        strict_assert_discovery_step_allowed,
    )
    monkeypatch.setattr(
        run_exp123_job,
        "run_model_optimization_job",
        lambda **kwargs: {"status": "completed", "kwargs": kwargs},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_exp123_job.py",
            "--config",
            str(config_path),
            "--run-id",
            "contract_run",
            "--job",
            "optimize-model",
            "--task",
            "exp1",
            "--model",
            "decision_tree",
            "--search-stages",
            "smoke",
        ],
    )

    assert run_exp123_job.main() == 0

    expected_run_dir = run_root / "contract_run"
    assert calls["preflight"] == {
        "project_root": run_exp123_job.PROJECT_ROOT,
        "config_path": config_path.resolve(),
        "run_dir": expected_run_dir,
        "verify_dataset_hashes": False,
    }
    assert calls["discovery"] == expected_run_dir


def test_failure_diagnostics_classifies_internal_api_contract() -> None:
    from los_baseline.failure_diagnostics import classify_failure

    log_text = (
        "Traceback (most recent call last):\n"
        "TypeError: verify_run_preflight() got an unexpected keyword argument 'config'\n"
    )
    assert classify_failure(None, log_text) == "internal_api_contract"
