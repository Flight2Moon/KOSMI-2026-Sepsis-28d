from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EXP123 = PROJECT_ROOT / "run_exp123.py"


def test_run_exp123_main_does_not_shadow_module_json() -> None:
    """Prevent branch-local imports from making json local to main()."""
    tree = ast.parse(RUN_EXP123.read_text(encoding="utf-8"))
    main_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    local_json_bindings: list[int] = []
    for node in ast.walk(main_node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name) == "json":
                    local_json_bindings.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == "json":
                    local_json_bindings.append(node.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "json":
                    local_json_bindings.append(node.lineno)

    assert not local_json_bindings, (
        "run_exp123.main() must use the module-level json import; local bindings "
        f"would break other CLI branches. Found at lines: {local_json_bindings}"
    )


def test_outcomes_cli_reads_manifest_with_module_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import argparse
    import json

    import run_exp123

    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "runs" / "run" / "common" / "outcomes"
    output_dir.mkdir(parents=True)
    (output_dir / "outcome_step_manifest.json").write_text(
        json.dumps(
            {
                "discovery_status": "ready",
                "external_validation_status": "on_hold",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        config=str(config_path),
        run_id="run",
        step="outcomes",
        dataset="both",
        overwrite=False,
        models=None,
        search_stages=None,
        allow_blocking=False,
    )
    monkeypatch.setattr(run_exp123, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_exp123,
        "load_config",
        lambda _: {
            "paths": {"runs_dir": str(tmp_path / "runs")},
            "progress": {},
        },
    )
    monkeypatch.setattr(
        run_exp123,
        "resolve_path",
        lambda config, value: Path(value),
    )
    monkeypatch.setattr(run_exp123, "configure_progress", lambda **_: None)
    monkeypatch.setattr(run_exp123, "configure_torch_runtime", lambda _: None)
    monkeypatch.setattr(run_exp123, "write_hardware_report", lambda *_, **__: None)
    monkeypatch.setattr(run_exp123, "emit_progress", lambda *_, **__: None)
    monkeypatch.setattr(
        run_exp123,
        "_require_clinical_execution_ready",
        lambda **_: None,
    )
    monkeypatch.setattr(
        run_exp123,
        "run_outcome_step",
        lambda **_: (output_dir, 0),
    )

    assert run_exp123.main() == 0
