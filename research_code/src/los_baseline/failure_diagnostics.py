from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .utils import atomic_write_json

DIRECT_FAILURE_STATUSES = {
    "failed",
    "resource_exceeded",
    "gpu_out_of_memory",
    "system_out_of_memory",
    "cancelled",
}
TERMINAL_STATUSES = DIRECT_FAILURE_STATUSES | {"completed", "blocked"}

_FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "internal_api_contract",
        (
            r"unexpected keyword argument",
            r"missing .* required positional argument",
            r"takes .* positional argument",
            r"got multiple values for argument",
        ),
    ),
    (
        "preflight_or_release_contract",
        (
            r"preflight",
            r"frozen release",
            r"release verification failed",
            r"configuration checksum changed",
            r"live experiment config does not match",
            r"release manifest changed",
        ),
    ),
    (
        "model_interface_error",
        (
            r"svc.*no attribute ['\"]predict_proba",
            r"predict_proba is not available when probability=false",
            r"decision_function.*shape",
        ),
    ),
    (
        "model_configuration_error",
        (
            r"label and prediction size not match",
            r"use merror or mlogloss for multi-class classification",
            r"invalid multiclass probability shape",
            r"invalid multiclass prediction shape",
        ),
    ),
    (
        "sqlite_concurrency",
        (
            r"database is locked",
            r"sqlite3\.operationalerror",
        ),
    ),
    (
        "cuda_out_of_memory",
        (
            r"cuda out of memory",
            r"torch\.outofmemoryerror",
            r"cublas_status_alloc_failed",
        ),
    ),
    (
        "gpu_resource_limit",
        (
            r"gpu .* process memory .* exceeded",
            r"cuda_vram_or_slot",
            r"gpu memory",
            r"invalid device ordinal",
            r"no visible gpu",
        ),
    ),
    (
        "system_memory_limit",
        (
            r"rss .* exceeded configured",
            r"system_out_of_memory",
            r"memoryerror",
            r"killed process",
            r"out of memory",
        ),
    ),
    (
        "artifact_integrity",
        (
            r"artifact",
            r"reloaded artifact predictions differ",
            r"decision mismatch",
            r"sha256_mismatch",
            r"completion=.*invalid",
        ),
    ),
    (
        "xgboost_cuda",
        (
            r"xgboosterror",
            r"xgboost cuda",
            r"device.*cuda",
            r"fallback to dmatrix",
        ),
    ),
    (
        "missing_file_or_path",
        (
            r"filenotfounderror",
            r"no such file or directory",
            r"was not found",
            r"missing_completion_artifact",
        ),
    ),
    (
        "python_exception",
        (
            r"traceback \(most recent call last\)",
            r"runtimeerror",
            r"valueerror",
            r"keyerror",
            r"typeerror",
            r"assertionerror",
        ),
    ),
)


def read_log_tail(path: Path, *, max_lines: int = 160) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def extract_error_excerpt(text: str, *, max_lines: int = 80) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    traceback_positions = [
        index
        for index, line in enumerate(lines)
        if "Traceback (most recent call last)" in line
    ]
    if traceback_positions:
        start = traceback_positions[-1]
        return "\n".join(lines[start : start + max_lines])

    error_pattern = re.compile(
        r"error|exception|failed|failure|out of memory|resource_exceeded|"
        r"missing_completion|checksum|database is locked|traceback",
        re.IGNORECASE,
    )
    matched = [line for line in lines if error_pattern.search(line)]
    if matched:
        return "\n".join(matched[-max_lines:])
    return "\n".join(lines[-min(max_lines, len(lines)):])


def classify_failure(message: str | None, log_text: str | None = None) -> str:
    combined = f"{message or ''}\n{log_text or ''}".lower()
    for category, patterns in _FAILURE_PATTERNS:
        if any(re.search(pattern, combined, re.IGNORECASE) for pattern in patterns):
            return category
    return "unknown"


def _failed_dependencies(
    job_name: str,
    *,
    jobs: Mapping[str, Mapping[str, Any]],
    status: Mapping[str, str],
) -> list[str]:
    dependencies = list(jobs.get(job_name, {}).get("dependencies", []) or [])
    return [
        dependency
        for dependency in dependencies
        if status.get(dependency) in DIRECT_FAILURE_STATUSES | {"blocked"}
    ]


def _root_dependencies(
    job_name: str,
    *,
    jobs: Mapping[str, Mapping[str, Any]],
    status: Mapping[str, str],
    seen: set[str] | None = None,
) -> list[str]:
    seen = set(seen or ())
    if job_name in seen:
        return []
    seen.add(job_name)
    state = status.get(job_name)
    if state in DIRECT_FAILURE_STATUSES:
        return [job_name]
    roots: list[str] = []
    for dependency in _failed_dependencies(job_name, jobs=jobs, status=status):
        roots.extend(
            _root_dependencies(
                dependency,
                jobs=jobs,
                status=status,
                seen=seen,
            )
        )
    return sorted(set(roots))


def build_protocol_result(
    *,
    run_id: str,
    mode: str,
    elapsed_seconds: float,
    jobs: Mapping[str, Mapping[str, Any]],
    status: Mapping[str, str],
    attempts: Mapping[str, int],
    failures: Mapping[str, str],
    log_dir: Path,
) -> dict[str, Any]:
    normalized_status = {str(name): str(value) for name, value in status.items()}
    counts = Counter(normalized_status.values())
    direct_names = [
        name
        for name, state in normalized_status.items()
        if state in DIRECT_FAILURE_STATUSES
    ]
    blocked_names = [
        name for name, state in normalized_status.items() if state == "blocked"
    ]
    completed_names = [
        name for name, state in normalized_status.items() if state == "completed"
    ]

    job_details: dict[str, dict[str, Any]] = {}
    direct_failures: list[dict[str, Any]] = []
    blocked_jobs: list[dict[str, Any]] = []

    for name, state in normalized_status.items():
        job = dict(jobs.get(name, {}))
        log_path = log_dir / f"{name}.log"
        log_tail = read_log_tail(log_path)
        message = failures.get(name)
        detail = {
            "name": name,
            "status": state,
            "attempts": int(attempts.get(name, 0)),
            "dependencies": list(job.get("dependencies", []) or []),
            "failed_dependencies": _failed_dependencies(
                name,
                jobs=jobs,
                status=normalized_status,
            ),
            "root_failed_jobs": _root_dependencies(
                name,
                jobs=jobs,
                status=normalized_status,
            ),
            "failure": message,
            "failure_category": (
                classify_failure(message, log_tail)
                if state in DIRECT_FAILURE_STATUSES
                else None
            ),
            "log_path": str(log_path),
            "log_exists": log_path.is_file(),
            "error_excerpt": (
                extract_error_excerpt(log_tail)
                if state in DIRECT_FAILURE_STATUSES
                else ""
            ),
            "completion_path": job.get("completion_path"),
            "request": job.get("request", {}),
            "command": job.get("command", []),
            "task_key": job.get("task_key"),
            "model_name": job.get("model_name"),
            "kind": job.get("kind"),
        }
        job_details[name] = detail
        if state in DIRECT_FAILURE_STATUSES:
            direct_failures.append(detail)
        elif state == "blocked":
            blocked_jobs.append(detail)

    categories = Counter(
        detail["failure_category"] or "unknown" for detail in direct_failures
    )
    total_failed_or_blocked = len(direct_names) + len(blocked_names)

    return {
        "schema_version": "exp123_resource_protocol_result_v2",
        "run_id": run_id,
        "mode": mode,
        "elapsed_seconds": float(elapsed_seconds),
        "success": total_failed_or_blocked == 0,
        "counts": {
            "total": len(normalized_status),
            "completed": len(completed_names),
            "direct_failed": len(direct_names),
            "blocked": len(blocked_names),
            "failed_or_blocked": total_failed_or_blocked,
            **{key: int(value) for key, value in sorted(counts.items())},
        },
        "status": normalized_status,
        "attempts": {str(name): int(value) for name, value in attempts.items()},
        "failures": dict(failures),
        "completed_jobs": completed_names,
        "direct_failed_jobs": direct_names,
        "blocked_job_names": blocked_names,
        "failure_categories": dict(sorted(categories.items())),
        "root_failures": direct_failures,
        "blocked_jobs": blocked_jobs,
        "jobs": job_details,
        # Backward-compatible integer fields used by existing scripts/tests.
        "completed": len(completed_names),
        "failed": total_failed_or_blocked,
    }


def _markdown_report(result: Mapping[str, Any]) -> str:
    counts = dict(result.get("counts", {}))
    lines = [
        "# EXP123 Resource Protocol Failure Diagnosis",
        "",
        f"- Run ID: `{result.get('run_id')}`",
        f"- Mode: `{result.get('mode', 'unknown')}`",
        f"- Completed: {counts.get('completed', result.get('completed', 0))}",
        f"- Direct failures: {counts.get('direct_failed', 0)}",
        f"- Blocked jobs: {counts.get('blocked', 0)}",
        "",
        "## Root failures",
        "",
    ]
    roots = list(result.get("root_failures", []) or [])
    if not roots:
        lines.append("No direct root failure was recorded.")
    for root in roots:
        lines.extend(
            [
                f"### {root.get('name')}",
                "",
                f"- Status: `{root.get('status')}`",
                f"- Category: `{root.get('failure_category')}`",
                f"- Attempts: {root.get('attempts')}",
                f"- Failure: `{root.get('failure')}`",
                f"- Log: `{root.get('log_path')}`",
                "",
            ]
        )
        excerpt = str(root.get("error_excerpt") or "").strip()
        if excerpt:
            lines.extend(["```text", excerpt, "```", ""])

    lines.extend(["## Blocked jobs", ""])
    blocked = list(result.get("blocked_jobs", []) or [])
    if not blocked:
        lines.append("No blocked jobs were recorded.")
    else:
        for item in blocked:
            roots_text = ", ".join(item.get("root_failed_jobs", []) or []) or "unknown"
            lines.append(f"- `{item.get('name')}` — root: `{roots_text}`")
    lines.append("")
    return "\n".join(lines)


def write_diagnostic_reports(
    result: Mapping[str, Any],
    *,
    progress_dir: Path,
    mode: str,
) -> dict[str, str]:
    progress_dir.mkdir(parents=True, exist_ok=True)
    json_path = progress_dir / f"resource_{mode}_failure_diagnosis.json"
    markdown_path = progress_dir / f"resource_{mode}_failure_diagnosis.md"
    atomic_write_json(json_path, dict(result))
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def normalize_existing_result(
    *,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run_dir: Path,
    mode: str,
) -> dict[str, Any]:
    if result.get("schema_version") == "exp123_resource_protocol_result_v2":
        return dict(result)
    jobs = {
        str(item["name"]): dict(item)
        for item in manifest.get("jobs", [])
        if isinstance(item, Mapping) and item.get("name")
    }
    return build_protocol_result(
        run_id=str(result.get("run_id", run_dir.name)),
        mode=mode,
        elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
        jobs=jobs,
        status=dict(result.get("status", {})),
        attempts=dict(result.get("attempts", {})),
        failures=dict(result.get("failures", {})),
        log_dir=run_dir / "common" / "progress" / "step_logs",
    )
