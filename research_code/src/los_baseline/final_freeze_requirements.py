from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Iterable


def _normalize_distribution_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def _public_version(value: str) -> str:
    return value.split("+", 1)[0]


def parse_exact_requirements(path: Path) -> dict[str, str]:
    """Parse direct requirements and reject ranges or unpinned packages."""
    if not path.is_file():
        raise FileNotFoundError(f"requirements.txt was not found: {path}")
    requirements: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement")):
            raise ValueError(
                "Nested requirement files are not supported in the release lock: "
                f"line {line_number}: {line}"
            )
        if "==" not in line or any(
            token in line for token in (">=", "<=", "~=", "!=", " @ ")
        ):
            raise ValueError(
                "Every direct dependency must use an exact == pin: "
                f"line {line_number}: {line}"
            )
        name, version = line.split("==", 1)
        name = _normalize_distribution_name(name)
        version = version.strip()
        if not name or not version:
            raise ValueError(
                f"Invalid pinned requirement at line {line_number}: {line}"
            )
        if name in requirements:
            raise ValueError(f"Duplicate requirement: {name}")
        requirements[name] = version
    if not requirements:
        raise ValueError("requirements.txt does not contain pinned dependencies.")
    return requirements


def collect_installed_versions(
    requirement_names: Iterable[str],
) -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    normalized_names = {
        _normalize_distribution_name(value) for value in requirement_names
    }
    for name in sorted(normalized_names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return versions, missing


def validate_installed_versions(
    expected: dict[str, str],
    installed: dict[str, str],
    *,
    allow_local_build_suffix: bool = True,
) -> list[str]:
    errors: list[str] = []
    for name, expected_version in sorted(expected.items()):
        actual = installed.get(name)
        if actual is None:
            errors.append(f"Missing required package: {name}=={expected_version}")
            continue
        comparable_actual = (
            _public_version(actual) if allow_local_build_suffix else actual
        )
        comparable_expected = (
            _public_version(expected_version)
            if allow_local_build_suffix
            else expected_version
        )
        if comparable_actual != comparable_expected:
            errors.append(
                "Package version mismatch: "
                f"{name} expected={expected_version}, actual={actual}"
            )
    return errors
