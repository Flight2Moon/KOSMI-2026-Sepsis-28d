from __future__ import annotations

import pytest

from los_baseline.study28.audit import assert_required_external_sites


def test_external_completion_guard_rejects_missing_nwicu():
    with pytest.raises(RuntimeError, match="nwicu"):
        assert_required_external_sites(
            ["eicu_alt", "eicu_primary", "sicdb"],
            ["eicu_alt", "eicu_primary", "sicdb", "nwicu"],
        )


def test_external_completion_guard_accepts_all_required_sites():
    assert_required_external_sites(
        ["nwicu", "eicu_alt", "sicdb", "eicu_primary"],
        ["eicu_alt", "eicu_primary", "sicdb", "nwicu"],
    )
