"""Scheduled core-minor signal in scripts/check_core_minor.py: flag a PyPI mloda release at or above the cap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script
from tests.toml_loader import load_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
chk = load_script("check_core_minor", _REPO_ROOT / "scripts" / "check_core_minor.py")

_CAPPED = "mloda>=0.20.0,<0.21.0"
_PYPI_JSON: dict[str, Any] = {
    "info": {"name": "mloda", "version": "0.21.0"},
    "releases": {"0.20.0": [], "0.21.0": []},
}


class TestCap:
    def test_plain_specifier(self) -> None:
        assert chk.cap(_CAPPED) == (0, 21, 0)

    def test_spaces_tolerated(self) -> None:
        assert chk.cap("mloda >= 0.20.0, < 0.21.0") == (0, 21, 0)

    @pytest.mark.parametrize("spec", ["mloda>=0.20.0", "mloda>=0.20.0,<=0.21.0"])
    def test_without_strict_upper_bound_raises(self, spec: str) -> None:
        with pytest.raises(ValueError):
            chk.cap(spec)


class TestAboveCap:
    @pytest.mark.parametrize(
        "latest, expected",
        [
            ("0.20.5", False),
            ("0.21.0", True),
            ("0.21.3", True),
            ("1.0.0", True),
            ("0.21", True),
            ("0.21.0rc1", False),
        ],
    )
    def test_latest_against_cap(self, latest: str, expected: bool) -> None:
        assert chk.above_cap(_CAPPED, latest) is expected

    def test_real_shared_cap_with_version_just_below(self) -> None:
        core = load_toml(_REPO_ROOT / "config" / "shared.toml")["defaults"]["core_dependency"]
        upper = chk.cap(core)
        just_below = ".".join(str(n) for n in (*upper[:-2], upper[-2] - 1, 99))
        assert chk.above_cap(core, just_below) is False


class TestLatestVersion:
    def test_returns_info_version(self) -> None:
        assert chk.latest_version(_PYPI_JSON) == "0.21.0"

    @pytest.mark.parametrize("payload", [{}, {"releases": {}}, {"info": {"name": "mloda"}}])
    def test_missing_info_or_version_raises(self, payload: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="version|info"):
            chk.latest_version(payload)
