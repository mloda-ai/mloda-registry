"""Tests for scripts/check_pypi_names.py: every configured name is registered on PyPI and owned by us."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_pypi_names.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "package-integrity.yaml"
_TOX_INI = _REPO_ROOT / "tox.ini"

_Fetch = Callable[[str], tuple[int, dict[str, Any] | None]]


def _script() -> Any:
    """The script module, loaded by file path."""
    assert _SCRIPT.exists(), f"{_SCRIPT} is missing; it must check PyPI name ownership"
    return load_script("check_pypi_names", _SCRIPT)


def _payload(*users: str) -> dict[str, Any]:
    """A PyPI JSON payload whose ownership roles list the given users."""
    return {"ownership": {"roles": [{"role": "Owner", "user": user} for user in users]}}


def _no_sleep(_seconds: float) -> None:
    """Sleep stand-in so retries cost nothing."""


def _check(fetch: _Fetch, names: list[str], retries: int = 1) -> Any:
    """Run check_names with the injected fetcher and a no-op sleep."""
    return _script().check_names(names, fetch, retries=retries, sleep=_no_sleep)


def _problems(report: Any) -> tuple[list[str], list[str], list[str]]:
    """The (missing, foreign, unverifiable) lists of a report."""
    return list(report.missing), list(report.foreign), list(report.unverifiable)


def test_owned_project_is_ok() -> None:
    """A 200 listing the expected owner lands in no problem list."""
    owner = _script().EXPECTED_OWNER
    assert _problems(_check(lambda _name: (200, _payload("someone", owner)), ["mloda-foo"])) == ([], [], [])


def test_404_is_missing_and_not_retried() -> None:
    """A 404 is reported missing after exactly one fetch."""
    calls: list[str] = []

    def fetch(name: str) -> tuple[int, dict[str, Any] | None]:
        calls.append(name)
        return 404, None

    assert _problems(_check(fetch, ["mloda-foo"], retries=3)) == (["mloda-foo"], [], [])
    assert calls == ["mloda-foo"]


def test_other_owner_is_foreign() -> None:
    """A 200 without the expected owner is foreign."""
    assert _problems(_check(lambda _name: (200, _payload("stranger")), ["mloda-foo"])) == ([], ["mloda-foo"], [])


def test_missing_ownership_key_is_foreign() -> None:
    """A 200 payload with no ownership data cannot prove ownership, so it is foreign."""
    assert _problems(_check(lambda _name: (200, {"info": {}}), ["mloda-foo"])) == ([], ["mloda-foo"], [])


def test_503_then_200_passes_after_retry() -> None:
    """A transient 503 followed by an owned 200 is ok."""
    owner = _script().EXPECTED_OWNER
    responses = iter([(503, None), (200, _payload(owner))])
    assert _problems(_check(lambda _name: next(responses), ["mloda-foo"])) == ([], [], [])


def _always_503(_name: str) -> tuple[int, dict[str, Any] | None]:
    return 503, None


def _always_timeout(_name: str) -> tuple[int, dict[str, Any] | None]:
    raise TimeoutError("timed out")


@pytest.mark.parametrize("fetch", [_always_503, _always_timeout], ids=["status-503", "timeout"])
def test_persistent_failure_is_unverifiable(fetch: _Fetch) -> None:
    """A name that keeps failing after the retries is unverifiable."""
    assert _problems(_check(fetch, ["mloda-foo"])) == ([], [], ["mloda-foo"])


def test_fetch_receives_the_canonical_name() -> None:
    """The fetcher is called with the PEP 503 normalized name, the report keeps the configured one."""
    seen: list[str] = []

    def fetch(name: str) -> tuple[int, dict[str, Any] | None]:
        seen.append(name)
        return 404, None

    assert _script().canonical_name("Mloda_Foo.Bar") == "mloda-foo-bar"
    assert _problems(_check(fetch, ["Mloda_Foo.Bar"])) == (["Mloda_Foo.Bar"], [], [])
    assert seen == ["mloda-foo-bar"]


def test_main_returns_1_when_a_name_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() fails when any configured name is not on PyPI."""
    script = _script()
    monkeypatch.setattr(script, "configured_names", lambda: ["mloda-foo"])
    monkeypatch.setattr(script, "fetch_project", lambda _name: (404, None))
    assert script.main() == 1


def test_main_returns_0_when_all_names_are_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() passes when every configured name is owned by us."""
    script = _script()
    owner = script.EXPECTED_OWNER
    monkeypatch.setattr(script, "configured_names", lambda: ["mloda-foo", "mloda-bar"])
    monkeypatch.setattr(script, "fetch_project", lambda _name: (200, _payload(owner)))
    assert script.main() == 0


def test_package_integrity_workflow_runs_the_script() -> None:
    """The check runs in CI through the package-integrity workflow."""
    assert "scripts/check_pypi_names.py" in _WORKFLOW.read_text()


def test_tox_never_runs_the_network_check() -> None:
    """The check needs the network, so tox must not reference it."""
    assert "check_pypi_names" not in _TOX_INI.read_text()
