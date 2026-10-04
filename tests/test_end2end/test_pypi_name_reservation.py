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


def test_404_is_missing_and_not_retried() -> None:
    """A 404 is reported missing after exactly one fetch."""
    calls: list[str] = []

    def fetch(name: str) -> tuple[int, dict[str, Any] | None]:
        calls.append(name)
        return 404, None

    assert _problems(_check(fetch, ["mloda-foo"], retries=3)) == (["mloda-foo"], [], [])
    assert calls == ["mloda-foo"]


_OWNER = "<expected owner>"
_Response = tuple[int, dict[str, Any] | None] | Exception
_NONE: tuple[list[str], list[str], list[str]] = ([], [], [])

_FETCH_CASES: list[Any] = [
    pytest.param([(200, _payload("someone", _OWNER))], _NONE, id="owned"),
    pytest.param([(200, _payload("stranger"))], ([], ["mloda-foo"], []), id="other-owner-is-foreign"),
    pytest.param([(200, {"info": {}})], ([], ["mloda-foo"], []), id="missing-ownership-key-is-foreign"),
    pytest.param([(503, None), (200, _payload(_OWNER))], _NONE, id="503-then-owned-passes-after-retry"),
    pytest.param([(503, None), (503, None)], ([], [], ["mloda-foo"]), id="persistent-503-is-unverifiable"),
    pytest.param(
        [TimeoutError("timed out"), TimeoutError("timed out")], ([], [], ["mloda-foo"]), id="persistent-timeout"
    ),
]


@pytest.mark.parametrize(("responses", "expected"), _FETCH_CASES)
def test_check_names_classifies_fetch_outcomes(
    responses: list[_Response], expected: tuple[list[str], list[str], list[str]]
) -> None:
    """Replayed fetch responses land the name in the expected (missing, foreign, unverifiable) lists."""
    owner = _script().EXPECTED_OWNER
    replay = iter(responses)

    def fetch(_name: str) -> tuple[int, dict[str, Any] | None]:
        response = next(replay)
        if isinstance(response, Exception):
            raise response
        status, payload = response
        if payload is not None and "ownership" in payload:
            roles = [
                {**role, "user": owner if role["user"] == _OWNER else role["user"]}
                for role in payload["ownership"]["roles"]
            ]
            payload = {"ownership": {"roles": roles}}
        return status, payload

    assert _problems(_check(fetch, ["mloda-foo"])) == expected


def test_fetch_receives_the_canonical_name() -> None:
    """The fetcher is called with the PEP 503 normalized name, the report keeps the configured one."""
    seen: list[str] = []

    def fetch(name: str) -> tuple[int, dict[str, Any] | None]:
        seen.append(name)
        return 404, None

    assert _script().canonical_name("Mloda_Foo.Bar") == "mloda-foo-bar"
    assert _problems(_check(fetch, ["Mloda_Foo.Bar"])) == (["Mloda_Foo.Bar"], [], [])
    assert seen == ["mloda-foo-bar"]


@pytest.mark.parametrize(
    ("result", "exit_code"),
    [((404, None), 1), ((200, _payload(_OWNER)), 0)],
    ids=["missing-name-fails", "all-owned-passes"],
)
def test_main_exit_code(
    monkeypatch: pytest.MonkeyPatch, result: tuple[int, dict[str, Any] | None], exit_code: int
) -> None:
    """main() exits 1 when a name is missing and 0 when every name is owned."""
    script = _script()
    if result[1] is not None:
        result = (result[0], _payload(script.EXPECTED_OWNER))
    monkeypatch.setattr(script, "configured_names", lambda: ["mloda-foo", "mloda-bar"])
    monkeypatch.setattr(script, "fetch_project", lambda _name: result)
    assert script.main() == exit_code


def test_package_integrity_workflow_runs_the_script() -> None:
    """The check runs in CI through the package-integrity workflow."""
    assert "scripts/check_pypi_names.py" in _WORKFLOW.read_text()


def test_tox_never_runs_the_network_check() -> None:
    """The check needs the network, so tox must not reference it."""
    assert "check_pypi_names" not in _TOX_INI.read_text()
