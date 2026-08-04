"""Typed-install probe in scripts/verify_typed_install.py. Against an install missing py.typed, mypy
sees the plugin as Any and exits 0 with 'Success: no issues found'; only judging the captured output
distinguishes that silent success from a genuinely typed install."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify_typed_install.py"

_PROBE_MODULE = "mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation"

# Lines a real mypy run over the probe produces against a typed install.
_TYPED_REVEAL = 'probe.py:4: note: Revealed type is "def (compute_framework: Any =) -> frozenset[builtins.str]"'
_CALL_ARG_ERROR = (
    'probe.py:5: error: Unexpected keyword argument "secondary" for "supported_subtypes" '
    'of "PyArrowAggregation"  [call-arg]'
)
_ASSIGNMENT_ERROR = (
    'probe.py:5: error: Incompatible types in assignment (expression has type "frozenset[str]", '
    'variable has type "int")  [assignment]'
)

_GOOD_OUTPUT = "\n".join(
    [_TYPED_REVEAL, _CALL_ARG_ERROR, _ASSIGNMENT_ERROR, "Found 2 errors in 1 file (checked 1 source file)"]
)

# The silent-untyped case: mypy reveals bare Any, reports nothing, and exits 0.
_UNTYPED_OUTPUT = "\n".join(['probe.py:4: note: Revealed type is "Any"', "Success: no issues found in 1 source file"])

_NO_REVEAL_OUTPUT = "Success: no issues found in 1 source file"

_MISSING_CALL_ARG_OUTPUT = "\n".join(
    [_TYPED_REVEAL, _ASSIGNMENT_ERROR, "Found 1 error in 1 file (checked 1 source file)"]
)

_MISSING_ASSIGNMENT_OUTPUT = "\n".join(
    [_TYPED_REVEAL, _CALL_ARG_ERROR, "Found 1 error in 1 file (checked 1 source file)"]
)

# Any as a parameter annotation inside the revealed signature, not the bare 'Revealed type is "Any"'.
_ANY_IN_SIGNATURE_OUTPUT = "\n".join(
    [
        'probe.py:4: note: Revealed type is "def (compute_framework: Any) -> frozenset[str]"',
        _CALL_ARG_ERROR,
        _ASSIGNMENT_ERROR,
        "Found 2 errors in 1 file (checked 1 source file)",
    ]
)


def _load() -> ModuleType:
    """Import scripts/verify_typed_install.py, the probe runner the typed-install gate uses."""
    assert _SCRIPT_PATH.exists(), f"{_SCRIPT_PATH} is missing; it is what proves the published wheels install typed"
    return load_script("verify_typed_install", _SCRIPT_PATH)


def _probe_source() -> str:
    """The PROBE_SOURCE constant verify_typed_install must expose."""
    source = getattr(_load(), "PROBE_SOURCE", None)
    assert isinstance(source, str), "verify_typed_install.PROBE_SOURCE must be a str constant"
    return source


def _mypy_output_problems() -> Callable[[str], list[str]]:
    """The pure output judge verify_typed_install must expose."""
    judge: Callable[[str], list[str]] | None = getattr(_load(), "mypy_output_problems", None)
    assert callable(judge), "verify_typed_install.mypy_output_problems(output) must be a callable"
    return judge


def test_probe_source_imports_reveals_and_miscalls_the_plugin() -> None:
    """The probe must import the installed class, reveal its type, and provoke two typed-only errors."""
    source = _probe_source()
    assert _PROBE_MODULE in source, f"PROBE_SOURCE must import from {_PROBE_MODULE}"
    assert "PyArrowAggregation" in source, "PROBE_SOURCE must import PyArrowAggregation"
    assert "reveal_type(" in source, "PROBE_SOURCE must reveal_type the supported_subtypes attribute"
    assert "secondary=123" in source, "PROBE_SOURCE must call supported_subtypes(secondary=123) to provoke [call-arg]"


def test_a_bare_any_reveal_is_a_problem() -> None:
    """'Revealed type is "Any"' is exactly the silent-untyped case that still exits 0."""
    problems = _mypy_output_problems()(_UNTYPED_OUTPUT)
    assert problems, f"a bare Any reveal must be flagged, output:\n{_UNTYPED_OUTPUT}"
    assert any("Any" in problem for problem in problems), f"the problem must mention Any, got {problems!r}"


@pytest.mark.parametrize(
    ("output", "missing"),
    [
        (_NO_REVEAL_OUTPUT, "a 'Revealed type is' line, so the probe never ran"),
        (_MISSING_CALL_ARG_OUTPUT, "the expected 'Unexpected keyword argument \"secondary\"' error"),
        (_MISSING_ASSIGNMENT_OUTPUT, "an '[assignment]' error"),
    ],
    ids=["no-reveal", "no-call-arg-error", "no-assignment-error"],
)
def test_an_incomplete_output_is_a_problem(output: str, missing: str) -> None:
    """Each expected line proves one thing; with any of them missing the install is not proven typed."""
    problems = _mypy_output_problems()(output)
    assert problems, f"output missing {missing} must yield a problem, but judged clean:\n{output}"


def test_a_fully_typed_output_yields_no_problems() -> None:
    """A concrete reveal plus both provoked errors is the proof that the install is typed."""
    problems = _mypy_output_problems()(_GOOD_OUTPUT)
    assert problems == [], f"a fully typed output must judge clean, got {problems!r}"


def test_any_inside_a_revealed_signature_is_not_flagged() -> None:
    """Only the bare 'Revealed type is "Any"' is untyped; Any as a parameter annotation is legitimate."""
    problems = _mypy_output_problems()(_ANY_IN_SIGNATURE_OUTPUT)
    assert problems == [], f"'compute_framework: Any' inside a signature must not be flagged, got {problems!r}"
