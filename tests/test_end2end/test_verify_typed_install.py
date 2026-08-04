"""Typed-install probe in scripts/verify_typed_install.py. Against an install missing py.typed, mypy
sees the plugin as Any and exits 0 with 'Success: no issues found'; only judging the captured output
distinguishes that silent success from a genuinely typed install."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify_typed_install.py"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# The probe targets supported_op_subtypes, a symbol the leaf itself defines; supported_subtypes
# comes from mloda core, so revealing it proves less about the leaf's own typing.
_PROBE_MODULE = "mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation"
_PROBE_LEAF = "mloda-community-aggregation"
_PROBE_BASE = "mloda-community-data-operations"

# Shaped like a real mypy run over the probe against a typed install (line 4 reveals, line 5 miscalls);
# the Green agent replaces these with genuinely captured lines.
_TYPED_REVEAL = (
    "probe.py:4: note: Revealed type is "
    '"def (secondary: Union[builtins.str, None] =) -> Union[frozenset[builtins.str], None]"'
)
_ARG_TYPE_ERROR = (
    'probe.py:5: error: Argument "secondary" to "supported_op_subtypes" of "PyArrowAggregation" '
    'has incompatible type "int"; expected "str | None"  [arg-type]'
)
_ASSIGNMENT_ERROR = (
    'probe.py:5: error: Incompatible types in assignment (expression has type "frozenset[str] | None", '
    'variable has type "int")  [assignment]'
)

_GOOD_OUTPUT = "\n".join(
    [_TYPED_REVEAL, _ARG_TYPE_ERROR, _ASSIGNMENT_ERROR, "Found 2 errors in 1 file (checked 1 source file)"]
)

# The silent-untyped case: mypy reveals bare Any, reports nothing, and exits 0.
_UNTYPED_OUTPUT = "\n".join(['probe.py:4: note: Revealed type is "Any"', "Success: no issues found in 1 source file"])

_NO_REVEAL_OUTPUT = "Success: no issues found in 1 source file"

_MISSING_ARG_TYPE_OUTPUT = "\n".join(
    [_TYPED_REVEAL, _ASSIGNMENT_ERROR, "Found 1 error in 1 file (checked 1 source file)"]
)

_MISSING_ASSIGNMENT_OUTPUT = "\n".join(
    [_TYPED_REVEAL, _ARG_TYPE_ERROR, "Found 1 error in 1 file (checked 1 source file)"]
)

# Any as a parameter annotation inside the revealed signature, not the bare 'Revealed type is "Any"'.
_ANY_IN_SIGNATURE_OUTPUT = "\n".join(
    [
        'probe.py:4: note: Revealed type is "def (secondary: Any =) -> Union[frozenset[builtins.str], None]"',
        _ARG_TYPE_ERROR,
        _ASSIGNMENT_ERROR,
        "Found 2 errors in 1 file (checked 1 source file)",
    ]
)


def _load() -> ModuleType:
    """Import scripts/verify_typed_install.py, the probe runner the typed-install gate uses."""
    assert _SCRIPT_PATH.exists(), f"{_SCRIPT_PATH} is missing; it is what proves the published wheels install typed"
    return load_script("verify_typed_install", _SCRIPT_PATH)


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


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
    """The probe must import the installed class, reveal a leaf-defined symbol, and provoke two typed-only errors."""
    source = _probe_source()
    assert _PROBE_MODULE in source, f"PROBE_SOURCE must import from {_PROBE_MODULE}"
    assert "PyArrowAggregation" in source, "PROBE_SOURCE must import PyArrowAggregation"
    assert "supported_op_subtypes" in source, (
        "PROBE_SOURCE must probe supported_op_subtypes, the symbol the leaf itself defines"
    )
    assert "reveal_type(" in source, "PROBE_SOURCE must reveal_type the supported_op_subtypes attribute"
    assert "secondary=123" in source, (
        "PROBE_SOURCE must call supported_op_subtypes(secondary=123) to provoke [arg-type]"
    )


def test_probe_distributions_and_module_track_the_config() -> None:
    """A delisted or moved probe package must fail loudly here, not silently probe a stale wheel."""
    packages = _load_toml(_PACKAGES_CONFIG)["packages"]
    distributions = getattr(_load(), "PROBE_DISTRIBUTIONS", None)
    assert isinstance(distributions, tuple), "verify_typed_install.PROBE_DISTRIBUTIONS must be a tuple constant"
    assert {_PROBE_LEAF, _PROBE_BASE} <= set(distributions), (
        f"PROBE_DISTRIBUTIONS must pin at least the leaf and its base, got {distributions!r}"
    )
    for name in distributions:
        assert packages.get(name, {}).get("published") is True, (
            f"PROBE_DISTRIBUTIONS lists {name!r}, which config/packages.toml does not flag published = true"
        )
    leaf_root = str(packages[_PROBE_LEAF]["path"]).replace("/", ".")
    assert _PROBE_MODULE.startswith(f"{leaf_root}."), (
        f"the probe module {_PROBE_MODULE!r} must live under the configured {_PROBE_LEAF} path {leaf_root!r}"
    )


def test_a_bare_any_reveal_is_a_problem() -> None:
    """'Revealed type is "Any"' is exactly the silent-untyped case that still exits 0."""
    problems = _mypy_output_problems()(_UNTYPED_OUTPUT)
    assert problems, f"a bare Any reveal must be flagged, output:\n{_UNTYPED_OUTPUT}"
    assert any("Any" in problem for problem in problems), f"the problem must mention Any, got {problems!r}"


@pytest.mark.parametrize(
    ("output", "missing"),
    [
        (_NO_REVEAL_OUTPUT, "a 'Revealed type is' line, so the probe never ran"),
        (_MISSING_ARG_TYPE_OUTPUT, "the provoked '[arg-type]' error"),
        (_MISSING_ASSIGNMENT_OUTPUT, "an '[assignment]' error"),
    ],
    ids=["no-reveal", "no-arg-type-error", "no-assignment-error"],
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
    assert problems == [], f"'secondary: Any' inside a signature must not be flagged, got {problems!r}"
