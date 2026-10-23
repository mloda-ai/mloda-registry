"""Typed-install probe in scripts/verify_typed_install.py. Against an install missing py.typed, mypy
sees the plugin as Any and exits 0 with 'Success: no issues found'; only judging the captured output
distinguishes that silent success from a genuinely typed install."""

from __future__ import annotations

import subprocess  # nosec
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.script_loader import load_script
from tests.test_end2end.test_verify_independent_installs import _FakeCompletedProcess
from tests.toml_loader import load_toml
from tests.tox_ini import tox_deps

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify_typed_install.py"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# PyArrowAggregation is an Implementation: the probe reveals and miscalls `calculate`, which the leaf itself
# defines, and misassigns `supported`, which the leaf declares. Inherited names (e.g. `divergent`) would prove less.
_PROBE_MODULE = "mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation"
_PROBE_LEAF = "mloda-community-aggregation"
_PROBE_BASE = "mloda-community-data-operations"

# Shaped like a real mypy run over the probe against a typed install (line 5 reveals, line 6 misassigns,
# line 7 miscalls); captured from a real mypy run.
_TYPED_REVEAL = (
    "probe.py:5: note: Revealed type is "
    '"def (group: type[mloda.community.feature_groups.data_operations.aggregation.base.AggregationFeatureGroup], '
    "data: Any, features: mloda.core.abstract_plugins.components.feature_set.FeatureSet, "
    'framework: mloda.core.abstract_plugins.compute_framework.ComputeFramework | None) -> Any"'
)
_ARG_TYPE_ERROR = (
    'probe.py:7: error: Argument 1 to "calculate" of "PyArrowAggregation" '
    'has incompatible type "int"; expected "type[AggregationFeatureGroup]"  [arg-type]'
)
_ASSIGNMENT_ERROR = (
    'probe.py:6: error: Incompatible types in assignment (expression has type "frozenset[str]", '
    'variable has type "int")  [assignment]'
)

_GOOD_OUTPUT = "\n".join(
    [_TYPED_REVEAL, _ARG_TYPE_ERROR, _ASSIGNMENT_ERROR, "Found 2 errors in 1 file (checked 1 source file)"]
)

# The silent-untyped case: mypy reveals bare Any, reports nothing, and exits 0.
_UNTYPED_OUTPUT = "\n".join(['probe.py:5: note: Revealed type is "Any"', "Success: no issues found in 1 source file"])

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
        'probe.py:5: note: Revealed type is "def (group: Any, data: Any, features: Any, framework: Any) -> Any"',
        _ARG_TYPE_ERROR,
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
    """The probe must import the installed class, reveal a leaf-defined symbol, and provoke two typed-only errors."""
    source = _probe_source()
    assert _PROBE_MODULE in source, f"PROBE_SOURCE must import from {_PROBE_MODULE}"
    assert "PyArrowAggregation" in source, "PROBE_SOURCE must import PyArrowAggregation"
    assert "supported_op_subtypes" not in source, "supported_op_subtypes is gone from the Implementation leaf"
    assert "reveal_type(PyArrowAggregation.calculate)" in source, (
        "PROBE_SOURCE must reveal_type PyArrowAggregation.calculate, the classmethod the leaf itself defines"
    )
    assert "BAD: int = PyArrowAggregation.supported\n" in source, (
        "PROBE_SOURCE must assign PyArrowAggregation.supported to an int to provoke [assignment]"
    )
    assert "PyArrowAggregation.calculate(123," in source, (
        "PROBE_SOURCE must call PyArrowAggregation.calculate(123, ...) to provoke [arg-type]"
    )


def test_probe_distributions_and_module_track_the_config() -> None:
    """A delisted or moved probe package must fail loudly here, not silently probe a stale wheel."""
    packages = load_toml(_PACKAGES_CONFIG)["packages"]
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


def _run_main(monkeypatch: pytest.MonkeyPatch, install_error: str | None) -> tuple[ModuleType, int, dict[str, Any]]:
    """Run main() in-process with install and subprocess.run faked; returns (script, exit code, recorded calls)."""
    script = _load()
    seen: dict[str, Any] = {"commands": []}

    def fake_install(requirements: list[str], tmpdir: str) -> str | None:
        seen["requirements"], seen["tmpdir"] = requirements, tmpdir
        return install_error

    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        seen["commands"].append((command, kwargs))
        return _FakeCompletedProcess(stdout=_GOOD_OUTPUT)

    monkeypatch.setattr(sys, "argv", ["verify_typed_install.py", "9.9.9"])
    monkeypatch.setattr(script, "install", fake_install)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return script, script.main(), seen


def test_main_reports_a_failed_install_and_skips_mypy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing shared install ends main() with 1 and its error, before any mypy run."""
    _, code, seen = _run_main(monkeypatch, "install exploded")

    assert code == 1, f"main() must return 1 when install fails, got {code}"
    assert "install exploded" in capsys.readouterr().out, "the install error must be printed"
    assert seen["commands"] == [], f"mypy must not run after a failed install, got {seen['commands']}"


def test_main_installs_pins_and_mypy_then_runs_mypy_from_the_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() installs the pinned probe distributions plus mypy via the shared helper, then runs the venv's mypy."""
    script, code, seen = _run_main(monkeypatch, None)

    assert code == 0, f"main() must return 0 on a typed install, got {code}"
    expected = [f"{name}==9.9.9" for name in script.PROBE_DISTRIBUTIONS] + ["mypy"]
    assert seen["requirements"] == expected, f"install got {seen['requirements']}, expected {expected}"
    assert len(seen["commands"]) == 1, f"expected one mypy command, got {seen['commands']}"
    command, kwargs = seen["commands"][0]
    assert command[0] == str(Path(seen["tmpdir"]) / "venv" / "bin" / "python"), f"mypy must use the venv: {command}"
    assert "mypy" in command and "--strict" in command, f"expected a strict mypy run, got {command}"
    assert Path(kwargs["cwd"]) == Path(seen["tmpdir"]), f"mypy must run in the install tmpdir, got {kwargs['cwd']}"


def test_tox_env_installs_a_toml_parser() -> None:
    """Python 3.10 has no tomllib and verify_common pulls in a toml parser, so the env must install tomli."""
    deps = tox_deps("verify-typed-install")
    assert "tomli" in deps, f"verify-typed-install must install tomli, got {deps}"
