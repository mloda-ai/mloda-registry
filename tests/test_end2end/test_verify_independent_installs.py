"""Tests that scripts/verify_independent_installs.py runs its per-distribution installs concurrently
through an extracted, isolated ``verify_distribution`` function, and that the workflow step timeout
was not raised to paper over the serial loop it replaces."""

from __future__ import annotations

import re
import subprocess  # nosec
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script
from tests.toml_loader import load_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify_independent_installs.py"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "verify-published.yaml"

_LICENSE_KIT_KEY = "mloda-enterprise[ed25519]==9.9.9 mloda-testing[binary-model]==9.9.9"
_LICENSE_KIT_PREFIX = "mloda.enterprise.licensing.testing"

_INDEPENDENT_STEP = "Verify packages install independently"

# The one Verify step allowed to run unconditionally; every step after it must run regardless.
_FIRST_VERIFY_STEP = "Verify published packages (all together)"

# Requirement specs (distributions and kit job requirements) whose 'uv pip install' the fakes below fail.
_FAILING_DISTRIBUTIONS = [
    "mloda-community-ema",
    "mloda-community-rank",
    "mloda-testing[otel]",
    "mloda-enterprise[ed25519]",
]

# A step name line, followed by its body up to the next step at the same indent or end of file.
_STEP_RE = re.compile(
    r"^(?P<indent>[ \t]*)- name: (?P<name>[^\n]+)\n(?P<body>.*?)(?=^(?P=indent)- name:|\Z)",
    re.MULTILINE | re.DOTALL,
)
_TIMEOUT_RE = re.compile(r"timeout-minutes:\s*(\d+)")

module = load_script("verify_independent_installs", _SCRIPT_PATH)
kit_map = load_script("kit_modules", _SCRIPTS_DIR / "kit_modules.py")


class _FakeCompletedProcess:
    """Stand-in for subprocess.CompletedProcess; the script reads returncode, stdout and stderr."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _verify_distribution_fn() -> Callable[[str, str, list[str]], str | None]:
    """The per-distribution install-and-probe cycle that main() must delegate to."""
    fn: Callable[[str, str, list[str]], str | None] | None = getattr(module, "verify_distribution", None)
    assert callable(fn), "verify_independent_installs.verify_distribution must be a callable"
    return fn


def _run_main(monkeypatch: pytest.MonkeyPatch, version: str = "9.9.9") -> int:
    """Run main() in-process the way tox invokes the script, with a fake version."""
    monkeypatch.syspath_prepend(str(_SCRIPTS_DIR))
    monkeypatch.setattr(sys, "argv", ["verify_independent_installs.py", version])
    main: Callable[[], int] | None = getattr(module, "main", None)
    assert callable(main), "verify_independent_installs.main must be a callable returning an exit code"
    return main()


def _verify_step_timeouts() -> dict[str, int]:
    """timeout-minutes declared by every 'Verify ...' step in the weekly verification workflow."""
    text = _WORKFLOW_PATH.read_text()
    steps: dict[str, int] = {}
    for match in _STEP_RE.finditer(text):
        name = match.group("name").strip()
        if not name.startswith("Verify "):
            continue
        timeout_match = _TIMEOUT_RE.search(match.group("body"))
        assert timeout_match is not None, (
            f".github/workflows/verify-published.yaml step '{name}' has no 'timeout-minutes'"
        )
        steps[name] = int(timeout_match.group(1))
    return steps


def test_verify_distribution_returns_none_and_runs_from_its_own_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful run reports no error and runs its three commands from an isolated directory."""
    fn = _verify_distribution_fn()
    calls: list[list[str]] = []

    def _fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        cwd = Path(kwargs["cwd"])
        assert cwd.exists(), f"verify_distribution ran a command with cwd={cwd}, which does not exist"
        assert cwd != _REPO_ROOT, "verify_distribution must run in its own temporary directory, not the repo root"
        calls.append(command)
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    error = fn("mloda-registry", "9.9.9", ["mloda.registry"])

    assert error is None, f"verify_distribution() returned {error!r} for an all-success fake, expected None"
    assert len(calls) == 3, f"verify_distribution() ran {len(calls)} command(s), expected exactly 3"


def test_verify_distribution_stops_at_the_first_failing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing 'uv venv' stops before installing or probing; the error names the distribution and uv."""
    fn = _verify_distribution_fn()
    calls: list[list[str]] = []

    def _fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        calls.append(command)
        return _FakeCompletedProcess(returncode=1, stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    error = fn("mloda-registry", "9.9.9", ["mloda.registry"])

    assert len(calls) == 1, f"verify_distribution() ran {len(calls)} command(s) after the first one failed, expected 1"
    assert error is not None, "verify_distribution() returned None for a failing first command"
    assert "mloda-registry" in error, f"error {error!r} does not name the failing distribution"
    assert "uv" in error, f"error {error!r} does not name the failing command's first word 'uv'"
    assert "boom" in error, f"error {error!r} does not contain the failing command's stderr"


def test_main_runs_the_per_distribution_loop_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """More than one distribution's subprocess calls must be in flight at the same time."""
    lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0}

    def _fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        with lock:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    exit_code = _run_main(monkeypatch)

    assert exit_code == 0, f"main() exited {exit_code!r} against an all-success fake, expected 0"
    assert state["max_in_flight"] > 1, (
        f"main() never ran more than {state['max_in_flight']} subprocess call(s) at once; the "
        "per-distribution cycle must run through a bounded thread pool, not the current serial loop"
    )


def test_main_reports_every_failing_distribution_while_running_concurrently(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One distribution's install failure must not stop the others, even while several run at once."""
    lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0}

    def _fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        with lock:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        joined = " ".join(command)
        if command[:3] == ["uv", "pip", "install"] and any(f"{name}==" in joined for name in _FAILING_DISTRIBUTIONS):
            return _FakeCompletedProcess(returncode=1, stderr="boom")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    exit_code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert state["max_in_flight"] > 1, (
        f"main() never ran more than {state['max_in_flight']} subprocess call(s) at once; the "
        "per-distribution cycle must run through a bounded thread pool, not the current serial loop"
    )
    assert exit_code == 1, f"main() exited {exit_code!r} with failing distributions (including a kit job), expected 1"
    for name in _FAILING_DISTRIBUTIONS:
        assert f"✗ {name}" in out, f"main() output does not report failing distribution {name!r}: {out!r}"
    assert f"✗ {_LICENSE_KIT_KEY}" in out, (
        f"main() output does not report the failing kit job {_LICENSE_KIT_KEY!r}: {out!r}"
    )


def test_later_verify_steps_run_independently_of_earlier_failures() -> None:
    """A failing probe must not skip the Verify steps after it, so each carries an '!cancelled()' 'if:'."""
    text = _WORKFLOW_PATH.read_text()
    verify_steps = [match for match in _STEP_RE.finditer(text) if match.group("name").strip().startswith("Verify ")]
    names = [match.group("name").strip() for match in verify_steps]
    assert _FIRST_VERIFY_STEP in names, (
        f".github/workflows/verify-published.yaml has no '- name: {_FIRST_VERIFY_STEP}'; this test cannot "
        "tell the first Verify step from the rest without it"
    )

    missing_if = []
    for match in verify_steps:
        name = match.group("name").strip()
        if name == _FIRST_VERIFY_STEP:
            continue
        if_match = re.search(r"^\s*if:\s*(.+)$", match.group("body"), re.MULTILINE)
        if (
            if_match is None
            or "!cancelled()" not in if_match.group(1)
            or ("steps.get_version.outcome == 'success'" not in if_match.group(1))
        ):
            missing_if.append(name)

    assert missing_if == [], (
        f".github/workflows/verify-published.yaml steps {missing_if} lack an 'if:' containing both "
        "'!cancelled()' and \"steps.get_version.outcome == 'success'\", so a failure in an earlier "
        "Verify step (or in 'Get latest release version') skips them instead of running independently"
    )


def test_independent_install_step_has_no_longer_timeout_than_the_other_verify_steps() -> None:
    """A longer timeout papered over the serial loop; parallelizing it is the fix, not a bigger budget."""
    steps = _verify_step_timeouts()
    assert _INDEPENDENT_STEP in steps, f".github/workflows/verify-published.yaml has no '- name: {_INDEPENDENT_STEP}'"
    others = {name: timeout for name, timeout in steps.items() if name != _INDEPENDENT_STEP}
    assert others, "fixture assumption: the workflow has other 'Verify ...' steps to compare against"
    assert steps[_INDEPENDENT_STEP] <= min(others.values()), (
        f"'{_INDEPENDENT_STEP}' has timeout-minutes={steps[_INDEPENDENT_STEP]}, higher than the other "
        f"Verify steps' minimum {min(others.values())} ({others}); parallelize the per-distribution "
        "loop instead of extending the timeout."
    )


def _kit_jobs() -> dict[str, str]:
    """Requirements joined with one space to probe source for every kit job main() must run."""
    fn: Callable[[str], list[tuple[tuple[str, ...], str]]] | None = getattr(module, "kit_jobs", None)
    assert callable(fn), "verify_independent_installs.kit_jobs must be a callable"
    jobs = fn("9.9.9")
    keys = [" ".join(requirements) for requirements, _probe in jobs]
    assert len(keys) == len(set(keys)), f"two kit jobs share a key: {sorted(keys)}"
    return dict(zip(keys, (probe for _requirements, probe in jobs), strict=True))


def _probe_imports(probe: str) -> list[str]:
    return [line.split()[1] for line in probe.splitlines() if line.startswith("import ")]


def test_bare_kit_job_probes_ungated_modules_with_no_plugin_package_present() -> None:
    """The bare mloda-testing probe asserts no community/enterprise package and imports only ungated modules."""
    probe = _kit_jobs()["mloda-testing==9.9.9"]
    for package in ("mloda.community", "mloda.enterprise"):
        assert f'find_spec("{package}") is None' in probe or f"find_spec('{package}') is None" in probe, (
            f"bare kit probe does not assert find_spec({package!r}) is None: {probe!r}"
        )
    ungated = kit_map.kit_modules()[None]
    assert sorted(_probe_imports(probe)) == sorted(ungated), "bare kit probe must import exactly the ungated modules"


@pytest.mark.parametrize("spec", sorted(set(kit_map.kit_modules()) - {None}))
def test_gated_kit_job_probes_no_plugin_package_present(spec: str) -> None:
    """Every kit job asserts no enterprise package is importable; all but binary-model also no community package."""
    probe = _kit_jobs()[f"mloda-testing[{spec}]==9.9.9"]

    def asserted(package: str) -> bool:
        return (
            f'find_spec("{package}") is None, "{package} must be absent"' in probe
            or f"find_spec('{package}') is None, '{package} must be absent'" in probe
        )

    assert asserted("mloda.enterprise"), f"[{spec}] kit probe does not assert mloda.enterprise is absent: {probe!r}"
    assert asserted("mloda.community") == (spec != "binary-model"), (
        f"[{spec}] kit probe has the wrong mloda.community absence assertion: {probe!r}"
    )


def test_one_kit_job_per_gate_extra_imports_exactly_that_extras_modules() -> None:
    """Besides the bare and gated mloda-testing jobs, the only other kit job is the license kit job."""
    jobs = _kit_jobs()
    gated = {gate: modules for gate, modules in kit_map.kit_modules().items() if gate is not None}
    assert set(jobs) == (
        {"mloda-testing==9.9.9", _LICENSE_KIT_KEY} | {f"mloda-testing[{gate}]==9.9.9" for gate in gated}
    ), sorted(jobs)
    for gate, modules in gated.items():
        imported = sorted(_probe_imports(jobs[f"mloda-testing[{gate}]==9.9.9"]))
        assert imported == sorted(modules), f"the [{gate}] kit job imports {imported}, expected {sorted(modules)}"


def test_license_kit_job_probes_the_shipped_modules_of_published_extras() -> None:
    """The license kit job imports the shipped modules and names published packages' declared extras."""
    probe = _kit_jobs()[_LICENSE_KIT_KEY]
    imported = _probe_imports(probe)
    expected = [_LICENSE_KIT_PREFIX] + [
        f"{_LICENSE_KIT_PREFIX}.{name}" for name in ("conformance", "license_vectors", "licensed_binary")
    ]
    assert set(expected) <= set(imported), f"license kit probe imports {imported}, expected at least {expected}"
    outside = [
        name for name in imported if name != _LICENSE_KIT_PREFIX and not name.startswith(f"{_LICENSE_KIT_PREFIX}.")
    ]
    assert outside == [], f"license kit probe imports modules outside {_LICENSE_KIT_PREFIX}: {outside}"
    assert "find_spec" not in probe, f"license kit probe must not assert module absence: {probe!r}"

    packages = load_toml(_PACKAGES_CONFIG)["packages"]
    for requirement in _LICENSE_KIT_KEY.split(" "):
        name, _, rest = requirement.partition("[")
        extra = rest.split("]")[0]
        assert name in packages and packages[name].get("published") is True, f"{name} is not a published package"
        assert extra in packages[name].get("optional_dependencies", {}), f"{name} declares no extra {extra!r}"


def test_kit_jobs_reject_a_license_kit_path_with_no_shipped_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty license kit probe would pass without checking anything, so kit_jobs must refuse it."""
    missing = "mloda/enterprise/no_such_kit"
    monkeypatch.setattr(module, "LICENSE_KIT_PATH", missing)
    with pytest.raises(ValueError, match=re.escape(missing)):
        module.kit_jobs("9.9.9")


def _record_installs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Fake subprocess.run; the list collects the requirements of each 'uv pip install' call."""
    requirements: list[list[str]] = []
    lock = threading.Lock()

    def _fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        if command[:3] == ["uv", "pip", "install"]:
            with lock:
                requirements.append(command[5:])
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    return requirements


def test_main_installs_every_kit_job_specifier(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requirements = _record_installs(monkeypatch)
    exit_code = _run_main(monkeypatch)
    header = next(line for line in capsys.readouterr().out.splitlines() if "Installing" in line)
    assert f"+ {len(_kit_jobs())} kit jobs" in header, f"main() header does not count the kit jobs: {header!r}"
    assert exit_code == 0, f"main() exited {exit_code!r} against an all-success fake, expected 0"
    installed = [requirement for call in requirements for requirement in call]
    missing = [spec for key in _kit_jobs() for spec in key.split(" ") if spec not in installed]
    assert missing == [], f"main() never installed the kit job requirement(s) {missing}"
    license_requirements = _LICENSE_KIT_KEY.split(" ")
    assert any(all(spec in call for spec in license_requirements) for call in requirements), (
        f"main() did not install {license_requirements} in one 'uv pip install' call: {requirements}"
    )
