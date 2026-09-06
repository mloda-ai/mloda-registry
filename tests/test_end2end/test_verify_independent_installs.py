"""Tests that scripts/verify_independent_installs.py runs its per-distribution installs concurrently
through an extracted, isolated ``verify_distribution`` function, and that the workflow step timeout
was not raised to paper over the serial loop it replaces."""

from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify_independent_installs.py"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "verify-published.yaml"

_INDEPENDENT_STEP = "Verify packages install independently"

# Two published distributions whose 'uv pip install' the fakes below fail, to check isolation.
_FAILING_DISTRIBUTIONS = ["mloda-community-ema", "mloda-community-rank"]

# A step name line, followed by its body up to the next step at the same indent or end of file.
_STEP_RE = re.compile(
    r"^(?P<indent>[ \t]*)- name: (?P<name>[^\n]+)\n(?P<body>.*?)(?=^(?P=indent)- name:|\Z)",
    re.MULTILINE | re.DOTALL,
)
_TIMEOUT_RE = re.compile(r"timeout-minutes:\s*(\d+)")

module = load_script("verify_independent_installs", _SCRIPT_PATH)


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

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

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

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

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

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

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

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    exit_code = _run_main(monkeypatch)
    out = capsys.readouterr().out

    assert state["max_in_flight"] > 1, (
        f"main() never ran more than {state['max_in_flight']} subprocess call(s) at once; the "
        "per-distribution cycle must run through a bounded thread pool, not the current serial loop"
    )
    assert exit_code == 1, f"main() exited {exit_code!r} with two failing distributions, expected 1"
    for name in _FAILING_DISTRIBUTIONS:
        assert name in out, f"main() output does not name failing distribution {name!r}: {out!r}"


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
