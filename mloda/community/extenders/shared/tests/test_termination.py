"""Tests for termination.py: the opt-in SIGTERM handler. Every case runs in a child process because the
handler must never be installed inside the pytest/xdist process."""

from __future__ import annotations

import signal
import subprocess  # nosec
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

_TIMEOUT = 60.0

_PRELUDE = """
import atexit, os, signal, sys, time
from mloda.community.extenders.shared import termination

def _ready():
    print("ready", flush=True)
    time.sleep(120)
"""

_CASES = {
    "chains_once_and_exits_143": """
calls = []
def previous(signum, frame):
    calls.append(signum)
signal.signal(signal.SIGTERM, previous)
termination.install_sigterm_handler()
termination.install_sigterm_handler()
atexit.register(lambda: print("atexit-ran calls=%d" % len(calls), flush=True))
_ready()
""",
    "raising_previous_handler_does_not_block": """
def previous(signum, frame):
    raise RuntimeError("boom")
signal.signal(signal.SIGTERM, previous)
termination.install_sigterm_handler()
atexit.register(lambda: print("atexit-ran", flush=True))
_ready()
""",
    "watchdog_caps_a_slow_atexit": """
termination.install_sigterm_handler(grace=0.5)
atexit.register(lambda: time.sleep(60))
_ready()
""",
    "sigterm_during_atexit": """
termination.install_sigterm_handler(grace=300.0)
def hook():
    print("ready", flush=True)
    time.sleep(2)
    open(sys.argv[1], "w").write("done")
atexit.register(hook)
""",
    "terminating_flag": """
print("before=%s" % termination.terminating(), flush=True)
termination.install_sigterm_handler()
atexit.register(lambda: print("after=%s" % termination.terminating(), flush=True))
_ready()
""",
}


def _run(case: str, marker: Path | None = None) -> tuple[int, str]:
    proc = subprocess.Popen(  # nosec
        [sys.executable, "-c", textwrap.dedent(_PRELUDE + _CASES[case]), str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        seen = ""
        while "ready" not in seen:
            line = proc.stdout.readline()
            assert line, f"child exited before ready: {proc.stderr.read() if proc.stderr else ''}"
            seen += line
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=_TIMEOUT)
        return proc.returncode, seen + out
    finally:
        proc.kill()
        proc.wait()


class TestInstallSigtermHandler:
    def test_chains_to_the_previous_handler_once_even_when_installed_twice_and_exits_143(self) -> None:
        code, out = _run("chains_once_and_exits_143")

        assert code == 143
        assert "atexit-ran calls=1" in out

    def test_a_raising_previous_handler_does_not_block_the_exit(self) -> None:
        code, out = _run("raising_previous_handler_does_not_block")

        assert code == 143
        assert "atexit-ran" in out

    def test_the_watchdog_ends_a_slow_atexit_hook_after_grace_with_143(self) -> None:
        start = time.monotonic()
        code, _ = _run("watchdog_caps_a_slow_atexit")

        assert code == 143
        assert time.monotonic() - start < 45

    def test_terminating_is_false_before_and_true_after_the_signal(self) -> None:
        code, out = _run("terminating_flag")

        assert code == 143
        assert "before=False" in out
        assert "after=True" in out

    def test_sigterm_during_atexit_does_not_cut_the_hook_short_and_keeps_status_0(self, tmp_path: Path) -> None:
        marker = tmp_path / "hook-done"
        start = time.monotonic()

        code, _ = _run("sigterm_during_atexit", marker)

        assert marker.read_text() == "done"
        assert code == 0
        assert time.monotonic() - start < 45
