"""Runs a fake binary's ``__main__`` entry point in a fresh interpreter and reports which of a
given set of heavy modules it loaded, so tests can pin that a CLI path never imports them.

Not a test module (no ``test_`` prefix): shared by ``test_binary_model_conformance.py`` and the
binary-model FeatureGroup's own tests.
"""

from __future__ import annotations

import json
import subprocess  # nosec
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_MODULE_PROBE_TEMPLATE = """
import json
import runpy
import sys

sys.argv = [{module!r}, *{argv!r}]
try:
    runpy.run_module({module!r}, run_name="__main__", alter_sys=True)
    code = 0
except SystemExit as exc:
    code = exc.code
loaded = sorted(m for m in {watched!r} if m in sys.modules)
print(json.dumps({{"code": code, "loaded": loaded}}))
"""


def run_module_probe(
    module: str, argv: Sequence[str], watched: Sequence[str], env: Mapping[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run ``python -m module *argv`` in a subprocess and return it with a summary of its exit code
    and which ``watched`` modules ended up in ``sys.modules``. The summary is the last stdout line,
    so callers must route binary output to ``--output`` rather than stdout."""
    script = _MODULE_PROBE_TEMPLATE.format(module=module, argv=list(argv), watched=tuple(watched))
    completed = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=None if env is None else dict(env),
    )
    assert completed.returncode == 0, f"probe subprocess failed: {completed.stderr}"
    summary: dict[str, Any] = json.loads(completed.stdout.splitlines()[-1])
    return completed, summary
