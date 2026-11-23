"""Runs a fake binary's ``__main__`` entry point in a fresh interpreter and reports which of a
given set of heavy modules it loaded, so tests can pin that a CLI path never imports them.

Not a test module (no ``test_`` prefix): shared by ``test_binary_model_conformance.py`` and the
binary-model FeatureGroup's own tests.
"""

from __future__ import annotations

import subprocess  # nosec
from collections.abc import Mapping, Sequence
from typing import Any

from mloda.testing.import_isolation import _run_probe

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
    return _run_probe(script, [], env)
