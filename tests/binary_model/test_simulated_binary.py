"""Wires the conformance kit to our own simulated binary.

``simulated_binary.py`` is a pure-Python CLI stand-in for a future Rust-compiled binary model: a
private test fixture, never published as a package. Every test collected here comes from
``tests.binary_model.conformance_kit``; the ``binary_cmd`` fixture in this directory's
``conftest.py`` points it at ``simulated_binary.py`` via
``[sys.executable, str(simulated_binary_path)]``. A future conformance run against a real binary
is meant to reuse the same ``conformance_kit`` functions unmodified, by supplying its own
``binary_cmd`` fixture instead.
"""

from __future__ import annotations

from tests.binary_model.conftest import SIMULATED_BINARY_PATH

# Star import is deliberate: conformance_kit.py's test_* functions are collected as part of this
# module so pytest picks them up here, wired to our own simulated_binary.py through the
# binary_cmd fixture in this directory's conftest.py.
from tests.binary_model.conformance_kit import *  # noqa: F401,F403


def test_simulated_binary_module_exists() -> None:
    """Sanity check for the wiring itself: the stub the conformance kit invokes is a real file."""
    assert SIMULATED_BINARY_PATH.is_file(), f"expected a file at {SIMULATED_BINARY_PATH}"
