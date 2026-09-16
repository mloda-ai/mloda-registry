"""Repo-wide pytest fixtures. Lives at the repo root (not under mloda/) so it can freely import
mloda.core internals without tripping tests/test_end2end/test_no_internal_core_imports.py, which
only scans the mloda/ tree.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer


@pytest.fixture(scope="module")
def flight_server() -> Iterator[ParallelRunnerFlightServer]:
    """Required by ParallelizationMode.MULTIPROCESSING; shared across every test module that needs it."""
    server = ParallelRunnerFlightServer()
    yield server
    server.end_flight_server_process()
