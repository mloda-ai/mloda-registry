"""A picklable injected OpenLineageClient must reach a real spawned MULTIPROCESSING worker: the
client rides along inside the pickled OpenLineageExtender itself, unlike OTel's ambient
tracer_provider resolution, which needs a child_bootstrap.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("openlineage.client")

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.user import ParallelizationMode
from openlineage.client.client import Event, OpenLineageClient
from openlineage.client.event_v2 import RunEvent
from openlineage.client.transport.transport import Config, Transport

from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.testing.extenders.runners import expected_value_int, run_value_int


class _FileTransport(Transport):
    """Appends one line per emitted event's type to marker_path.

    Used from inside a real spawned MULTIPROCESSING worker: a RecordingTransport's in-memory list
    lives only in the worker's own memory and is never visible back in the parent (pytest) process,
    so a plain file is the only cross-process-visible sink here.
    """

    kind = "file-transport"
    config_class = Config

    def __init__(self, marker_path: Path) -> None:
        self._marker_path = marker_path

    def emit(self, event: Event) -> None:
        if not isinstance(event, RunEvent):
            raise TypeError(f"_FileTransport only records RunEvent, got {type(event).__name__}")
        if event.eventType is None:
            return
        with open(self._marker_path, "a") as handle:
            handle.write(f"{event.eventType.value}\n")


@pytest.fixture(scope="module")
def flight_server() -> Iterator[ParallelRunnerFlightServer]:
    """Required by ParallelizationMode.MULTIPROCESSING."""
    server = ParallelRunnerFlightServer()
    yield server
    server.end_flight_server_process()


def test_injected_client_emits_into_a_real_spawned_worker(
    tmp_path: Path, flight_server: ParallelRunnerFlightServer
) -> None:
    marker_path = tmp_path / "openlineage_multiprocessing_events.txt"
    client = OpenLineageClient(transport=_FileTransport(marker_path))

    values = run_value_int(
        OpenLineageExtender(client=client),
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
    )

    assert values == expected_value_int()
    assert marker_path.exists(), (
        "the injected client's transport never wrote an event marker file; the spawned worker "
        "never emitted an OpenLineage event for the injected client"
    )
    event_types = marker_path.read_text().splitlines()
    assert "START" in event_types
    assert "COMPLETE" in event_types
