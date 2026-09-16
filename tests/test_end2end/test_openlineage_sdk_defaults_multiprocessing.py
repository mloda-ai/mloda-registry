"""The supported OpenLineage pattern under MULTIPROCESSING: no child_bootstrap-equivalent seam exists,
so OpenLineageExtender(use_sdk_defaults=True) must resolve its own client from OPENLINEAGE__* env vars
inherited by the spawned worker at process start; proven with a real spawned worker.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("openlineage.client")

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.user import ParallelizationMode

from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.testing.extenders.runners import expected_value_int, run_value_int


def test_use_sdk_defaults_client_emits_inside_the_spawned_worker(
    tmp_path: Path,
    flight_server: ParallelRunnerFlightServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / "openlineage_sdk_defaults_events.txt"
    # Clear ambient config: OPENLINEAGE_DISABLED/OPENLINEAGE_CONFIG are unset, and chdir plus a
    # fresh HOME rule out an openlineage.yml being picked up from either lookup location, so only
    # the OPENLINEAGE__TRANSPORT__* env vars set below can resolve the transport.
    monkeypatch.delenv("OPENLINEAGE_DISABLED", raising=False)
    monkeypatch.delenv("OPENLINEAGE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENLINEAGE__TRANSPORT__TYPE", "mloda.testing.extenders.openlineage._PidTaggedFileTransport")
    monkeypatch.setenv("OPENLINEAGE__TRANSPORT__LOG_FILE_PATH", str(marker_path))

    values = run_value_int(
        OpenLineageExtender(use_sdk_defaults=True),
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
    )

    assert values == expected_value_int()
    assert marker_path.exists(), (
        "OpenLineageExtender(use_sdk_defaults=True)'s lazily-built client never wrote an event marker "
        "file; the spawned worker never emitted an OpenLineage event"
    )
    lines = [line for line in marker_path.read_text().splitlines() if line.strip()]
    assert lines, "marker file exists but has no content"
    pids = set()
    event_types = set()
    for line in lines:
        pid_str, _, event_type = line.partition(":")
        pids.add(int(pid_str))
        event_types.add(event_type)
    assert os.getpid() not in pids, (
        "OpenLineage events were emitted by the parent test process, not a spawned worker: "
        f"parent pid {os.getpid()} appeared among emitting pids {pids}"
    )
    assert {"START", "COMPLETE"} <= event_types
