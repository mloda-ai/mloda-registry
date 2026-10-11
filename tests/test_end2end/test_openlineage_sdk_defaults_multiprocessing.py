"""The supported OpenLineage pattern under MULTIPROCESSING: no child_bootstrap-equivalent seam exists,
so OpenLineageExtender(use_sdk_defaults=True) must resolve its own client from OPENLINEAGE__* env vars
inherited by the spawned worker at process start; proven with a real spawned worker.
It also covers SIGTERM to the parent of a MULTIPROCESSING run, with and without the handler in the worker bootstrap.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("openlineage.client")

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.user import ParallelizationMode

from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.testing.extenders.runners import expected_value_int, run_until_ready_then_sigterm, run_value_int


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
    worker_event_types = set()
    for line in lines:
        pid_str, _, event_type = line.partition(":")
        # The parent emits the run's own START/COMPLETE; only a worker's events prove the worker built a client.
        if int(pid_str) != os.getpid():
            worker_event_types.add(event_type)
    assert {"START", "COMPLETE"} <= worker_event_types, (
        f"no spawned worker emitted a START and COMPLETE event (parent pid {os.getpid()}): {lines}"
    )

    parent_lines = [line for line in lines if line.split(":", 1)[0] == str(os.getpid())]
    assert [line.split(":", 1)[1] for line in parent_lines] == ["START", "COMPLETE"], (
        f"the parent process must emit exactly the run START and COMPLETE, got {parent_lines}"
    )


PARENT_GRACE = 30.0
# Below PARENT_GRACE minus core's graceful_shutdown_timeout, so the worker watchdog never races the parent's.
WORKER_GRACE = 15.0

_SIGTERM_TRANSPORT_MODULE = """
import json, os
from pathlib import Path
from openlineage.client.transport.transport import Config, Transport


class SigtermTransport(Transport):
    kind = "sigterm-transport"
    config_class = Config

    def __init__(self, config):
        self._dir = Path(os.environ["MLODA_TEST_DIR"])

    def emit(self, event):
        record = {"pid": os.getpid(), "job": event.job.name, "type": event.eventType.value}
        with open(self._dir / "events.ndjson", "a") as handle:
            handle.write(json.dumps(record) + "\\n")

    def close(self, timeout=-1.0):
        (self._dir / f"closed.{os.getpid()}.marker").write_text("closed")
        return True
"""

_SIGTERM_MULTIPROCESSING_CHILD = """
import json, os, time
from pathlib import Path
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.provider import FeatureGroup
from mloda.user import ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.community.extenders.otel import OtelExtender
from mloda.community.extenders.shared.termination import install_sigterm_handler
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

PARENT_GRACE = __PARENT_GRACE__
WORKER_GRACE = __WORKER_GRACE__


class MlodaTestingSigtermSleeper(FeatureGroup):
    @classmethod
    def input_data(cls):
        return None

    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
        return str(getattr(feature_name, "name", feature_name)) == "sigterm_sleeping"

    @classmethod
    def compute_framework_rule(cls):
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data, features):
        print("ready", flush=True)
        time.sleep(30)
        return {"sigterm_sleeping": [1]}


class MlodaTestingStatusSpanExporter(SpanExporter):
    def export(self, spans):
        with open(Path(os.environ["MLODA_TEST_DIR"]) / "spans.ndjson", "a") as handle:
            for span in spans:
                record = {"pid": os.getpid(), "name": span.name, "status": span.status.status_code.name}
                handle.write(json.dumps(record) + "\\n")
        return SpanExportResult.SUCCESS


def child_bootstrap():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(MlodaTestingStatusSpanExporter()))
    trace.set_tracer_provider(provider)
    if os.environ["MLODA_TEST_WORKER_HANDLER"] == "1":
        install_sigterm_handler(grace=WORKER_GRACE)


if __name__ == "__main__":
    install_sigterm_handler(grace=PARENT_GRACE)
    server = ParallelRunnerFlightServer()
    try:
        mloda.run_all(
            ["sigterm_sleeping"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({MlodaTestingSigtermSleeper}),
            function_extender={OpenLineageExtender(use_sdk_defaults=True), OtelExtender(use_sdk_defaults=True)},
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=server,
            child_bootstrap=child_bootstrap,
        )
    finally:
        server.end_flight_server_process()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("worker_handler", [True, False], ids=["with_handler", "control_without_handler"])
def test_sigterm_to_parent_aborts_the_worker_step_only_when_the_worker_installs_the_handler(
    tmp_path: Path, worker_handler: bool
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    script = tmp_path / "sigterm_multiprocessing_child.py"
    script.write_text(
        _SIGTERM_MULTIPROCESSING_CHILD.replace("__PARENT_GRACE__", repr(PARENT_GRACE)).replace(
            "__WORKER_GRACE__", repr(WORKER_GRACE)
        ),
        encoding="utf-8",
    )
    (tmp_path / "sigterm_transport.py").write_text(_SIGTERM_TRANSPORT_MODULE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENLINEAGE")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    env["MLODA_TEST_DIR"] = str(tmp_path)
    env["MLODA_TEST_WORKER_HANDLER"] = "1" if worker_handler else "0"
    env["OPENLINEAGE__TRANSPORT__TYPE"] = "sigterm_transport.SigtermTransport"

    returncode, output = run_until_ready_then_sigterm([sys.executable, str(script)], env=env, timeout=120)

    assert returncode == 143, output
    events = [json.loads(line) for line in (tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()]
    root = [e for e in events if e["job"] == "mloda.run_all"]
    step = [e for e in events if e["job"] != "mloda.run_all"]
    assert [e["type"] for e in root] == ["START", "ABORT"], events
    parent_pid = root[0]["pid"]
    assert {e["pid"] for e in root} == {parent_pid}, events
    assert {e["pid"] for e in step} - {parent_pid}, f"no worker pid emitted a step event: {events}"
    worker_pid = next(e["pid"] for e in step if e["pid"] != parent_pid)
    worker_step = [e["type"] for e in step if e["pid"] == worker_pid]
    spans_path = tmp_path / "spans.ndjson"
    worker_spans = (
        [r for r in map(json.loads, spans_path.read_text(encoding="utf-8").splitlines()) if r["pid"] == worker_pid]
        if spans_path.exists()
        else []
    )
    calculate_statuses = [r["status"] for r in worker_spans if r["name"].startswith("calculate ")]
    if worker_handler:
        assert worker_step == ["START", "ABORT"], events
        assert (tmp_path / f"closed.{worker_pid}.marker").exists(), output
        assert calculate_statuses == ["ERROR"], worker_spans
    else:
        assert worker_step == ["START"], events
        assert calculate_statuses == [], worker_spans
        assert not (tmp_path / f"closed.{worker_pid}.marker").exists(), output
