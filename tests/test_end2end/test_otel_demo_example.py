"""End to end: the local OTel + OpenLineage demo (examples/otel_demo) runs on every source and its stack files parse."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk.trace")
yaml = pytest.importorskip("yaml")

from mloda.steward import Extender  # noqa: E402
from openlineage.client.event_v2 import RunEvent, RunState  # noqa: E402

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender  # noqa: E402
from mloda.community.extenders.otel.otel_extender import OtelExtender  # noqa: E402
from mloda.testing.extenders.openlineage import ROOT_JOB_NAME, make_recording_client  # noqa: E402
from mloda.testing.extenders.otel import make_span_capture  # noqa: E402
from tests.script_loader import load_script  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[2] / "examples" / "otel_demo"
SOURCES = ("csv", "parquet", "memory")
LOAD_FORMATS = {"csv": "CsvReader", "parquet": "ParquetReader"}


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    """The demo module, loaded lazily so a missing file fails each test instead of collection."""
    return load_script("otel_demo_features", DEMO_DIR / "demo_features.py")


def _run(demo: ModuleType, source: str, data_dir: Path, fail: bool = False) -> tuple[list[Any], Any, Any]:
    client, transport = make_recording_client()
    provider, exporter = make_span_capture()
    extenders: set[Extender] = {OtelExtender(tracer_provider=provider), OpenLineageExtender(client=client)}
    values: list[Any] = demo.run_pipeline(source, extenders, data_dir, fail=fail)
    return values, exporter, transport


def _spans_with_operation(exporter: Any, operation: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if (s.attributes or {}).get("mloda.operation.name") == operation]


@pytest.mark.parametrize("source", SOURCES)
def test_every_source_has_a_join_span_and_two_calculate_spans(demo: ModuleType, tmp_path: Path, source: str) -> None:
    _, exporter, _ = _run(demo, source, tmp_path)

    assert len(_spans_with_operation(exporter, "join")) >= 1
    assert len(_spans_with_operation(exporter, "calculate")) >= 2


def test_every_source_returns_the_same_values(demo: ModuleType, tmp_path: Path) -> None:
    results = {}
    for source in SOURCES:
        data_dir = tmp_path / source
        data_dir.mkdir()
        values, _, _ = _run(demo, source, data_dir)
        assert values, source
        assert sorted(values) == sorted(["NORTH:5.00", "SOUTH:7.00", "NORTH:9.00", "NORTH:11.00"]), source
        results[source] = values

    assert results["csv"] == results["parquet"] == results["memory"]


@pytest.mark.parametrize("source", sorted(LOAD_FORMATS))
def test_file_sources_have_one_load_span_with_their_reader_format(
    demo: ModuleType, tmp_path: Path, source: str
) -> None:
    _, exporter, _ = _run(demo, source, tmp_path)

    load_spans = _spans_with_operation(exporter, "load")
    assert len(load_spans) == 1
    assert (load_spans[0].attributes or {}).get("mloda.data_access.format") == LOAD_FORMATS[source]


def test_csv_and_parquet_load_identities_differ(demo: ModuleType, tmp_path: Path) -> None:
    identities = {}
    for source in LOAD_FORMATS:
        data_dir = tmp_path / source
        data_dir.mkdir()
        _, exporter, _ = _run(demo, source, data_dir)
        identities[source] = (_spans_with_operation(exporter, "load")[0].attributes or {}).get(
            "mloda.data_access.identity"
        )

    assert identities["csv"] is not None
    assert identities["parquet"] is not None
    assert identities["csv"] != identities["parquet"]


def test_memory_source_has_no_load_span(demo: ModuleType, tmp_path: Path) -> None:
    _, exporter, _ = _run(demo, "memory", tmp_path)

    assert _spans_with_operation(exporter, "load") == []


def test_write_orders_writes_files_for_csv_and_parquet_and_nothing_for_memory(demo: ModuleType, tmp_path: Path) -> None:
    for source in LOAD_FORMATS:
        path = demo.write_orders(tmp_path, source)
        assert path is not None
        assert path.exists()
        assert path.parent == tmp_path
    assert demo.write_orders(tmp_path, "memory") is None


def test_default_data_dir_is_stable(demo: ModuleType) -> None:
    assert demo.default_data_dir() == demo.default_data_dir()
    assert demo.default_data_dir().name == "mloda-otel-demo"


def test_a_step_run_event_carries_mloda_trace(demo: ModuleType, tmp_path: Path) -> None:
    _, _, transport = _run(demo, "csv", tmp_path)

    step_events: list[RunEvent] = [
        e for e in transport.events if e.eventType == RunState.COMPLETE and e.job.name != ROOT_JOB_NAME
    ]
    assert step_events
    assert any("mlodaTrace" in (e.run.facets or {}) for e in step_events)


@pytest.mark.parametrize("source", SOURCES)
def test_fail_raises_and_leaves_a_step_span_with_error_type(demo: ModuleType, tmp_path: Path, source: str) -> None:
    client, _ = make_recording_client()
    provider, exporter = make_span_capture()
    extenders: set[Extender] = {OtelExtender(tracer_provider=provider), OpenLineageExtender(client=client)}

    with pytest.raises(ValueError, match="simulated failure"):
        demo.run_pipeline(source, extenders, tmp_path, fail=True)

    failed = [s for s in exporter.get_finished_spans() if "error.type" in (s.attributes or {})]
    assert failed
    assert any((s.attributes or {}).get("mloda.operation.name") == "calculate" for s in failed)


def _stack_yaml_files() -> Iterator[Path]:
    yield from sorted(p for pattern in ("*.yaml", "*.yml") for p in DEMO_DIR.rglob(pattern))


def test_stack_directory_has_the_expected_files() -> None:
    for name in ("compose.yaml", "otel-collector.yaml", "tempo.yaml", "prometheus.yaml"):
        assert (DEMO_DIR / name).is_file(), name
    assert (DEMO_DIR / "grafana" / "dashboards" / "mloda.json").is_file()


def test_every_yaml_file_parses() -> None:
    paths = list(_stack_yaml_files())
    assert paths, f"no yaml files under {DEMO_DIR}"
    for path in paths:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path


def test_dashboard_json_parses() -> None:
    path = DEMO_DIR / "grafana" / "dashboards" / "mloda.json"
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_every_compose_image_has_an_explicit_non_latest_tag() -> None:
    compose = yaml.safe_load((DEMO_DIR / "compose.yaml").read_text(encoding="utf-8"))
    images = [service["image"] for service in compose["services"].values() if "image" in service]
    assert images
    for image in images:
        tag = image.rsplit("/", 1)[-1].partition(":")[2]
        assert tag, image
        assert tag != "latest", image
