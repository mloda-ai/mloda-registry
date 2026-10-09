"""End to end: the local OTel + OpenLineage demo (examples/otel_demo) runs on every source and its stack files parse."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from mloda.steward import Extender  # noqa: E402
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader  # noqa: E402
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader  # noqa: E402
from openlineage.client.event_v2 import RunEvent, RunState  # noqa: E402

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender  # noqa: E402
from mloda.community.extenders.otel.otel_extender import OtelExtender  # noqa: E402
from mloda.community.extenders.otel.otel_metrics_extender import OtelMetricsExtender  # noqa: E402
from mloda.testing.extenders.openlineage import ROOT_JOB_NAME, make_recording_client  # noqa: E402
from mloda.testing.extenders.otel import make_metric_capture, make_span_capture  # noqa: E402
from tests.script_loader import load_script  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[2] / "examples" / "otel_demo"
SOURCES = ("csv", "parquet", "memory")
LOAD_FORMATS = {"csv": "CsvReader", "parquet": "ParquetReader"}


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    """The demo module, loaded lazily so a missing file fails each test instead of collection."""
    return load_script("otel_demo_features", DEMO_DIR / "demo_features.py")


def _capture() -> tuple[set[Extender], Any, Any]:
    client, transport = make_recording_client()
    provider, exporter = make_span_capture()
    extenders: set[Extender] = {OtelExtender(tracer_provider=provider), OpenLineageExtender(client=client)}
    return extenders, exporter, transport


def _run(demo: ModuleType, source: str, data_dir: Path) -> tuple[list[Any], Any, Any]:
    extenders, exporter, transport = _capture()
    values: list[Any] = demo.run_pipeline(source, extenders, data_dir)
    return values, exporter, transport


def _spans_with_operation(exporter: Any, operation: str) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if (s.attributes or {}).get("mloda.operation.name") == operation]


@pytest.mark.parametrize("source", SOURCES)
def test_every_source_returns_the_pinned_values_with_a_join_and_two_calculate_spans(
    demo: ModuleType, tmp_path: Path, source: str
) -> None:
    values, exporter, _ = _run(demo, source, tmp_path)

    assert sorted(values) == sorted(["NORTH:5.00", "SOUTH:7.00", "NORTH:9.00", "NORTH:11.00"])
    assert len(_spans_with_operation(exporter, "join")) >= 1
    assert len(_spans_with_operation(exporter, "calculate")) >= 2


def test_file_sources_have_one_load_span_with_their_reader_format_and_distinct_identities(
    demo: ModuleType, tmp_path: Path
) -> None:
    identities = {}
    for source, reader in LOAD_FORMATS.items():
        data_dir = tmp_path / source
        data_dir.mkdir()
        _, exporter, _ = _run(demo, source, data_dir)
        load_spans = _spans_with_operation(exporter, "load")
        assert len(load_spans) == 1, source
        attributes = load_spans[0].attributes or {}
        assert attributes.get("mloda.data_access.format") == reader
        identities[source] = attributes.get("mloda.data_access.identity")

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
    extenders, exporter, _ = _capture()

    with pytest.raises(ValueError, match="simulated failure"):
        demo.run_pipeline(source, extenders, tmp_path, fail=True)

    failed = [s for s in exporter.get_finished_spans() if "error.type" in (s.attributes or {})]
    assert failed
    assert any((s.attributes or {}).get("mloda.operation.name") == "calculate" for s in failed)


def _stack_yaml_files() -> Iterator[Path]:
    yield from sorted(p for pattern in ("*.yaml", "*.yml") for p in DEMO_DIR.rglob(pattern))


EXPECTED_YAML_FILES = {
    "compose.yaml",
    "otel-collector.yaml",
    "tempo.yaml",
    "prometheus.yaml",
    "datasources.yaml",
    "dashboards.yaml",
}


def test_every_yaml_file_parses() -> None:
    yaml = pytest.importorskip("yaml")
    paths = list(_stack_yaml_files())
    assert {p.name for p in paths} == EXPECTED_YAML_FILES
    for path in paths:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path


def test_dashboard_json_parses() -> None:
    path = DEMO_DIR / "grafana" / "dashboards" / "mloda.json"
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_every_compose_image_has_an_explicit_non_latest_tag() -> None:
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((DEMO_DIR / "compose.yaml").read_text(encoding="utf-8"))
    images = [service["image"] for service in compose["services"].values() if "image" in service]
    assert images
    for image in images:
        tag = image.rsplit("/", 1)[-1].partition(":")[2]
        assert tag, image
        assert tag != "latest", image


def test_dashboard_datasource_uids_exist_in_the_provisioned_datasources() -> None:
    yaml = pytest.importorskip("yaml")
    provisioned = yaml.safe_load(
        (DEMO_DIR / "grafana" / "provisioning" / "datasources" / "datasources.yaml").read_text(encoding="utf-8")
    )
    known = {source["uid"] for source in provisioned["datasources"]}
    dashboard = json.loads((DEMO_DIR / "grafana" / "dashboards" / "mloda.json").read_text(encoding="utf-8"))

    used = set()
    for panel in dashboard["panels"]:
        if "datasource" in panel:
            used.add(panel["datasource"]["uid"])
        for target in panel.get("targets", []):
            used.add(target["datasource"]["uid"])
    assert used
    assert used <= known


def _normalised(text: str) -> str:
    return " ".join(text.split())


def test_dashboard_queries_equal_the_export_guide_queries() -> None:
    guide = (Path(__file__).resolve().parents[2] / "docs" / "guides" / "12-export-telemetry.md").read_text(
        encoding="utf-8"
    )
    block = re.search(r"```promql\n(.*?)```", guide, re.DOTALL)
    assert block is not None
    guide_queries = [_normalised(q) for q in block.group(1).split("\n\n") if q.strip()]
    dashboard = json.loads((DEMO_DIR / "grafana" / "dashboards" / "mloda.json").read_text(encoding="utf-8"))
    exprs = [_normalised(t["expr"]) for panel in dashboard["panels"] for t in panel.get("targets", [])]

    assert len(guide_queries) == 2
    assert exprs == guide_queries


def _step_duration_points(demo: ModuleType, tmp_path: Path, fail: bool) -> list[Any]:
    extenders, _, _ = _capture()
    meter_provider, reader = make_metric_capture()
    extenders.add(OtelMetricsExtender(meter_provider=meter_provider))
    if fail:
        with pytest.raises(ValueError, match="simulated failure"):
            demo.run_pipeline("memory", extenders, tmp_path, fail=True)
    else:
        demo.run_pipeline("memory", extenders, tmp_path)
    data = reader.get_metrics_data()
    assert data is not None
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == "mloda.step.duration"
        for point in metric.data.data_points
    ]


def test_a_successful_run_records_step_duration_per_demo_feature_group(demo: ModuleType, tmp_path: Path) -> None:
    points = _step_duration_points(demo, tmp_path, fail=False)

    groups = {str((p.attributes or {}).get("mloda.feature_group.name")) for p in points}
    assert any(g.endswith("OtelDemoRegionAmount") for g in groups), groups
    assert any(g.endswith("OtelDemoTaxedRegion") for g in groups), groups


def test_a_failed_run_records_error_type_on_a_step_duration_point(demo: ModuleType, tmp_path: Path) -> None:
    points = _step_duration_points(demo, tmp_path, fail=True)

    assert any("error.type" in (p.attributes or {}) for p in points)


@pytest.mark.parametrize("source", SOURCES)
def test_requested_feature_carries_fail_and_reader_options(demo: ModuleType, tmp_path: Path, source: str) -> None:
    path = demo.write_orders(tmp_path, source)

    feature = demo.requested_feature(source, tmp_path, fail=True)

    assert feature.options.context.get("otel_demo_fail") is True
    if source == "memory":
        return
    reader = CsvReader if source == "csv" else ParquetReader
    assert feature.options.group.get(reader.__name__) == str(path)


def test_requested_feature_fail_defaults_to_false(demo: ModuleType, tmp_path: Path) -> None:
    demo.write_orders(tmp_path, "memory")

    assert demo.requested_feature("memory", tmp_path).options.context.get("otel_demo_fail") in (False, None)


def test_demo_keeps_no_module_level_run_state(demo: ModuleType) -> None:
    assert not hasattr(demo, "_STATE")
