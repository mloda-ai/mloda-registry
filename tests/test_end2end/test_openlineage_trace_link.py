"""End to end: a step RunEvent's mlodaTrace points at the OTel calculate span of the same step."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from mloda.steward import Extender  # noqa: E402
from openlineage.client.event_v2 import RunEvent, RunState  # noqa: E402

from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender  # noqa: E402
from mloda.community.extenders.otel.otel_extender import OtelExtender  # noqa: E402
from mloda.enterprise.extenders.lineage.lineage_extender import LineageFacetsExtender  # noqa: E402
from mloda.testing.extenders.openlineage import ROOT_JOB_NAME, make_recording_client  # noqa: E402
from mloda.testing.extenders.otel import make_span_capture  # noqa: E402
from mloda.testing.extenders.runners import run_two_features  # noqa: E402

_OPENLINEAGE_EXTENDERS = [
    pytest.param(OpenLineageExtender, id="openlineage"),
    pytest.param(LineageFacetsExtender, id="lineage_facets"),
]


def _step_complete_events(events: list[RunEvent]) -> list[RunEvent]:
    return [e for e in events if e.eventType == RunState.COMPLETE and e.job.name != ROOT_JOB_NAME]


@pytest.mark.parametrize("extender_class", _OPENLINEAGE_EXTENDERS)
def test_every_calculate_complete_event_points_at_its_own_calculate_span(
    extender_class: type[OpenLineageExtender],
) -> None:
    client, transport = make_recording_client()
    provider, exporter = make_span_capture()

    run_two_features(OtelExtender(tracer_provider=provider), extender_class(client=client))

    spans_by_step_run_id = {
        span.attributes["mloda.step.run_id"]: span
        for span in exporter.get_finished_spans()
        if span.attributes is not None and "mloda.step.run_id" in span.attributes
    }
    events = _step_complete_events(transport.events)
    assert len(events) == 2
    for event in events:
        span = spans_by_step_run_id[event.run.runId]
        assert span.context is not None
        facet = (event.run.facets or {}).get("mlodaTrace")
        assert facet is not None
        assert facet.traceId == format(span.context.trace_id, "032x")  # type: ignore[attr-defined]
        assert facet.spanId == format(span.context.span_id, "016x")  # type: ignore[attr-defined]


@pytest.mark.parametrize("extender_class", _OPENLINEAGE_EXTENDERS)
def test_root_run_events_carry_no_mloda_trace(extender_class: type[OpenLineageExtender]) -> None:
    client, transport = make_recording_client()
    provider, _ = make_span_capture()

    run_two_features(OtelExtender(tracer_provider=provider), extender_class(client=client))

    root_events = [e for e in transport.events if e.job.name == ROOT_JOB_NAME]
    assert root_events
    for event in root_events:
        assert "mlodaTrace" not in (event.run.facets or {})


@pytest.mark.parametrize("extender_class", _OPENLINEAGE_EXTENDERS)
def test_openlineage_forced_outside_otel_gives_no_mloda_trace(extender_class: type[OpenLineageExtender]) -> None:
    client, transport = make_recording_client()
    provider, _ = make_span_capture()
    openlineage: Extender = extender_class(client=client)
    openlineage.priority = 50

    run_two_features(OtelExtender(tracer_provider=provider), openlineage)

    events = _step_complete_events(transport.events)
    assert len(events) == 2
    for event in events:
        assert "mlodaTrace" not in (event.run.facets or {})


@pytest.mark.parametrize("extender_class", _OPENLINEAGE_EXTENDERS)
def test_an_inert_otel_extender_gives_no_mloda_trace(extender_class: type[OpenLineageExtender]) -> None:
    client, transport = make_recording_client()

    run_two_features(OtelExtender(), extender_class(client=client))

    events = _step_complete_events(transport.events)
    assert len(events) == 2
    for event in events:
        assert "mlodaTrace" not in (event.run.facets or {})


def _app_span_ids(
    extender_class: type[OpenLineageExtender], *, otel: bool, outside: bool
) -> tuple[list[RunEvent], dict[str, tuple[str, str]], tuple[str, str]]:
    """Run under an application span; return the step COMPLETE events, step span ids by step runId, app span ids."""
    client, transport = make_recording_client()
    provider, exporter = make_span_capture()
    openlineage: Extender = extender_class(client=client)
    if outside:
        openlineage.priority = 50
    extenders: list[Extender] = [openlineage]
    if otel:
        extenders.append(OtelExtender(tracer_provider=provider))

    with provider.get_tracer("app").start_as_current_span("app-request") as app_span:
        app_context = app_span.get_span_context()
        run_two_features(*extenders)

    step_ids = {
        span.attributes["mloda.step.run_id"]: (
            format(span.context.trace_id, "032x"),
            format(span.context.span_id, "016x"),
        )
        for span in exporter.get_finished_spans()
        if span.attributes is not None and span.context is not None and "mloda.step.run_id" in span.attributes
    }
    app_ids = (format(app_context.trace_id, "032x"), format(app_context.span_id, "016x"))
    return _step_complete_events(transport.events), step_ids, app_ids


@pytest.mark.parametrize("extender_class", _OPENLINEAGE_EXTENDERS)
def test_under_an_application_span_each_event_still_points_at_its_own_calculate_span(
    extender_class: type[OpenLineageExtender],
) -> None:
    events, step_ids, app_ids = _app_span_ids(extender_class, otel=True, outside=False)

    assert len(events) == 2
    for event in events:
        facet = (event.run.facets or {}).get("mlodaTrace")
        assert facet is not None
        assert (facet.traceId, facet.spanId) == step_ids[event.run.runId]  # type: ignore[attr-defined]
        assert facet.spanId != app_ids[1]  # type: ignore[attr-defined]


@pytest.mark.parametrize("extender_class", _OPENLINEAGE_EXTENDERS)
@pytest.mark.parametrize(
    ("otel", "outside"),
    [pytest.param(True, True, id="openlineage_outside_otel"), pytest.param(False, False, id="no_otel_extender")],
)
def test_under_an_application_span_events_carry_no_mloda_trace_without_the_step_span(
    extender_class: type[OpenLineageExtender], otel: bool, outside: bool
) -> None:
    events, _, _ = _app_span_ids(extender_class, otel=otel, outside=outside)

    assert len(events) == 2
    for event in events:
        assert "mlodaTrace" not in (event.run.facets or {})
