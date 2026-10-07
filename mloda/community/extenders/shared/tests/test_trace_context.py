"""Tests for active_span_ids, the optional-opentelemetry lookup of the current span's ids."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry import trace  # noqa: E402

from mloda.community.extenders.shared.trace_context import active_span_ids  # noqa: E402
from mloda.testing.extenders.otel import make_non_recording_span, make_span_capture  # noqa: E402


@pytest.mark.parametrize("recording_only", [False, True], ids=["any_valid", "recording_only"])
def test_a_recording_span_gives_its_lowercase_hex_ids(recording_only: bool) -> None:
    provider, _ = make_span_capture()
    tracer = provider.get_tracer("trace-context-test")

    with tracer.start_as_current_span("step") as span:
        ctx = span.get_span_context()
        ids = active_span_ids(recording_only=recording_only)

    assert ids == (format(ctx.trace_id, "032x"), format(ctx.span_id, "016x"))
    assert ids is not None
    assert len(ids[0]) == 32 and len(ids[1]) == 16
    assert ids == (ids[0].lower(), ids[1].lower())


def test_a_non_recording_valid_span_gives_ids_by_default() -> None:
    with trace.use_span(make_non_recording_span()):
        assert active_span_ids() == ("1234567890abcdef1234567890abcdef", "1234567890abcdef")


def test_a_non_recording_valid_span_gives_none_when_recording_only() -> None:
    with trace.use_span(make_non_recording_span()):
        assert active_span_ids(recording_only=True) is None


@pytest.mark.parametrize("recording_only", [False, True], ids=["any_valid", "recording_only"])
def test_no_span_gives_none(recording_only: bool) -> None:
    assert active_span_ids(recording_only=recording_only) is None


@pytest.mark.parametrize("recording_only", [False, True], ids=["any_valid", "recording_only"])
def test_unimportable_opentelemetry_gives_none(monkeypatch: pytest.MonkeyPatch, recording_only: bool) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)

    assert active_span_ids(recording_only=recording_only) is None


_STEP_RUN_ID = "step-run-1"


def _step_ids(step_run_id: str) -> tuple[str, str] | None:
    from mloda.community.extenders.shared.trace_context import active_step_span_ids

    return active_step_span_ids(step_run_id)


def test_the_step_lookup_gives_ids_for_a_recording_span_with_the_matching_step_run_id() -> None:
    provider, _ = make_span_capture()

    with provider.get_tracer("trace-context-test").start_as_current_span(
        "step", attributes={"mloda.step.run_id": _STEP_RUN_ID}
    ) as span:
        ctx = span.get_span_context()
        ids = _step_ids(_STEP_RUN_ID)

    assert ids == (format(ctx.trace_id, "032x"), format(ctx.span_id, "016x"))


@pytest.mark.parametrize(
    "attributes", [{"mloda.step.run_id": "other"}, {}], ids=["mismatched_step_run_id", "missing_attribute"]
)
def test_the_step_lookup_gives_none_for_a_recording_span_without_the_matching_step_run_id(
    attributes: dict[str, str],
) -> None:
    provider, _ = make_span_capture()

    with provider.get_tracer("trace-context-test").start_as_current_span("step", attributes=attributes):
        assert _step_ids(_STEP_RUN_ID) is None


def test_the_step_lookup_gives_none_for_a_non_recording_span() -> None:
    with trace.use_span(make_non_recording_span()):
        assert _step_ids(_STEP_RUN_ID) is None


def test_the_step_lookup_gives_none_without_a_span() -> None:
    assert _step_ids(_STEP_RUN_ID) is None


def test_the_step_lookup_gives_none_when_opentelemetry_is_unimportable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)

    assert _step_ids(_STEP_RUN_ID) is None
