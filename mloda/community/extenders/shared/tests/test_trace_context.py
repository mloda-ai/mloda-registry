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
