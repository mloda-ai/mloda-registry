"""Tests for teardown.py: force_flush, CLOSE_TIMEOUT and to_timeout_millis, the shared bounded-flush
primitives every registry extender's close() builds on. Imports no opentelemetry: the two force_flush
cases that need the real SDK/API live in otel/tests/test_otel_multiprocessing.py instead, exercised
through that module's force_flush re-export."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT, force_flush
from mloda.testing.extenders.flush import blocking_flush_provider, call_with_join_timeout


class TestCloseTimeout:
    def test_value_is_one_second(self) -> None:
        assert CLOSE_TIMEOUT == 1.0


class TestForceFlush:
    """force_flush() duck-types on a provider's optional force_flush method."""

    def test_returns_none_for_object_without_force_flush(self) -> None:
        """No force_flush at all means "cannot be asked to flush", distinct from a real False result."""
        result = force_flush(object())

        assert result is None

    def test_returns_false_when_provider_force_flush_returns_false(self) -> None:
        """The real SDK TracerProvider.force_flush() returns False when the flush times out and spans
        were dropped; that signal must not be discarded in favor of an unconditional True."""
        stub = Mock(force_flush=Mock(return_value=False))

        result = force_flush(stub)

        assert result is False, (
            "force_flush() must propagate provider.force_flush()'s own return value instead of always "
            "returning True whenever the method exists and is callable"
        )

    def test_returns_true_when_provider_force_flush_returns_true(self) -> None:
        """Sanity check the positive case: a fix must not accidentally invert the boolean."""
        stub = Mock(force_flush=Mock(return_value=True))

        result = force_flush(stub)

        assert result is True

    def test_passes_timeout_millis_as_keyword_when_given(self) -> None:
        stub = Mock(force_flush=Mock(return_value=True))

        force_flush(stub, timeout_millis=5000)

        stub.force_flush.assert_called_once_with(timeout_millis=5000)

    def test_calls_with_no_args_when_timeout_millis_omitted(self) -> None:
        stub = Mock(force_flush=Mock(return_value=True))

        force_flush(stub)

        stub.force_flush.assert_called_once_with()


class TestForceFlushBoundedByTimeoutThread:
    """opentelemetry-sdk 1.44 BatchProcessor.force_flush(timeout_millis) ignores the timeout (upstream
    TODO) and exports the whole queue synchronously, so force_flush() must run the provider's own
    force_flush on a bounded-wait background thread rather than trusting it to honor timeout_millis."""

    def test_blocking_force_flush_returns_false_within_the_timeout(self) -> None:
        with blocking_flush_provider() as provider:
            start = time.monotonic()
            still_running, outcome = call_with_join_timeout(
                lambda: force_flush(provider, timeout_millis=100), join_timeout=1.0
            )
            elapsed = time.monotonic() - start

        assert not still_running, "force_flush(timeout_millis=100) did not return within 1.0s"
        if "error" in outcome:
            raise outcome["error"]
        assert outcome["result"] is False
        assert elapsed < 1.0, elapsed

    @pytest.mark.parametrize("timeout_millis", [1000, None], ids=["with_timeout", "no_timeout"])
    def test_a_raising_force_flush_propagates(self, timeout_millis: int | None) -> None:
        provider = Mock(force_flush=Mock(side_effect=RuntimeError("flush boom")))

        with pytest.raises(RuntimeError, match="flush boom"):
            force_flush(provider, timeout_millis=timeout_millis)


class TestToTimeoutMillis:
    """to_timeout_millis(seconds) -> the millisecond value force_flush's timeout_millis wants, None
    meaning no cap, matching OpenLineage's own "negative waits with no limit" convention."""

    def test_one_second_converts_to_one_thousand_millis(self) -> None:
        from mloda.community.extenders.shared.teardown import to_timeout_millis

        assert to_timeout_millis(1.0) == 1000

    def test_quarter_second_converts_to_two_hundred_fifty_millis(self) -> None:
        from mloda.community.extenders.shared.teardown import to_timeout_millis

        assert to_timeout_millis(0.25) == 250

    def test_negative_seconds_means_no_cap(self) -> None:
        from mloda.community.extenders.shared.teardown import to_timeout_millis

        assert to_timeout_millis(-1.0) is None

    def test_infinite_seconds_means_no_cap(self) -> None:
        from mloda.community.extenders.shared.teardown import to_timeout_millis

        assert to_timeout_millis(float("inf")) is None
