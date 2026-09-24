"""Tests for teardown.py: force_flush and CLOSE_TIMEOUT, the shared bounded-flush primitives every
registry extender's close() builds on."""

from __future__ import annotations

from unittest.mock import Mock, patch

from opentelemetry import trace as otel_trace_api
from opentelemetry.sdk.trace import TracerProvider

from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT, force_flush


class TestCloseTimeout:
    def test_value_is_one_second(self) -> None:
        assert CLOSE_TIMEOUT == 1.0


class TestForceFlush:
    """force_flush() duck-types on a provider's optional force_flush method."""

    def test_calls_and_returns_true_for_real_sdk_provider(self) -> None:
        provider = TracerProvider()

        with patch.object(provider, "force_flush") as mock_force_flush:
            result = force_flush(provider)

        mock_force_flush.assert_called_once()
        assert result is True

    def test_returns_none_for_object_without_force_flush(self) -> None:
        """No force_flush at all means "cannot be asked to flush", distinct from a real False result."""
        result = force_flush(object())

        assert result is None

    def test_returns_none_for_default_proxy_tracer_provider(self) -> None:
        """The API-only ProxyTracerProvider lacks force_flush entirely.

        Constructed directly rather than via get_tracer_provider(), which returns whatever the
        process-global provider is set to and would make this test order/environment-dependent.
        """
        proxy_provider = otel_trace_api.ProxyTracerProvider()
        assert not hasattr(proxy_provider, "force_flush")  # precondition this test relies on

        result = force_flush(proxy_provider)

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
