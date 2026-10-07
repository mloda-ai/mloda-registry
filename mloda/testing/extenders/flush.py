"""Bounded-wait test helpers for a force_flush(timeout_millis=...) that may not honor its timeout.
Must not import opentelemetry: mloda-community-extenders-shared's own tests have no OTel dependency."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
from mloda.steward import CloseContext, Extender


def call_with_join_timeout(func: Callable[[], Any], *, join_timeout: float) -> tuple[bool, dict[str, Any]]:
    """Run func() on a daemon thread, joined for join_timeout; returns (still_running, outcome with
    'result' or 'error'). A still-running thread is left running so the test never hangs."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = func()
        except BaseException as exc:  # noqa: BLE001 - captured to re-raise on the calling thread
            outcome["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(join_timeout)
    return thread.is_alive(), outcome


@contextmanager
def active_close_context(remaining: float) -> Iterator[CloseContext]:
    """Activate a CloseContext with `remaining` seconds of budget for the scope (negative is already expired)."""
    ctx = CloseContext(deadline=time.monotonic() + remaining, reason="stop")
    with ctx.activate():
        yield ctx


@contextmanager
def blocking_flush_provider() -> Iterator[Mock]:
    """Yield a Mock whose force_flush(timeout_millis=...) ignores the timeout and blocks until the
    context exits, like the OTel SDK's BatchProcessor.force_flush. The Event is always released on
    exit, so no thread outlives the test."""
    release = threading.Event()

    def blocking_force_flush(timeout_millis: int) -> bool:
        release.wait()  # ignores timeout_millis, exactly like the real SDK's BatchProcessor
        return True

    provider = Mock(force_flush=Mock(side_effect=blocking_force_flush))
    try:
        yield provider
    finally:
        release.set()


class ProviderCloseTestMixin:
    """close() contract for an extender that flushes one provider (constructor kwarg sink_noun()) within
    close_timeout. Reuses ExtenderContractTestMixin's extender_class, sink_noun, make_unconfigured_extender and
    make_sdk_defaults_extender; adds ambient_provider_getter and flushed_signal. Must not import opentelemetry."""

    if TYPE_CHECKING:  # declared for mypy only, so the contract mixin's implementations are never shadowed

        @classmethod
        def extender_class(cls) -> type[Extender]: ...

        @classmethod
        def sink_noun(cls) -> str | None: ...

        def make_unconfigured_extender(self) -> Extender: ...

        def make_sdk_defaults_extender(self) -> Extender: ...

    @classmethod
    def ambient_provider_getter(cls) -> str:
        """Patch target of the ambient provider lookup, e.g. 'opentelemetry.metrics.get_meter_provider'."""
        raise NotImplementedError

    @classmethod
    def flushed_signal(cls) -> str:
        """The word the not-flushed warning uses for what a flush drains ('spans', 'metrics')."""
        raise NotImplementedError

    def _noun(self) -> str:
        noun = self.sink_noun()
        assert noun, "ProviderCloseTestMixin needs sink_noun()"
        return noun

    def _injected(self, provider: Any) -> Extender:
        return self.extender_class()(**{self._noun(): provider})

    @staticmethod
    def _set_close_timeout(extender: Extender, value: Any) -> None:
        setattr(extender, "close_timeout", value)

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]

    def test_close_flushes_the_injected_provider_with_timeout_millis(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        extender = self._injected(provider)

        extender.close()

        provider.force_flush.assert_called_once_with(timeout_millis=int(getattr(extender, "close_timeout") * 1000))

    def test_close_caps_flush_timeout_to_the_active_close_context(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        extender = self._injected(provider)
        self._set_close_timeout(extender, 5.0)

        with active_close_context(3.0):
            extender.close()

        provider.force_flush.assert_called_once()
        assert 0 < provider.force_flush.call_args.kwargs["timeout_millis"] <= 3000

    def test_close_flushes_the_global_provider_under_use_sdk_defaults(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with patch(self.ambient_provider_getter(), return_value=provider):
            self.make_sdk_defaults_extender().close()

        provider.force_flush.assert_called_once()

    def test_inert_extender_close_touches_no_provider(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with patch(self.ambient_provider_getter(), return_value=provider) as getter:
            self.make_unconfigured_extender().close()  # no injected provider, use_sdk_defaults False: inert

        provider.force_flush.assert_not_called()
        assert getter.call_count == 0

    def test_close_never_calls_shutdown(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        self._injected(provider).close()

        provider.shutdown.assert_not_called()

    def test_close_swallows_a_raising_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(side_effect=RuntimeError("flush boom")))
        extender = self._injected(provider)

        with caplog.at_level(logging.WARNING):
            extender.close()

        name = self.extender_class().__name__
        assert self._warnings(caplog) == [f"{name} failed to flush {self._noun()}: RuntimeError"]
        assert "flush boom" not in caplog.text

    def test_close_logs_a_warning_when_force_flush_returns_false(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(return_value=False))
        extender = self._injected(provider)

        with caplog.at_level(logging.WARNING):
            extender.close()

        name = self.extender_class().__name__
        assert self._warnings(caplog) == [f"{name} did not flush all {self.flushed_signal()} within its close budget"]

    def test_close_logs_nothing_when_provider_has_no_force_flush(self, caplog: pytest.LogCaptureFixture) -> None:
        class _NoFlushProvider:
            pass

        extender = self._injected(_NoFlushProvider())

        with caplog.at_level(logging.WARNING):
            extender.close()

        assert caplog.records == []

    def test_close_timeout_override_is_honored(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        extender = self._injected(provider)
        self._set_close_timeout(extender, 5.0)

        extender.close()

        provider.force_flush.assert_called_once_with(timeout_millis=5000)

    def test_close_bounds_a_blocking_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """opentelemetry-sdk's BatchProcessor.force_flush(timeout_millis) currently ignores the timeout
        and exports synchronously; close() must still return well under a second."""
        with blocking_flush_provider() as provider:
            extender = self._injected(provider)
            self._set_close_timeout(extender, 0.1)

            start = time.monotonic()
            with caplog.at_level(logging.WARNING):
                still_running, outcome = call_with_join_timeout(extender.close, join_timeout=1.0)
            elapsed = time.monotonic() - start

        assert not still_running, "close() did not return within 1.0s while force_flush blocked past close_timeout"
        if "error" in outcome:
            raise outcome["error"]
        assert elapsed < 1.0, elapsed

        name = self.extender_class().__name__
        assert any(name in message for message in self._warnings(caplog)), self._warnings(caplog)

    def test_negative_close_timeout_calls_force_flush_with_no_args(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        extender = self._injected(provider)
        self._set_close_timeout(extender, -1.0)

        extender.close()

        provider.force_flush.assert_called_once_with()

    def test_close_with_a_nonsensical_close_timeout_never_raises_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = Mock(force_flush=Mock(return_value=True))
        extender = self._injected(provider)
        self._set_close_timeout(extender, None)

        with caplog.at_level(logging.WARNING):
            extender.close()

        name = self.extender_class().__name__
        assert any(name in message and "Error" in message for message in self._warnings(caplog)), self._warnings(caplog)

    def test_inert_close_never_raises_even_with_a_nonsensical_close_timeout(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender = self.make_unconfigured_extender()
        self._set_close_timeout(extender, None)

        with patch(self.ambient_provider_getter()) as getter, caplog.at_level(logging.WARNING):
            extender.close()

        assert getter.call_count == 0
        assert self._warnings(caplog) == []

    def test_close_never_raises_when_resolving_the_ambient_provider_fails(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender = self.make_sdk_defaults_extender()

        with patch(self.ambient_provider_getter(), side_effect=RuntimeError("resolve boom")):
            with caplog.at_level(logging.WARNING):
                extender.close()

        name = self.extender_class().__name__
        assert any(name in message and "RuntimeError" in message for message in self._warnings(caplog)), caplog.text
        assert "resolve boom" not in caplog.text
