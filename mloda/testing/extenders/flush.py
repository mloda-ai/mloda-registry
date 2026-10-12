"""Bounded-wait test helpers for a force_flush(timeout_millis=...) that may not honor its timeout.
Must not import opentelemetry: mloda-community-extenders-shared's own tests have no OTel dependency."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
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


class ProviderFlushTestMixin:
    """Flush cases for an object that flushes one provider within close_timeout via flush_on_close. Hooks:
    make_flusher, flush, flush_noun, flushed_signal; optional provider_in_effect, flush_logger_name.
    Cases keep their test_close_* names so extender hosts keep their test ids. Must not import opentelemetry."""

    def make_flusher(self, provider: Any) -> Any:
        """Build the flushing object (it has a close_timeout attribute) for provider."""
        raise NotImplementedError

    def provider_in_effect(self, provider: Any) -> AbstractContextManager[Any]:
        """Scope in which provider is the one the flusher flushes; a no-op when make_flusher injects it."""
        return nullcontext()

    def flush(self, flusher: Any) -> None:
        """Call the flusher's flush entry point."""
        raise NotImplementedError

    @classmethod
    def flush_noun(cls) -> str:
        """The noun the failure warning uses ('failed to flush <noun>: ErrorType')."""
        raise NotImplementedError

    @classmethod
    def flushed_signal(cls) -> str:
        """The word the not-flushed warning uses for what a flush drains ('spans', 'metrics')."""
        raise NotImplementedError

    @classmethod
    def flush_logger_name(cls) -> str | None:
        """When set, only records from this logger count as warnings."""
        return None

    @classmethod
    def _not_flushed_text(cls, name: str) -> str:
        return f"{name} did not flush all {cls.flushed_signal()} within its close budget"

    @staticmethod
    def _set_close_timeout(flusher: Any, value: Any) -> None:
        setattr(flusher, "close_timeout", value)

    def _warnings(self, caplog: pytest.LogCaptureFixture) -> list[str]:
        logger_name = self.flush_logger_name()
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and (logger_name is None or r.name == logger_name)
        ]

    def assert_blocking_flush_warned(self, caplog: pytest.LogCaptureFixture, name: str) -> None:
        """The warning check of the blocking-flush case; exact text by default."""
        assert self._warnings(caplog) == [self._not_flushed_text(name)]

    def test_close_flushes_the_injected_provider_with_timeout_millis(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            self.flush(flusher)

        provider.force_flush.assert_called_once_with(timeout_millis=int(getattr(flusher, "close_timeout") * 1000))

    def test_close_caps_flush_timeout_to_the_active_close_context(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            self._set_close_timeout(flusher, 5.0)
            with active_close_context(3.0):
                self.flush(flusher)

        provider.force_flush.assert_called_once()
        assert 0 < provider.force_flush.call_args.kwargs["timeout_millis"] <= 3000

    def test_close_never_calls_shutdown(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            self.flush(flusher)

        provider.shutdown.assert_not_called()

    def test_close_swallows_a_raising_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(side_effect=RuntimeError("flush boom")))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            with caplog.at_level(logging.WARNING):
                self.flush(flusher)

        name = type(flusher).__name__
        assert self._warnings(caplog) == [f"{name} failed to flush {self.flush_noun()}: RuntimeError"]
        assert "flush boom" not in caplog.text

    def test_close_logs_a_warning_when_force_flush_returns_false(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = Mock(force_flush=Mock(return_value=False))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            with caplog.at_level(logging.WARNING):
                self.flush(flusher)

        assert self._warnings(caplog) == [self._not_flushed_text(type(flusher).__name__)]

    def test_close_logs_nothing_when_provider_has_no_force_flush(self, caplog: pytest.LogCaptureFixture) -> None:
        class _NoFlushProvider:
            pass

        provider = _NoFlushProvider()

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            with caplog.at_level(logging.WARNING):
                self.flush(flusher)

        assert caplog.records == []

    def test_close_timeout_override_is_honored(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            self._set_close_timeout(flusher, 5.0)
            self.flush(flusher)

        provider.force_flush.assert_called_once_with(timeout_millis=5000)

    def test_close_bounds_a_blocking_force_flush_and_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """opentelemetry-sdk's BatchProcessor.force_flush(timeout_millis) currently ignores the timeout
        and exports synchronously; the flush must still return well under a second."""
        with blocking_flush_provider() as provider:
            start = time.monotonic()
            with self.provider_in_effect(provider):
                flusher = self.make_flusher(provider)
                self._set_close_timeout(flusher, 0.1)
                with caplog.at_level(logging.WARNING):
                    still_running, outcome = call_with_join_timeout(lambda: self.flush(flusher), join_timeout=1.0)
            elapsed = time.monotonic() - start

        assert not still_running, "flush did not return within 1.0s while force_flush blocked past close_timeout"
        if "error" in outcome:
            raise outcome["error"]
        assert elapsed < 1.0, elapsed

        self.assert_blocking_flush_warned(caplog, type(flusher).__name__)

    def test_negative_close_timeout_calls_force_flush_with_no_args(self) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            self._set_close_timeout(flusher, -1.0)
            self.flush(flusher)

        provider.force_flush.assert_called_once_with()

    def test_close_with_a_nonsensical_close_timeout_never_raises_and_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = Mock(force_flush=Mock(return_value=True))

        with self.provider_in_effect(provider):
            flusher = self.make_flusher(provider)
            self._set_close_timeout(flusher, None)
            with caplog.at_level(logging.WARNING):
                self.flush(flusher)

        name = type(flusher).__name__
        assert any(name in message and "Error" in message for message in self._warnings(caplog)), self._warnings(caplog)


class ProviderCloseTestMixin(ProviderFlushTestMixin):
    """close() contract for an extender that flushes one provider (constructor kwarg sink_noun()) within
    close_timeout. Reuses ExtenderContractTestMixin's extender_class, sink_noun, make_unconfigured_extender and
    make_sdk_defaults_extender; the flush cases and flushed_signal come from ProviderFlushTestMixin, this class
    adds ambient_provider_getter. Must not import opentelemetry."""

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
    def flush_noun(cls) -> str:
        noun = cls.sink_noun()
        assert noun, "ProviderCloseTestMixin needs sink_noun()"
        return noun

    def _noun(self) -> str:
        return self.flush_noun()

    def _injected(self, provider: Any) -> Extender:
        return self.extender_class()(**{self._noun(): provider})

    def make_flusher(self, provider: Any) -> Any:
        return self._injected(provider)

    def flush(self, flusher: Any) -> None:
        flusher.close()

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
