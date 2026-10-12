"""Inert, no-SDK-provider and pickled-copy warning tests shared by OtelExtender and OtelMetricsExtender.
Local to the otel tests, deliberately not in mloda.testing (the span no-SDK tests stay out of the published mixin)."""

from __future__ import annotations

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest
from mloda.steward import Extender

from mloda.testing.extenders.hook_context import make_hook_context


class AmbientProvider:
    """What the patched ambient provider getter returns; reassign .provider to switch mid-test, read .calls."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.calls = 0

    def get(self) -> Any:
        self.calls += 1
        return self.provider


def patch_ambient_provider(monkeypatch: pytest.MonkeyPatch, target: str, provider: Any) -> AmbientProvider:
    """Patch the ambient provider getter at `target` so no test installs a real global provider."""
    holder = AmbientProvider(provider)
    monkeypatch.setattr(target, holder.get)
    return holder


def call_once(extender: Extender) -> Any:
    with make_hook_context().activate():
        return extender(lambda: 42)


class ProviderWarningTestMixin:
    """Inert, no-SDK and pickled-copy warnings of an extender with one provider (constructor kwarg sink_noun()).
    Reuses the contract hosts' extender_class, sink_noun, ambient_provider_getter, make_unconfigured_extender,
    make_sdk_defaults_extender, make_injected_and_sdk_defaults_extender and make_unpicklable_sink_extender."""

    if TYPE_CHECKING:  # declared for mypy only, so the hosts' implementations are never shadowed

        @classmethod
        def extender_class(cls) -> type[Extender]: ...

        @classmethod
        def sink_noun(cls) -> str | None: ...

        @classmethod
        def ambient_provider_getter(cls) -> str: ...

        def make_unconfigured_extender(self) -> Extender: ...

        def make_sdk_defaults_extender(self) -> Extender: ...

        def make_injected_and_sdk_defaults_extender(self) -> Extender: ...

        def make_unpicklable_sink_extender(self) -> Extender: ...

    def make_proxy_provider(self) -> Any:
        """The API default proxy provider (a factory: the metrics proxy has no public class)."""
        raise NotImplementedError

    def make_noop_provider(self) -> Any:
        """The API default no-op provider."""
        raise NotImplementedError

    def api_default_in_effect(self) -> bool:
        """True while the unpatched ambient provider is still the API default (no global provider installed)."""
        raise NotImplementedError

    def no_sdk_marker(self) -> str:
        """Text only the no-SDK warning contains."""
        raise NotImplementedError

    def module_logger(self) -> logging.Logger:
        """The logger the extender's module warns on."""
        raise NotImplementedError

    def inert_guidance_substrings(self) -> tuple[str, ...]:
        """Substrings only the inert warning's guidance text contains."""
        raise NotImplementedError

    def make_real_provider_and_signal_check(self) -> tuple[Any, Callable[[], None]]:
        """A real provider plus a callable asserting the signal (a span, a metric) reached it."""
        raise NotImplementedError

    @pytest.fixture
    def ambient(self, monkeypatch: pytest.MonkeyPatch) -> AmbientProvider:
        return patch_ambient_provider(monkeypatch, self.ambient_provider_getter(), self.make_proxy_provider())

    def _name(self) -> str:
        return self.extender_class().__name__

    def _marker_records(self, caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if self.no_sdk_marker() in r.getMessage()]

    def _inert_records(self, caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
        return [r for r in caplog.records if "inert" in r.getMessage().lower()]

    def _extender_with_injected(self, provider: Any) -> Extender:
        noun = self.sink_noun()
        assert noun, "ProviderWarningTestMixin needs sink_noun()"
        return self.extender_class()(**{noun: provider})

    def test_inert_extender_warns_once_per_instance_and_still_calls_through(
        self, ambient: AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = self.make_unconfigured_extender()

        with caplog.at_level(logging.WARNING):
            assert call_once(otel) == 42
            assert call_once(otel) == 42
            records = self._inert_records(caplog)
            assert len(records) == 1, caplog.records
            message = records[0].getMessage()
            noun = self.sink_noun()
            assert noun is not None
            assert self._name() in message and noun in message
            assert "use_sdk_defaults=True" in message

            call_once(self.make_unconfigured_extender())

        assert len(self._inert_records(caplog)) == 2, caplog.records
        assert ambient.calls == 0

    def test_injected_provider_is_not_inert(self, caplog: pytest.LogCaptureFixture) -> None:
        provider, _ = self.make_real_provider_and_signal_check()

        with caplog.at_level(logging.WARNING):
            call_once(self._extender_with_injected(provider))

        assert self._inert_records(caplog) == [], caplog.records

    def test_pickled_copy_warns_for_its_own_inert_state(self, caplog: pytest.LogCaptureFixture) -> None:
        otel = self.make_unconfigured_extender()
        with caplog.at_level(logging.WARNING):
            call_once(otel)
            copy = pickle.loads(pickle.dumps(otel))  # nosec
            call_once(copy)

        assert len(self._inert_records(caplog)) == 2, caplog.records

    def test_pickled_copy_that_dropped_its_injected_provider_logs_its_own_inert_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = self.make_unpicklable_sink_extender()
        getattr(otel, "_inert_warning").warn_once(lambda: None)

        copy = pickle.loads(pickle.dumps(otel))  # nosec

        with caplog.at_level(logging.WARNING):
            result = call_once(copy)

        assert result == 42
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        # The pickle-drop warning also says "inert"; the guidance substrings only appear in the copy's own warning.
        assert any(
            self._name() in r.message
            and "inert" in r.message.lower()
            and all(text in r.message for text in self.inert_guidance_substrings())
            for r in warning_records
        ), warning_records
        assert self._marker_records(caplog) == [], caplog.records

    @pytest.mark.parametrize("use_sdk_defaults", [False, True], ids=["inert", "sdk_defaults"])
    def test_concurrent_first_calls_log_exactly_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        ambient: AmbientProvider,
        use_sdk_defaults: bool,
    ) -> None:
        # ambient defaults to the API default provider, so the sdk_defaults case warns.
        otel = self.make_sdk_defaults_extender() if use_sdk_defaults else self.make_unconfigured_extender()
        logger = self.module_logger()
        original_warning = logger.warning
        warning_calls: list[str] = []

        # Slow the one-shot warning to widen the check-then-log race window.
        def slow_warning(msg: str, *args: Any, **kwargs: Any) -> None:
            warning_calls.append(msg)
            time.sleep(0.05)
            original_warning(msg, *args, **kwargs)

        monkeypatch.setattr(logger, "warning", slow_warning)

        thread_count = 32
        barrier = threading.Barrier(thread_count)

        def worker() -> None:
            barrier.wait(timeout=5)
            with make_hook_context().activate():
                otel(lambda: None)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(thread_count)]
        with caplog.at_level(logging.WARNING):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        if use_sdk_defaults:
            matching_records = self._marker_records(caplog)
        else:
            matching_records = self._inert_records(caplog)
        assert len(matching_records) == 1, matching_records
        # Self-check: the slow_warning patch was actually hit.
        assert len(warning_calls) == 1, warning_calls

    @pytest.mark.parametrize("default_kind", ["proxy", "noop"])
    def test_warns_once_per_instance_when_ambient_provider_is_the_api_default(
        self, ambient: AmbientProvider, caplog: pytest.LogCaptureFixture, default_kind: str
    ) -> None:
        if default_kind == "noop":
            ambient.provider = self.make_noop_provider()  # the fixture already starts on the proxy
        otel = self.make_sdk_defaults_extender()

        with caplog.at_level(logging.WARNING):
            first_result = call_once(otel)
            second_result = call_once(otel)

            records = self._marker_records(caplog)
            assert len(records) == 1, caplog.records
            assert records[0].levelno == logging.WARNING
            assert records[0].name == self.module_logger().name
            message = records[0].getMessage()
            assert self._name() in message and "opentelemetry-sdk" in message
            assert "inert" not in message.lower()

            call_once(self.make_sdk_defaults_extender())

        assert first_result == 42
        assert second_result == 42
        assert len(self._marker_records(caplog)) == 2, caplog.records

    def test_warns_when_the_unpatched_api_default_provider_is_in_effect(self, caplog: pytest.LogCaptureFixture) -> None:
        if not self.api_default_in_effect():
            pytest.skip("global provider already installed in this process")
        otel = self.make_sdk_defaults_extender()

        with caplog.at_level(logging.WARNING):
            call_once(otel)

        assert len(self._marker_records(caplog)) == 1, caplog.records

    @pytest.mark.parametrize(
        "install_before_construction", [True, False], ids=["before_construction", "after_construction"]
    )
    def test_real_ambient_provider_does_not_warn_and_receives_the_signal(
        self, ambient: AmbientProvider, caplog: pytest.LogCaptureFixture, install_before_construction: bool
    ) -> None:
        provider, assert_signal_arrived = self.make_real_provider_and_signal_check()
        if install_before_construction:
            ambient.provider = provider
        otel = self.make_sdk_defaults_extender()
        if not install_before_construction:
            ambient.provider = provider

        with caplog.at_level(logging.WARNING):
            result = call_once(otel)

        assert result == 42
        assert self._marker_records(caplog) == []
        assert_signal_arrived()

    def test_switching_to_a_real_provider_after_the_warning_emits_the_signal_without_warning_again(
        self, ambient: AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, assert_signal_arrived = self.make_real_provider_and_signal_check()
        otel = self.make_sdk_defaults_extender()

        with caplog.at_level(logging.WARNING):
            call_once(otel)
            assert len(self._marker_records(caplog)) == 1, caplog.records

            ambient.provider = provider
            call_once(otel)

        assert len(self._marker_records(caplog)) == 1, caplog.records
        assert_signal_arrived()

    def test_no_sdk_pickled_copy_warns_again(self, ambient: AmbientProvider, caplog: pytest.LogCaptureFixture) -> None:
        otel = self.make_sdk_defaults_extender()

        with caplog.at_level(logging.WARNING):
            call_once(otel)
            assert len(self._marker_records(caplog)) == 1, caplog.records

            copy = pickle.loads(pickle.dumps(otel))  # nosec
            call_once(copy)

        assert len(self._marker_records(caplog)) == 2, caplog.records

    def test_pickled_copy_that_dropped_its_injected_provider_warns_under_sdk_defaults(
        self, ambient: AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The SDK provider holds locks, so pickling drops it; the copy then resolves the ambient provider.
        otel = self.make_injected_and_sdk_defaults_extender()

        with caplog.at_level(logging.WARNING):
            call_once(otel)
            assert self._marker_records(caplog) == [], caplog.records

            copy = pickle.loads(pickle.dumps(otel))  # nosec
            assert getattr(copy, f"_{self.sink_noun()}") is None
            call_once(copy)

        assert len(self._marker_records(caplog)) == 1, caplog.records

    def test_inert_instance_does_not_log_the_no_sdk_marker(
        self, ambient: AmbientProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        otel = self.make_unconfigured_extender()

        with caplog.at_level(logging.WARNING):
            result = call_once(otel)

        assert result == 42
        assert any("inert" in r.getMessage().lower() for r in caplog.records), caplog.records
        assert self._marker_records(caplog) == []
