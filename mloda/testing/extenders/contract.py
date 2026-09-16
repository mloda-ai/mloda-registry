"""Contract every Extender implementation keeps, exercised through its own test suite."""

from __future__ import annotations

import logging
import pickle  # nosec
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pytest
from mloda.steward import CompositeExtender, Extender, ExtenderHook, HookContext
from mloda.user import ParallelizationMode

from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import (
    CountingExtender,
    expected_value_int,
    failing_feature_group,
    run_failing_feature,
    run_value_int,
)


class ExtenderContractTestMixin:
    """Contract every Extender keeps. Host provides extender_class, make_extender and own_failure."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        raise NotImplementedError

    def make_extender(self, *, raise_on_error: bool | None = None) -> Extender:
        """Return an instance wired to an in-memory backend (no network); None means the extender's own default."""
        raise NotImplementedError

    def own_failure(self) -> AbstractContextManager[Any]:
        """A context manager making the extender's OWN instrumentation raise RuntimeError."""
        raise NotImplementedError

    @classmethod
    def raise_on_error_default(cls) -> bool:
        """Core's Extender default; observability extenders override this to False."""
        return True

    @classmethod
    def supports_warning_only(cls) -> bool:
        """Hosts of extenders that have no raise_on_error=False mode return False."""
        return True

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        """None skips the exact-set check; override to pin the exact hooks wraps() returns."""
        return None

    def pickled_copy_environment(self) -> AbstractContextManager[Any]:
        """Context active around a call made through a pickled copy; default is a no-op."""
        return nullcontext()

    @classmethod
    def has_backend_sink(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def sink_noun(cls) -> str | None:
        """The word this host's drop-warning uses for its sink ('tracer_provider', 'client'). None
        skips the noun-specific assertions below; only a host with supports_unpicklable_sink_degrade()
        or supports_pickled_sink_capture() True need override this."""
        return None

    @classmethod
    def unpicklable_sink_failure_type(cls) -> str | None:
        """The pickle-failure exception type name this host's drop-warning names. Both current hosts
        produce TypeError (pickling a threading.Lock always raises TypeError). None skips the
        assertion, e.g. for a host whose drop-warning doesn't name a type."""
        return "TypeError"

    def make_unconfigured_extender(self) -> Extender:
        return self.extender_class()()

    def make_sdk_defaults_extender(self) -> Extender:
        return self.extender_class()(use_sdk_defaults=True)  # type: ignore[call-arg]

    def make_injected_and_sdk_defaults_extender(self) -> Extender:
        raise NotImplementedError

    @classmethod
    def injected_and_sdk_defaults_sink_survives_pickling(cls) -> bool:
        """Whether make_injected_and_sdk_defaults_extender()'s sink is expected to survive a pickle
        round trip and stay in direct use, rather than being dropped and re-resolved from ambient
        sdk_defaults configuration. Default False (a drop is expected); a host whose injected sink
        there is picklable overrides this to True."""
        return False

    def make_unpicklable_sink_extender(self) -> Extender:
        """Return an instance wired to a sink that cannot survive plain pickling."""
        raise NotImplementedError

    @classmethod
    def supports_unpicklable_sink_degrade(cls) -> bool:
        """False unless the host trial-pickles its sink and drops+warns instead of hard-failing pickling."""
        return False

    def ambient_sink_environment(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def sink_resolution_spy(self) -> AbstractContextManager[list[Any]]:
        raise NotImplementedError

    def make_extender_with_sink_probe(self) -> tuple[Extender, Callable[[], Any]]:
        """Return an extender wired to a fresh, per-instance in-memory sink, plus a zero-arg
        callable returning whatever that exact sink instance captured. Must be per-instance state
        (built alongside the extender), never shared/class-level state, or the identity test below
        is vacuous."""
        raise NotImplementedError

    def sink_probe_expected_content(self) -> set[str] | None:
        """Content make_extender_with_sink_probe()'s captured sink, or a real worker's marker file,
        must contain. None skips the check (default; override once probe()'s return type is a
        comparable string collection)."""
        return None

    @classmethod
    def supports_pickled_sink_capture(cls) -> bool:
        """False unless the host's injected sink survives pickling."""
        return False

    def injected_sink_capture(self) -> AbstractContextManager[list[Any]]:
        """Context yielding a list of what a picklable, injected sink actually received after the pickle round trip."""
        raise NotImplementedError

    @classmethod
    def supports_real_worker_sink(cls) -> bool:
        """True when the host provides a file-backed sink observable from a real spawned
        MULTIPROCESSING worker. Default False: opt-in per concrete host, not inherited automatically
        by a backend mixin (unlike has_backend_sink), since it requires a real flight_server and a
        spawned subprocess, real overhead a lightweight probe host should not pay."""
        return False

    def make_real_worker_extender_and_marker(self, tmp_path: Path) -> tuple[Extender, Path]:
        """Return an extender wired to a file-backed sink under tmp_path, plus the marker file path a
        real spawned worker's emission writes to. Only called when supports_real_worker_sink() is True."""
        raise NotImplementedError

    def context_hook(self) -> ExtenderHook:
        """FEATURE_GROUP_CALCULATE_FEATURE when wrapped, else the wrapped hook with the smallest value."""
        wraps = self.make_extender().wraps()
        if ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in wraps:
            return ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        return min(wraps, key=lambda hook: hook.value)

    def test_contract_extender_pickles(self) -> None:
        extender = self.make_extender()
        copy = pickle.loads(pickle.dumps(extender))  # nosec
        assert isinstance(copy, self.extender_class())
        assert copy.wraps() == extender.wraps()
        assert copy.raise_on_error == extender.raise_on_error

    def test_contract_wraps_only_known_hooks(self) -> None:
        extender = self.make_extender()
        wraps = extender.wraps()
        assert wraps
        assert wraps <= set(ExtenderHook)
        assert extender.wraps() == wraps

    def test_contract_raise_on_error_default(self) -> None:
        assert self.make_extender().raise_on_error is self.raise_on_error_default()

    def test_contract_call_returns_wrapped_result_unchanged(self) -> None:
        calls = 0

        def func(a: int, b: int) -> int:
            nonlocal calls
            calls += 1
            return a + b

        with make_hook_context(hook=self.context_hook()).activate():
            assert self.make_extender()(func, 3, 4) == 7
        assert calls == 1

    def test_contract_wrapped_failure_propagates_and_runs_once(self) -> None:
        calls = 0

        def func(*args: Any) -> Any:
            nonlocal calls
            calls += 1
            raise RuntimeError("inner boom")

        with make_hook_context(hook=self.context_hook()).activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                self.make_extender()(func, 3, 4)
        assert calls == 1

    def test_contract_own_failure_falls_back_when_raise_on_error_false(self, caplog: pytest.LogCaptureFixture) -> None:
        if not self.supports_warning_only():
            pytest.skip("extender is breaking-only")
        extender = self.make_extender(raise_on_error=False)
        assert extender.raise_on_error is False
        composite = CompositeExtender([extender])
        with make_hook_context(hook=self.context_hook()).activate():
            with self.own_failure():
                with caplog.at_level(logging.WARNING):
                    result = composite(lambda a, b: a + b, 3, 4)
        assert result == 7
        name = self.extender_class().__name__
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(name in message for message in warnings), (
            f"{name}: own_failure() did not fault the extender's own code (see docs/guides/11-create-extender.md)"
        )

    def test_contract_own_failure_propagates_when_raise_on_error_true(self) -> None:
        composite = CompositeExtender([self.make_extender(raise_on_error=True)])
        with make_hook_context(hook=self.context_hook()).activate():
            with self.own_failure():
                with pytest.raises(RuntimeError):
                    composite(lambda a, b: a + b, 3, 4)

    def test_contract_run_all_leaves_result_unchanged(self) -> None:
        assert run_value_int(self.make_extender()) == expected_value_int()

    def test_contract_run_all_wrapped_failure_propagates_and_runs_once(self) -> None:
        fg = failing_feature_group(f"{self.extender_class().__name__.lower()}_boom_feature")
        with pytest.raises(Exception, match="inner boom"):
            run_failing_feature(fg, self.make_extender(raise_on_error=False))
        assert fg.calls == 1

    def test_contract_wraps_expected_hooks(self) -> None:
        expected = self.expected_hooks()
        if expected is None:
            pytest.skip("no expected_hooks declared")
        assert self.make_extender().wraps() == expected

    def test_contract_raise_on_error_is_configurable(self) -> None:
        assert self.make_extender(raise_on_error=True).raise_on_error is True
        if not self.supports_warning_only():
            pytest.skip("extender is breaking-only")
        extender = self.make_extender(raise_on_error=False)
        assert extender.raise_on_error is False

    def test_contract_call_without_hook_context_passes_through(self) -> None:
        assert HookContext.current() is None
        calls = 0

        def func(a: int, b: int) -> int:
            nonlocal calls
            calls += 1
            return a + b

        assert self.make_extender()(func, 3, 4) == 7
        assert calls == 1

    def test_contract_pickled_copy_still_wraps(self) -> None:
        copy = pickle.loads(pickle.dumps(self.make_extender()))  # nosec
        with self.pickled_copy_environment():
            with make_hook_context(hook=self.context_hook()).activate():
                assert copy(lambda a, b: a + b, 3, 4) == 7

    def test_contract_pickled_copy_with_picklable_sink_still_emits(self, caplog: pytest.LogCaptureFixture) -> None:
        if not self.supports_pickled_sink_capture():
            pytest.skip("host does not support pickled-sink capture")
        with caplog.at_level(logging.WARNING):
            with self.injected_sink_capture() as captured:
                copy = pickle.loads(pickle.dumps(self.make_extender()))  # nosec
                with self.pickled_copy_environment():
                    with make_hook_context(hook=self.context_hook()).activate():
                        assert copy(lambda a, b: a + b, 3, 4) == 7
                assert captured  # the pickled copy actually emitted into the shared/captured sink

        noun = self.sink_noun()
        if noun is not None:
            name = self.extender_class().__name__
            drop_warnings = [
                r.message
                for r in caplog.records
                if r.levelno >= logging.WARNING and name in r.message and noun in r.message
            ]
            assert drop_warnings == [], drop_warnings

    def test_contract_unpicklable_sink_drops_and_warns_once_per_instance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        if not self.supports_unpicklable_sink_degrade():
            pytest.skip("host does not support unpicklable-sink degrade")
        extender = self.make_unpicklable_sink_extender()

        with caplog.at_level(logging.WARNING):
            pickle.loads(pickle.dumps(extender))  # nosec
            copy = pickle.loads(pickle.dumps(extender))  # nosec

        name = self.extender_class().__name__
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING and name in r.message]
        assert len(warnings) == 1, warnings

        failure_type = self.unpicklable_sink_failure_type()
        if failure_type is not None:
            assert any(failure_type in message for message in warnings), warnings
        noun = self.sink_noun()
        if noun is not None:
            assert any(noun in message for message in warnings), warnings

        with self.pickled_copy_environment():
            with make_hook_context(hook=self.context_hook()).activate():
                assert copy(lambda a, b: a + b, 3, 4) == 7

    def test_contract_own_failure_does_not_stop_chained_extender(self, caplog: pytest.LogCaptureFixture) -> None:
        if not self.supports_warning_only():
            pytest.skip("extender is breaking-only")
        extender = self.make_extender(raise_on_error=False)
        assert extender.raise_on_error is False
        counting = CountingExtender()
        composite = CompositeExtender([extender, counting])
        with make_hook_context(hook=self.context_hook()).activate():
            with self.own_failure():
                with caplog.at_level(logging.WARNING):
                    assert composite(lambda a, b: a + b, 3, 4) == 7
        assert counting.calls == 1
        name = self.extender_class().__name__
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(name in message for message in warnings), (
            f"{name}: own_failure() did not fault the extender's own code (see docs/guides/11-create-extender.md)"
        )

    def test_contract_run_all_own_failure_falls_back_when_raise_on_error_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        if not self.supports_warning_only():
            pytest.skip("extender is breaking-only")
        extender = self.make_extender(raise_on_error=False)
        assert extender.raise_on_error is False
        counting = CountingExtender()
        with self.own_failure():
            with caplog.at_level(logging.WARNING):
                assert run_value_int(extender, counting) == expected_value_int()
        assert counting.calls >= 1
        name = self.extender_class().__name__
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(name in message for message in warnings), (
            f"{name}: own_failure() did not fault the extender's own code (see docs/guides/11-create-extender.md)"
        )

    def test_contract_unconfigured_extender_emits_nothing(self) -> None:
        if not self.has_backend_sink():
            pytest.skip("extender has no external sink")
        extender = self.make_unconfigured_extender()
        calls = 0

        def func(a: int, b: int) -> int:
            nonlocal calls
            calls += 1
            return a + b

        with self.ambient_sink_environment(), self.sink_resolution_spy() as spy:
            with make_hook_context(hook=self.context_hook()).activate():
                assert extender(func, 3, 4) == 7
            assert spy == []
        assert calls == 1

    def test_contract_unconfigured_extender_emits_nothing_in_run_all(self) -> None:
        if not self.has_backend_sink():
            pytest.skip("extender has no external sink")
        extender = self.make_unconfigured_extender()

        with self.ambient_sink_environment(), self.sink_resolution_spy() as spy:
            assert run_value_int(extender) == expected_value_int()
            assert spy == []

    def test_contract_sdk_defaults_resolves_sink(self) -> None:
        if not self.has_backend_sink():
            pytest.skip("extender has no external sink")
        extender = self.make_sdk_defaults_extender()

        with self.ambient_sink_environment(), self.sink_resolution_spy() as spy:
            with make_hook_context(hook=self.context_hook()).activate():
                extender(lambda: None)
            assert spy != []

    def test_contract_pickled_copy_with_sdk_defaults_resolves_ambient_sink(self) -> None:
        if not self.has_backend_sink():
            pytest.skip("extender has no external sink")
        copy = pickle.loads(pickle.dumps(self.make_injected_and_sdk_defaults_extender()))  # nosec
        with self.ambient_sink_environment(), self.sink_resolution_spy() as spy:
            with self.pickled_copy_environment():
                with make_hook_context(hook=self.context_hook()).activate():
                    copy(lambda: None)
            if self.injected_and_sdk_defaults_sink_survives_pickling():
                # The injected sink survived pickling and the copy kept using it directly (never
                # touching ambient config), which is itself a legitimate resolves-through-pickling outcome.
                assert spy == []
            else:
                # The injected sink didn't survive pickling; the copy must have re-resolved from
                # ambient sdk_defaults configuration instead of silently staying inert.
                assert spy != [], "sink was dropped on pickling but never re-resolved from ambient sdk defaults"

    def test_contract_injected_sink_ignores_ambient(self) -> None:
        if not self.has_backend_sink():
            pytest.skip("extender has no external sink")
        extender = self.make_extender()

        with self.ambient_sink_environment(), self.sink_resolution_spy() as spy:
            with make_hook_context(hook=self.context_hook()).activate():
                extender(lambda: None)
            assert spy == []

    def test_contract_injected_sink_wins_even_with_sdk_defaults_true(self) -> None:
        if not self.has_backend_sink():
            pytest.skip("extender has no external sink")
        extender = self.make_injected_and_sdk_defaults_extender()

        with self.ambient_sink_environment(), self.sink_resolution_spy() as spy:
            with make_hook_context(hook=self.context_hook()).activate():
                with self.own_failure():
                    with pytest.raises(RuntimeError):
                        extender(lambda: None)
            assert spy == []

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_contract_run_all_emits_into_the_exact_injected_sink(
        self, mode: ParallelizationMode, caplog: pytest.LogCaptureFixture
    ) -> None:
        if not self.has_backend_sink():
            pytest.skip("extender has no external sink")
        extender, probe = self.make_extender_with_sink_probe()
        name = self.extender_class().__name__
        with caplog.at_level(logging.WARNING):
            values = run_value_int(extender, parallelization_modes={mode})
        assert values == expected_value_int()
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING and name in r.message]
        assert warnings == [], warnings
        assert probe(), "no observation reached the injected sink; identity was not preserved"
        expected = self.sink_probe_expected_content()
        if expected is not None:
            assert expected <= set(probe()), probe()

    def test_contract_real_worker_multiprocessing_emits_into_the_exact_injected_sink(
        self, tmp_path: Path, request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
    ) -> None:
        if not self.supports_real_worker_sink():
            pytest.skip("host does not support a real-worker MULTIPROCESSING sink")
        flight_server = request.getfixturevalue("flight_server")
        extender, marker_path = self.make_real_worker_extender_and_marker(tmp_path)
        name = self.extender_class().__name__
        with caplog.at_level(logging.WARNING):
            values = run_value_int(
                extender, parallelization_modes={ParallelizationMode.MULTIPROCESSING}, flight_server=flight_server
            )
        assert values == expected_value_int()
        assert marker_path.exists(), "no marker written; the spawned worker never emitted through the injected sink"
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING and name in r.message]
        assert warnings == [], warnings
        expected = self.sink_probe_expected_content()
        if expected is not None:
            assert expected <= set(marker_path.read_text().splitlines()), marker_path.read_text()

    def test_contract_real_worker_multiprocessing_unpicklable_sink_degrades_gracefully(
        self, request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
    ) -> None:
        if not self.supports_real_worker_sink():
            pytest.skip("host does not support a real-worker MULTIPROCESSING sink")
        if not self.supports_unpicklable_sink_degrade():
            pytest.skip("host does not support unpicklable-sink degrade")
        flight_server = request.getfixturevalue("flight_server")
        extender = self.make_unpicklable_sink_extender()
        name = self.extender_class().__name__
        with caplog.at_level(logging.WARNING):
            values = run_value_int(
                extender, parallelization_modes={ParallelizationMode.MULTIPROCESSING}, flight_server=flight_server
            )
        assert values == expected_value_int()
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING and name in r.message]
        assert warnings, "expected a drop-and-warn message when the injected sink could not survive pickling"
