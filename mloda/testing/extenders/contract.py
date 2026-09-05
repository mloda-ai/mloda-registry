"""Contract every Extender implementation keeps, exercised through its own test suite."""

from __future__ import annotations

import logging
import pickle  # nosec
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest
from mloda.core.abstract_plugins.function_extender import _CompositeExtender
from mloda.steward import Extender, ExtenderHook, HookContext

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
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        """None skips the exact-set check; override to pin the exact hooks wraps() returns."""
        return None

    def pickled_copy_environment(self) -> AbstractContextManager[Any]:
        """Context active around a call made through a pickled copy; default is a no-op."""
        return nullcontext()

    def _context_hook(self) -> ExtenderHook:
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
        with make_hook_context(hook=self._context_hook()).activate():
            assert self.make_extender()(lambda a, b: a + b, 3, 4) == 7

    def test_contract_wrapped_failure_propagates_and_runs_once(self) -> None:
        calls = 0

        def func(*args: Any) -> Any:
            nonlocal calls
            calls += 1
            raise RuntimeError("inner boom")

        with make_hook_context(hook=self._context_hook()).activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                self.make_extender()(func, 3, 4)
        assert calls == 1

    def test_contract_own_failure_falls_back_when_raise_on_error_false(self, caplog: pytest.LogCaptureFixture) -> None:
        extender = self.make_extender(raise_on_error=False)
        if extender.raise_on_error is not False:
            pytest.skip("extender is breaking-only")
        composite = _CompositeExtender([extender])
        with make_hook_context(hook=self._context_hook()).activate():
            with self.own_failure():
                with caplog.at_level(logging.WARNING):
                    result = composite(lambda a, b: a + b, 3, 4)
        assert result == 7
        assert any(self.extender_class().__name__ in message for message in caplog.messages)

    def test_contract_own_failure_propagates_when_raise_on_error_true(self) -> None:
        composite = _CompositeExtender([self.make_extender(raise_on_error=True)])
        with make_hook_context(hook=self._context_hook()).activate():
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
        extender = self.make_extender(raise_on_error=False)
        if extender.raise_on_error is not False:
            pytest.skip("extender is breaking-only")
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
            with make_hook_context(hook=self._context_hook()).activate():
                assert copy(lambda a, b: a + b, 3, 4) == 7

    def test_contract_own_failure_does_not_stop_chained_extender(self) -> None:
        extender = self.make_extender(raise_on_error=False)
        if extender.raise_on_error is not False:
            pytest.skip("extender is breaking-only")
        counting = CountingExtender()
        composite = _CompositeExtender([extender, counting])
        with make_hook_context(hook=self._context_hook()).activate():
            with self.own_failure():
                assert composite(lambda a, b: a + b, 3, 4) == 7
        assert counting.calls == 1

    def test_contract_run_all_own_failure_falls_back_when_raise_on_error_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        extender = self.make_extender(raise_on_error=False)
        if extender.raise_on_error is not False:
            pytest.skip("extender is breaking-only")
        with self.own_failure():
            with caplog.at_level(logging.WARNING):
                assert run_value_int(extender) == expected_value_int()
        assert any(self.extender_class().__name__ in message for message in caplog.messages)
