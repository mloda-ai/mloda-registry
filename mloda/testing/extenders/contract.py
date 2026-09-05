"""Contract every Extender implementation keeps, exercised through its own test suite."""

from __future__ import annotations

import logging
import pickle  # nosec
from contextlib import AbstractContextManager
from typing import Any

import pytest
from mloda.core.abstract_plugins.function_extender import _CompositeExtender
from mloda.steward import Extender, ExtenderHook

from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import (
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
