"""Contract every Extender implementation keeps, exercised through its own test suite."""

from __future__ import annotations

import logging
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

    def make_extender(self, *, raise_on_error: bool = False) -> Extender:
        """Return an instance wired to an in-memory backend (no network)."""
        raise NotImplementedError

    def own_failure(self) -> AbstractContextManager[Any]:
        """A context manager making the extender's OWN instrumentation raise RuntimeError."""
        raise NotImplementedError

    @classmethod
    def raise_on_error_default(cls) -> bool:
        """Observability extenders default to warning-only."""
        return False

    def test_contract_is_extender_subclass(self) -> None:
        assert issubclass(self.extender_class(), Extender)

    def test_contract_wraps_only_known_hooks(self) -> None:
        wraps = self.extender_class()().wraps()
        assert wraps
        assert wraps <= set(ExtenderHook)

    def test_contract_raise_on_error_default(self) -> None:
        assert self.extender_class()().raise_on_error is self.raise_on_error_default()

    def test_contract_call_returns_wrapped_result_unchanged(self) -> None:
        with make_hook_context().activate():
            assert self.make_extender()(lambda a, b: a + b, 3, 4) == 7

    def test_contract_wrapped_failure_propagates_and_runs_once(self) -> None:
        calls = 0

        def func(*args: Any) -> Any:
            nonlocal calls
            calls += 1
            raise RuntimeError("inner boom")

        with make_hook_context().activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                self.make_extender()(func, 3, 4)
        assert calls == 1

    def test_contract_own_failure_falls_back_when_raise_on_error_false(self, caplog: pytest.LogCaptureFixture) -> None:
        composite = _CompositeExtender([self.make_extender(raise_on_error=False)])
        with make_hook_context().activate():
            with self.own_failure():
                with caplog.at_level(logging.WARNING):
                    result = composite(lambda a, b: a + b, 3, 4)
        assert result == 7
        assert any(self.extender_class().__name__ in message for message in caplog.messages)

    def test_contract_own_failure_propagates_when_raise_on_error_true(self) -> None:
        composite = _CompositeExtender([self.make_extender(raise_on_error=True)])
        with make_hook_context().activate():
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
