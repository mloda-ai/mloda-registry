"""Tests for mloda.testing package."""

from typing import Any

import pytest
from mloda.steward import Extender, ExtenderHook

from mloda.testing.base import FeatureGroupTestBase
from mloda.testing.extenders.contract import ExtenderContractTestMixin


def test_feature_group_test_base_import() -> None:
    """Verify FeatureGroupTestBase can be imported."""
    assert FeatureGroupTestBase is not None


def test_contract_mixin_assert_reports_compared_values() -> None:
    """Self-test: a failing contract assert should show the compared values, not an empty message."""

    class _OffByOneExtender(Extender):
        def __init__(self) -> None:
            self.raise_on_error = True

        def wraps(self) -> set[ExtenderHook]:
            return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs) + 1

    class _Host(ExtenderContractTestMixin):
        def make_extender(self, *, raise_on_error: bool | None = None) -> _OffByOneExtender:
            return _OffByOneExtender()

    with pytest.raises(AssertionError, match=r"assert 8 == 7"):
        _Host().test_contract_call_returns_wrapped_result_unchanged()
