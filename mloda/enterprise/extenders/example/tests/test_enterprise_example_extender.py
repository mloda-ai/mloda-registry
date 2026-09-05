"""Tests for EnterpriseExampleExtender."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, ExtenderHook

from mloda.enterprise.extenders.example import EnterpriseExampleExtender
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.runners import run_value_int


class FailingEnterpriseExampleExtender(EnterpriseExampleExtender):
    """Deliberately failing extender: its own code raises before delegating to func."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("extender boom")


class TestEnterpriseExampleExtenderContract(ExtenderContractTestMixin):
    """EnterpriseExampleExtender satisfies the shared Extender contract."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return EnterpriseExampleExtender

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def make_extender(self, *, raise_on_error: bool | None = None) -> EnterpriseExampleExtender:
        if raise_on_error is None:
            return EnterpriseExampleExtender()
        return EnterpriseExampleExtender(raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(EnterpriseExampleExtender, "__call__", side_effect=RuntimeError("extender boom"))


class TestEnterpriseExampleExtenderRunAll:
    """The one raise_on_error semantic the shared contract does not pin."""

    def test_failing_extender_breaks_run_by_default(self) -> None:
        """raise_on_error=True (default): the extender failure propagates out of run_all."""
        with pytest.raises(Exception, match="extender boom"):
            run_value_int(FailingEnterpriseExampleExtender())
