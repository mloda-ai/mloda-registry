"""Tests for CommunityExampleExtender."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, ExtenderHook

from mloda.community.extenders.example import CommunityExampleExtender
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.runners import failing_feature_group, run_failing_feature, run_value_int


class FailingCommunityExampleExtender(CommunityExampleExtender):
    """Deliberately failing extender: its own code raises before delegating to func."""

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("extender boom")


class TestCommunityExampleExtenderContract(ExtenderContractTestMixin):
    """CommunityExampleExtender satisfies the shared Extender contract."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return CommunityExampleExtender

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def make_extender(self, *, raise_on_error: bool | None = None) -> CommunityExampleExtender:
        if raise_on_error is None:
            return CommunityExampleExtender()
        return CommunityExampleExtender(raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(CommunityExampleExtender, "__call__", side_effect=RuntimeError("extender boom"))


class TestCommunityExampleExtenderRunAll:
    """The one raise_on_error semantic the shared contract does not pin."""

    def test_failing_extender_breaks_run_by_default(self) -> None:
        """raise_on_error=True (default): the extender failure propagates out of run_all."""
        with pytest.raises(Exception, match="extender boom"):
            run_value_int(FailingCommunityExampleExtender())

    def test_wrapped_failure_is_not_logged_as_extender_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """A wrapped func's own failure is not the extender's failure; it must not be logged as one."""
        fg = failing_feature_group("community_example_boom_feature")
        extender = CommunityExampleExtender(raise_on_error=False)

        with caplog.at_level(logging.WARNING):
            with pytest.raises(Exception, match="inner boom"):
                run_failing_feature(fg, extender)

        assert fg.calls == 1
        assert not any("inner boom" in r.message for r in caplog.records if r.levelno == logging.WARNING)
