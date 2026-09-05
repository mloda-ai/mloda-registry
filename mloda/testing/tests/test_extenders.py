"""Tests for mloda.testing.extenders (hook_context, runners, contract)."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, ExtenderHook

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import (
    expected_value_int,
    failing_feature_group,
    run_failing_feature,
    run_value_int,
)


class _ProbeExtender(Extender):
    """Pass-through observability probe; own_failure() patches `explode` to fault its instrumentation."""

    explode = False

    def __init__(self, raise_on_error: bool = False, sink: list[str] | None = None, explode: bool = False) -> None:
        self.raise_on_error = raise_on_error
        self.sink = sink if sink is not None else []
        if explode:
            self.explode = explode

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self.explode:
            raise RuntimeError("probe instrumentation boom")
        self.sink.append("before")
        result = func(*args, **kwargs)
        self.sink.append("after")
        return result


class TestMakeHookContextDefaults:
    def test_defaults(self) -> None:
        context = make_hook_context()

        assert context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        assert context.feature_names == ("value_int",)
        assert context.compute_framework_name == "PyArrowTable"
        assert context.run_id is None


class TestMakeHookContextOverrides:
    """Every keyword override must land unchanged on the built HookContext."""

    def test_hook_override(self) -> None:
        context = make_hook_context(hook=ExtenderHook.INPUT_DATA_LOAD)
        assert context.hook == ExtenderHook.INPUT_DATA_LOAD

    def test_feature_names_override(self) -> None:
        context = make_hook_context(feature_names=("other_feature",))
        assert context.feature_names == ("other_feature",)

    def test_run_id_override(self) -> None:
        context = make_hook_context(run_id="run-123")
        assert context.run_id == "run-123"

    def test_data_access_identity_override(self) -> None:
        context = make_hook_context(data_access_identity="s3://bucket/key")
        assert context.data_access_identity == "s3://bucket/key"

    def test_carrier_override(self) -> None:
        context = make_hook_context(carrier={"traceparent": "00-abc"})
        assert context.carrier == {"traceparent": "00-abc"}

    def test_worker_index_override(self) -> None:
        context = make_hook_context(worker_index=3)
        assert context.worker_index == 3


class TestExpectedValueInt:
    def test_matches_raw_data_creator(self) -> None:
        assert expected_value_int() == PyArrowDataOpsTestDataCreator.get_raw_data()["value_int"]


class TestRunValueInt:
    def test_no_extenders_returns_expected(self) -> None:
        assert run_value_int() == expected_value_int()

    def test_pass_through_extender_returns_expected(self) -> None:
        probe = _ProbeExtender(sink=[])
        assert run_value_int(probe) == expected_value_int()


class TestFailingFeatureGroup:
    def test_distinct_names_produce_distinct_classes(self) -> None:
        first = failing_feature_group("boom_one")
        second = failing_feature_group("boom_two")

        assert first.__name__ != second.__name__
        assert first.__qualname__ != second.__qualname__
        assert first.feature_name == "boom_one"
        assert second.feature_name == "boom_two"
        assert first.calls == 0
        assert second.calls == 0

    def test_run_failing_feature_raises_and_runs_once(self) -> None:
        fg = failing_feature_group("boom_three")

        with pytest.raises(Exception, match="inner boom"):
            run_failing_feature(fg)

        assert fg.calls == 1


class TestProbeExtenderContract(ExtenderContractTestMixin):
    """Self-test: _ProbeExtender must satisfy every contract test the mixin defines."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _ProbeExtender

    def make_extender(self, *, raise_on_error: bool = False) -> Extender:
        return _ProbeExtender(raise_on_error=raise_on_error, sink=[])

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(_ProbeExtender, "explode", True, create=True)
