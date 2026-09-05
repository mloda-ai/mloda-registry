"""Tests for mloda.testing.extenders (hook_context, runners, contract)."""

from __future__ import annotations

import dataclasses
import inspect
import pickle  # nosec
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, ExtenderHook, HookContext

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import (
    CountingExtender,
    FailingFeatureGroup,
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


class _BreakingProbeExtender(Extender):
    """Breaking-by-default probe with no zero-arg constructor: `sink` is required."""

    explode = False

    def __init__(self, sink: list[str], raise_on_error: bool = True, explode: bool = False) -> None:
        self.sink = sink
        self.raise_on_error = raise_on_error
        if explode:
            self.explode = explode

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self.explode:
            raise RuntimeError("breaking probe instrumentation boom")
        self.sink.append("before")
        result = func(*args, **kwargs)
        self.sink.append("after")
        return result


class _ValidateOnlyProbeExtender(Extender):
    """Wraps VALIDATE_OUTPUT_FEATURE only; records the active HookContext.hook on every call."""

    explode = False

    def __init__(self, sink: list[Any] | None = None, raise_on_error: bool = False, explode: bool = False) -> None:
        self.raise_on_error = raise_on_error
        self.sink = sink if sink is not None else []
        if explode:
            self.explode = explode

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_OUTPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self.explode:
            raise RuntimeError("validate-only probe instrumentation boom")
        context = HookContext.current()
        self.sink.append(context.hook if context is not None else None)
        result = func(*args, **kwargs)
        return result


class TestMakeHookContextDefaults:
    def test_defaults(self) -> None:
        context = make_hook_context()

        assert context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        assert context.feature_names == ("value_int",)
        assert context.compute_framework_name == "PyArrowTable"
        assert context.run_id is None

    def test_keywords_mirror_hook_context_fields(self) -> None:
        """Guards that a HookContext field added upstream stays reachable through the helper."""
        helper_keywords = set(inspect.signature(make_hook_context).parameters)
        # HookContext is a kw_only dataclass with no init=False fields, so every field is constructor-settable.
        hook_context_fields = {field.name for field in dataclasses.fields(HookContext) if field.init}

        assert helper_keywords == hook_context_fields, (
            f"make_hook_context keywords {helper_keywords} vs HookContext fields {hook_context_fields}"
        )


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

    def test_tenant_id_override(self) -> None:
        context = make_hook_context(tenant_id="tenant-1")
        assert context.tenant_id == "tenant-1"

    def test_project_id_override(self) -> None:
        context = make_hook_context(project_id="project-1")
        assert context.project_id == "project-1"

    def test_principal_override(self) -> None:
        context = make_hook_context(principal="user-1")
        assert context.principal == "user-1"

    def test_join_type_override(self) -> None:
        context = make_hook_context(join_type="inner")
        assert context.join_type == "inner"

    def test_join_keys_override(self) -> None:
        context = make_hook_context(join_keys=("id",))
        assert context.join_keys == ("id",)

    def test_plan_depth_override(self) -> None:
        context = make_hook_context(plan_depth=2)
        assert context.plan_depth == 2

    def test_duration_seconds_override(self) -> None:
        context = make_hook_context(duration_seconds=1.5)
        assert context.duration_seconds == 1.5

    def test_status_override(self) -> None:
        context = make_hook_context(status="ok")
        assert context.status == "ok"


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

    def test_base_feature_name_is_never_requested_sentinel(self) -> None:
        assert FailingFeatureGroup.feature_name == "mloda_testing_never_requested"

    def test_same_name_calls_produce_distinct_qualnames(self) -> None:
        first = failing_feature_group("same")
        second = failing_feature_group("same")

        assert first.__qualname__ != second.__qualname__
        assert first.__qualname__.startswith("FailingFeatureGroup_same_")
        assert second.__qualname__.startswith("FailingFeatureGroup_same_")


class TestProbeExtenderContract(ExtenderContractTestMixin):
    """Self-test: _ProbeExtender must satisfy every contract test the mixin defines."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _ProbeExtender

    @classmethod
    def raise_on_error_default(cls) -> bool:
        return False

    def make_extender(self, *, raise_on_error: bool | None = None) -> _ProbeExtender:
        if raise_on_error is None:
            return _ProbeExtender(sink=[])
        return _ProbeExtender(raise_on_error=raise_on_error, sink=[])

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(_ProbeExtender, "explode", True, create=True)


class TestBreakingProbeExtenderContract(ExtenderContractTestMixin):
    """Proves the mixin only ever builds instances via make_extender(), never extender_class()()."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _BreakingProbeExtender

    def make_extender(self, *, raise_on_error: bool | None = None) -> _BreakingProbeExtender:
        if raise_on_error is None:
            return _BreakingProbeExtender(sink=[])
        return _BreakingProbeExtender(sink=[], raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(_BreakingProbeExtender, "explode", True, create=True)


class TestValidateOnlyProbeContract(ExtenderContractTestMixin):
    """Self-test: a probe wrapping only VALIDATE_OUTPUT_FEATURE, pinning context_hook()."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _ValidateOnlyProbeExtender

    @classmethod
    def raise_on_error_default(cls) -> bool:
        return False

    def make_extender(self, *, raise_on_error: bool | None = None) -> _ValidateOnlyProbeExtender:
        if raise_on_error is None:
            return _ValidateOnlyProbeExtender(sink=[])
        return _ValidateOnlyProbeExtender(sink=[], raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(_ValidateOnlyProbeExtender, "explode", True, create=True)

    def test_contract_context_uses_wrapped_hook(self) -> None:
        extender = self.make_extender()
        with make_hook_context(hook=self.context_hook()).activate():
            extender(lambda: None)
        assert extender.sink[-1] == ExtenderHook.VALIDATE_OUTPUT_FEATURE


class TestCountingExtender:
    """CountingExtender: breaking pass-through probe that counts its own invocations."""

    def test_fresh_instance_defaults(self) -> None:
        extender = CountingExtender()
        assert extender.calls == 0
        assert extender.raise_on_error is True

    def test_wraps_calculate_feature_hook(self) -> None:
        assert CountingExtender().wraps() == {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def test_call_passes_through_and_counts(self) -> None:
        extender = CountingExtender()
        assert extender(lambda a, b: a + b, 3, 4) == 7
        assert extender.calls == 1

    def test_survives_pickle_roundtrip(self) -> None:
        copy = pickle.loads(pickle.dumps(CountingExtender()))  # nosec
        assert copy.calls == 0
        assert copy.wraps() == {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}


class TestExtenderContractTestMixinShape:
    """The mixin's own raise_on_error default, and the pickle-vs-subclass test swap."""

    def test_raise_on_error_default_is_true(self) -> None:
        assert ExtenderContractTestMixin.raise_on_error_default() is True

    def test_pickle_contract_test_replaces_subclass_check(self) -> None:
        assert not hasattr(ExtenderContractTestMixin, "test_contract_is_extender_subclass")
        assert hasattr(ExtenderContractTestMixin, "test_contract_extender_pickles")

    def test_expected_hooks_defaults_to_none(self) -> None:
        assert ExtenderContractTestMixin.expected_hooks() is None

    def test_pickled_copy_environment_is_a_context_manager(self) -> None:
        with ExtenderContractTestMixin().pickled_copy_environment():
            pass

    @pytest.mark.parametrize(
        "name",
        [
            "test_contract_wraps_expected_hooks",
            "test_contract_raise_on_error_is_configurable",
            "test_contract_call_without_hook_context_passes_through",
            "test_contract_pickled_copy_still_wraps",
            "test_contract_own_failure_does_not_stop_chained_extender",
            "test_contract_run_all_own_failure_falls_back_when_raise_on_error_false",
        ],
    )
    def test_new_contract_tests_exist(self, name: str) -> None:
        assert hasattr(ExtenderContractTestMixin, name)


class TestProbeExtenderDeclaredHooks(ExtenderContractTestMixin):
    """Mirrors TestProbeExtenderContract but declares expected_hooks so that test also runs."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _ProbeExtender

    @classmethod
    def raise_on_error_default(cls) -> bool:
        return False

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def make_extender(self, *, raise_on_error: bool | None = None) -> _ProbeExtender:
        if raise_on_error is None:
            return _ProbeExtender(sink=[])
        return _ProbeExtender(raise_on_error=raise_on_error, sink=[])

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(_ProbeExtender, "explode", True, create=True)
