"""Tests for LookupFeatureGroup base class: match rules, subclass guard, type rule."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    ON_UNMATCHED_POLICIES,
    LookupFeatureGroup,
    LookupUnmatchedError,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import PyArrowLookup
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    FULL_STR_MAP,
    PARTIAL_INT_MAP,
    PARTIAL_STR_MAP,
    STR_TO_BOOL_MAP,
    STR_TO_FLOAT_MAP,
    STR_TO_INT_MAP,
    NutsMappingHook,
    create_lookup_arrow_table,
)


class _HookLookup(NutsMappingHook, PyArrowLookup):
    pass


class _InvalidMappingHookLookup(NutsMappingHook, PyArrowLookup):
    PREFIX_PATTERN = r".*__badmap$"

    @classmethod
    def resolve_mapping(cls, feature: Any) -> dict[Any, Any]:
        return {"a": 1, "b": "x"}


class _FailingHookLookup(NutsMappingHook, PyArrowLookup):
    PREFIX_PATTERN = r".*__failing$"

    @classmethod
    def resolve_mapping(cls, feature: Any) -> dict[Any, Any]:
        raise OSError("mapping file unreadable")


def _opts(**context: Any) -> Options:
    return Options(context=context)


class TestConstants:
    def test_option_keys_and_policies(self) -> None:
        assert LookupFeatureGroup.LOOKUP_MAPPING == "lookup_mapping"
        assert LookupFeatureGroup.ON_UNMATCHED == "on_unmatched"
        assert LookupFeatureGroup.MAPPING_FROM_OPTION is True
        assert set(ON_UNMATCHED_POLICIES) == {"raise", "null", "keep"}

    def test_unmatched_error_is_value_error(self) -> None:
        assert issubclass(LookupUnmatchedError, ValueError)


class TestMatchRules:
    def test_name_with_mapping_matches(self) -> None:
        assert PyArrowLookup.match_feature_group_criteria("x__lookup", _opts(lookup_mapping=PARTIAL_STR_MAP))

    def test_name_without_mapping_does_not_match(self) -> None:
        assert not PyArrowLookup.match_feature_group_criteria("x__lookup", Options())

    def test_config_path_with_mapping_matches(self) -> None:
        options = _opts(in_features="x", lookup_mapping=PARTIAL_STR_MAP)
        assert PyArrowLookup.match_feature_group_criteria("my_result", options)

    def test_config_path_without_mapping_does_not_match(self) -> None:
        assert not PyArrowLookup.match_feature_group_criteria("my_result", _opts(in_features="x"))

    def test_unrelated_name_does_not_match(self) -> None:
        assert not PyArrowLookup.match_feature_group_criteria("x__ffill", _opts(lookup_mapping=PARTIAL_STR_MAP))

    def test_hook_subclass_does_not_match_generic_name(self) -> None:
        assert not _HookLookup.match_feature_group_criteria("x__lookup", _opts(lookup_mapping=PARTIAL_STR_MAP))

    def test_generic_does_not_match_hook_name(self) -> None:
        assert not PyArrowLookup.match_feature_group_criteria("x__nuts2024", _opts(lookup_mapping=PARTIAL_STR_MAP))

    def test_hook_subclass_matches_by_name_without_mapping(self) -> None:
        assert _HookLookup.match_feature_group_criteria("x__nuts2024", Options())

    def test_hook_subclass_declines_foreign_config_features(self) -> None:
        for context in (
            {"in_features": "x"},
            {"in_features": "x", "lookup_mapping": PARTIAL_STR_MAP},
            {"in_features": "x", "partition_by": ["g"], "order_by": "t"},
        ):
            assert not _HookLookup.match_feature_group_criteria("my_result", _opts(**context))

    @pytest.mark.parametrize(
        "mapping",
        [
            pytest.param({"a": "x"}, id="str_str"),
            pytest.param({1: 10}, id="int_int"),
            pytest.param({"a": 1}, id="str_int"),
            pytest.param({"a": 1.5, "b": 2.5}, id="floats"),
            pytest.param({"a": True, "b": False}, id="bools"),
        ],
    )
    def test_valid_mappings_accepted(self, mapping: dict[Any, Any]) -> None:
        assert PyArrowLookup.match_feature_group_criteria("x__lookup", _opts(lookup_mapping=mapping))

    @pytest.mark.parametrize(
        "mapping",
        [
            pytest.param({}, id="empty"),
            pytest.param([("a", "b")], id="not_a_mapping"),
            pytest.param({"a": "x", 1: "y"}, id="mixed_key_families"),
            pytest.param({True: "x"}, id="bool_keys"),
            pytest.param({None: "x"}, id="null_key"),
            pytest.param({1.5: "x"}, id="float_keys"),
            pytest.param({"a": "x", "b": 1}, id="mixed_value_families_str_int"),
            pytest.param({"a": 1, "b": 2.5}, id="mixed_value_families_int_float"),
            pytest.param({"a": True, "b": 1}, id="mixed_value_families_bool_int"),
            pytest.param({"a": None}, id="null_value"),
            pytest.param({"a": object()}, id="unsupported_value"),
        ],
    )
    def test_invalid_mappings_rejected(self, mapping: Any) -> None:
        assert not PyArrowLookup.match_feature_group_criteria("x__lookup", _opts(lookup_mapping=mapping))

    @pytest.mark.parametrize("policy", ["raise", "null", "keep"])
    def test_valid_policies_accepted(self, policy: str) -> None:
        options = _opts(lookup_mapping=PARTIAL_STR_MAP, on_unmatched=policy)
        assert PyArrowLookup.match_feature_group_criteria("x__lookup", options)

    @pytest.mark.parametrize("policy", ["drop", "RAISE", "", 1])
    def test_unknown_policy_rejected(self, policy: Any) -> None:
        options = _opts(lookup_mapping=PARTIAL_STR_MAP, on_unmatched=policy)
        assert not PyArrowLookup.match_feature_group_criteria("x__lookup", options)


class TestRejectionReason:
    def test_invalid_mapping_reason_names_option_without_echoing_mapping(self) -> None:
        big = {f"key{i}": "x" for i in range(200)} | {"bad": 1}
        reason = PyArrowLookup._strict_validation_rejection_reason("x__lookup", _opts(lookup_mapping=big))
        assert reason is not None
        assert "'lookup_mapping'" in reason
        assert "key150" not in reason

    def test_unknown_policy_reason_names_option(self) -> None:
        options = _opts(lookup_mapping=PARTIAL_STR_MAP, on_unmatched="drop")
        reason = PyArrowLookup._strict_validation_rejection_reason("x__lookup", options)
        assert reason is not None
        assert "'on_unmatched'" in reason


class TestSubclassGuard:
    def test_hook_subclass_keeping_base_pattern_raises_type_error(self) -> None:
        with pytest.raises(TypeError):

            class _KeepsBasePattern(PyArrowLookup):
                MAPPING_FROM_OPTION = False

    def test_hook_subclass_with_own_pattern_is_accepted(self) -> None:
        assert _HookLookup.PREFIX_PATTERN != LookupFeatureGroup.PREFIX_PATTERN


class TestHooks:
    def test_default_resolve_mapping_unwraps_option(self) -> None:
        feature = Feature("x__lookup", options=_opts(lookup_mapping=PARTIAL_STR_MAP))
        assert dict(PyArrowLookup.resolve_mapping(feature)) == PARTIAL_STR_MAP

    def test_default_reasons_empty(self) -> None:
        feature = Feature("x__lookup", options=_opts(lookup_mapping=PARTIAL_STR_MAP))
        assert dict(PyArrowLookup.resolve_unmatched_reasons(feature)) == {}

    def test_compute_lookup_signature(self) -> None:
        result = PyArrowLookup._compute_lookup(
            create_lookup_arrow_table(), "product_code__lookup", "product_code", FULL_STR_MAP, "raise", {}
        )
        assert result.column("product_code__lookup").to_pylist() == [
            "Alpha",
            "Beta",
            "Alpha",
            "Gamma",
            None,
            "Delta",
            "Gamma",
        ]


class TestReturnDataTypeRule:
    @pytest.mark.parametrize(
        ("mapping", "expected"),
        [
            pytest.param(PARTIAL_STR_MAP, DataType.STRING, id="str"),
            pytest.param(STR_TO_INT_MAP, DataType.INT64, id="int"),
            pytest.param(PARTIAL_INT_MAP, DataType.INT64, id="int_keys_int_values"),
            pytest.param(STR_TO_FLOAT_MAP, DataType.DOUBLE, id="float"),
            pytest.param(STR_TO_BOOL_MAP, DataType.BOOLEAN, id="bool"),
        ],
    )
    def test_by_value_family(self, mapping: dict[Any, Any], expected: DataType) -> None:
        feature = Feature("x__lookup", options=_opts(lookup_mapping=mapping))
        assert PyArrowLookup.return_data_type_rule(feature) == expected

    def test_invalid_mapping_returns_none(self) -> None:
        feature = Feature("x__lookup", options=_opts(lookup_mapping={"a": "x", "b": 1}))
        assert PyArrowLookup.return_data_type_rule(feature) is None

    def test_hook_mapping_typed(self) -> None:
        assert _HookLookup.return_data_type_rule(Feature("x__nuts2024", options=Options())) == DataType.STRING

    def test_invalid_hook_mapping_returns_none(self) -> None:
        feature = Feature("x__badmap", options=Options())
        assert _InvalidMappingHookLookup.return_data_type_rule(feature) is None

    def test_hook_io_error_propagates(self) -> None:
        with pytest.raises(OSError):
            _FailingHookLookup.return_data_type_rule(Feature("x__failing", options=Options()))
