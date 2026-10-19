"""Tests for LookupFeatureGroup base class: match rules, subclass guard, type rule."""

from __future__ import annotations

import copy
import pickle  # nosec
from typing import Any

import pytest
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import (
    ON_UNMATCHED_POLICIES,
    LookupFeatureGroup,
    LookupUnmatchedError,
    assert_key_family,
    build_unmatched_error,
    key_family,
    value_family,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import PyArrowLookup
from mloda.testing.feature_groups.data_operations.helpers import extract_column, feature_set_for
from mloda.testing.feature_groups.data_operations.match_validation import ScalarArityTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    FULL_STR_MAP,
    INT_TO_STR_MAP,
    PARTIAL_INT_MAP,
    PARTIAL_STR_MAP,
    STR_TO_BOOL_MAP,
    STR_TO_INT_MAP,
    NutsMappingHook,
    create_lookup_arrow_table,
)


class _HookLookup(NutsMappingHook, PyArrowLookup):
    pass


class _BackToOption(_HookLookup):
    MAPPING_FROM_OPTION = True  # type: ignore[misc]
    PREFIX_PATTERN = r".*__lookup$"


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


_UNSAFE_MAPPINGS: list[Any] = [
    pytest.param({"a": float("nan"), "b": 1.0}, id="nan_value"),
    pytest.param({2**63: "x"}, id="int_key_above_int64"),
    pytest.param({-(2**63) - 1: "x"}, id="int_key_below_int64"),
    pytest.param({1: 2**63}, id="int_value_above_int64"),
    pytest.param({1: -(2**63) - 1}, id="int_value_below_int64"),
]


def _opts(**context: Any) -> Options:
    return Options(context=context)


def _inherited_context_opts(**context: Any) -> Options:
    options = Options(context=context)
    options.inherited_context_keys = frozenset(context)
    return options


def _forwarded_opts(**group: Any) -> Options:
    options = Options(group=group)
    options.inherited_group_keys = frozenset(group)
    return options


class TestConstants:
    def test_option_keys_and_policies(self) -> None:
        assert LookupFeatureGroup.LOOKUP_MAPPING == "lookup_mapping"
        assert LookupFeatureGroup.ON_UNMATCHED == "on_unmatched"
        assert LookupFeatureGroup.MAPPING_FROM_OPTION is True
        assert set(ON_UNMATCHED_POLICIES) == {"raise", "null", "keep"}

    def test_unmatched_error_is_value_error(self) -> None:
        assert issubclass(LookupUnmatchedError, ValueError)

    @pytest.mark.parametrize("clone", [lambda e: pickle.loads(pickle.dumps(e)), copy.copy], ids=["pickle", "copy"])  # nosec
    def test_unmatched_error_survives_clone(self, clone: Any) -> None:
        error = build_unmatched_error("f__lookup", 3, ["x", "y"], {"x": "why"})
        cloned = clone(error)
        assert type(cloned) is LookupUnmatchedError
        assert str(cloned) == str(error)
        for attr in ("feature_name", "unmatched_rows", "distinct_keys", "sample", "reasons"):
            assert getattr(cloned, attr) == getattr(error, attr)

    @pytest.mark.parametrize(
        ("rows", "keys", "expected"),
        [
            pytest.param(1, ["a"], "f: 1 row, 1 distinct key not in the mapping", id="1_row_1_key"),
            pytest.param(3, ["a"], "f: 3 rows, 1 distinct key not in the mapping", id="3_rows_1_key"),
            pytest.param(3, ["a", "b"], "f: 3 rows, 2 distinct keys not in the mapping", id="3_rows_2_keys"),
        ],
    )
    def test_unmatched_message_pluralization(self, rows: int, keys: list[str], expected: str) -> None:
        assert str(build_unmatched_error("f", rows, keys, {})).startswith(expected + ".")

    def test_family_mismatch_names_source_dtype(self) -> None:
        with pytest.raises(ValueError, match="mapping keys are str") as info:
            assert_key_family("f__lookup", "v", {"a": "x"}, "other", source_dtype="list<item: int64>")
        assert "list<item: int64>" in str(info.value)
        assert "other" not in str(info.value)

    def test_family_mismatch_without_dtype_falls_back_to_family(self) -> None:
        with pytest.raises(ValueError, match="mapping keys are str but source column 'v' is int"):
            assert_key_family("f__lookup", "v", {"a": "x"}, "int")

    def test_key_family_invalid_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid lookup mapping"):
            key_family({None: "x"})

    def test_value_family_invalid_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid lookup mapping"):
            value_family({"a": None})


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
            *_UNSAFE_MAPPINGS,
        ],
    )
    def test_invalid_mappings_rejected(self, mapping: Any) -> None:
        assert not PyArrowLookup.match_feature_group_criteria("x__lookup", _opts(lookup_mapping=mapping))

    @pytest.mark.parametrize("policy", ["raise", "null", "keep"])
    def test_valid_policies_accepted(self, policy: str) -> None:
        options = _opts(lookup_mapping=PARTIAL_STR_MAP, on_unmatched=policy)
        assert PyArrowLookup.match_feature_group_criteria("x__lookup", options)

    @pytest.mark.parametrize(
        "policy",
        [
            pytest.param("drop", id="unknown_word"),
            pytest.param("RAISE", id="wrong_case"),
            pytest.param("", id="empty"),
            pytest.param(1, id="not_a_str"),
        ],
    )
    def test_unknown_policy_rejected(self, policy: Any) -> None:
        options = _opts(lookup_mapping=PARTIAL_STR_MAP, on_unmatched=policy)
        assert not PyArrowLookup.match_feature_group_criteria("x__lookup", options)


class TestMatchAbort:
    @pytest.mark.parametrize("mapping", [STR_TO_INT_MAP, INT_TO_STR_MAP, STR_TO_BOOL_MAP])
    def test_keep_with_mismatched_families_aborts_match(self, mapping: dict[Any, Any]) -> None:
        options = _opts(lookup_mapping=mapping, on_unmatched="keep")
        with pytest.raises(ValueError, match="keep"):
            PyArrowLookup.match_feature_group_criteria("x__lookup", options)

    def test_keep_with_invalid_mapping_declines_quietly(self) -> None:
        options = _opts(lookup_mapping={"a": 1, "b": "x"}, on_unmatched="keep")
        assert not PyArrowLookup.match_feature_group_criteria("x__lookup", options)

    def test_keep_without_mapping_declines_quietly(self) -> None:
        assert not PyArrowLookup.match_feature_group_criteria("x__lookup", _opts(on_unmatched="keep"))

    @pytest.mark.parametrize("mapping", [PARTIAL_STR_MAP, {"a": 1, "b": "x"}], ids=["valid", "invalid"])
    def test_hook_own_mapping_option_aborts_match(self, mapping: dict[Any, Any]) -> None:
        with pytest.raises(ValueError, match="lookup_mapping"):
            _HookLookup.match_feature_group_criteria("x__nuts2024", _opts(lookup_mapping=mapping))

    @pytest.mark.parametrize("mapping", [PARTIAL_STR_MAP, {"a": 1, "b": "x"}], ids=["valid", "invalid"])
    def test_hook_ignores_forwarded_mapping(self, mapping: dict[Any, Any]) -> None:
        assert _HookLookup.match_feature_group_criteria("x__nuts2024", _forwarded_opts(lookup_mapping=mapping))

    @pytest.mark.parametrize("mapping", [PARTIAL_STR_MAP, {"a": 1, "b": "x"}], ids=["valid", "invalid"])
    def test_hook_ignores_context_inherited_mapping(self, mapping: dict[Any, Any]) -> None:
        assert _HookLookup.match_feature_group_criteria("x__nuts2024", _inherited_context_opts(lookup_mapping=mapping))

    def test_hook_over_generic_lookup_source_does_not_abort(self) -> None:
        options = _opts(lookup_mapping=PARTIAL_STR_MAP)
        assert _HookLookup.match_feature_group_criteria("x__lookup__nuts2024", options)

    def test_subclass_restoring_mapping_option_regains_guard(self) -> None:
        invalid = _opts(lookup_mapping={"a": 1, "b": "x"}, on_unmatched="keep")
        assert not _BackToOption.match_feature_group_criteria("x__lookup", invalid)
        assert _BackToOption.match_feature_group_criteria("x__lookup", _opts(lookup_mapping=PARTIAL_STR_MAP))

    def test_hook_config_path_with_mapping_declines(self) -> None:
        options = _opts(in_features="x", lookup_mapping=PARTIAL_STR_MAP)
        assert not _HookLookup.match_feature_group_criteria("my_result", options)


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
        mapping = PyArrowLookup.resolve_mapping(feature)
        assert mapping is not None
        assert dict(mapping) == PARTIAL_STR_MAP

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
            # str/int/float/bool value families are pinned by COMPLETENESS_CASES in
            # test_return_data_type_rule_invariants.py; only the int-keyed case is added here.
            pytest.param(PARTIAL_INT_MAP, DataType.INT64, id="int_keys_int_values"),
        ],
    )
    def test_by_value_family(self, mapping: dict[Any, Any], expected: DataType) -> None:
        feature = Feature("x__lookup", options=_opts(lookup_mapping=mapping))
        assert PyArrowLookup.return_data_type_rule(feature) == expected

    def test_invalid_mapping_returns_none(self) -> None:
        feature = Feature("x__lookup", options=_opts(lookup_mapping={"a": "x", "b": 1}))
        assert PyArrowLookup.return_data_type_rule(feature) is None

    @pytest.mark.parametrize("mapping", _UNSAFE_MAPPINGS)
    def test_unsafe_mapping_returns_none(self, mapping: dict[Any, Any]) -> None:
        assert PyArrowLookup.return_data_type_rule(Feature("x__lookup", options=_opts(lookup_mapping=mapping))) is None

    def test_int64_bounds_accepted(self) -> None:
        mapping = {-(2**63): 2**63 - 1, 2**63 - 1: -(2**63)}
        assert PyArrowLookup.match_feature_group_criteria("x__lookup", _opts(lookup_mapping=mapping))

    def test_hook_mapping_typed(self) -> None:
        assert _HookLookup.return_data_type_rule(Feature("x__nuts2024", options=Options())) == DataType.STRING

    def test_invalid_hook_mapping_returns_none(self) -> None:
        feature = Feature("x__badmap", options=Options())
        assert _InvalidMappingHookLookup.return_data_type_rule(feature) is None

    def test_hook_io_error_propagates(self) -> None:
        with pytest.raises(OSError):
            _FailingHookLookup.return_data_type_rule(Feature("x__failing", options=Options()))


class TestOnUnmatchedArity(ScalarArityTestBase):
    """``on_unmatched`` is a scalar key: a multi-element container must not match."""

    @classmethod
    def feature_group_class(cls) -> Any:
        return LookupFeatureGroup

    @classmethod
    def match_class(cls) -> Any:
        return PyArrowLookup

    @classmethod
    def base_context(cls) -> dict[str, Any]:
        return {"in_features": "product_code", "lookup_mapping": PARTIAL_STR_MAP}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        return [TokenCase("on_unmatched", "null", "keep")]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return [LookupFeatureGroup._extract_policy(Feature("my_result", options=options))]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        result = PyArrowLookup.calculate_feature(create_lookup_arrow_table(), feature_set_for("my_result", options))
        return extract_column(result, "my_result")
