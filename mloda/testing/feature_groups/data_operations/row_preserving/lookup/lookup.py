"""Shared test base, dedicated fixture, and expected values for key-lookup tests.

Expected values are pinned as literals so the PyArrow reference is checked against the spec.

Fixture (``id`` is the row-order witness)::

    id | product_code | store_id
    ---+--------------+---------
    0  | A100         | 1
    1  | B200         | 2
    2  | A100         | 1       (repeated matched key)
    3  | C300         | 3       (unmatched)
    4  | None         | None    (null key)
    5  | D400         | 4       (unmatched)
    6  | C300         | 3       (repeated unmatched key)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.lookup.base import LookupUnmatchedError
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import assert_values_with_nulls, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.input_validation import (
    InputValidationCase,
    InputValidationTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.output_contract import OutputContractTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin

_IDS: list[int] = [0, 1, 2, 3, 4, 5, 6]
_PRODUCT_CODES: list[str | None] = ["A100", "B200", "A100", "C300", None, "D400", "C300"]
_STORE_IDS: list[int | None] = [1, 2, 1, 3, None, 4, 3]


def create_lookup_arrow_table() -> pa.Table:
    """The 7-row table used by every lookup test (str key column plus int key column)."""
    return pa.table(
        {
            "id": pa.array(_IDS, type=pa.int64()),
            "product_code": pa.array(_PRODUCT_CODES, type=pa.string()),
            "store_id": pa.array(_STORE_IDS, type=pa.int64()),
        }
    )


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

PARTIAL_STR_MAP: dict[str, str] = {"A100": "Alpha", "B200": "Beta"}
FULL_STR_MAP: dict[str, str] = {"A100": "Alpha", "B200": "Beta", "C300": "Gamma", "D400": "Delta"}
PARTIAL_INT_MAP: dict[int, int] = {1: 10, 2: 20}
STR_TO_INT_MAP: dict[str, int] = {"A100": 1, "B200": 2}
STR_TO_FLOAT_MAP: dict[str, float] = {"A100": 1.5, "B200": 2.5}
STR_TO_BOOL_MAP: dict[str, bool] = {"A100": True, "B200": False}
INT_TO_STR_MAP: dict[int, str] = {1: "one", 2: "two"}

# Reasons are only supplied through the resolve_unmatched_reasons hook.
HOOK_SUFFIX = "nuts2024"
HOOK_MAPPING: dict[str, str] = PARTIAL_STR_MAP
HOOK_REASONS: dict[str, str] = {"C300": "discontinued"}


class NutsMappingHook:
    """Hook-path mixin: combine with a framework class, e.g. ``class X(NutsMappingHook, PyArrowLookup)``."""

    MAPPING_FROM_OPTION = False
    PREFIX_PATTERN = rf".*__{HOOK_SUFFIX}$"

    @classmethod
    def resolve_mapping(cls, feature: Feature) -> dict[str, str]:
        return dict(HOOK_MAPPING)

    @classmethod
    def resolve_unmatched_reasons(cls, feature: Feature) -> dict[str, str]:
        return dict(HOOK_REASONS)


class EmptyMappingHook:
    """Mixin for an invalid hook: resolves an empty mapping, which must raise ``ValueError``."""

    MAPPING_FROM_OPTION = False
    PREFIX_PATTERN = r".*__emptymap$"

    @classmethod
    def resolve_mapping(cls, feature: Feature) -> dict[str, str]:
        return {}


# ---------------------------------------------------------------------------
# Expected values (literals)
# ---------------------------------------------------------------------------

NULL_STR_PARTIAL: list[Any] = ["Alpha", "Beta", "Alpha", None, None, None, None]
KEEP_STR_PARTIAL: list[Any] = ["Alpha", "Beta", "Alpha", "C300", None, "D400", "C300"]
RAISE_STR_FULL: list[Any] = ["Alpha", "Beta", "Alpha", "Gamma", None, "Delta", "Gamma"]
NULL_INT_PARTIAL: list[Any] = [10, 20, 10, None, None, None, None]
KEEP_INT_PARTIAL: list[Any] = [10, 20, 10, 3, None, 4, 3]


class LookupCase(NamedTuple):
    """One value case: the feature, its mapping, the policy and the pinned output."""

    feature_name: str
    mapping: dict[Any, Any]
    policy: str
    expected: list[Any]
    approx: bool = False


VALUE_CASES: list[Any] = [
    pytest.param(LookupCase("product_code__lookup", PARTIAL_STR_MAP, "null", NULL_STR_PARTIAL), id="str_null"),
    pytest.param(LookupCase("product_code__lookup", PARTIAL_STR_MAP, "keep", KEEP_STR_PARTIAL), id="str_keep"),
    pytest.param(LookupCase("product_code__lookup", FULL_STR_MAP, "raise", RAISE_STR_FULL), id="str_raise_all_matched"),
    pytest.param(LookupCase("store_id__lookup", PARTIAL_INT_MAP, "null", NULL_INT_PARTIAL), id="int_null"),
    pytest.param(LookupCase("store_id__lookup", PARTIAL_INT_MAP, "keep", KEEP_INT_PARTIAL), id="int_keep"),
    pytest.param(
        LookupCase("product_code__lookup", STR_TO_INT_MAP, "null", [1, 2, 1, None, None, None, None]),
        id="str_to_int",
    ),
    pytest.param(
        LookupCase("product_code__lookup", STR_TO_FLOAT_MAP, "null", [1.5, 2.5, 1.5, None, None, None, None], True),
        id="str_to_float",
    ),
    pytest.param(
        LookupCase("product_code__lookup", STR_TO_BOOL_MAP, "null", [True, False, True, None, None, None, None]),
        id="str_to_bool",
    ),
    pytest.param(
        LookupCase("store_id__lookup", INT_TO_STR_MAP, "null", ["one", "two", "one", None, None, None, None]),
        id="int_to_str",
    ),
]


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class LookupTestBase(InputValidationTestMixin, OutputContractTestMixin, ReservedColumnsTestMixin, DataOpsTestBase):
    """Abstract base class for key-lookup framework tests."""

    # -- Hook path configuration -------------------------------------------------

    @classmethod
    def hook_implementation_class(cls) -> Any:
        """The module-level ``NutsMappingHook`` subclass for this framework, or None to skip hook tests."""
        return None

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        """The module-level ``EmptyMappingHook`` subclass for this framework, or None to skip."""
        return None

    # -- OutputContractTestMixin configuration ----------------------------------

    def output_contract_feature_set(self) -> FeatureSet:
        return make_feature_set("product_code__lookup", lookup_mapping=FULL_STR_MAP)

    # -- ReservedColumnsTestMixin configuration --------------------------------

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "product_code__lookup"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None

    @classmethod
    def reserved_columns_extra_context(cls) -> dict[str, Any]:
        return {"lookup_mapping": FULL_STR_MAP, "on_unmatched": "null"}

    @classmethod
    def reserved_columns_helper_names(cls) -> set[str]:
        return {"__mloda_rn__", "__mloda_rn"}

    @classmethod
    def reference_implementation_class(cls) -> Any:
        # Lazy import so the testing module imports cleanly before the production code exists.
        from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import (
            PyArrowLookup,
        )

        return PyArrowLookup

    @classmethod
    def source_arrow_table(cls) -> pa.Table:
        return create_lookup_arrow_table()

    # -- InputValidationTestMixin configuration ---------------------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, Any]:
        without_source = create_lookup_arrow_table().drop_columns(["product_code"])
        return {
            "multi_column_in_features": InputValidationCase(
                "bad_multi_col",
                {"in_features": ["product_code", "store_id"], "lookup_mapping": FULL_STR_MAP},
                "at most 1",
            ),
            "missing_source_column": InputValidationCase(
                "product_code__lookup",
                {"lookup_mapping": FULL_STR_MAP},
                "Source column 'product_code' is not present",
                table=without_source,
            ),
            "empty_partition_by": None,
            "missing_partition_by_column": None,
            "missing_order_by_column": None,
        }

    # -- Helpers ------------------------------------------------------------

    def _run(self, feature_name: str, mapping: dict[Any, Any], policy: str | None = None, data: Any = None) -> Any:
        context: dict[str, Any] = {"lookup_mapping": mapping}
        if policy is not None:
            context["on_unmatched"] = policy
        fs = make_feature_set(feature_name, **context)
        return self.implementation_class().calculate_feature(self.test_data if data is None else data, fs)

    def _unmatched_error(self, feature_name: str, mapping: dict[Any, Any], policy: str | None = None) -> Any:
        with pytest.raises(LookupUnmatchedError) as excinfo:
            self._run(feature_name, mapping, policy)
        return excinfo.value

    # -- Pinned values ------------------------------------------------------

    @pytest.mark.parametrize("case", VALUE_CASES)
    def test_pinned_values(self, case: LookupCase) -> None:
        result = self._run(case.feature_name, case.mapping, case.policy)
        assert isinstance(result, self.get_expected_type())
        col = self.extract_column(result, case.feature_name)
        assert_values_with_nulls(col, case.expected, approx=case.approx, nan_is_null=True)

    @pytest.mark.parametrize("case", VALUE_CASES)
    def test_cross_framework(self, case: LookupCase) -> None:
        self._compare_with_reference(
            case.feature_name,
            use_approx=case.approx,
            lookup_mapping=case.mapping,
            on_unmatched=case.policy,
        )

    @pytest.mark.parametrize(
        ("policy", "mapping"),
        [
            pytest.param("raise", FULL_STR_MAP, id="raise"),
            pytest.param("null", PARTIAL_STR_MAP, id="null"),
            pytest.param("keep", PARTIAL_STR_MAP, id="keep"),
        ],
    )
    def test_null_key_stays_null(self, policy: str, mapping: dict[str, str]) -> None:
        col = self.extract_column(self._run("product_code__lookup", mapping, policy), "product_code__lookup")
        assert col[4] is None, f"null key must stay null under {policy!r}, got {col[4]!r}"

    def test_null_key_not_counted_as_unmatched(self) -> None:
        err = self._unmatched_error("product_code__lookup", PARTIAL_STR_MAP, "raise")
        assert err.unmatched_rows == 3
        assert err.distinct_keys == 2

    # -- Row-preserving semantics -------------------------------------------

    def test_original_row_order_and_source_preserved(self) -> None:
        result = self._run("product_code__lookup", PARTIAL_STR_MAP, "null")
        assert [int(v) for v in self.extract_column(result, "id")] == _IDS
        assert self.extract_column(result, "product_code") == _PRODUCT_CODES

    def test_empty_table(self) -> None:
        empty = self.create_test_data(self._arrow_table.slice(0, 0))
        for policy in ("raise", "null", "keep"):
            result = self._run("product_code__lookup", PARTIAL_STR_MAP, policy, data=empty)
            assert self.get_row_count(result) == 0
            assert self.extract_column(result, "product_code__lookup") == []

    # -- Backend robustness (values passed as data, not code) -----------------

    def test_int_output_with_nulls_stays_integer(self) -> None:
        col = self.extract_column(self._run("product_code__lookup", STR_TO_INT_MAP, "null"), "product_code__lookup")
        assert col == [1, 2, 1, None, None, None, None]
        assert all(type(v) is int for v in col if v is not None), f"expected ints, got {col!r}"

    def test_quote_and_semicolon_round_trip_exactly(self) -> None:
        key, value = "O'Brien; DROP", 'it\'s; "x" -- done'
        table = self.create_test_data(pa.table({"k": pa.array([key, "plain", None, key], type=pa.string())}))
        result = self._run("k__lookup", {key: value, "plain": "p"}, "raise", data=table)
        assert self.extract_column(result, "k__lookup") == [value, "p", None, value]

    def test_mapping_larger_than_sql_bound_variable_limit(self) -> None:
        size = 40000
        mapping = {i: i + 1 for i in range(size)}
        table = self.create_test_data(pa.table({"k": pa.array([0, 7, None, size - 1], type=pa.int64())}))
        result = self._run("k__lookup", mapping, "raise", data=table)
        assert self.extract_column(result, "k__lookup") == [1, 8, None, size]

    # -- Option-based configuration -----------------------------------------

    def test_option_based_lookup(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(
                context={"in_features": "product_code", "lookup_mapping": PARTIAL_STR_MAP, "on_unmatched": "null"}
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert_values_with_nulls(self.extract_column(result, "my_result"), NULL_STR_PARTIAL)

    # -- on_unmatched="raise" ------------------------------------------------

    def test_default_policy_is_raise(self) -> None:
        self._unmatched_error("product_code__lookup", PARTIAL_STR_MAP)

    @pytest.mark.parametrize(
        ("feature_name", "mapping", "rows", "distinct", "sample", "message"),
        [
            pytest.param(
                "product_code__lookup",
                PARTIAL_STR_MAP,
                3,
                2,
                ("C300", "D400"),
                "product_code__lookup: 3 rows, 2 distinct keys not in the mapping. Sample: ['C300', 'D400']",
                id="str_keys",
            ),
            pytest.param(
                "store_id__lookup",
                PARTIAL_INT_MAP,
                3,
                2,
                (3, 4),
                "store_id__lookup: 3 rows, 2 distinct keys not in the mapping. Sample: [3, 4]",
                id="int_keys",
            ),
        ],
    )
    def test_raise_error_attributes_and_message(
        self,
        feature_name: str,
        mapping: dict[Any, Any],
        rows: int,
        distinct: int,
        sample: tuple[Any, ...],
        message: str,
    ) -> None:
        err = self._unmatched_error(feature_name, mapping, "raise")
        assert err.feature_name == feature_name
        assert (err.unmatched_rows, err.distinct_keys, err.sample) == (rows, distinct, sample)
        assert dict(err.reasons) == {}
        assert str(err) == message

    def test_raise_sample_is_sorted_and_capped_at_ten(self) -> None:
        keys = [f"K{i:02d}" for i in range(12)]
        table = self.create_test_data(pa.table({"k": pa.array(["K11", *keys, None], type=pa.string())}))
        fs = make_feature_set("k__lookup", lookup_mapping={"zzz": "x"}, on_unmatched="raise")
        with pytest.raises(LookupUnmatchedError) as excinfo:
            self.implementation_class().calculate_feature(table, fs)
        err = excinfo.value
        assert err.unmatched_rows == 13
        assert err.distinct_keys == 12
        assert err.sample == tuple(keys[:10])

    # -- Policy and family rules ----------------------------------------------

    @pytest.mark.parametrize(
        ("feature_name", "mapping"),
        [
            pytest.param("product_code__lookup", STR_TO_INT_MAP, id="str_to_int"),
            pytest.param("store_id__lookup", INT_TO_STR_MAP, id="int_to_str"),
            pytest.param("product_code__lookup", STR_TO_BOOL_MAP, id="str_to_bool"),
        ],
    )
    def test_keep_with_mismatched_families_raises(self, feature_name: str, mapping: dict[Any, Any]) -> None:
        with pytest.raises(ValueError, match="keep"):
            self._run(feature_name, mapping, "keep")

    @pytest.mark.parametrize(
        ("feature_name", "mapping"),
        [
            pytest.param("store_id__lookup", PARTIAL_STR_MAP, id="str_keys_on_int_column"),
            pytest.param("product_code__lookup", PARTIAL_INT_MAP, id="int_keys_on_str_column"),
        ],
    )
    @pytest.mark.parametrize("policy", ["raise", "null", "keep"])
    def test_key_family_must_match_source_dtype(self, feature_name: str, mapping: dict[Any, Any], policy: str) -> None:
        source = feature_name.removesuffix("__lookup")
        with pytest.raises(ValueError, match=source) as excinfo:
            self._run(feature_name, mapping, policy)
        assert not isinstance(excinfo.value, LookupUnmatchedError)

    # -- Hook path --------------------------------------------------------------

    def _hook(self) -> Any:
        hook = self.hook_implementation_class()
        if hook is None:
            pytest.skip("no hook subclass declared for this framework")
        return hook

    def test_hook_matches_by_name_without_mapping_option(self) -> None:
        hook = self._hook()
        assert hook.match_feature_group_criteria(f"product_code__{HOOK_SUFFIX}", Options())
        assert not hook.match_feature_group_criteria("product_code__lookup", Options())

    @pytest.mark.parametrize(
        "context",
        [
            pytest.param({"in_features": "product_code"}, id="plain_config"),
            pytest.param({"in_features": "product_code", "lookup_mapping": PARTIAL_STR_MAP}, id="with_mapping"),
            pytest.param(
                {"in_features": "product_code", "partition_by": ["region"], "order_by": "ts"}, id="ffill_shaped"
            ),
        ],
    )
    def test_hook_declines_config_path(self, context: dict[str, Any]) -> None:
        hook = self._hook()
        assert not hook.match_feature_group_criteria("my_result", Options(context=context))

    def test_hook_output(self) -> None:
        hook = self._hook()
        fs = make_feature_set(f"product_code__{HOOK_SUFFIX}", on_unmatched="null")
        result = hook.calculate_feature(self.test_data, fs)
        col = self.extract_column(result, f"product_code__{HOOK_SUFFIX}")
        assert_values_with_nulls(col, NULL_STR_PARTIAL)

    def test_invalid_hook_mapping_raises_value_error(self) -> None:
        hook = self.invalid_hook_implementation_class()
        if hook is None:
            pytest.skip("no invalid hook subclass declared for this framework")
        fs = make_feature_set("product_code__emptymap", on_unmatched="null")
        with pytest.raises(ValueError):
            hook.calculate_feature(self.test_data, fs)

    def test_hook_reasons_in_error(self) -> None:
        hook = self._hook()
        name = f"product_code__{HOOK_SUFFIX}"
        with pytest.raises(LookupUnmatchedError) as excinfo:
            hook.calculate_feature(self.test_data, make_feature_set(name))
        err = excinfo.value
        assert dict(err.reasons) == HOOK_REASONS
        assert str(err) == (
            f"{name}: 3 rows, 2 distinct keys not in the mapping. Sample: ['C300' (discontinued), 'D400']"
        )
