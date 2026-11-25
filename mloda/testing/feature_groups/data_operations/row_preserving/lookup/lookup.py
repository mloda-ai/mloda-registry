"""Shared test base, dedicated fixture, and expected values for key-lookup tests.

Expected values are pinned as literals; cross-framework results are compared to the pure Python reference.

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

import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest
from mloda.provider import FeatureSet, Implementation
from mloda.user import DataType, Feature, Options

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    calculate_with,
    empty_like,
    extract_column,
    make_feature_set,
    make_multi_feature_set,
    typed_data_without,
)
from mloda.testing.feature_groups.data_operations.mixins.input_validation import (
    InputValidationCase,
    InputValidationTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.output_contract import OutputContractTestMixin
from mloda.testing.feature_groups.data_operations.mixins.pinned_rows import PinnedRowTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.result_type import ResultTypeTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningTable
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.pinning_rows import LOOKUP_PINNING
from mloda.testing.import_isolation import require_pyarrow
from mloda.testing.typed_data import TypedTestData

if TYPE_CHECKING:
    import pyarrow as pa

_IDS: list[int] = [0, 1, 2, 3, 4, 5, 6]
_PRODUCT_CODES: list[str | None] = ["A100", "B200", "A100", "C300", None, "D400", "C300"]
_STORE_IDS: list[int | None] = [1, 2, 1, 3, None, 4, 3]


def create_lookup_data() -> TypedTestData:
    """The data used by every lookup test (str key column plus int key column)."""
    return TypedTestData(
        {
            "id": (DataType.INT64, _IDS),
            "product_code": (DataType.STRING, _PRODUCT_CODES),
            "store_id": (DataType.INT64, _STORE_IDS),
        }
    )


def _single(data_type: DataType, values: list[Any]) -> TypedTestData:
    """One column named ``v``."""
    return TypedTestData({"v": (data_type, values)})


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
    """Hook-path mixin: combine with the group, e.g. ``class X(NutsMappingHook, LookupFeatureGroup)``."""

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


class LookupTestBase(
    PinnedRowTestMixin,
    InputValidationTestMixin,
    OutputContractTestMixin,
    ReservedColumnsTestMixin,
    ResultTypeTestMixin,
    DataOpsTestBase,
):
    """Abstract base class for key-lookup framework tests."""

    @classmethod
    def pinning_table(cls) -> PinningTable:
        return LOOKUP_PINNING

    # -- Hook path configuration -------------------------------------------------

    @classmethod
    def hook_group_class(cls) -> Any:
        """A ``NutsMappingHook`` ``LookupFeatureGroup`` subclass with its own ``UNMATCHED_ERROR``, or None to skip."""
        return None

    @classmethod
    def invalid_hook_group_class(cls) -> Any:
        """An ``EmptyMappingHook`` subclass of ``LookupFeatureGroup``, or None to skip."""
        return None

    @classmethod
    def untyped_null_source_becomes_int(cls) -> bool:
        """True if the fixture turns an Arrow null-typed column into an integer column (DuckDB)."""
        return False

    @classmethod
    def typed_null_source_becomes_untyped(cls) -> bool:
        """True if an all-null int64 column loses its type in the fixture (pandas float64 fixture, python dict)."""
        return False

    @classmethod
    def typed_null_string_becomes_untyped(cls) -> bool:
        """True if an all-null string column loses its type in the fixture."""
        return False

    @classmethod
    def int_with_nulls_becomes_float(cls) -> bool:
        """True if the fixture stores an int64 column with nulls as float64 (pandas)."""
        return False

    # -- OutputContractTestMixin configuration ----------------------------------

    def output_contract_feature_set(self) -> FeatureSet:
        return make_feature_set("product_code__lookup", lookup_mapping=FULL_STR_MAP)

    # -- ResultTypeTestMixin configuration --------------------------------------

    @classmethod
    def result_type_cases(cls) -> dict[str, FeatureSet]:
        """One mapping per value family read off ``product_code`` (holds a null), plus one raising case."""
        mappings = {
            "STRING": PARTIAL_STR_MAP,
            "INT64": STR_TO_INT_MAP,
            "DOUBLE": STR_TO_FLOAT_MAP,
            "BOOLEAN": STR_TO_BOOL_MAP,
        }
        cases = {
            family: make_feature_set("product_code__lookup", lookup_mapping=mapping, on_unmatched="null")
            for family, mapping in mappings.items()
        }
        # Every fixture key matches; core's perturbed run does not, so error_values inspects the raised text.
        cases["STRING_raise"] = make_feature_set(
            "product_code__lookup", lookup_mapping=FULL_STR_MAP, on_unmatched="raise"
        )
        return cases

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
        return {"__mloda_rn__", "__mloda_rn", "__mloda_rn0__", "__mloda_lk0__", "__mloda_lv0__"}

    @classmethod
    def unmatched_error(cls) -> Any:
        """Exception class the group under test raises for unmatched keys."""
        return cls.feature_group_class().UNMATCHED_ERROR

    @classmethod
    def reference_calculate(cls, table: pa.Table, features: Any) -> Any:
        """Run the pure Python reference function on the canonical table."""
        from mloda.testing.feature_groups.data_operations.row_preserving.lookup.reference import reference_lookup

        return reference_lookup(cls.feature_group_class(), table, features)

    @classmethod
    def source_data(cls) -> TypedTestData:
        return create_lookup_data()

    # -- InputValidationTestMixin configuration ---------------------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, Any]:
        full = create_lookup_data()
        without_source = typed_data_without(full, "product_code")
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
                data=without_source,
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
        return calculate_with(self.implementation_class(), self.test_data if data is None else data, fs)

    def _unmatched_error(self, feature_name: str, mapping: dict[Any, Any], policy: str | None = None) -> Any:
        with pytest.raises(self.unmatched_error()) as excinfo:
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

    def test_two_features_in_one_feature_set(self) -> None:
        specs: list[tuple[str, dict[str, Any]]] = [
            ("product_code__lookup", {"lookup_mapping": PARTIAL_STR_MAP, "on_unmatched": "null"}),
            ("store_id__lookup", {"lookup_mapping": PARTIAL_INT_MAP, "on_unmatched": "keep"}),
        ]
        fs = make_multi_feature_set(specs)
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        ref = self.reference_calculate(self._arrow_table, fs)
        for name, expected in (("product_code__lookup", NULL_STR_PARTIAL), ("store_id__lookup", KEEP_INT_PARTIAL)):
            col = self.extract_column(result, name)
            assert_values_with_nulls(col, extract_column(ref, name), nan_is_null=True)
            assert_values_with_nulls(col, expected, nan_is_null=True)
        assert [int(v) for v in self.extract_column(result, "id")] == _IDS

    def test_empty_table(self) -> None:
        empty = self.create_test_data(empty_like(self._source))
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
        table = self.create_test_data(TypedTestData({"k": (DataType.STRING, [key, "plain", None, key])}))
        result = self._run("k__lookup", {key: value, "plain": "p"}, "raise", data=table)
        assert self.extract_column(result, "k__lookup") == [value, "p", None, value]

    def test_mapping_larger_than_sql_bound_variable_limit(self) -> None:
        size = 40000
        mapping = {i: i + 1 for i in range(size)}
        table = self.create_test_data(TypedTestData({"k": (DataType.INT64, [0, 7, None, size - 1])}))
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
        result = calculate_with(self.implementation_class(), self.test_data, fs)
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
                "product_code__lookup: 3 rows, 2 distinct keys not in the mapping. "
                "A sorted sample of up to 10 keys is on the error's sample attribute.",
                id="str_keys",
            ),
            pytest.param(
                "store_id__lookup",
                PARTIAL_INT_MAP,
                3,
                2,
                (3, 4),
                "store_id__lookup: 3 rows, 2 distinct keys not in the mapping. "
                "A sorted sample of up to 10 keys is on the error's sample attribute.",
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
        table = self.create_test_data(TypedTestData({"k": (DataType.STRING, ["K11", *keys, None])}))
        fs = make_feature_set("k__lookup", lookup_mapping={"zzz": "x"}, on_unmatched="raise")
        with pytest.raises(self.unmatched_error()) as excinfo:
            calculate_with(self.implementation_class(), table, fs)
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
        assert not isinstance(excinfo.value, self.unmatched_error())

    @pytest.mark.parametrize("policy", ["raise", "null", "keep"])
    @pytest.mark.parametrize(
        ("column", "mapping"),
        [
            pytest.param(
                (DataType.DATE, [datetime.date(2020, 1, 1), None, datetime.date(2020, 1, 2)]),
                PARTIAL_STR_MAP,
                id="date_source_str_keys",
            ),
            pytest.param((DataType.BOOLEAN, [True, None, False]), PARTIAL_STR_MAP, id="bool_with_nulls_str_keys"),
            pytest.param((DataType.BOOLEAN, [True, None, False]), PARTIAL_INT_MAP, id="bool_with_nulls_int_keys"),
        ],
    )
    def test_non_key_source_dtype_raises(self, column: Any, mapping: dict[Any, Any], policy: str) -> None:
        table = self.create_test_data(_single(*column))
        with pytest.raises(ValueError, match="'v'") as excinfo:
            self._run("v__lookup", mapping, policy, data=table)
        assert not isinstance(excinfo.value, self.unmatched_error())

    # -- Integer width and range ----------------------------------------------

    @pytest.mark.parametrize("policy", ["null", "keep"])
    @pytest.mark.parametrize(
        ("data_type", "mapping", "values", "null_expected", "keep_expected"),
        [
            pytest.param(
                DataType.INT32,
                {1: 10, 2**40: 20},
                [1, 5, None],
                [10, None, None],
                [10, 5, None],
                id="int32_key_above_width",
            ),
            pytest.param(
                DataType.INT8,
                {1: 10, -(2**40): 20},
                [1, -5, None],
                [10, None, None],
                [10, -5, None],
                id="int8_key_below_width",
            ),
            pytest.param(
                DataType.UINT8, {1: 10, -1: 20}, [1, 5, None], [10, None, None], [10, 5, None], id="uint8_negative_key"
            ),
        ],
    )
    def test_int_key_outside_source_width(
        self,
        data_type: DataType,
        mapping: dict[Any, Any],
        values: list[Any],
        null_expected: list[Any],
        keep_expected: list[Any],
        policy: str,
    ) -> None:
        table = self.create_test_data(_single(data_type, values))
        col = self.extract_column(self._run("v__lookup", mapping, policy, data=table), "v__lookup")
        assert_values_with_nulls(col, null_expected if policy == "null" else keep_expected, nan_is_null=True)

    @pytest.mark.parametrize(
        ("policy", "mapping", "expected"),
        [
            pytest.param("null", {2**60: "lo", 2**60 + 1: "hi"}, ["lo", "hi", None], id="null"),
            pytest.param("raise", {2**60: "lo", 2**60 + 1: "hi"}, ["lo", "hi", None], id="raise"),
            pytest.param("keep", {2**60: 1, 2**60 + 1: 2}, [1, 2, None], id="keep"),
        ],
    )
    def test_large_int_keys_map_exactly(self, policy: str, mapping: dict[Any, Any], expected: list[Any]) -> None:
        if self.int_with_nulls_becomes_float():
            pytest.skip("fixture stores int64 with nulls as float64, which cannot hold 2**60 + 1")
        table = self.create_test_data(_single(DataType.INT64, [2**60, 2**60 + 1, None]))
        col = self.extract_column(self._run("v__lookup", mapping, policy, data=table), "v__lookup")
        assert col == expected

    def test_int_key_outside_source_width_still_matches_in_range_key_under_raise(self) -> None:
        table = self.create_test_data(_single(DataType.INT32, [1, 2, None]))
        with pytest.raises(self.unmatched_error()) as excinfo:
            self._run("v__lookup", {1: 10, 2**40: 20}, "raise", data=table)
        err = excinfo.value
        assert (err.unmatched_rows, err.distinct_keys, err.sample) == (1, 1, (2,))

    @pytest.mark.parametrize("policy", ["raise", "null", "keep"])
    def test_uint64_above_int64_is_rejected(self, policy: str) -> None:
        big = 2**63 + 5
        try:
            table = self.create_test_data(_single(DataType.UINT64, [1, big, None]))
        except OverflowError:
            pytest.skip("framework cannot hold uint64 values above int64")
        with pytest.raises(ValueError) as excinfo:
            self._run("v__lookup", {1: 10}, policy, data=table)
        assert not isinstance(excinfo.value, self.unmatched_error())

    # -- All-null sources ---------------------------------------------------------

    @pytest.mark.parametrize("policy", ["raise", "null", "keep"])
    @pytest.mark.parametrize(
        "mapping",
        [pytest.param(PARTIAL_STR_MAP, id="str_keys"), pytest.param(PARTIAL_INT_MAP, id="int_keys")],
    )
    def test_untyped_all_null_source(self, mapping: dict[Any, Any], policy: str) -> None:
        # DataType has no null type; the arrow override keeps the column null-typed
        require_pyarrow()
        import pyarrow as pa

        table = self.create_test_data(_single(DataType.INT64, [None, None, None]), {"v": pa.null()})
        if self.untyped_null_source_becomes_int() and isinstance(next(iter(mapping)), str):
            with pytest.raises(ValueError) as excinfo:
                self._run("v__lookup", mapping, policy, data=table)
            assert not isinstance(excinfo.value, self.unmatched_error())
            return
        result = self._run("v__lookup", mapping, policy, data=table)
        assert self.get_row_count(result) == 3
        assert_values_with_nulls(self.extract_column(result, "v__lookup"), [None, None, None], nan_is_null=True)

    @pytest.mark.parametrize("policy", ["raise", "null", "keep"])
    @pytest.mark.parametrize(
        ("data_type", "mapping"),
        [
            pytest.param(DataType.INT64, PARTIAL_STR_MAP, id="int64_str_keys"),
            pytest.param(DataType.STRING, PARTIAL_INT_MAP, id="string_int_keys"),
        ],
    )
    def test_typed_all_null_source_checks_key_family(
        self, data_type: DataType, mapping: dict[Any, Any], policy: str
    ) -> None:
        table = self.create_test_data(_single(data_type, [None, None, None]))
        if data_type == DataType.INT64:
            untyped = self.typed_null_source_becomes_untyped()
        else:
            untyped = self.typed_null_string_becomes_untyped()
        if untyped:
            result = self._run("v__lookup", mapping, policy, data=table)
            assert self.get_row_count(result) == 3
            assert_values_with_nulls(self.extract_column(result, "v__lookup"), [None, None, None], nan_is_null=True)
            return
        with pytest.raises(ValueError, match="'v'") as excinfo:
            self._run("v__lookup", mapping, policy, data=table)
        assert not isinstance(excinfo.value, self.unmatched_error())

    # -- Declared divergences ------------------------------------------------------

    def test_divergent_flags_are_declared(self) -> None:
        impl = self.implementation_class()
        if not (isinstance(impl, type) and issubclass(impl, Implementation)):
            pytest.skip("implementation under test is not an Implementation class")
        flags = (
            self.untyped_null_source_becomes_int(),
            self.typed_null_source_becomes_untyped(),
            self.typed_null_string_becomes_untyped(),
            self.int_with_nulls_becomes_float(),
        )
        if any(flags):
            assert impl.divergences

    # -- Hook path --------------------------------------------------------------

    def _hook(self) -> Any:
        hook = self.hook_group_class()
        if hook is None:
            pytest.skip("no hook group declared for this framework")
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
        result = calculate_with(self.implementation_class(), self.test_data, fs, group=hook)
        col = self.extract_column(result, f"product_code__{HOOK_SUFFIX}")
        assert_values_with_nulls(col, NULL_STR_PARTIAL)

    def test_invalid_hook_mapping_raises_value_error(self) -> None:
        hook = self.invalid_hook_group_class()
        if hook is None:
            pytest.skip("no invalid hook group declared for this framework")
        fs = make_feature_set("product_code__emptymap", on_unmatched="null")
        with pytest.raises(ValueError, match="invalid lookup mapping"):
            calculate_with(self.implementation_class(), self.test_data, fs, group=hook)

    def test_hook_reasons_in_error(self) -> None:
        hook = self._hook()
        name = f"product_code__{HOOK_SUFFIX}"
        with pytest.raises(hook.UNMATCHED_ERROR) as excinfo:
            calculate_with(self.implementation_class(), self.test_data, make_feature_set(name), group=hook)
        err = excinfo.value
        assert type(err) is hook.UNMATCHED_ERROR
        assert dict(err.reasons) == HOOK_REASONS
        assert str(err) == (
            f"{name}: 3 rows, 2 distinct keys not in the mapping. "
            "A sorted sample of up to 10 keys is on the error's sample attribute."
        )
