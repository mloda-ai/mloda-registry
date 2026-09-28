"""Arity contract of the shared scalar match_guards.

Sibling of ``test_op_token_guard.py``: what that file pins for ``is_op_token``, this
one pins for the remaining scalar ``PROPERTY_MAPPING`` keys. Core unwraps a singleton
container when it reads a property value (``feature_chain_parser._unpack_property_value``),
so ``("timestamp",)`` is valid caller syntax for one column reference and ``(5,)`` for one
number. Every key that is read back as a SINGLE value must accept that form, dispatch it
to the bare value, and reject multi-element containers, empty containers and wrong types
at match time rather than inside ``calculate_feature``.
"""

from __future__ import annotations

from typing import Any, Callable

import pyarrow as pa
import pytest
from mloda.user import Options

from mloda.community.feature_groups.data_operations.base import (
    assert_key_columns_present,
    assert_source_columns_present,
    available_columns,
    column_ref_value,
    is_column_ref,
    is_positive_int,
    is_scalar_number,
    positive_int_value,
    scalar_number_value,
)
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import BinningFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import PyArrowFfill
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup

# ---------------------------------------------------------------------------
# available_columns / assert_source_columns_present: one data factory per
# framework input type, lazily importing (and skipping) its optional dep.
# ---------------------------------------------------------------------------


_DEFAULT_COLUMNS: dict[str, list[int]] = {"a": [1], "b": [2]}


def _dict_data(columns: dict[str, list[int]] | None = None) -> dict[str, list[int]]:
    return columns or _DEFAULT_COLUMNS


def _pandas_data(columns: dict[str, list[int]] | None = None) -> Any:
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(columns or _DEFAULT_COLUMNS)


def _pyarrow_data(columns: dict[str, list[int]] | None = None) -> Any:
    return pa.table(columns or _DEFAULT_COLUMNS)


def _polars_lazy_data(columns: dict[str, list[int]] | None = None) -> Any:
    pl = pytest.importorskip("polars")
    return pl.from_arrow(pa.table(columns or _DEFAULT_COLUMNS)).lazy()


def _duckdb_data(columns: dict[str, list[int]] | None = None) -> Any:
    duckdb = pytest.importorskip("duckdb")
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

    return DuckdbRelation.from_arrow(duckdb.connect(), pa.table(columns or _DEFAULT_COLUMNS))


def _sqlite_data(columns: dict[str, list[int]] | None = None) -> Any:
    import sqlite3

    from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

    return SqliteRelation.from_arrow(sqlite3.connect(":memory:"), pa.table(columns or _DEFAULT_COLUMNS))


class TestIsColumnRefAccepts:
    @pytest.mark.parametrize(
        "value",
        [
            "timestamp",
            ("timestamp",),
            ["timestamp"],
            {"timestamp"},
            frozenset({"timestamp"}),
        ],
    )
    def test_plain_and_singleton_accepted(self, value: Any) -> None:
        """A plain column name and any single-element container are valid caller syntax."""
        assert is_column_ref(value) is True


class TestIsColumnRefRejects:
    @pytest.mark.parametrize(
        "value",
        [
            ["timestamp", "region"],
            ("timestamp", "region"),
            {"timestamp", "region"},
            frozenset({"timestamp", "region"}),
        ],
    )
    def test_multi_element_container_rejected(self, value: Any) -> None:
        """More than one column is a composite value, not a single column reference."""
        assert is_column_ref(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            123,
            True,
            None,
            "",
            [],
            (),
            set(),
            frozenset(),
            [123],
            (None,),
        ],
    )
    def test_non_str_and_empty_rejected(self, value: Any) -> None:
        """Non-string, empty-string and empty containers are never column references."""
        assert is_column_ref(value) is False


class TestColumnRefValue:
    @pytest.mark.parametrize(
        "value",
        [
            "timestamp",
            ("timestamp",),
            ["timestamp"],
            {"timestamp"},
            frozenset({"timestamp"}),
        ],
    )
    def test_unwraps_to_bare_column_name(self, value: Any) -> None:
        """The unwrapper must yield the column name itself, never the container's string form."""
        assert column_ref_value(value) == "timestamp"


class TestIsScalarNumberAccepts:
    @pytest.mark.parametrize(
        "value",
        [
            5,
            -5,
            0,
            0.75,
            (5,),
            [5],
            {5},
            frozenset({5}),
            (0.75,),
            [0.75],
        ],
    )
    def test_plain_and_singleton_accepted(self, value: Any) -> None:
        assert is_scalar_number(value) is True


class TestIsScalarNumberRejects:
    @pytest.mark.parametrize(
        "value",
        [
            [5, 10],
            (5, 10),
            {5, 10},
            frozenset({5, 10}),
        ],
    )
    def test_multi_element_container_rejected(self, value: Any) -> None:
        assert is_scalar_number(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            "5",
            None,
            [],
            (),
            set(),
            frozenset(),
            ["5"],
            (None,),
        ],
    )
    def test_non_number_and_empty_rejected(self, value: Any) -> None:
        assert is_scalar_number(value) is False

    @pytest.mark.parametrize("value", [True, False, (True,), [False]])
    def test_bool_rejected(self, value: Any) -> None:
        """bool subclasses int but is not a number here, the same rule is_positive_int applies."""
        assert is_scalar_number(value) is False


class TestScalarNumberValue:
    @pytest.mark.parametrize("value", [5, (5,), [5], {5}, frozenset({5})])
    def test_unwraps_to_bare_int(self, value: Any) -> None:
        assert scalar_number_value(value) == 5

    @pytest.mark.parametrize("value", [0.75, (0.75,), [0.75]])
    def test_unwraps_to_bare_float(self, value: Any) -> None:
        assert scalar_number_value(value) == 0.75


class TestIsPositiveIntSingleton:
    """is_positive_int gains the same arity contract; its value space is unchanged."""

    @pytest.mark.parametrize("value", [4, (4,), [4], {4}, frozenset({4})])
    def test_plain_and_singleton_accepted(self, value: Any) -> None:
        assert is_positive_int(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            0,
            (0,),
            [0],
            -1,
            (-1,),
            (4, 5),
            [4, 5],
            (),
            [],
            True,
            (True,),
            "4",
            ("4",),
            (4.0,),
        ],
    )
    def test_non_positive_multi_and_wrong_type_rejected(self, value: Any) -> None:
        assert is_positive_int(value) is False


class TestPositiveIntValue:
    @pytest.mark.parametrize("value", [4, (4,), [4], {4}, frozenset({4})])
    def test_unwraps_to_bare_int(self, value: Any) -> None:
        assert positive_int_value(value) == 4


class TestSingletonMatchesEndToEnd:
    """The guards' arity contract must hold through ``match_feature_group_criteria``."""

    def test_singleton_n_bins_matches(self) -> None:
        options = Options(context={"binning_op": "bin", "n_bins": (5,), "in_features": "value_int"})
        assert BinningFeatureGroup.match_feature_group_criteria("my_result", options, None) is True

    def test_multi_element_n_bins_still_rejected(self) -> None:
        options = Options(context={"binning_op": "bin", "n_bins": [5, 10], "in_features": "value_int"})
        assert BinningFeatureGroup.match_feature_group_criteria("my_result", options, None) is False

    def test_singleton_order_by_matches(self) -> None:
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": ("value_int",),
            }
        )
        assert RankFeatureGroup.match_feature_group_criteria("my_result", options, None) is True

    def test_multi_element_order_by_rejected(self) -> None:
        options = Options(context={"order_by": ["timestamp", "region"], "partition_by": ["region"]})
        assert PyArrowFfill.match_feature_group_criteria("amount__ffill", options, None) is False

    def test_non_string_order_by_rejected(self) -> None:
        options = Options(context={"order_by": 123, "partition_by": ["region"]})
        assert PyArrowFfill.match_feature_group_criteria("amount__ffill", options, None) is False


class TestAvailableColumns:
    """``available_columns`` dispatches on the input type, one factory per framework."""

    @pytest.mark.parametrize(
        "make_data",
        [
            pytest.param(_dict_data, id="dict"),
            pytest.param(_pandas_data, id="pandas"),
            pytest.param(_pyarrow_data, id="pyarrow"),
            pytest.param(_polars_lazy_data, id="polars_lazy"),
            pytest.param(_duckdb_data, id="duckdb"),
            pytest.param(_sqlite_data, id="sqlite"),
        ],
    )
    def test_returns_column_names_per_framework(self, make_data: Callable[[], Any]) -> None:
        assert available_columns(make_data()) == ["a", "b"]

    def test_pandas_column_name_collision_with_dispatch_helper_names(self) -> None:
        """Dispatch is on the class, so a pandas column named ``column_names`` or
        ``collect_schema`` must not hijack the pandas branch."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"column_names": [1], "collect_schema": [2]})
        assert available_columns(df) == ["column_names", "collect_schema"]


class TestAssertSourceColumnsPresent:
    """``assert_source_columns_present`` raises on the first missing column, with a fixed message shape."""

    def test_passes_when_all_columns_present(self) -> None:
        assert_source_columns_present({"a": [1], "b": [2]}, ["a", "b"])

    def test_raises_for_first_missing_column(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"Source column 'c' is not present in the dict input; available: \['a', 'b'\]",
        ):
            assert_source_columns_present({"a": [1], "b": [2]}, ["a", "c"])

    def test_message_names_the_input_type(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match=r"is not present in the DataFrame input; available: \['a'\]"):
            assert_source_columns_present(df, ["missing"])

    # -- Case sensitivity: exact match everywhere, SQL relations included ---

    @pytest.mark.parametrize(
        "make_data",
        [
            pytest.param(lambda: _dict_data({"Val": [1]}), id="dict"),
            pytest.param(lambda: _pandas_data({"Val": [1]}), id="pandas"),
            pytest.param(lambda: _pyarrow_data({"Val": [1]}), id="pyarrow"),
            pytest.param(lambda: _polars_lazy_data({"Val": [1]}), id="polars_lazy"),
            pytest.param(lambda: _duckdb_data({"Val": [1]}), id="duckdb"),
            pytest.param(lambda: _sqlite_data({"Val": [1]}), id="sqlite"),
        ],
    )
    def test_case_mismatch_rejected_for_every_framework(self, make_data: Callable[[], Any]) -> None:
        """Only column 'Val' exists; requesting 'val' must be rejected (exact-name match, no SQL case folding)."""
        with pytest.raises(ValueError, match=r"Source column 'val' is not present"):
            assert_source_columns_present(make_data(), ["val"])

    def test_label_overrides_the_message_prefix(self) -> None:
        with pytest.raises(ValueError, match=r"time_column 'ts' is not present in the dict input"):
            assert_source_columns_present({"a": [1]}, ["ts"], label="time_column")


class TestAssertKeyColumnsPresent:
    """``assert_key_columns_present`` checks partition, order and mask columns with per-role labels."""

    def test_passes_when_nothing_to_check(self) -> None:
        assert_key_columns_present({"a": [1]})
        assert_key_columns_present({"a": [1]}, partition_by=[], order_by=None, mask_spec=[])

    def test_passes_when_all_columns_present(self) -> None:
        assert_key_columns_present(
            {"a": [1], "b": [2], "c": [3]}, partition_by=["a"], order_by="b", mask_spec=[("c", "equal", 1)]
        )

    def test_missing_partition_by_column_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"partition_by 'x' is not present in the dict input"):
            assert_key_columns_present({"a": [1]}, partition_by=["a", "x"])

    def test_missing_order_by_column_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"order_by 'x' is not present in the dict input"):
            assert_key_columns_present({"a": [1]}, order_by="x")

    def test_missing_mask_column_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"mask column 'x' is not present in the dict input"):
            assert_key_columns_present({"a": [1]}, mask_spec=[("a", "equal", 1), ("x", "equal", 1)])

    def test_order_label_overrides_the_order_by_label(self) -> None:
        with pytest.raises(ValueError, match=r"time_column 'ts' is not present in the dict input"):
            assert_key_columns_present({"a": [1]}, order_by="ts", order_label="time_column")
