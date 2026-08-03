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

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.base import (
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
