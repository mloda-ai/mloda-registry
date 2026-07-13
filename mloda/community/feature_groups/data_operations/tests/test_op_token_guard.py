"""Arity contract of the shared ``is_op_token`` match_guard.

Core unwraps a singleton container when reading a property value (see
``feature_chain_parser._unpack_property_value`` and ``FeatureGroup.resolve_subtype``),
so ``("sum",)`` is valid caller syntax for a single op token. The guard must accept it
while still rejecting multi-element containers and non-string values.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.base import is_op_token


class TestIsOpTokenAccepts:
    @pytest.mark.parametrize(
        "value",
        [
            "sum",
            ("sum",),
            ["sum"],
            {"sum"},
            frozenset({"sum"}),
        ],
    )
    def test_plain_and_singleton_accepted(self, value: Any) -> None:
        """A plain token and any single-element container are valid caller syntax."""
        assert is_op_token(value) is True


class TestIsOpTokenRejects:
    @pytest.mark.parametrize(
        "value",
        [
            ["sum", "max"],
            ("sum", "max"),
            {"sum", "max"},
            frozenset({"sum", "max"}),
        ],
    )
    def test_multi_element_container_rejected(self, value: Any) -> None:
        """More than one token is a composite form, not a single op token."""
        assert is_op_token(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            123,
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
        """Non-string, empty-string and empty containers are never op tokens."""
        assert is_op_token(value) is False


class TestSingletonMatchesEndToEnd:
    """The guard's arity contract must hold through ``match_feature_group_criteria``."""

    def test_singleton_tuple_matches(self) -> None:
        options = Options(
            context={"aggregation_type": ("sum",), "in_features": "value_int", "partition_by": ["region"]}
        )
        result = AggregationFeatureGroup().match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_multi_element_list_still_rejected(self) -> None:
        options = Options(
            context={"aggregation_type": ["sum", "max"], "in_features": "value_int", "partition_by": ["region"]}
        )
        result = AggregationFeatureGroup().match_feature_group_criteria("my_result", options, None)
        assert result is False
