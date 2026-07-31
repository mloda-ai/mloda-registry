"""Tests for FfillFeatureGroup base class.

Covers the arity contract of the scalar ``order_by`` key: core unwraps a
one-element container when it reads a property value, so ``("timestamp",)`` is
valid caller syntax for one column and must dispatch to ``"timestamp"`` rather
than to the container's string form. Match tests use ``PyArrowFfill`` (the ffill
convention, mirroring test_integration).
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import PyArrowFfill


class TestOrderByArity:
    """``order_by`` is a scalar key: one column, bare or in a single-element container.

    ffill is a single-op family with no operation config key at all, so
    ``MatchValidationTestBase`` does not fit it and these checks are hand-rolled.
    """

    def _options(self, order_by: Any) -> Options:
        return Options(context={"order_by": order_by, "partition_by": ["region"]})

    @pytest.mark.parametrize("order_by", ["timestamp", ("timestamp",), ["timestamp"]])
    def test_singleton_matches(self, order_by: Any) -> None:
        assert PyArrowFfill.match_feature_group_criteria("amount__ffill", self._options(order_by)) is True

    @pytest.mark.parametrize("order_by", [["timestamp", "region"], ("timestamp", "region")])
    def test_multi_element_rejected(self, order_by: Any) -> None:
        assert PyArrowFfill.match_feature_group_criteria("amount__ffill", self._options(order_by)) is False

    def test_wrong_type_rejected(self) -> None:
        assert PyArrowFfill.match_feature_group_criteria("amount__ffill", self._options(123)) is False

    @pytest.mark.parametrize("order_by", ["timestamp", ("timestamp",), ["timestamp"]])
    def test_extract_order_by_unwraps_to_bare_column(self, order_by: Any) -> None:
        feature = Feature("amount__ffill", options=self._options(order_by))
        assert FfillFeatureGroup._extract_order_by(feature) == "timestamp"

    def test_extract_order_by_raises_when_missing(self) -> None:
        feature = Feature("amount__ffill", options=Options())
        with pytest.raises(ValueError, match="order_by"):
            FfillFeatureGroup._extract_order_by(feature)

    def test_singleton_order_by_dispatches_like_bare(self) -> None:
        table = PyArrowDataOpsTestDataCreator.create()

        def _compute(order_by: Any) -> list[Any]:
            fs = FeatureSet()
            fs.add(Feature("amount__ffill", options=self._options(order_by)))
            return list(PyArrowFfill.calculate_feature(table, fs).column("amount__ffill").to_pylist())

        assert _compute(("timestamp",)) == _compute("timestamp")
