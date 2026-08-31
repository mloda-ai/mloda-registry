"""Tests for EmaFeatureGroup base class.

Covers the arity contract of the scalar ``order_by`` key: core unwraps a
one-element container when it reads a property value, so ``("timestamp",)`` is
valid caller syntax for one column and must dispatch to ``"timestamp"`` rather
than to the container's string form. Match tests use ``PandasEma`` (the ema
convention, mirroring ``TestEmaMatchFeatureGroupCriteria`` in test_integration).
"""

from __future__ import annotations

from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.user import Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ema.pandas_ema import PandasEma
from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column, feature_set_for
from mloda.testing.feature_groups.data_operations.match_validation import ScalarArityTestBase, TokenCase

FEATURE_NAME = "value_float__ema_2"


class TestOrderByArity(ScalarArityTestBase):
    """``order_by`` is a scalar key: one column, bare or in a single-element container.

    ema declares no operation config key (the span is part of the feature name), so it
    declares the arity base directly instead of ``MatchValidationTestBase``.
    """

    @classmethod
    def feature_group_class(cls) -> Any:
        return EmaFeatureGroup

    @classmethod
    def match_class(cls) -> Any:
        return PandasEma

    @classmethod
    def match_feature_name(cls) -> str:
        return FEATURE_NAME

    @classmethod
    def base_context(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        return [TokenCase("order_by", "timestamp", "region", required=True)]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return [EmaFeatureGroup._extract_order_by(Feature(FEATURE_NAME, options=options))]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        # Compared as strings so that NaN, which never equals itself, stays comparable.
        df = pd.DataFrame(DataOperationsTestDataCreator.get_raw_data())
        result = PandasEma.calculate_feature(df, feature_set_for(FEATURE_NAME, options))
        return [str(value) for value in extract_column(result, FEATURE_NAME)]
