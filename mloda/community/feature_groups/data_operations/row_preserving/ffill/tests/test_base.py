"""Tests for FfillFeatureGroup base class.

Covers the arity contract of the scalar ``order_by`` key: core unwraps a
one-element container when it reads a property value, so ``("timestamp",)`` is
valid caller syntax for one column and must dispatch to ``"timestamp"`` rather
than to the container's string form. Match tests use ``PyArrowFfill`` (the ffill
convention, mirroring test_integration).
"""

from __future__ import annotations

from typing import Any

from mloda.user import Feature, Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column, feature_set_for
from mloda.testing.feature_groups.data_operations.match_validation import ScalarArityTestBase, TokenCase

from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import PyArrowFfill

FEATURE_NAME = "amount__ffill"


class TestOrderByArity(ScalarArityTestBase):
    """``order_by`` is a scalar key: one column, bare or in a single-element container.

    ffill is a single-op family with no operation config key at all, so it declares the
    arity base directly instead of ``MatchValidationTestBase``.
    """

    @classmethod
    def feature_group_class(cls) -> Any:
        return FfillFeatureGroup

    @classmethod
    def match_class(cls) -> Any:
        return PyArrowFfill

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
        return [FfillFeatureGroup._extract_order_by(Feature(FEATURE_NAME, options=options))]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        result = PyArrowFfill.calculate_feature(
            PyArrowDataOpsTestDataCreator.create(), feature_set_for(FEATURE_NAME, options)
        )
        return extract_column(result, FEATURE_NAME)
