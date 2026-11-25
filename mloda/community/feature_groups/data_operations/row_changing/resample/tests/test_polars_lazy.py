"""Tests for PolarsLazyResample compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_changing.resample.polars_lazy_resample import (
    PolarsLazyResample,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    ResampleTestBase,
)


class TestPolarsLazyResample(PolarsLazyTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyResample

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        # The first-seen negative zero partition key loses its sign.
        return {"27-partition_keys-v__resample_1_hour_count": {"p": [0.0]}}
