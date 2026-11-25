"""Known result-type finding shared by the Pandas aggregation and window tests."""

from __future__ import annotations

from mloda.testing.feature_groups.data_operations.mixins.result_type import KnownFinding

PANDAS_INPUT_DTYPE_FINDING = KnownFinding(
    "changes type on zero rows",
    "Pandas keeps the input dtype: the null-bearing int64 source is float64 with rows and int64 on zero rows "
    "(https://github.com/mloda-ai/mloda-registry/issues/1671)",
)
