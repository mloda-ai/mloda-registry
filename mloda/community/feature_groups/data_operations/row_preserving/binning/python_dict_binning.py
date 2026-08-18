"""PythonDict implementation for binning feature groups."""

from __future__ import annotations

import math
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class PythonDictBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_binning(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> dict[str, list[Any]]:
        data = dict(data)
        col = data[source_col]

        non_null = [(i, v) for i, v in enumerate(col) if not cls._is_null(v)]

        if not non_null:
            data[feature_name] = [None] * len(col)
            return data

        if op == "bin":
            result = cls._equal_width_bin(col, non_null, n_bins)
        elif op == "qbin":
            result = cls._quantile_bin(col, non_null, n_bins)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        data[feature_name] = result
        return data

    @staticmethod
    def _is_null(value: Any) -> bool:
        return value is None or (isinstance(value, float) and math.isnan(value))

    @classmethod
    def _equal_width_bin(cls, col: list[Any], non_null: list[tuple[int, Any]], n_bins: int) -> list[Any]:
        values = [v for _, v in non_null]
        col_min = min(values)
        col_max = max(values)

        result: list[Any] = [None] * len(col)

        if col_min == col_max:
            for i, _ in non_null:
                result[i] = 0
            return result

        bin_width = (col_max - col_min) / n_bins
        max_bin = n_bins - 1

        for i, v in non_null:
            bin_idx = math.floor((v - col_min) / bin_width)
            result[i] = min(bin_idx, max_bin)

        return result

    @classmethod
    def _quantile_bin(cls, col: list[Any], non_null: list[tuple[int, Any]], n_bins: int) -> list[Any]:
        n = len(non_null)
        ordered = sorted(non_null, key=lambda item: item[1])

        result: list[Any] = [None] * len(col)

        for rank, (original_idx, _) in enumerate(ordered):
            result[original_idx] = rank * n_bins // n

        return result
