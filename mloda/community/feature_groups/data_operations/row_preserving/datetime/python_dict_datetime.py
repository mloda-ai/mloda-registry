"""PythonDict implementation for datetime extraction feature groups."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)


class PythonDictDateTimeExtraction(DateTimeFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_datetime(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        op: str,
    ) -> dict[str, list[Any]]:
        data = dict(data)
        col = data[source_col]

        data[feature_name] = [cls._extract(value, op) for value in col]
        return data

    @staticmethod
    def _extract(value: datetime | date | None, op: str) -> int | None:
        if value is None:
            return None

        if op == "year":
            return value.year
        elif op == "month":
            return value.month
        elif op == "day":
            return value.day
        elif op == "hour":
            return getattr(value, "hour", 0)
        elif op == "minute":
            return getattr(value, "minute", 0)
        elif op == "second":
            return getattr(value, "second", 0)
        elif op == "dayofweek":
            return value.weekday()
        elif op == "is_weekend":
            return 1 if value.weekday() >= 5 else 0
        elif op == "quarter":
            return (value.month - 1) // 3 + 1
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")
