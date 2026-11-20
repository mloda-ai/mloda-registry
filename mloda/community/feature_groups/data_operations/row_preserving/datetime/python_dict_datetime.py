"""PythonDict implementation for datetime extraction feature groups."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)

_TIME_OF_DAY_OPS = frozenset({"hour", "minute", "second"})


class PythonDictDateTimeExtraction(Implementation[DateTimeFeatureGroup]):
    feature_group = DateTimeFeatureGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls, group: type[DateTimeFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.extract_datetime(data, features, compute_datetime=cls._compute_datetime)

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

        cls._assert_source_column_is_datetime(col, source_col, op)

        data[feature_name] = [cls._extract(value, op) for value in col]
        return data

    @staticmethod
    def _assert_source_column_is_datetime(col: list[Any], source_col: str, op: str) -> None:
        """Reject non-date values, and bare dates for time-of-day ops."""
        for value in col:
            if value is None:
                continue
            if not isinstance(value, date):
                raise ValueError(
                    f"Column {source_col!r} must contain datetime.date or datetime.datetime values to extract "
                    f"datetime components; got {type(value).__name__} ({value!r})."
                )
            if op in _TIME_OF_DAY_OPS and not isinstance(value, datetime):
                raise ValueError(
                    f"Column {source_col!r} holds a bare date, but {op!r} needs a time of day (datetime or timestamp)."
                )

    @staticmethod
    def _extract(value: date | None, op: str) -> int | None:
        if value is None:
            return None

        if op == "year":
            return value.year
        elif op == "month":
            return value.month
        elif op == "day":
            return value.day
        elif op == "hour":
            return value.hour if isinstance(value, datetime) else None
        elif op == "minute":
            return value.minute if isinstance(value, datetime) else None
        elif op == "second":
            return value.second if isinstance(value, datetime) else None
        elif op == "dayofweek":
            return value.weekday()
        elif op == "is_weekend":
            return 1 if value.weekday() >= 5 else 0
        elif op == "quarter":
            return (value.month - 1) // 3 + 1
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")
