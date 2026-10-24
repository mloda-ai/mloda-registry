"""PythonDict implementation for datetime extraction feature groups."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)


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

        cls._assert_source_column_is_datetime(col, source_col)

        data[feature_name] = [cls._extract(value, op) for value in col]
        return data

    @staticmethod
    def _assert_source_column_is_datetime(col: list[Any], source_col: str) -> None:
        """Reject any non-null value that is not a real ``datetime.datetime``.

        A bare ``datetime.date`` (no time component) or a wholly unrelated
        type (e.g. ``str``) cannot be used to compute any datetime op here:
        ``datetime.datetime`` is a subclass of ``datetime.date``, so this
        check accepts real datetimes while rejecting bare dates.
        """
        for value in col:
            if value is None:
                continue
            if not isinstance(value, datetime):
                raise ValueError(
                    f"Column {source_col!r} must contain datetime.datetime values to extract "
                    f"datetime components; got {type(value).__name__} ({value!r})."
                )

    @staticmethod
    def _extract(value: datetime | None, op: str) -> int | None:
        # The column-wide guard in _assert_source_column_is_datetime already ran, so
        # every non-None value here is a real datetime.
        if value is None:
            return None

        if op == "year":
            return value.year
        elif op == "month":
            return value.month
        elif op == "day":
            return value.day
        elif op == "hour":
            return value.hour
        elif op == "minute":
            return value.minute
        elif op == "second":
            return value.second
        elif op == "dayofweek":
            return value.weekday()
        elif op == "is_weekend":
            return 1 if value.weekday() >= 5 else 0
        elif op == "quarter":
            return (value.month - 1) // 3 + 1
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")
