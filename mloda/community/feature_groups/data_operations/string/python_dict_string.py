"""PythonDict implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)


class PythonDictStringOps(StringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_string(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        op: str,
    ) -> dict[str, list[Any]]:
        col = data[source_col]

        if op == "upper":
            result = [None if v is None else v.upper() for v in col]
        elif op == "lower":
            result = [None if v is None else v.lower() for v in col]
        elif op == "trim":
            result = [None if v is None else v.strip() for v in col]
        elif op == "length":
            result = [None if v is None else len(v) for v in col]
        elif op == "reverse":
            result = [None if v is None else v[::-1] for v in col]
        else:
            raise ValueError(f"Unsupported string operation: {op}")

        data = dict(data)
        data[feature_name] = result
        return data
