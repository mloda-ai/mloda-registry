"""Tests for PythonDictResample compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_changing.resample.python_dict_resample import (
    PythonDictResample,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    ResampleTestBase,
)


class TestPythonDictResample(PythonDictTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictResample
