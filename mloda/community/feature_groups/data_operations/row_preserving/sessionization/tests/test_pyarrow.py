"""Tests for PyArrowSessionization compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pyarrow_sessionization import (
    PyArrowSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPyArrowSessionization(PyArrowTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowSessionization
