"""Tests for PyArrowEma: EMA is REJECTED, not computed.

PyArrow has no exponentially weighted (EWM) compute kernel, so the PyArrow
backend must reject EMA up-front with a clear ValueError rather than emulate it
in Python (forbidden by the CFW-backend rule). This file inherits only the
single rejection assertion from ``EmaRejectionTestBase`` -- none of the value
tests.
"""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ema.pyarrow_ema import (
    PyArrowEma,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaRejectionTestBase,
)


class TestPyArrowEmaRejected(PyArrowTestMixin, EmaRejectionTestBase):
    """EMA on PyArrow raises a clear ValueError."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowEma
