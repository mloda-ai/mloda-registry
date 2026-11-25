"""All-NaN min/max cases shared by the PyArrow aggregation, window and scalar-aggregate tests."""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

ALL_NAN_EXTREMA_CASES = [
    pytest.param([math.nan] * 3, math.nan, math.nan, id="all-nan"),
    pytest.param([math.nan, None, math.nan], math.nan, math.nan, id="nan-and-null"),
    pytest.param([math.nan, math.inf, None], math.inf, math.inf, id="nan-and-positive-infinity"),
    pytest.param([-math.inf, math.nan, math.inf], -math.inf, math.inf, id="both-infinities-and-nan"),
    pytest.param([math.nan, 3.0, -2.0], -2.0, 3.0, id="nan-and-finite"),
    pytest.param([None] * 3, None, None, id="all-null"),
]


def all_nan_case_control_table(values: list[float | None], arrow_type: pa.DataType) -> pa.Table:
    """A ``case`` group holding ``values`` next to a ``control`` group of 4.0, NaN, 1.0."""
    return pa.table(
        {
            "grp": ["case"] * 3 + ["control"] * 3,
            "value": pa.array([*values, 4.0, math.nan, 1.0], type=arrow_type),
        }
    )
