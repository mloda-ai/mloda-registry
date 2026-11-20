"""Hand-computed pinning rows of the binning reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

BINNING_RULES: Mapping[str, str] = MappingProxyType(
    {
        "none": "None in gives None out and is not counted, for both operations.",
        "nan": "NaN counts as missing: it gives None and takes no part in min, max or rank.",
        "all_missing": "An all-missing column gives all None; an empty input gives an empty int64 column.",
        "equal_width": (
            "bin splits [min, max] into n equal widths: bin = floor((x - min) / width), width = (max - min) / n."
        ),
        "max_value": "The maximum would land in bin n, so it is clamped into the last bin n - 1.",
        "float_rounding": (
            "The width is a float64 and the quotient is floored as is, so 0.1 with width 0.3 / 3 is bin 1."
        ),
        "int_float": "Integers and floats with the same values give the same bins.",
        "constant": "When min equals max every present value is bin 0.",
        "single_bin": "With n = 1 every present value is bin 0.",
        "wide_range": "Narrow integers are computed in float64, so int8 -100 to 100 does not overflow.",
        "float32": "A float32 column is computed in float64 from the float32 values, not in float32.",
        "qbin_rank": "qbin ranks the present values ascending and gives bin = rank * n // count.",
        "ties": (
            "Equal values, 0.0 and -0.0 included, keep input order in the rank, so they can land in different bins."
        ),
        "gaps": "With more bins than rows some bins stay empty: bin = rank * n // count skips them.",
        "infinity": "qbin accepts infinite values and integers beyond 2**53; they sort like any other value.",
        "dtype": "bin and qbin give int64.",
        "row_order": "Output rows keep the input order.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE

_Pairs = Sequence[tuple[Any, int | None]]


def _row(case: str, feature: str, pairs: _Pairs, source: DataType = _I) -> PinningRow:
    return PinningRow(
        case=case,
        feature=feature,
        inputs={"x": (source, tuple(value for value, _ in pairs))},
        expected={feature: (_I, tuple(bin_ for _, bin_ in pairs))},
    )


_NAN = float("nan")
_INF = float("inf")

BINNING_PINNING = PinningTable(
    operation="binning",
    rules=BINNING_RULES,
    rows=(
        _row("none", "x__bin_2", [(10, 0), (None, None), (20, 1), (30, 1)]),
        _row("none", "x__qbin_3", [(5, 2), (None, None), (1, 0), (3, 1)]),
        _row("nan", "x__bin_2", [(0.0, 0), (_NAN, None), (10.0, 1), (None, None)], _D),
        _row("nan", "x__qbin_2", [(2.0, 1), (_NAN, None), (1.0, 0)], _D),
        _row("all_missing", "x__bin_3", [(None, None), (None, None)]),
        _row("all_missing", "x__bin_3", [(None, None), (_NAN, None)], _D),
        _row("all_missing", "x__qbin_3", [(None, None), (_NAN, None)], _D),
        _row("all_missing", "x__qbin_3", [(None, None)]),
        _row("all_missing", "x__bin_3", []),
        _row("all_missing", "x__qbin_3", []),
        _row("all_missing", "x__bin_3", [], _D),
        _row("equal_width", "x__bin_5", [(i, b) for i, b in enumerate([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])]),
        _row("equal_width", "x__bin_4", [(-1.0, 0), (-0.5, 1), (0.0, 2), (0.5, 3), (1.0, 3)], _D),
        _row("max_value", "x__bin_2", [(0, 0), (5, 1), (10, 1)]),
        _row("float_rounding", "x__bin_3", [(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 2)], _D),
        _row(
            "int_float",
            "x__bin_5",
            [(float(i), b) for i, b in enumerate([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])],
            _D,
        ),
        _row("constant", "x__bin_3", [(7, 0), (7, 0), (None, None), (7, 0)]),
        _row("constant", "x__bin_4", [(2.5, 0), (2.5, 0)], _D),
        _row("single_bin", "x__bin_1", [(1, 0), (5, 0), (9, 0)]),
        _row("wide_range", "x__bin_3", [(-100, 0), (0, 1), (100, 2)], DataType.INT8),
        _row("wide_range", "x__bin_3", [(-(2**31), 0), (0, 1), (2**31 - 1, 2)], DataType.INT32),
        _row("float32", "x__bin_3", [(0.0, 0), (0.1, 0), (0.2, 1), (0.3, 2)], DataType.FLOAT),
        _row("qbin_rank", "x__qbin_2", [(30, 1), (10, 0), (20, 0), (40, 1)]),
        _row("qbin_rank", "x__qbin_3", [(5, 2), (None, None), (1, 0), (3, 1)]),
        _row("ties", "x__qbin_4", [(0.0, 0), (-0.0, 1), (1.0, 3), (-0.0, 2)], _D),
        _row("ties", "x__qbin_2", [(5, 0), (5, 1), (5, 1), (1, 0)]),
        _row("gaps", "x__qbin_5", [(3, 3), (1, 0), (2, 1)]),
        _row("infinity", "x__qbin_3", [(_INF, 2), (1.0, 1), (-_INF, 0)], _D),
        _row("infinity", "x__qbin_3", [(2**53 + 1, 2), (1, 0), (2**53, 1)]),
        _row("infinity", "x__qbin_2", [(2**63 - 1, 1), (-(2**63), 0)]),
        _row("dtype", "x__bin_2", [(1, 0), (2, 1)]),
        _row("dtype", "x__qbin_2", [(1, 0), (2, 1)]),
        _row("row_order", "x__bin_3", [(9, 2), (0, 0), (5, 1)]),
        _row("row_order", "x__qbin_3", [(9, 2), (0, 0), (5, 1)]),
    ),
)
