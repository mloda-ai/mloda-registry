"""PythonDict implementation for frame aggregate feature groups.

Pure-Python, dependency-free implementation with no engine-level rolling/window
limitations, so it targets FULL support: all four frame types (rolling, time,
cumulative, expanding), all seven time units (including month/year), and all
eight aggregation types (including std/var/median) -- matching DuckDB and
Polars-lazy, the two fullest-support existing backends. No
``SUPPORTED_TIME_UNITS`` / ``SUPPORTED_FRAME_TYPES`` / ``supported_op_subtypes()``
override is needed since the base class already declares the full sets.

The algorithm mirrors ``ReferenceFrameAggregate`` (the PyArrow-over-pure-Python
ground truth used across every backend's cross-framework tests): build
per-partition-group row lists (stable-sorted by ``order_by``, nulls last),
then for each row compute a window of prior/current rows per the frame type,
then reduce the window's values with the requested aggregation type.

Month/year time windows use calendar-aware subtraction with day-of-month
clamping (``_subtract_months``, stdlib ``calendar`` only), matching
``dateutil.relativedelta`` semantics without adding a dependency (the
``python_dict`` extras group is intentionally empty).
"""

from __future__ import annotations

import calendar
import statistics
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.errors import (
    unsupported_agg_type_error,
    unsupported_frame_type_error,
)
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

# Frame aggregation's order-independent aggregation-type subset (no mode/nunique/
# first/last, no ddof-variant spellings); mirrors base.py's ``_AGGREGATION_TYPES``.
_SUPPORTED_AGG_TYPES: frozenset[str] = frozenset({"sum", "avg", "count", "min", "max", "std", "var", "median"})


def _day_delta(n: int) -> timedelta:
    """Fallback factory (also the ``day`` unit itself) for ``_TIMEDELTA_FACTORIES.get``.

    A named, explicitly-typed function rather than an inline lambda default: mypy
    --strict cannot infer a bare lambda passed as ``dict.get``'s default argument.
    """
    return timedelta(days=n)


_TIMEDELTA_FACTORIES: dict[str, Callable[[int], timedelta]] = {
    "second": lambda n: timedelta(seconds=n),
    "minute": lambda n: timedelta(minutes=n),
    "hour": lambda n: timedelta(hours=n),
    "day": _day_delta,
    "week": lambda n: timedelta(weeks=n),
}


def _subtract_months(dt: datetime, months: int) -> datetime:
    """Subtract *months* calendar months from *dt*, clamping the day to the target month's length.

    Matches ``dateutil.relativedelta(months=months)`` semantics, e.g. Mar 31 minus
    1 month = Feb 28 (or Feb 29 in a leap year). A year offset is expressed as
    ``months = years * 12``, which reduces to a plain year decrement with the same
    day-of-month clamping.
    """
    total = dt.month - 1 - months
    year = dt.year + total // 12
    month = total % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


class PythonDictFrameAggregate(FrameAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_frame(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: int | None = None,
        frame_unit: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> dict[str, list[Any]]:
        if frame_type not in cls.SUPPORTED_FRAME_TYPES:
            raise unsupported_frame_type_error(frame_type, cls.SUPPORTED_FRAME_TYPES, framework="PythonDict")
        if agg_type not in _SUPPORTED_AGG_TYPES:
            raise unsupported_agg_type_error(
                agg_type, _SUPPORTED_AGG_TYPES, framework="PythonDict", operation="frame aggregate"
            )

        if frame_type == "time" and mask_spec is not None and source_col == order_by:
            # Mask + source_col == order_by: the reference treats masked rows as
            # having null order_by (mask writes null into source_col, which is
            # also order_by). This backend keeps order_by and the masked source
            # as two independently-built lists, so it cannot reproduce that
            # coupling without special-casing the combination. Reject to match
            # pandas/polars/DuckDB/SQLite. See known-divergences.md.
            raise ValueError(
                "PythonDict frame aggregate (time frame): mask + source_col == order_by "
                f"({source_col!r}) is unsupported. The reference semantic requires "
                "treating masked rows as having null order_by, which this backend does "
                "not special-case. See known-divergences.md."
            )

        partition_by = list(partition_by)
        num_rows = row_count(data)

        order_vals = data[order_by]
        source_values = data[source_col]

        if mask_spec is not None:
            mask = build_mask_from_spec(PythonDictMaskEngine, data, mask_spec)
            source_values = [v if m else None for v, m in zip(source_values, mask)]

        partition_cols = [data[col] for col in partition_by]

        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = tuple(col[i] for col in partition_cols)
            groups.setdefault(key, []).append((i, order_vals[i], source_values[i]))

        for rows in groups.values():
            rows.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0))

        result_values: list[Any] = [None] * num_rows

        for rows in groups.values():
            for pos, (orig_idx, order_val, _val) in enumerate(rows):
                if frame_type == "rolling":
                    wsize = int(frame_size) if frame_size is not None else 1
                    window_start = max(0, pos - wsize + 1)
                    window = [r[2] for r in rows[window_start : pos + 1]]
                elif frame_type in ("cumulative", "expanding"):
                    window = [r[2] for r in rows[: pos + 1]]
                elif frame_type == "time":
                    window = cls._time_window(rows, pos, order_val, frame_size or 1, str(frame_unit or "day"))
                else:  # pragma: no cover - guarded by the SUPPORTED_FRAME_TYPES check above
                    raise unsupported_frame_type_error(frame_type, cls.SUPPORTED_FRAME_TYPES, framework="PythonDict")

                result_values[orig_idx] = cls._reduce_window(window, agg_type)

        result = dict(data)
        result[feature_name] = result_values
        return result

    @classmethod
    def _time_window(
        cls,
        rows: list[tuple[int, Any, Any]],
        pos: int,
        current_order: Any,
        size: int,
        unit: str,
    ) -> list[Any]:
        """Collect values within a time-based window ending at the current row.

        Ports ``ReferenceFrameAggregate._time_window`` directly: second/minute/
        hour/day/week use ``timedelta``; month/year use ``_subtract_months`` for
        calendar-accurate arithmetic (no ``dateutil`` dependency). A null
        ``current_order`` returns just the row's own value, matching the
        reference's null-cutoff behavior.
        """
        if current_order is None:
            return [rows[pos][2]]

        if unit in ("month", "year"):
            months = size * 12 if unit == "year" else size
            window_start = _subtract_months(current_order, months)
        else:
            factory = _TIMEDELTA_FACTORIES.get(unit, _day_delta)
            window_start = current_order - factory(size)

        return [r[2] for r in rows[: pos + 1] if r[1] is not None and r[1] >= window_start]

    @classmethod
    def _reduce_window(cls, values: list[Any], agg_type: str) -> Any:
        """Reduce one window's raw (possibly null-containing) values to a single result.

        Null-skipping / all-null-returns-null semantics, duplicated locally (not
        imported from the aggregation operation package or the testing-only
        ``aggregation_helpers`` module) to keep this backend self-contained.
        """
        non_null = [v for v in values if v is not None]

        if not non_null:
            return 0 if agg_type == "count" else None

        if agg_type == "sum":
            return sum(non_null)
        if agg_type == "avg":
            return sum(non_null) / len(non_null)
        if agg_type == "count":
            return len(non_null)
        if agg_type == "min":
            return min(non_null)
        if agg_type == "max":
            return max(non_null)
        if agg_type == "median":
            return statistics.median(non_null)
        if agg_type in ("std", "var"):
            # Population variant (ddof=0), matching DuckDB's STDDEV_POP/VAR_POP
            # mapping and Polars' rolling_std/var(ddof=0).
            mean = sum(non_null) / len(non_null)
            variance = sum((x - mean) ** 2 for x in non_null) / len(non_null)
            return variance**0.5 if agg_type == "std" else variance

        raise unsupported_agg_type_error(
            agg_type, _SUPPORTED_AGG_TYPES, framework="PythonDict", operation="frame aggregate"
        )
