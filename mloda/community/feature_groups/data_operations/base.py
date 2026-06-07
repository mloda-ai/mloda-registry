"""Shared constants and utilities for all data operation feature groups."""

from __future__ import annotations

from mloda.core.abstract_plugins.components.feature import Feature


# ---------------------------------------------------------------------------
# Shared config key constants
# ---------------------------------------------------------------------------
# These are the keys used in Options to configure data operation feature groups.
# Core's DefaultOptionKeys defines standard keys like "order_by".
# Use DefaultOptionKeys directly for keys that exist in core.
# The constants below are data-operations-specific keys not in DefaultOptionKeys.

PARTITION_BY = "partition_by"
"""Config key for partitioning columns (list of column names).

Used by: window_aggregation, aggregation, rank, offset, frame_aggregate.
"""

FRAME_TYPE = "frame_type"
"""Config key for frame type in frame_aggregate operations.

Valid values: "rows", "time", "expanding", "cumulative".
"""

FRAME_SIZE = "frame_size"
"""Config key for frame size (positive integer).

Used by frame_aggregate with frame_type "rows" or "time".
"""

FRAME_UNIT = "frame_unit"
"""Config key for time unit in time-interval frames.

Used by frame_aggregate with frame_type "time".
Valid values: second, minute, hour, day, week, month, year.
"""


# ---------------------------------------------------------------------------
# Shared mixins
# ---------------------------------------------------------------------------


class PartitionByMixin:
    """Shared ``_extract_partition_by`` for partition-aware row operations.

    Mixed into the ffill/ema/sessionization/resample bases, which each define
    a ``PARTITION_BY`` options key. Reads that key from the feature options,
    returning ``[]`` when absent.
    """

    PARTITION_BY: str

    @classmethod
    def _extract_partition_by(cls, feature: Feature) -> list[str]:
        """Return ``partition_by`` as a list (defaulting to ``[]`` when absent)."""
        partition_by = feature.options.get(cls.PARTITION_BY)
        if partition_by is None:
            return []
        return list(partition_by)
