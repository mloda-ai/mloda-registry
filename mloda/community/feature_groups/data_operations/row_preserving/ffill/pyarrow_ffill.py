"""PyArrow implementation of ffill-by-time (production AND reference)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup


class PyArrowFfill(FfillFeatureGroup):
    """PyArrow backend; also the cross-framework reference implementation.

    ``pyarrow.compute.fill_null_forward`` does NOT respect partition boundaries,
    so the fill is applied PER PARTITION. Rows are tagged with their original
    position, sorted by ``[*partition_by, order_by]`` ascending, filled within
    each partition's contiguous (sorted) slice, then scattered back to the
    original row order. No per-row Python loop is used; only a loop over the
    (small) set of partition groups.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _assert_source_column_present(cls, data: pa.Table, source_col: str) -> None:
        if source_col not in data.schema.names:
            raise ValueError(
                f"Source column {source_col!r} is not present in the PyArrow table; available: {data.schema.names}."
            )

    @classmethod
    def _compute_ffill(
        cls,
        data: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
    ) -> pa.Table:
        # Sort the whole table by [*partition_by, order_by] ascending so that
        # rows of one partition form a contiguous, time-ordered slice.
        sort_keys = [(col, "ascending") for col in (*partition_by, order_by)]
        sorted_indices = pc.sort_indices(data, sort_keys=sort_keys)

        source = data.column(source_col)
        sorted_source = pc.take(source, sorted_indices)

        if partition_by:
            # Build, in sorted order, an integer group id that increments at each
            # partition boundary, then fill within each contiguous group.
            sorted_partition_cols = [pc.take(data.column(col), sorted_indices) for col in partition_by]
            filled_sorted = cls._fill_per_partition(sorted_source, sorted_partition_cols)
        else:
            filled_sorted = pc.fill_null_forward(sorted_source)

        # Scatter the filled values back to their original row positions.
        scatter = pc.sort_indices(sorted_indices)
        filled = pc.take(filled_sorted, scatter)

        return data.append_column(feature_name, filled)

    @staticmethod
    def _fill_per_partition(sorted_source: pa.Array, sorted_partition_cols: list[pa.Array]) -> pa.Array:
        """Forward-fill ``sorted_source`` independently within each contiguous partition.

        ``sorted_source`` and ``sorted_partition_cols`` are already ordered so
        that rows of one partition are contiguous. A boundary mask (computed with
        Arrow vector ops, no per-row Python loop) marks where a new partition
        starts; the source is sliced at those boundaries and each slice is filled
        independently, then the slices are concatenated. The only Python loop is
        over the (small) number of partition groups.
        """
        n = len(sorted_source)
        if n == 0:
            return sorted_source

        # Build a boundary mask of length n where mask[i] is True iff row i begins
        # a new partition (its key differs from row i-1). Row 0 is always a boundary.
        # Comparing the shifted columns and OR-ing across partition columns is all
        # done with Arrow compute kernels.
        changed: pa.Array = pa.array([False] * (n - 1), type=pa.bool_())
        for col in sorted_partition_cols:
            prev = col.slice(0, n - 1)
            curr = col.slice(1, n - 1)
            # not equal OR exactly one side null -> key changed at this position
            neq = pc.not_equal(curr, prev)
            null_diff = pc.not_equal(pc.is_null(curr), pc.is_null(prev))
            col_changed = pc.fill_null(pc.or_(neq, null_diff), True)
            changed = pc.or_(changed, col_changed)

        boundary_mask = pa.concat_arrays([pa.array([True], type=pa.bool_()), changed.combine_chunks()])
        boundaries: list[int] = pc.indices_nonzero(boundary_mask).to_pylist()

        pieces: list[pa.Array] = []
        for start, end in zip(boundaries, boundaries[1:] + [n]):
            piece = sorted_source.slice(start, end - start)
            pieces.append(pc.fill_null_forward(piece))

        return pa.chunked_array(pieces, type=sorted_source.type).combine_chunks()
