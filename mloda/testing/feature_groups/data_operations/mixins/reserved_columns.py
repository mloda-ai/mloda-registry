"""Reusable reserved-column test mixin for data-operations feature groups.

Provides two standardized tests, run across every framework implementation of a
data-operation, that together prove there is no reserved namespace any more:
helper columns are made collision-free at runtime, so a user column of any name
(including a ``__mloda_``-prefixed one) is processed normally.

- ``test_mixin_reserved_column_collision_accepted``: a ``__mloda_``-prefixed
  user column is ACCEPTED (the call returns a non-``None`` result).
- ``test_mixin_helper_column_name_collision_survives``: user columns named like
  the FG's actual internal helpers SURVIVE unchanged in input row order. This is
  a real collision regression for row-preserving ops. It is opt-in: it runs only
  when ``reserved_columns_helper_names`` returns a non-empty set (default empty
  skips), so ops that reduce rows or need extra options are unaffected.

Each feature-group test base mixes this in and overrides the abstract
configuration methods to adapt the generic tests to its specific semantics
(feature name, partition keys, order key, helper name).

The test methods use a ``test_mixin_`` prefix to match the existing
convention for shared mixin tests (see ``MaskTestMixin``).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.helpers import make_feature_set


class ReservedColumnsTestMixin:
    """Mixin providing a standardized reserved-column acceptance test.

    Feature-group test bases mix this in and implement the configuration
    methods below to adapt the generic test to their semantics.

    Every backend now accepts ``__mloda_``-prefixed user columns because helper
    columns are made collision-free at runtime; no reserved namespace exists.

    Requires the host class to provide (from DataOpsTestBase):
    - ``implementation_class()``
    - ``_arrow_table`` attribute (set in setup_method)
    - ``create_test_data(arrow_table)``
    """

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        """Feature name for the reserved-columns test (e.g. 'value_int__sum_window')."""
        raise NotImplementedError

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        """Partition key(s) for the reserved-columns test.

        Defaults to ``["region"]``, the canonical partition column in the shared
        data-operations test dataset. Override to return ``None`` for ops that
        do not partition (e.g. scalar aggregate, binning, datetime, string) or
        to a different column list when the op uses a non-standard partition.
        """
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        """Order key for the reserved-columns test.

        Defaults to ``None``. Override for ops that require an order key
        (e.g. frame aggregate, offset).
        """
        return None

    @classmethod
    def reserved_columns_helper_names(cls) -> set[str]:
        """Internal helper column base names to exercise for collision survival.

        The survival test (below) appends a user column for EACH name in this
        set and asserts every one passes through unchanged. Different framework
        backends of the same op may pick different helper bases (e.g. offset:
        pandas ``__mloda_null_sort`` vs polars ``__mloda_orig_idx``; window
        aggregation: pyarrow ``__mloda_wa_idx__`` vs the SQL backends'
        ``__mloda_rn``). Listing all of them means that whichever backend the
        concrete test runs, that backend's own helper is among the injected
        columns and so is a real collision regression, while the others are
        harmless passthrough checks.

        Defaults to an empty set (the survival test is skipped). The test is
        opt-in because it only makes sense for row-preserving ops whose
        FeatureSet is buildable from ``feature_name`` + ``partition_by`` +
        ``order_by`` alone; ops that reduce rows or require extra options (e.g. a
        ``constant``) must leave this empty. Override to enable it, e.g.
        ``{"__mloda_rn__"}`` (ffill / ema / sessionization / frame_aggregate).
        """
        return set()

    # -- Concrete test method --------------------------------------------------

    def test_mixin_reserved_column_collision_accepted(self) -> None:
        """Mixin: a user input column matching the (formerly reserved) prefix is accepted.

        Uses an uppercase column name to verify case handling (SQLite and DuckDB
        unquoted identifiers fold case).

        No reserved namespace exists any more: helper columns are made
        collision-free at runtime, so every backend accepts and processes a
        ``__mloda_``-prefixed user column. The call therefore returns a
        non-``None`` result.
        """
        colliding_name = "__MLODA_USER_COL__"
        base_table: pa.Table = self._arrow_table  # type: ignore[attr-defined]
        colliding_table = base_table.append_column(
            colliding_name,
            pa.array([0] * base_table.num_rows, type=pa.int64()),
        )
        colliding_data = self.create_test_data(colliding_table)  # type: ignore[attr-defined]
        fs = make_feature_set(
            self.reserved_columns_feature_name(),
            partition_by=self.reserved_columns_partition_by(),
            order_by=self.reserved_columns_order_by(),
        )
        result = self.implementation_class().calculate_feature(colliding_data, fs)  # type: ignore[attr-defined]
        assert result is not None

    def test_mixin_helper_column_name_collision_survives(self) -> None:
        """Mixin: user input columns named like internal helpers survive unchanged.

        The pandas / polars / pyarrow backends pick a runtime row helper via
        ``unique_helper_name`` (the SQL backends via ``pick_helper_column_name``);
        a user column sharing that base name used to be clobbered (pandas
        overwrote it with ``range(len)`` then dropped it, polars
        ``with_row_index`` raised a DuplicateError). Because these ops are
        row-preserving, a user column of any name must pass through with its
        original values in input row order. The exercised helper names come from
        ``reserved_columns_helper_names``; an empty set skips this op. All names
        are injected at once so whichever backend runs sees its own helper among
        them.
        """
        helper_names = self.reserved_columns_helper_names()
        if not helper_names:
            pytest.skip("op opts out of the helper-column survival test")

        base_table: pa.Table = self._arrow_table  # type: ignore[attr-defined]
        n = base_table.num_rows
        # A distinct value range per helper column so a cross-column mix-up is caught too.
        expected = {name: [1000 * (j + 1) + i for i in range(n)] for j, name in enumerate(sorted(helper_names))}
        colliding_table = base_table
        for name, values in expected.items():
            colliding_table = colliding_table.append_column(name, pa.array(values, type=pa.int64()))
        data = self.create_test_data(colliding_table)  # type: ignore[attr-defined]
        fs = make_feature_set(
            self.reserved_columns_feature_name(),
            partition_by=self.reserved_columns_partition_by(),
            order_by=self.reserved_columns_order_by(),
        )
        result = self.implementation_class().calculate_feature(data, fs)  # type: ignore[attr-defined]
        assert self.get_row_count(result) == n  # type: ignore[attr-defined]
        for name, values in expected.items():
            survived = [int(v) for v in self.extract_column(result, name)]  # type: ignore[attr-defined]
            assert survived == values, f"user column {name} changed: {survived!r}"
