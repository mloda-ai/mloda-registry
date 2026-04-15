"""Reusable helper-column collision test mixin for data-operations feature groups.

A collision test enriches the canonical 12-row ``DataOpsTestBase.test_data``
with a user-supplied column whose name matches one of the implementation's
internal helper columns (``__mloda_rn__``, ``__mloda_null_sort``,
``__mloda_masked_src__`` etc.), runs the implementation via the public
``calculate_feature(test_data, fs)`` path, and verifies that:

* for row-preserving feature groups, the user column survives unchanged and
  the output column matches the same implementation's baseline output (the
  same call without the enriched column);
* for reducing feature groups, the helper column does not leak into the
  grouped result and the output column is still produced.

The mixin reuses ``DataOpsTestBase.create_test_data`` (framework-specific
conversion of a PyArrow table) and the framework adapters
``extract_column``/``get_row_count`` provided by the per-framework mixins.
Assertion helpers live in
``mloda.testing.feature_groups.data_operations.collision``.

Host classes must provide (already from ``DataOpsTestBase`` +
framework mixin):
- ``implementation_class()``
- ``test_data`` attribute and ``_arrow_table`` attribute (set in
  ``DataOpsTestBase.setup_method``)
- ``create_test_data(arrow_table)``
- ``extract_column(result, column_name)``
- ``get_row_count(result)``

Host classes must implement the ``collision_*`` configuration hooks below.
For operations that already mix in ``MaskTestMixin``, the defaults delegate
to the matching ``mask_*`` hook so the only new declaration is
``collision_order_by()`` when it differs.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa


from mloda.testing.feature_groups.data_operations.collision import (
    assert_column_absent,
    assert_output_column,
    assert_user_column_preserved,
    column_names,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

# Canonical 12-row dataset has 12 rows; user values are generated per-row.
_CANONICAL_ROW_COUNT = 12
_CANONICAL_MASK_SPEC: tuple[str, str, Any] = ("category", "equal", "X")


def _user_values(n: int = _CANONICAL_ROW_COUNT) -> list[str]:
    return [f"u{i}" for i in range(n)]


class CollisionTestMixin:
    """Mixin providing standardized helper-column collision tests.

    Concrete framework test classes (e.g. ``TestPandasOffset``) call
    ``self._run_collision_case(helper_name)`` from a one-line test method
    per helper name their implementation uses.
    """

    # -- Configuration hooks ---------------------------------------------------
    # Default implementations delegate to the matching MaskTestMixin hook when
    # present, so mask-capable ops don't have to duplicate the configuration.

    @classmethod
    def collision_feature_name(cls) -> str:
        """Feature name to compute for collision tests."""
        mask_hook = getattr(cls, "mask_feature_name", None)
        if mask_hook is not None:
            return str(mask_hook())
        raise NotImplementedError(
            f"{cls.__name__} must implement collision_feature_name() or inherit from MaskTestMixin"
        )

    @classmethod
    def collision_partition_by(cls) -> list[str] | None:
        mask_hook = getattr(cls, "mask_partition_by", None)
        if mask_hook is not None:
            return mask_hook()  # type: ignore[no-any-return]
        return None

    @classmethod
    def collision_order_by(cls) -> str | None:
        mask_hook = getattr(cls, "mask_order_by", None)
        if mask_hook is not None:
            return mask_hook()  # type: ignore[no-any-return]
        return None

    @classmethod
    def collision_is_reducing(cls) -> bool:
        """True for group-by aggregation (row count collapses); False for row-preserving."""
        mask_hook = getattr(cls, "mask_is_reducing", None)
        if mask_hook is not None:
            return bool(mask_hook())
        return False

    # -- Runner ----------------------------------------------------------------

    def _run_collision_case(
        self,
        helper_name: str,
        *,
        use_mask: bool = False,
        feature_name: str | None = None,
        partition_by: list[str] | None = None,
        order_by: str | None = None,
    ) -> None:
        """Verify that a user column named *helper_name* survives the computation.

        Builds an enriched copy of the canonical arrow table with a
        pass-through column and runs the implementation's
        ``calculate_feature`` on the framework-native conversion.

        ``feature_name``/``partition_by``/``order_by`` override the class
        defaults when a collision helper is only materialized on a specific
        code path (e.g. DuckDB window first/last uses ``__mloda_rn__``, not
        the default sum path).
        """
        feature_name = feature_name if feature_name is not None else type(self).collision_feature_name()
        partition_by = partition_by if partition_by is not None else type(self).collision_partition_by()
        order_by = order_by if order_by is not None else type(self).collision_order_by()
        mask = _CANONICAL_MASK_SPEC if use_mask else None

        # Baseline is computed first so the enriched relation stays "fresh"
        # when we read from it. Some frameworks (notably DuckDB) return lazy
        # relations whose column resolution can be disturbed by creating
        # further relations on the same connection.
        baseline_fs = make_feature_set(feature_name, partition_by, order_by, mask=mask)
        baseline_result = self.implementation_class().calculate_feature(self.test_data, baseline_fs)  # type: ignore[attr-defined]
        baseline_values = self.extract_column(baseline_result, feature_name)  # type: ignore[attr-defined]

        enriched_table = _append_passthrough_column(self._arrow_table, helper_name)  # type: ignore[attr-defined]
        enriched_data = self.create_test_data(enriched_table)  # type: ignore[attr-defined]
        fs = make_feature_set(feature_name, partition_by, order_by, mask=mask)
        result = self.implementation_class().calculate_feature(enriched_data, fs)  # type: ignore[attr-defined]

        if type(self).collision_is_reducing():
            # Helper column must not leak into the grouped result and output
            # column must still be present. Value correctness is already
            # covered by MaskTestMixin's own tests on the same call.
            assert_column_absent(result, helper_name)
            assert feature_name in column_names(result), (
                f"output column '{feature_name}' missing after collision scenario"
            )
            return

        assert_user_column_preserved(result, helper_name, _user_values())
        assert_output_column(result, feature_name, baseline_values)

    # -- Inner pytest test --------------------------------------------------
    # Intentionally no ``test_mixin_*`` method is defined here: the coverage
    # decision is "one test per framework-specific helper name", which lives
    # on the concrete framework test class. Each concrete class adds a
    # one-line ``def test_collision_<name>(self): self._run_collision_case(...)``
    # for each helper name its implementation uses.


def _append_passthrough_column(table: pa.Table, column_name: str) -> pa.Table:
    """Return *table* with an extra string column ``[u0, u1, ...]``."""
    return table.append_column(column_name, pa.array(_user_values(table.num_rows), type=pa.string()))
