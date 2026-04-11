"""Reusable mask integration test mixin for data-operations feature groups.

Provides 3 integration test methods that verify masking works correctly
through mloda's full pipeline (run_all, PluginCollector, feature resolution).

Each integration test class mixes this in alongside DataOpsIntegrationTestBase
and overrides the configuration methods to adapt to its specific feature group.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature, mloda


def _is_null(value: Any) -> bool:
    """Check if a value is null (None or NaN)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


class MaskIntegrationTestMixin:
    """Mixin providing mask integration tests through mloda's full pipeline.

    Requires the host class to provide (from DataOpsIntegrationTestBase):
    - ``feature_group_class()``
    - ``data_creator_class()``
    - ``compute_framework_class()``
    - ``_plugin_collector()``
    """

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def mask_integration_feature_name(cls) -> str:
        """Feature name for mask integration tests."""
        raise NotImplementedError

    @classmethod
    def mask_integration_options(cls) -> dict[str, Any]:
        """Options context dict for mask integration tests (without mask key)."""
        raise NotImplementedError

    @classmethod
    def mask_integration_spec(cls) -> tuple[str, str, Any]:
        """Mask spec for the basic mask integration test."""
        return ("category", "equal", "X")

    @classmethod
    def mask_integration_expected(cls) -> list[Any]:
        """Expected column values for the basic masked feature."""
        raise NotImplementedError

    @classmethod
    def mask_integration_complex_spec(cls) -> list[tuple[str, str, Any]]:
        """Multi-condition mask spec for the complex mask test."""
        return [("category", "equal", "X"), ("value_int", "greater_equal", 10)]

    @classmethod
    def mask_integration_complex_expected(cls) -> list[Any]:
        """Expected column values for the complex masked feature."""
        raise NotImplementedError

    @classmethod
    def mask_integration_unmasked_expected(cls) -> list[Any]:
        """Expected column values for the unmasked feature (for the together test)."""
        raise NotImplementedError

    @classmethod
    def mask_integration_expected_row_count(cls) -> int:
        """Expected number of rows. Default: 12 (row-preserving)."""
        return 12

    @classmethod
    def mask_integration_use_approx(cls) -> bool:
        """Whether to use approximate comparison for floating-point results."""
        return False

    @classmethod
    def mask_integration_compare_sorted(cls) -> bool:
        """Whether to sort values before comparing (for reducing operations)."""
        return False

    # -- Assertion helpers -----------------------------------------------------

    def _assert_mask_integration_values(self, actual: list[Any], expected: list[Any]) -> None:
        """Compare actual vs expected, handling nulls, approx, and sorting."""
        if self.mask_integration_compare_sorted():
            actual = sorted(actual, key=lambda x: (x is None, x if x is not None else 0))
            expected = sorted(expected, key=lambda x: (x is None, x if x is not None else 0))

        assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            if _is_null(e):
                assert _is_null(a), f"row {i}: expected null, got {a}"
            elif self.mask_integration_use_approx() and isinstance(e, float):
                assert a == pytest.approx(e, rel=1e-3), f"row {i}: {a} != {e}"
            else:
                assert a == e, f"row {i}: {a} != {e}"

    def _run_masked_feature(
        self,
        name: str,
        options_ctx: dict[str, Any],
        mask_spec: Any,
    ) -> pa.Table:
        """Run a single masked feature through the pipeline."""
        ctx = dict(options_ctx)
        ctx["mask"] = mask_spec
        feature = Feature(name, options=Options(context=ctx))
        results = mloda.run_all(
            [feature],
            compute_frameworks={self.compute_framework_class()},  # type: ignore[attr-defined]
            plugin_collector=self._plugin_collector(),  # type: ignore[attr-defined]
        )
        assert len(results) >= 1

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and name in table.column_names:
                result_table = table
                break

        assert result_table is not None, f"No result table with {name} found"
        return result_table

    # -- Concrete test methods -------------------------------------------------

    def test_mask_feature_through_pipeline(self) -> None:
        """Run a masked feature through run_all and verify values."""
        result_table = self._run_masked_feature(
            self.mask_integration_feature_name(),
            self.mask_integration_options(),
            self.mask_integration_spec(),
        )
        assert result_table.num_rows == self.mask_integration_expected_row_count()

        result_col = result_table.column(self.mask_integration_feature_name()).to_pylist()
        self._assert_mask_integration_values(result_col, self.mask_integration_expected())

    def test_mask_complex_conditions_through_pipeline(self) -> None:
        """Run a multi-condition masked feature through run_all and verify values."""
        result_table = self._run_masked_feature(
            self.mask_integration_feature_name(),
            self.mask_integration_options(),
            self.mask_integration_complex_spec(),
        )
        assert result_table.num_rows == self.mask_integration_expected_row_count()

        result_col = result_table.column(self.mask_integration_feature_name()).to_pylist()
        self._assert_mask_integration_values(result_col, self.mask_integration_complex_expected())

    def test_masked_and_unmasked_consistent(self) -> None:
        """Run masked + unmasked features separately and verify both produce correct results.

        This exercises the complex feature composition path by confirming that
        the same feature produces different (correct) results with and without
        a mask through the full mloda pipeline. Features are run in separate
        pipeline calls because mloda resolves features of the same type to a
        shared feature group instance, which can cause mask config leakage.
        """
        name = self.mask_integration_feature_name()
        base_options = self.mask_integration_options()

        # Unmasked feature through pipeline
        result_unmasked = self._run_single_feature(name, base_options)  # type: ignore[attr-defined]
        assert result_unmasked.num_rows == self.mask_integration_expected_row_count()
        unmasked_col = result_unmasked.column(name).to_pylist()
        self._assert_mask_integration_values(unmasked_col, self.mask_integration_unmasked_expected())

        # Masked feature through pipeline (same name, different config)
        result_masked = self._run_masked_feature(name, base_options, self.mask_integration_spec())
        assert result_masked.num_rows == self.mask_integration_expected_row_count()
        masked_col = result_masked.column(name).to_pylist()
        self._assert_mask_integration_values(masked_col, self.mask_integration_expected())
