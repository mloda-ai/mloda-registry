"""Reusable single-value std/var test mixin for data-operations feature groups.

Provides ``test_mixin_single_value_std_var``: std/var of a group or window holding
exactly one value must be ``0.0`` (population, ddof=0), not null, on both the
reference and the framework under test. See
``docs/guides/data-operation-patterns/06-window-aggregation.md`` (std/var are
population, ddof=0).

The test method uses a ``test_mixin_`` prefix to match the existing convention
for shared mixin tests (see ``MaskTestMixin``, ``ReservedColumnsTestMixin``).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin

# grp A holds three values (Jan 1, 10, 11); grp B holds exactly one value (Jan 1),
# the single-value group each case's std/var must resolve to 0.0 for.
_SINGLE_VALUE_TABLE: pa.Table = pa.table(
    {
        "grp": ["A", "A", "A", "B"],
        "ts": [
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 10, tzinfo=timezone.utc),
            datetime(2023, 1, 11, tzinfo=timezone.utc),
            datetime(2023, 1, 1, tzinfo=timezone.utc),
        ],
        "val": [10, 20, 30, 40],
    }
)

# Population std/var of grp A's three values ([10, 20, 30], mean=20, ddof=0).
SINGLE_VALUE_STD = math.sqrt(200 / 3)
SINGLE_VALUE_VAR = 200 / 3


class SingleValueStdVarTestMixin(CaseParametrizationTestMixin):
    """Mixin providing a standardized single-value std/var test.

    Feature-group test bases mix this in and override the configuration methods
    below to adapt the generic test to their semantics. A missing override of
    ``single_value_cases()`` fails at collection time:
    ``CaseParametrizationTestMixin.pytest_generate_tests`` calls it at collection
    time to build the parametrize list.

    Requires the host class to provide (from DataOpsTestBase):
    - ``implementation_class()``
    - ``create_test_data(arrow_table)``
    - ``reference_implementation_class()``
    - ``extract_column(result, column_name)``
    - ``_skip_if_unsupported(op)`` (used by the default skip hook)
    """

    _case_fixtures: ClassVar[dict[str, str]] = {"single_value_case": "single_value_cases"}

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def single_value_cases(cls) -> dict[str, Any]:
        """Case id -> the expected value for that case. Required."""
        raise NotImplementedError

    @classmethod
    def single_value_feature_name(cls, case: str) -> str:
        """Feature name to run for the given case id. Required."""
        raise NotImplementedError

    @classmethod
    def single_value_agg_type(cls, case: str) -> str:
        """Agg type fed to the default skip hook. Default: the case id."""
        return case

    @classmethod
    def single_value_table(cls) -> pa.Table:
        """The fixture table to run cases against. Default: the shared grp/ts/val table."""
        return _SINGLE_VALUE_TABLE

    @classmethod
    def single_value_feature_set(cls, feature_name: str) -> FeatureSet:
        """FeatureSet to run for ``feature_name``. Default: partitioned by 'grp'."""
        return make_feature_set(feature_name, ["grp"])

    def single_value_skip_if_unsupported(self, case: str, agg_type: str, feature_name: str) -> None:
        """Skip the case if unsupported. Default: probe ``agg_type`` via ``_skip_if_unsupported``."""
        self._skip_if_unsupported(agg_type)  # type: ignore[attr-defined]

    def single_value_extract_values(
        self, result: Any, feature_name: str, column: Callable[[Any, str], list[Any]]
    ) -> Any:
        """Extract the comparable values from a result via ``column``. Default: a plain list column."""
        return column(result, feature_name)

    # -- Concrete test method ----------------------------------------------

    def test_mixin_single_value_std_var(self, single_value_case: str) -> None:
        """Reference and backend must resolve a single-value group's std/var to 0.0."""
        case = single_value_case
        agg_type = self.single_value_agg_type(case)
        feature_name = self.single_value_feature_name(case)
        self.single_value_skip_if_unsupported(case, agg_type, feature_name)

        table = self.single_value_table()
        expected = self.single_value_cases()[case]
        fs = self.single_value_feature_set(feature_name)

        ref = self.reference_implementation_class().calculate_feature(table, fs)  # type: ignore[attr-defined]
        ref_values = self.single_value_extract_values(ref, feature_name, _extract_column)
        assert ref_values == pytest.approx(expected, nan_ok=True), f"{case} reference: {ref_values!r}"

        result = self.implementation_class().calculate_feature(  # type: ignore[attr-defined]
            self.create_test_data(table),  # type: ignore[attr-defined]
            fs,
        )
        result_values = self.single_value_extract_values(result, feature_name, self.extract_column)  # type: ignore[attr-defined]
        assert result_values == pytest.approx(expected, nan_ok=True), f"{case} backend: {result_values!r}"
