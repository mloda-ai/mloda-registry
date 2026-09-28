"""Reusable NaN-policy test mixin for data-operations feature groups.

Provides ``test_mixin_nan_policy``: runs ``DataOpsTestBase.nan_policy_table()``
through the reference and the framework under test, and asserts both match the
reference policy, or a framework's pinned divergence. See
``docs/guides/data-operation-patterns/03-reference-implementation.md`` and
``docs/guides/data-operation-patterns/known-divergences.md``.

The test method uses a ``test_mixin_`` prefix to match the existing convention
for shared mixin tests (see ``MaskTestMixin``, ``ReservedColumnsTestMixin``).
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

import pytest
from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin


class NanPolicyTestMixin(CaseParametrizationTestMixin):
    """Mixin providing a standardized NaN-policy test.

    Feature-group test bases mix this in and override the configuration methods
    below to adapt the generic test to their semantics.

    Requires the host class to provide (from DataOpsTestBase):
    - ``nan_policy_table()``
    - ``reference_implementation_class()``
    - ``implementation_class()``
    - ``create_test_data(arrow_table)``
    - ``extract_column(result, column_name)``
    - ``nan_divergent_agg_types()``
    - ``_skip_if_unsupported(op)`` (only used by the default skip hook)

    A concrete class that mixes this in without overriding ``nan_policy_cases()``
    fails at collection time (``CaseParametrizationTestMixin.pytest_generate_tests``
    calls it at collection time to build the parametrization), not at test-run time.
    """

    _case_fixtures: ClassVar[dict[str, str]] = {"nan_policy_case": "nan_policy_cases"}

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def nan_policy_cases(cls) -> dict[str, Any]:
        """Case id -> the policy's expected value for that case. Required."""
        raise NotImplementedError

    @classmethod
    def nan_policy_divergent_cases(cls) -> dict[str, Any]:
        """Case id -> pinned divergent value. Default: no divergences."""
        return {}

    @classmethod
    def nan_policy_feature_name(cls, case: str) -> str:
        """Feature name to run for the given case id. Required."""
        raise NotImplementedError

    @classmethod
    def nan_policy_agg_type(cls, case: str) -> str:
        """Agg type fed to ``nan_divergent_agg_types()`` and the skip hook. Default: the case id."""
        return case

    @classmethod
    def nan_policy_feature_set(cls, feature_name: str) -> FeatureSet:
        """FeatureSet to run for ``feature_name``. Default: partitioned by 'grp'."""
        return make_feature_set(feature_name, ["grp"])

    def nan_policy_skip_if_unsupported(self, case: str, agg_type: str, feature_name: str) -> None:
        """Skip the case if unsupported. Default: probe ``agg_type`` via ``_skip_if_unsupported``."""
        self._skip_if_unsupported(agg_type)  # type: ignore[attr-defined]

    def nan_policy_extract_values(self, result: Any, feature_name: str, column: Callable[[Any, str], list[Any]]) -> Any:
        """Extract the comparable values from a result via ``column``. Default: a plain list column."""
        return column(result, feature_name)

    # -- Concrete test method ----------------------------------------------

    def test_mixin_nan_policy(self, nan_policy_case: str) -> None:
        """Reference and backend must match the NaN policy, or a pinned divergence."""
        case = nan_policy_case
        agg_type = self.nan_policy_agg_type(case)
        feature_name = self.nan_policy_feature_name(case)
        self.nan_policy_skip_if_unsupported(case, agg_type, feature_name)

        table = self.nan_policy_table()  # type: ignore[attr-defined]
        fs = self.nan_policy_feature_set(feature_name)
        policy_expected = self.nan_policy_cases()[case]

        ref = self.reference_implementation_class().calculate_feature(table, fs)  # type: ignore[attr-defined]
        ref_values = self.nan_policy_extract_values(ref, feature_name, _extract_column)
        assert ref_values == pytest.approx(policy_expected, nan_ok=True), f"{case} reference: {ref_values!r}"

        divergent_cases = self.nan_policy_divergent_cases()
        if agg_type in self.nan_divergent_agg_types():  # type: ignore[attr-defined]
            assert any(self.nan_policy_agg_type(divergent_case) == agg_type for divergent_case in divergent_cases), (
                f"{type(self).__name__} pins {agg_type!r} via nan_divergent_agg_types(), but no case in "
                "nan_policy_divergent_cases() has that agg type (stale hook)"
            )
        expected = (
            divergent_cases[case]
            if case in divergent_cases and agg_type in self.nan_divergent_agg_types()  # type: ignore[attr-defined]
            else policy_expected
        )

        result = self.implementation_class().calculate_feature(  # type: ignore[attr-defined]
            self.create_test_data(table),  # type: ignore[attr-defined]
            fs,
        )
        result_values = self.nan_policy_extract_values(
            result,
            feature_name,
            self.extract_column,  # type: ignore[attr-defined]
        )
        assert result_values == pytest.approx(expected, nan_ok=True), f"{case} backend: {result_values!r}"
