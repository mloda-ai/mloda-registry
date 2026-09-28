"""Shared output-contract test mixin for data-operations feature groups.

Runs the result-type, row-count, and new-column checks that used to be hand-duplicated per base.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.provider import FeatureSet


class OutputContractTestMixin:
    """Mixin providing the standard result-type / row-count / new-column tests.

    Requires the host class to provide (from DataOpsTestBase):
    - ``implementation_class()``
    - ``test_data`` attribute (set in setup_method)
    - ``extract_column(result, column_name)``
    - ``get_row_count(result)``
    - ``get_expected_type()``
    """

    # -- Configuration methods (override per feature group) --------------------

    def output_contract_feature_set(self) -> FeatureSet:
        """Return the FeatureSet exercised by the output-contract tests.

        An instance method (unlike the classmethod hooks of the other
        mixins) because some bases build their feature set via instance
        helpers (e.g. ``self._ema_feature_set(2)``).
        """
        raise NotImplementedError

    def output_contract_expected_row_count(self) -> int | None:
        """Expected output row count. Default: same as the input table (row-preserving).

        ``None`` means the op has no fixed output row count (e.g. resample,
        whose bucket count depends on the data); the row-count test skips.
        """
        return int(self._arrow_table.num_rows)  # type: ignore[attr-defined]

    # -- Helper ------------------------------------------------------------

    def _output_contract_result(self) -> tuple[Any, str]:
        fs = self.output_contract_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)  # type: ignore[attr-defined]
        return result, str(fs.get_name_of_one_feature())

    # -- Concrete test methods --------------------------------------------------

    def test_mixin_output_contract_result_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        result, _ = self._output_contract_result()
        assert isinstance(result, self.get_expected_type())  # type: ignore[attr-defined]

    def test_mixin_output_contract_row_count(self) -> None:
        """Output row count matches the op's contract, when the op has a fixed one."""
        expected = self.output_contract_expected_row_count()
        if expected is None:
            pytest.skip("op has no fixed output row count")
        result, _ = self._output_contract_result()
        assert self.get_row_count(result) == expected  # type: ignore[attr-defined]

    def test_mixin_output_contract_new_column(self) -> None:
        """The result column should be present in the output, one value per output row."""
        result, name = self._output_contract_result()
        col = self.extract_column(result, name)  # type: ignore[attr-defined]
        assert len(col) == self.get_row_count(result)  # type: ignore[attr-defined]
        expected = self.output_contract_expected_row_count()
        if expected is not None:
            assert len(col) == expected
