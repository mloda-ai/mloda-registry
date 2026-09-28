"""Reusable input-validation test mixin for data-operations feature groups.

Provides ``test_mixin_input_validation``: runs a declared ``InputValidationCase``
(or skips / omits it) for each of three input-rejection kinds shared across every
data-operations op: ``multi_column_in_features``, ``missing_source_column``, and
``empty_partition_by``. Also provides ``test_mixin_empty_in_features``, derived from
the ``multi_column_in_features`` case, which checks rejection of a zero-length ``in_features``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
from unittest.mock import patch

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin

# The three input-rejection kinds every op test base must address (declare, opt
# out with a reason, or mark not-applicable with None).
_INPUT_VALIDATION_KINDS: tuple[str, ...] = (
    "multi_column_in_features",
    "missing_source_column",
    "empty_partition_by",
)


@dataclass(frozen=True)
class InputValidationCase:
    """One input-rejection case: the feature to run, its context, and the expected error match.

    ``table`` is ``None`` to reuse ``self.test_data``, or a ``pa.Table`` to build fresh test
    data from (e.g. to omit a source column).
    """

    feature_name: str
    context: dict[str, Any]
    match: str
    table: pa.Table | None = None


class InputValidationTestMixin(CaseParametrizationTestMixin):
    """Mixin providing a standardized input-validation rejection test.

    Feature-group test bases mix this in and override ``input_validation_cases()``
    to declare the three kinds. A missing or unknown kind fails at collection time.

    Requires the host class to provide (from DataOpsTestBase):
    - ``implementation_class()``
    - ``create_test_data(arrow_table)``
    - ``test_data``
    """

    _case_fixtures: ClassVar[dict[str, str]] = {"input_validation_case": "input_validation_case_ids"}

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, InputValidationCase | str | None]:
        """Kind -> case (run it), a reason string (opt out, skip with that reason), or None (not applicable).

        Keys must be exactly ``multi_column_in_features``, ``missing_source_column``, and
        ``empty_partition_by``. Required. ``multi_column_in_features`` must be config-based or a
        reason string, since it also drives ``test_mixin_empty_in_features``.
        """
        raise NotImplementedError

    @classmethod
    def input_validation_case_ids(cls) -> list[str]:
        """Validate the declared kinds and return the ids to parametrize (``None`` values excluded)."""
        cases = cls.input_validation_cases()
        declared = set(cases)
        expected = set(_INPUT_VALIDATION_KINDS)
        missing = sorted(expected - declared)
        unknown = sorted(declared - expected)
        if missing or unknown:
            raise TypeError(
                f"{cls.__name__}.input_validation_cases() must declare exactly {sorted(expected)}; "
                f"missing {missing}, unknown {unknown}"
            )
        for kind, case in cases.items():
            if case is None or isinstance(case, InputValidationCase):
                continue
            if isinstance(case, str) and case:
                continue
            raise TypeError(
                f"{cls.__name__}.input_validation_cases()[{kind!r}] must be an InputValidationCase, "
                f"a non-empty reason str, or None; got {case!r}"
            )
        return [kind for kind, case in cases.items() if case is not None]

    # -- Shared helper -------------------------------------------------------

    def _case_test_data(self, case: InputValidationCase) -> Any:
        """Return the shared test data, or fresh data built from ``case.table``."""
        return self.test_data if case.table is None else self.create_test_data(case.table)  # type: ignore[attr-defined]

    # -- Concrete test methods ------------------------------------------------

    def test_mixin_input_validation(self, input_validation_case: str) -> None:
        """A declared case must raise ValueError matching its pinned message; an opt-out skips with its reason."""
        case = self.input_validation_cases()[input_validation_case]
        if isinstance(case, str):
            pytest.skip(case)
        assert case is not None, f"{input_validation_case!r} should not be parametrized when its case is None"

        data = self._case_test_data(case)
        fs = make_feature_set(case.feature_name, **case.context)
        with pytest.raises(ValueError, match=case.match):
            self.implementation_class().calculate_feature(data, fs)  # type: ignore[attr-defined]

    def test_mixin_empty_in_features(self) -> None:
        """Derive a zero-length in_features rejection from the multi_column_in_features case."""
        case = self.input_validation_cases()["multi_column_in_features"]
        if case is None:
            pytest.skip("no multi_column_in_features case to derive an empty in_features case from")
        if isinstance(case, str):
            pytest.skip(case)

        data = self._case_test_data(case)
        fs = make_feature_set(case.feature_name, **{**case.context, "in_features": []})
        feature = next(iter(fs.features))
        # Empty both the raw option and the resolved value, since some ops read the raw option instead.
        with patch.object(feature.options, "get_in_features", return_value=frozenset()):
            with pytest.raises(ValueError, match="at least"):
                self.implementation_class().calculate_feature(data, fs)  # type: ignore[attr-defined]
