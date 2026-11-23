"""Reusable result-type test mixin: each typed op's output column has the type its group declares.

Provides ``test_mixin_result_type``: runs core's ``check_feature_group_shape`` per case on source data with
nulls and asserts only the type outcome (exact on typed frameworks, value fits elsewhere).
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from mloda.provider import DivergenceKind, FeatureSet
from mloda.user import DataType, FeatureName, Options

from mloda.testing.feature_group_shape import check_feature_group_shape
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase, declares_nan_as_null
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin
from mloda.testing.typed_data import TypedTestData

_FAILURE_PREFIXES = ("type:", "run:", "outputs: run raised", "outputs: declared input", "outputs: requested name")
_MISSING = "is missing from the result"
_VALUE_ONLY = "checked by value only"


def _blocking(failures: list[str], notes: list[str], *, readback: bool, gap_covers: bool) -> list[str]:
    """Core messages that fail the test; all matching on core's wording lives here."""
    bad = [f for f in failures if f.startswith(_FAILURE_PREFIXES) or _MISSING in f]
    for note in notes:
        excused_exact = "declared gap divergence" in note and (
            ("is " in note and "expected" in note) or "lost its type" in note
        )
        if note.startswith("type:") and excused_exact:
            bad.append(note)
        elif _VALUE_ONLY in note and readback and not gap_covers:
            bad.append(note)
    return bad


class ResultTypeTestMixin(DataOpsTestBase, CaseParametrizationTestMixin):
    """Mixin checking each case's output type against the group's ``return_data_type_rule``."""

    _case_fixtures: ClassVar[dict[str, str]] = {"result_type_case": "result_type_cases"}

    @classmethod
    def result_type_cases(cls) -> dict[str, FeatureSet]:
        """Case id -> one-feature FeatureSet of a typed op. Required."""
        raise NotImplementedError("Subclass must override result_type_cases")

    @classmethod
    def result_type_data(cls) -> TypedTestData:
        """Source data the cases run on; every input column a case reads must hold a null. Default: source_data()."""
        return cls.source_data()

    def test_mixin_result_type(self, result_type_case: str) -> None:
        """The case's output column has the declared type, or a ``type`` record excuses it."""
        cases = self.result_type_cases()
        assert cases, f"{type(self).__name__}.result_type_cases() is empty"
        feature = next(iter(cases[result_type_case].features))
        name = str(feature.name)
        group = self.feature_group_class()
        impl = self.implementation_class()
        assert group.return_data_type_rule(feature) is not None, f"{name} has no return_data_type_rule"

        subtype = impl.canonical_subtype(group, name, feature.options)
        subtypes = frozenset({subtype}) if subtype else frozenset()
        if subtype is not None and subtype not in self.declared_supported(set()):
            pytest.skip(f"{impl.__name__} does not support {subtype!r}")

        data = self.result_type_data()
        null_free = [
            str(f.name)
            for f in group.declared_input_features(feature.options, FeatureName(name)) or ()
            if str(f.name) in data.types and None not in data.to_python_dict()[str(f.name)]
        ]
        assert not null_free, f"{name}: input column(s) {null_free} hold no null in result_type_data()"

        group_options = dict(feature.options.group)
        if declares_nan_as_null(impl, subtypes):
            group_options["allow_variants"] = "nan_as_null"
        options = Options(group=group_options, context=dict(feature.options.context))
        result = check_feature_group_shape(
            group, options, input_data=data, adapter=self.adapter, requests=[name], require_run=True
        )

        probe = self.adapter.from_typed(TypedTestData({"p": (DataType.INT64, [1])}))
        records = [*group.divergence_records(), *impl.divergence_records()]
        gap_covers = any(r.kind is DivergenceKind.GAP and r.applies_to(subtype, frozenset()) for r in records)
        bad = _blocking(
            list(result.failures),
            list(result.notes),
            readback=self.adapter.exact_type(probe, "p") is not None,
            gap_covers=gap_covers,
        )
        assert not bad, f"{name}: " + "; ".join(bad)
