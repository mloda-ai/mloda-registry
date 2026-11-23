"""Reusable result-type test mixin: each op passes core's shape check, and its output type matches the group's declared type if any.

Provides ``test_mixin_result_type``: runs core's ``check_feature_group_shape`` per case on source data with
nulls and asserts the full shape check on every group; on a typed group also the declared output type
(exact on typed frameworks, value fits elsewhere).
"""

from __future__ import annotations

import inspect
import re
from typing import ClassVar

import pytest
from mloda._testing.feature_group_shape import _same
from mloda.provider import DivergenceKind, FeatureGroup, FeatureSet
from mloda.user import DataType, FeatureName, Options

from mloda.testing.feature_group_shape import check_feature_group_shape
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase, declares_nan_as_null
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin
from mloda.testing.framework_adapter import to_arrow_table
from mloda.testing.typed_data import TypedTestData

_ZERO_ROWS_RETYPE = re.compile(r"^zero_rows: column '(.*)' changes type on zero rows$")
_INPUT_CHANGED = re.compile(r"^outputs: input column '(.*)' was changed$")
_VALUE_ONLY = "checked by value only"


def _blocking(
    failures: list[str],
    notes: list[str],
    *,
    readback: bool,
    gap_covers: bool,
    adapter_retypes: frozenset[str] = frozenset(),
    adapter_rewrites: frozenset[str] = frozenset(),
) -> list[str]:
    """Core messages that fail the test: every failure, except a zero-row retype or a changed input the adapter makes."""
    bad = []
    for f in failures:
        m = _ZERO_ROWS_RETYPE.match(f)
        c = _INPUT_CHANGED.match(f)
        if not (m and m.group(1) in adapter_retypes) and not (c and c.group(1) in adapter_rewrites):
            bad.append(f)
    for note in notes:
        excused_exact = "declared gap divergence" in note and (
            ("is " in note and "expected" in note) or "lost its type" in note
        )
        if note.startswith("type:") and (excused_exact or "do not fit" in note):
            bad.append(note)
        elif _VALUE_ONLY in note and readback and not gap_covers:
            bad.append(note)
    return bad


class ResultTypeTestMixin(DataOpsTestBase, CaseParametrizationTestMixin):
    """Mixin asserting the full shape check; on a typed group also the output type against ``return_data_type_rule``."""

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
        if inspect.getattr_static(group, "return_data_type_rule") is not inspect.getattr_static(
            FeatureGroup, "return_data_type_rule"
        ):
            assert group.return_data_type_rule(feature) is not None, f"{name} has no return_data_type_rule"

        subtype = impl.canonical_subtype(group, name, feature.options)
        subtypes = frozenset({subtype}) if subtype else frozenset()
        if subtype is not None and subtype not in self.declared_supported(set()):
            pytest.skip(f"{impl.__name__} does not support {subtype!r}")

        data = self.result_type_data()
        inputs = [
            str(f.name)
            for f in group.declared_input_features(feature.options, FeatureName(name)) or ()
            if str(f.name) in data.types
        ]
        named: set[str] = set()
        for value in feature.options.context.values():
            named.update(str(v) for v in (value if isinstance(value, (list, tuple, set, frozenset)) else [value]))
        sources = [col for col in inputs if col not in named]
        has_null = [col for col in sources if None in data.to_python_dict()[col]]
        assert has_null or not sources, (
            f"{name}: none of the source columns {sources} holds a null in result_type_data()"
        )

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
        empty = TypedTestData({n: (t, []) for n, t in data.types.items()})
        arrow_types = None
        list_columns = [n for n, t in data.types.items() if t == DataType.LIST]
        if list_columns and self.adapter.capabilities.typed_columns:
            schema = to_arrow_table(data).schema
            arrow_types = {n: schema.field(n).type for n in list_columns}
        full_frame = self.adapter.from_typed(data, arrow_types)
        empty_frame = self.adapter.from_typed(empty, arrow_types)
        adapter_retypes = frozenset(
            c for c in data.types if self.adapter.column_type(full_frame, c) != self.adapter.column_type(empty_frame, c)
        )
        source_values = data.to_python_dict()
        adapter_rewrites = frozenset(
            c for c in data.types if not _same(self.adapter.column_values(full_frame, c), source_values[c])
        )
        bad = _blocking(
            list(result.failures),
            list(result.notes),
            readback=self.adapter.exact_type(probe, "p") is not None,
            gap_covers=gap_covers,
            adapter_retypes=adapter_retypes,
            adapter_rewrites=adapter_rewrites,
        )
        assert not bad, f"{name}: " + "; ".join(bad)
