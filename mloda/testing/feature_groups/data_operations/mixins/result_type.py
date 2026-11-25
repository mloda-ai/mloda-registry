"""Reusable result-type test mixin: each op passes core's shape check and the group's declared output type.

Provides ``test_mixin_result_type``: runs core's ``check_feature_group_shape`` per case on source data with
nulls and asserts the full shape check on every group; on a typed group also the declared output type
(exact on typed frameworks, value fits elsewhere). Ops a typed group leaves untyped are marked in
``untyped_result_type_ids()`` and still run the shape check.
"""

from __future__ import annotations

import inspect
import re
from typing import ClassVar, NamedTuple

import pytest
from mloda.provider import DivergenceKind, FeatureGroup, FeatureSet
from mloda.user import DataType, Feature, FeatureName, Options

from mloda.testing.feature_group_shape import check_feature_group_shape
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase, declares_nan_as_null
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, same_values
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin
from mloda.testing.framework_adapter import to_arrow_table
from mloda.testing.typed_data import TypedTestData

_ZERO_ROWS_RETYPE = re.compile(r"^zero_rows: column '(.*)' changes type on zero rows$")
_INPUT_CHANGED = re.compile(r"^outputs: input column '(.*)' was changed$")
_VALUE_ONLY = "checked by value only"


class KnownFinding(NamedTuple):
    """A known failure of one case: ``match`` is a substring every blocking message contains, ``reason`` links the issue."""

    match: str
    reason: str


def _blocking(
    failures: list[str],
    notes: list[str],
    *,
    readback: bool,
    gap_covers: bool,
    adapter_retypes: frozenset[str] = frozenset(),
    adapter_rewrites: frozenset[str] = frozenset(),
) -> list[str]:
    """Core messages that fail the test: every failure except a zero-row retype or changed input the adapter makes."""
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
        """Case id -> one-feature FeatureSet: one per typed op, or per op if untyped.

        Ops a typed group leaves untyped are also listed in ``untyped_result_type_ids()``.
        """
        raise NotImplementedError("Subclass must override result_type_cases")

    @classmethod
    def untyped_result_type_ids(cls) -> frozenset[str]:
        """Case ids whose op declares no output type on a typed group. Default: none."""
        return frozenset()

    @classmethod
    def result_type_known_findings(cls) -> dict[str, KnownFinding]:
        """Case id -> ``KnownFinding``; the case xfails while it fails only with the named failure.

        Any blocking message that does not contain ``match`` fails the test, and so does a case that passes.
        """
        return {}

    @classmethod
    def result_type_data(cls) -> TypedTestData:
        """Source data the cases run on; every input column a case reads must hold a null. Default: source_data()."""
        return cls.source_data()

    def test_mixin_result_type(self, result_type_case: str) -> None:
        """The shape check passes; a typed output has its declared type unless a ``type`` record excuses it."""
        cases = self.result_type_cases()
        assert cases, f"{type(self).__name__}.result_type_cases() is empty"
        marked = self.untyped_result_type_ids()
        unknown_marked = sorted(marked - cases.keys())
        assert not unknown_marked, f"untyped_result_type_ids() names ids outside result_type_cases(): {unknown_marked}"
        findings = self.result_type_known_findings()
        unknown_findings = sorted(findings.keys() - cases.keys())
        assert not unknown_findings, (
            f"result_type_known_findings() names ids outside result_type_cases(): {unknown_findings}"
        )
        feature = next(iter(cases[result_type_case].features))
        name = str(feature.name)
        group = self.feature_group_class()
        impl = self.implementation_class()
        typed_group = inspect.getattr_static(group, "return_data_type_rule") is not inspect.getattr_static(
            FeatureGroup, "return_data_type_rule"
        )
        assert typed_group or not marked, (
            "untyped_result_type_ids() is non-empty but the group declares no return_data_type_rule"
        )
        if typed_group:
            rule = group.return_data_type_rule(feature)
            if result_type_case in marked:
                assert rule is None, (
                    f"{result_type_case} ({name}) declares {rule}: move it out of untyped_result_type_ids()"
                )
            else:
                assert rule is not None, f"{name} has no return_data_type_rule"
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
        candidates = [
            c for c in data.types if not same_values(self.adapter.column_values(full_frame, c), source_values[c])
        ]
        adapter_rewrites: frozenset[str] = frozenset()
        if candidates:
            features = FeatureSet()
            features.add(Feature(name, options=options))
            ran = calculate_with(impl, self.adapter.from_typed(data, arrow_types), features, group=group)
            adapter_rewrites = frozenset(
                c
                for c in candidates
                if same_values(self.adapter.column_values(ran, c), self.adapter.column_values(full_frame, c))
            )
        bad = _blocking(
            list(result.failures),
            list(result.notes),
            readback=self.adapter.exact_type(probe, "p") is not None,
            gap_covers=gap_covers,
            adapter_retypes=adapter_retypes,
            adapter_rewrites=adapter_rewrites,
        )
        if result_type_case in findings:
            finding = findings[result_type_case]
            if bad and all(finding.match in b for b in bad):
                pytest.xfail(finding.reason)
            if bad:
                raise AssertionError(f"{name}: known finding expects only {finding.match!r}, got: " + "; ".join(bad))
            raise AssertionError(
                f"{result_type_case} ({name}) is a known finding but no longer reproduces: "
                "remove it from result_type_known_findings()"
            )
        assert not bad, f"{name}: " + "; ".join(bad)
