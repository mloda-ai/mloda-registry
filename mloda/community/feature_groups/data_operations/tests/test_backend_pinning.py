"""Run all reference pinning tables through every supported shipped Implementation.

Reuse the twin suites' adapters and existing pinned runners, including declared arithmetic,
case-mapping and UTC expectations. Other rows compare every expected column strictly after
lossless representation conversions. Exact known limitations execute under strict xfail;
neither a generic divergence nor an unexpected exception can silently hide a new failure.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from mloda.provider import DivergenceKind, Implementation
from mloda.user import DataType, Options

from mloda.community.feature_groups.data_operations import DataOperationsCatalog
from mloda.community.feature_groups.data_operations.tests._backend_pinning_limitations import (
    LIMITATIONS,
    PinningLimitation,
)
from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import (
    FRAMEWORK_CATALOG_KEYS,
    FRAMEWORKS,
)
from mloda.community.feature_groups.data_operations.tests.test_reference_pinning import PINNED
from mloda.community.feature_groups.data_operations.tests.test_twin_catalog_consistency import (
    TWIN_SPECS,
    import_test_class,
)
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    calculate_with,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    _instant_and_offset,
)
from mloda.testing.typed_data import TypedTestData


@dataclass(frozen=True)
class BackendPinningCase:
    table: PinningTable
    row: PinningRow
    row_id: str
    framework: str
    host: type[DataOpsTestBase]
    implementation: type[Implementation[Any]]
    limitation: PinningLimitation | None


def _limitation_index() -> dict[tuple[str, str, str], PinningLimitation]:
    indexed: dict[tuple[str, str, str], PinningLimitation] = {}
    for limitation in LIMITATIONS:
        for framework in limitation.frameworks:
            for row_id in limitation.rows:
                key = (limitation.operation, framework, row_id)
                assert key not in indexed, f"duplicate pinning limitation: {key}"
                indexed[key] = limitation
    return indexed


def _limitation_mark(case: BackendPinningCase, subtype: str | None) -> pytest.MarkDecorator:
    limitation = case.limitation
    assert limitation is not None
    reason = limitation.reason
    if limitation.variant is not None:
        records = [
            record
            for record in case.implementation.divergences
            if record.kind is DivergenceKind.VARIANT
            and record.variant == limitation.variant
            and record.applies_to(subtype, None)
        ]
        assert records, f"{case.implementation.__name__}/{case.row_id}: missing {limitation.variant} declaration"
        reason += " Declared " + limitation.variant + ": " + "; ".join(record.reason for record in records)
    module, name = limitation.raises.rsplit(".", 1)
    exception = getattr(importlib.import_module(module), name)
    assert isinstance(exception, type) and issubclass(exception, Exception)
    return pytest.mark.xfail(reason=reason, strict=True, raises=exception)


def _backend_parameters() -> list[Any]:
    limitations = _limitation_index()
    available: set[tuple[str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    parameters: list[Any] = []
    catalog = {info.name: info for info in DataOperationsCatalog.list()}
    for group, _, table in PINNED:
        spec = TWIN_SPECS[table.operation]
        base = getattr(importlib.import_module(spec.base_module), spec.base_class)
        for framework, _ in FRAMEWORKS:
            if FRAMEWORK_CATALOG_KEYS[framework] not in catalog[table.operation].frameworks:
                continue
            twin = import_test_class(spec.tests_pkg, framework, base)
            assert twin is not None and issubclass(twin, DataOpsTestBase)
            host = twin
            implementation = host.implementation_class()
            assert implementation.feature_group is group
            available.add((table.operation, framework))
            for index, row in enumerate(table.rows):
                options = Options(context=dict(row.context))
                subtype = implementation.canonical_subtype(group, row.feature, options)
                if implementation.supported is not None and subtype not in implementation.supported:
                    continue
                if not implementation.accepts(row.feature, options):
                    continue
                row_id = f"{index}-{row.case}-{row.feature}"
                key = (table.operation, framework, row_id)
                seen.add(key)
                case = BackendPinningCase(table, row, row_id, framework, host, implementation, limitations.get(key))
                marks = [_limitation_mark(case, subtype)] if case.limitation is not None else []
                parameters.append(pytest.param(case, id="-".join(key), marks=marks))
    stale = [key for key in limitations if key[:2] in available and key not in seen]
    assert not stale, f"remove stale or unsupported pinning limitations: {stale}"
    assert {key[0] for key in seen} == {table.operation for _, _, table in PINNED}
    return parameters


@pytest.fixture
def backend(pinned_backend_case: BackendPinningCase) -> Iterator[DataOpsTestBase]:
    host = pinned_backend_case.host()
    host.setup_method()
    try:
        yield host
    finally:
        host.teardown_method()


def _lossless_int(value: Any) -> int:
    """Pandas nullable results and DuckDB HUGEINT/Decimal may represent an integer differently."""
    assert not isinstance(value, bool), "a boolean is not an integer result"
    converted = int(value)
    assert converted == value, f"integer conversion would change {value!r}"
    return converted


def _sqlite_boolean(value: Any) -> bool:
    """SQLite stores booleans as 0/1; reject every other integer instead of truthiness-casting it."""
    assert isinstance(value, (bool, int)) and value in (0, 1), f"invalid SQLite boolean {value!r}"
    return bool(value)


def _assert_columns(case: BackendPinningCase, backend: DataOpsTestBase, result: Any) -> None:
    for column, (data_type, values) in case.row.expected.items():
        actual = backend.extract_column(result, column)
        if data_type.name.startswith("TIMESTAMP"):
            assert_values_with_nulls(
                actual,
                [_instant_and_offset(value) for value in values],
                strict=True,
                parse_iso=True,
                cast=_instant_and_offset,
            )
        elif data_type in (DataType.INT8, DataType.INT32, DataType.INT64, DataType.UINT8, DataType.UINT64):
            assert_values_with_nulls(actual, list(values), strict=True, cast=_lossless_int)
        elif data_type is DataType.BOOLEAN and case.framework == "sqlite":
            assert_values_with_nulls(actual, list(values), strict=True, cast=_sqlite_boolean)
        else:
            assert_values_with_nulls(actual, list(values), strict=True)
    expected_columns = set(case.row.expected)
    if case.table.row_preserving:
        expected_columns |= set(case.row.inputs)
    assert set(backend.adapter.column_names(result)) == expected_columns


@pytest.mark.parametrize("pinned_backend_case", _backend_parameters())
def test_backend_matches_pinned_row(pinned_backend_case: BackendPinningCase, backend: DataOpsTestBase) -> None:
    case = pinned_backend_case
    runner = getattr(backend, "test_mixin_pinned_row", None) or getattr(backend, "test_pinned_row", None)
    if runner is not None and not (case.limitation and case.limitation.native):
        # Existing runners carry float32 input construction, declared value transforms, and
        # expected rejection checks. Keep those contracts instead of reimplementing them here.
        cases = getattr(backend, "pinned_row_cases")()
        key = next(key for key, row in cases.items() if row is case.row)
        runner(key)
        return
    result = calculate_with(
        case.implementation,
        backend.create_test_data(TypedTestData(case.row.inputs)),
        make_feature_set(case.row.feature, **case.row.context),
    )
    _assert_columns(case, backend, result)
