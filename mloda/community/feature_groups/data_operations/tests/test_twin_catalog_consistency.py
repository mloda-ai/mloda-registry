"""Consistency check: twin ``supported_*()`` sets must mirror DataOperationsCatalog.

The per-framework test classes (``{op}/tests/test_{framework}.py``) restrict which
subtypes their inherited reference tests run via ``supported_*()`` overrides. The
production capability is declared on the concrete classes and queried via
``DataOperationsCatalog``. These tests pin the twins to the catalog: a twin class
must exist exactly for the frameworks the catalog lists, and its supported set
must equal the catalog's frozenset (a twin without a ``supported_*()`` method
implicitly claims the full subtype universe). ``frame_aggregate`` is compared on
the flattened compound set (frame types plus ``time:<unit>``).
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

from mloda.community.feature_groups.data_operations import DataOperationsCatalog
from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import (
    FRAMEWORK_CATALOG_KEYS,
    FRAMEWORKS,
)

_COMMUNITY = "mloda.community.feature_groups.data_operations"
_TESTING = "mloda.testing.feature_groups.data_operations"

SUPPORT_METHODS: tuple[str, ...] = (
    "supported_agg_types",
    "supported_ops",
    "supported_frame_types",
    "supported_offset_types",
    "supported_rank_types",
)


@dataclass(frozen=True)
class TwinSpec:
    """Where to find an operation's twin test classes."""

    tests_pkg: str
    base_module: str
    base_class: str


TWIN_SPECS: dict[str, TwinSpec] = {
    "aggregation": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.aggregation.tests",
        base_module=f"{_TESTING}.aggregation.aggregation",
        base_class="AggregationTestBase",
    ),
    "binning": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.binning.tests",
        base_module=f"{_TESTING}.row_preserving.binning.binning",
        base_class="BinningTestBase",
    ),
    "datetime": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.datetime.tests",
        base_module=f"{_TESTING}.row_preserving.datetime.datetime",
        base_class="DateTimeTestBase",
    ),
    "ema": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.ema.tests",
        base_module=f"{_TESTING}.row_preserving.ema.ema",
        base_class="EmaTestBase",
    ),
    "ffill": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.ffill.tests",
        base_module=f"{_TESTING}.row_preserving.ffill.ffill",
        base_class="FfillTestBase",
    ),
    "frame_aggregate": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.frame_aggregate.tests",
        base_module=f"{_TESTING}.row_preserving.frame_aggregate.frame_aggregate",
        base_class="FrameAggregateTestBase",
    ),
    "offset": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.offset.tests",
        base_module=f"{_TESTING}.row_preserving.offset.offset",
        base_class="OffsetTestBase",
    ),
    "percentile": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.percentile.tests",
        base_module=f"{_TESTING}.row_preserving.percentile.percentile",
        base_class="PercentileTestBase",
    ),
    "point_arithmetic": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.point_arithmetic.tests",
        base_module=f"{_TESTING}.row_preserving.point_arithmetic.point_arithmetic",
        base_class="PointArithmeticTestBase",
    ),
    "rank": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.rank.tests",
        base_module=f"{_TESTING}.row_preserving.rank.rank",
        base_class="RankTestBase",
    ),
    "resample": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_changing.resample.tests",
        base_module=f"{_TESTING}.row_changing.resample.resample",
        base_class="ResampleTestBase",
    ),
    "scalar_aggregate": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.scalar_aggregate.tests",
        base_module=f"{_TESTING}.row_preserving.scalar_aggregate.scalar_aggregate",
        base_class="ScalarAggregateTestBase",
    ),
    "scalar_arithmetic": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.scalar_arithmetic.tests",
        base_module=f"{_TESTING}.row_preserving.scalar_arithmetic.scalar_arithmetic",
        base_class="ScalarArithmeticTestBase",
    ),
    "sessionization": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.sessionization.tests",
        base_module=f"{_TESTING}.row_preserving.sessionization.sessionization",
        base_class="SessionizationTestBase",
    ),
    "string": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.string.tests",
        base_module=f"{_TESTING}.string.string",
        base_class="StringTestBase",
    ),
    "time_bucketization": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.time_bucketization.tests",
        base_module=f"{_TESTING}.row_preserving.time_bucketization.time_bucketization",
        base_class="TimeBucketizationTestBase",
    ),
    "window_aggregation": TwinSpec(
        tests_pkg=f"{_COMMUNITY}.row_preserving.window_aggregation.tests",
        base_module=f"{_TESTING}.row_preserving.window_aggregation.window_aggregation",
        base_class="WindowAggregationTestBase",
    ),
}


def import_test_class(tests_pkg: str, framework: str, base_cls: type) -> type | None:
    """Import test_{framework}.py under *tests_pkg* and return the concrete test class.

    The concrete class is the one module-local subclass of *base_cls* (or ``None``
    if the module does not exist)."""
    mod_name = f"{tests_pkg}.test_{framework}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        # Only treat the test module (or a missing parent tests package) as
        # "framework absent". Broken imports *inside* an existing test module
        # name a different module in ``e.name`` and must surface loudly.
        if e.name and (e.name == mod_name or mod_name.startswith(e.name + ".")):
            return None
        raise
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != mod.__name__:
            continue
        if not issubclass(obj, base_cls):
            continue
        if obj is base_cls:
            continue
        return obj
    return None


def _twin_base_class(op_name: str) -> type:
    spec = TWIN_SPECS[op_name]
    base_cls: type = getattr(importlib.import_module(spec.base_module), spec.base_class)
    return base_cls


def _frame_aggregate_flat(cls: type) -> set[str]:
    """Flatten frame_aggregate's (frame_type, time_unit) axes into compound subtypes."""
    frame_types: set[str] = set(cls.supported_frame_types())  # type: ignore[attr-defined]
    time_units: set[str] = set(cls.supported_time_units())  # type: ignore[attr-defined]
    out: set[str] = set()
    for frame_type in frame_types:
        if frame_type == "time":
            out.update(f"time:{unit}" for unit in time_units)
        else:
            out.add(frame_type)
    return out


def twin_supported_set(op_name: str, cls: type, universe: tuple[str, ...]) -> set[str]:
    """The subtype set a twin class claims; no ``supported_*()`` method means the full universe."""
    if op_name == "frame_aggregate":
        return _frame_aggregate_flat(cls)
    for attr in SUPPORT_METHODS:
        method = getattr(cls, attr, None)
        if method is not None:
            return {str(item) for item in method()}
    return set(universe)


def test_twin_specs_cover_catalog() -> None:
    """TWIN_SPECS keys must be exactly the catalog operation names."""
    catalog_names = {info.name for info in DataOperationsCatalog.list()}
    assert set(TWIN_SPECS) == catalog_names, (
        f"TWIN_SPECS out of sync with DataOperationsCatalog: "
        f"missing={sorted(catalog_names - set(TWIN_SPECS))} extra={sorted(set(TWIN_SPECS) - catalog_names)}"
    )


def test_twin_classes_exist_exactly_for_catalog_frameworks() -> None:
    """Per operation, a twin test class exists iff the catalog lists the framework."""
    problems: list[str] = []
    for info in DataOperationsCatalog.list():
        spec = TWIN_SPECS[info.name]
        base_cls = _twin_base_class(info.name)
        present = {
            FRAMEWORK_CATALOG_KEYS[fw_key]
            for fw_key, _label in FRAMEWORKS
            if import_test_class(spec.tests_pkg, fw_key, base_cls) is not None
        }
        if present != set(info.frameworks):
            problems.append(
                f"{info.name}: twin classes for {sorted(present)} != catalog frameworks {sorted(info.frameworks)}"
            )
    assert problems == [], "twin presence diverges from DataOperationsCatalog:\n  " + "\n  ".join(problems)


def test_twin_supported_sets_equal_catalog() -> None:
    """Per operation and framework, the twin's supported set equals the catalog's frozenset."""
    problems: list[str] = []
    for info in DataOperationsCatalog.list():
        if info.subtypes is None:
            continue
        spec = TWIN_SPECS[info.name]
        base_cls = _twin_base_class(info.name)
        for fw_key, _label in FRAMEWORKS:
            supported = info.frameworks.get(FRAMEWORK_CATALOG_KEYS[fw_key])
            cls = import_test_class(spec.tests_pkg, fw_key, base_cls)
            if cls is None or supported is None:
                # Presence mismatches are covered by the twin-presence test.
                continue
            twin = twin_supported_set(info.name, cls, info.subtypes)
            if twin != set(supported):
                problems.append(
                    f"{info.name}/{FRAMEWORK_CATALOG_KEYS[fw_key]}: "
                    f"twin-only={sorted(twin - set(supported))} catalog-only={sorted(set(supported) - twin)}"
                )
    assert problems == [], "twin supported_*() sets diverge from DataOperationsCatalog:\n  " + "\n  ".join(problems)
