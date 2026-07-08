"""Runtime catalog of the built-in data operations and their per-framework capability (issue #247).

The catalog enumerates every built-in data operation from its production base
class (``PREFIX_PATTERN`` and the subtype universe) and derives per-framework
capability from the mloda match-time machinery: a subtype is supported on a
framework iff the concrete backend class both matches a probe feature
(``match_feature_group_criteria``) and accepts the framework
(``supports_compute_framework``). Nothing framework-heavy is imported at module
import time; backend modules are imported lazily and skipped when their
optional dependency (or their whole pip package) is missing.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType, ModuleType
from typing import Any

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.errors import (
    unsupported_op_error,
    unsupported_subtype_error,
)

_PKG = "mloda.community.feature_groups.data_operations"

#: Backend module filename prefixes, one per potential compute framework.
_FRAMEWORK_MODULE_PREFIXES: tuple[str, ...] = ("pyarrow", "pandas", "polars_lazy", "duckdb", "sqlite")

#: Order hints only; membership always comes from the base-class constants.
_FRAME_TYPE_ORDER: tuple[str, ...] = ("rolling", "time", "cumulative", "expanding")
_TIME_UNIT_ORDER: tuple[str, ...] = ("second", "minute", "hour", "day", "week", "month", "year")


@dataclass(frozen=True)
class OperationInfo:
    """Describes one built-in data operation and its per-framework capability."""

    name: str
    prefix_pattern: str
    subtype_label: str
    subtypes: tuple[str, ...] | None
    frameworks: Mapping[str, frozenset[str] | None]


_SubtypesFn = Callable[[ModuleType], tuple[str, ...]]
_ProbeFn = Callable[[ModuleType, str], tuple[str, Options]]


@dataclass(frozen=True)
class _OperationSpec:
    """Where to find an operation's production classes and how to probe its subtypes."""

    name: str
    package: str
    base_class: str
    subtype_label: str
    subtypes: _SubtypesFn | None = None
    probe: _ProbeFn | None = None


def _ordered(values: Any, hint: tuple[str, ...]) -> tuple[str, ...]:
    """Sort *values* by their position in *hint* (unknown entries last, alphabetically)."""
    index = {name: pos for pos, name in enumerate(hint)}
    return tuple(sorted((str(value) for value in values), key=lambda name: (index.get(name, len(hint)), name)))


def _partition_options() -> Options:
    """Options carrying the partition_by context the group-by families require to match."""
    return Options(context={"partition_by": ["region"]})


def _partition_order_options() -> Options:
    """Options carrying partition_by and order_by for the ordered-partition families."""
    return Options(context={"partition_by": ["region"], "order_by": "ts"})


# ---------------------------------------------------------------------------
# Per-operation subtype universes (from production base constants only)
# ---------------------------------------------------------------------------


def _keys(mapping: Any) -> tuple[str, ...]:
    """Tuple of a constant's keys in insertion order."""
    return tuple(str(key) for key in mapping)


def _aggregation_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.AggregationFeatureGroup.AGGREGATION_TYPES)


def _binning_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.BINNING_OPS)


def _datetime_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.DATETIME_OPS)


def _frame_aggregate_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    """Flatten (frame_type, time_unit) into compound subtypes such as ``time:month``."""
    cls = base_module.FrameAggregateFeatureGroup
    time_units = _ordered(cls.SUPPORTED_TIME_UNITS, _TIME_UNIT_ORDER)
    out: list[str] = []
    for frame_type in _ordered(cls.SUPPORTED_FRAME_TYPES, _FRAME_TYPE_ORDER):
        if frame_type == "time":
            out.extend(f"time:{unit}" for unit in time_units)
        else:
            out.append(frame_type)
    return tuple(out)


def _offset_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    cls = base_module.OffsetFeatureGroup
    return _keys(cls.PARAMETRIC_OFFSET_FAMILIES) + _keys(cls.OFFSET_TYPES)


def _arithmetic_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.ARITHMETIC_OPERATIONS)


def _rank_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    cls = base_module.RankFeatureGroup
    return _keys(cls.RANK_TYPES) + _keys(cls.PARAMETRIC_RANK_FAMILIES)


def _scalar_aggregate_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.ScalarAggregateFeatureGroup.AGGREGATION_TYPES)


def _string_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.STRING_OPS)


def _time_bucketization_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.TIME_BUCKETIZATION_OPS)


def _window_aggregation_subtypes(base_module: ModuleType) -> tuple[str, ...]:
    return _keys(base_module.WindowAggregationFeatureGroup.AGGREGATION_TYPES)


# ---------------------------------------------------------------------------
# Per-operation probes (feature name + Options shapes that reach match time)
# ---------------------------------------------------------------------------


def _aggregation_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"value__{subtype}_agg", _partition_options()


def _binning_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"value__{subtype}_5", Options()


def _datetime_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"ts__{subtype}", Options()


def _frame_aggregate_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    """Probe rolling via the string pattern; time/cumulative/expanding via config Options."""
    if subtype == "rolling":
        return "value__sum_rolling_3", _partition_order_options()
    context: dict[str, Any] = {
        "aggregation_type": "sum",
        "in_features": "value",
        "partition_by": ["region"],
        "order_by": "ts",
    }
    if subtype.startswith("time:"):
        context.update({"frame_type": "time", "frame_size": 3, "frame_unit": subtype.split(":", 1)[1]})
    else:
        context["frame_type"] = subtype
    return "value_frame_probe", Options(context=context)


def _offset_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    """Parametric families probe with N=1; static offset types probe verbatim."""
    if subtype in base_module.OffsetFeatureGroup.PARAMETRIC_OFFSET_FAMILIES:
        subtype = f"{subtype}_1"
    return f"value__{subtype}_offset", _partition_order_options()


def _point_arithmetic_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"a&b__{subtype}_point", Options()


#: Representative N per parametric rank family used when probing capability.
_RANK_FAMILY_PROBE_N: dict[str, int] = {"ntile": 2, "top": 3, "bottom": 2}


def _rank_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    """Parametric families probe with a representative N; named rank types probe verbatim."""
    if subtype in base_module.RankFeatureGroup.PARAMETRIC_RANK_FAMILIES:
        subtype = f"{subtype}_{_RANK_FAMILY_PROBE_N[subtype]}"
    return f"value__{subtype}_ranked", _partition_order_options()


def _scalar_aggregate_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"value__{subtype}_scalar", Options()


def _scalar_arithmetic_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"value__{subtype}_constant", Options()


def _string_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"value__{subtype}", Options()


def _time_bucketization_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"ts__{subtype}_1_day", Options()


def _window_aggregation_probe(base_module: ModuleType, subtype: str) -> tuple[str, Options]:
    return f"value__{subtype}_window", _partition_order_options()


_OPERATION_SPECS: tuple[_OperationSpec, ...] = (
    _OperationSpec(
        name="aggregation",
        package=f"{_PKG}.aggregation",
        base_class="AggregationFeatureGroup",
        subtype_label="agg type",
        subtypes=_aggregation_subtypes,
        probe=_aggregation_probe,
    ),
    _OperationSpec(
        name="binning",
        package=f"{_PKG}.row_preserving.binning",
        base_class="BinningFeatureGroup",
        subtype_label="op",
        subtypes=_binning_subtypes,
        probe=_binning_probe,
    ),
    _OperationSpec(
        name="datetime",
        package=f"{_PKG}.row_preserving.datetime",
        base_class="DateTimeFeatureGroup",
        subtype_label="op",
        subtypes=_datetime_subtypes,
        probe=_datetime_probe,
    ),
    _OperationSpec(
        name="ema",
        package=f"{_PKG}.row_preserving.ema",
        base_class="EmaFeatureGroup",
        subtype_label="op",
    ),
    _OperationSpec(
        name="ffill",
        package=f"{_PKG}.row_preserving.ffill",
        base_class="FfillFeatureGroup",
        subtype_label="op",
    ),
    _OperationSpec(
        name="frame_aggregate",
        package=f"{_PKG}.row_preserving.frame_aggregate",
        base_class="FrameAggregateFeatureGroup",
        subtype_label="frame type",
        subtypes=_frame_aggregate_subtypes,
        probe=_frame_aggregate_probe,
    ),
    _OperationSpec(
        name="offset",
        package=f"{_PKG}.row_preserving.offset",
        base_class="OffsetFeatureGroup",
        subtype_label="offset type",
        subtypes=_offset_subtypes,
        probe=_offset_probe,
    ),
    _OperationSpec(
        name="percentile",
        package=f"{_PKG}.row_preserving.percentile",
        base_class="PercentileFeatureGroup",
        subtype_label="op",
    ),
    _OperationSpec(
        name="point_arithmetic",
        package=f"{_PKG}.row_preserving.point_arithmetic",
        base_class="PointArithmeticFeatureGroup",
        subtype_label="op",
        subtypes=_arithmetic_subtypes,
        probe=_point_arithmetic_probe,
    ),
    _OperationSpec(
        name="rank",
        package=f"{_PKG}.row_preserving.rank",
        base_class="RankFeatureGroup",
        subtype_label="rank type",
        subtypes=_rank_subtypes,
        probe=_rank_probe,
    ),
    _OperationSpec(
        name="resample",
        package=f"{_PKG}.row_changing.resample",
        base_class="ResampleFeatureGroup",
        subtype_label="op",
    ),
    _OperationSpec(
        name="scalar_aggregate",
        package=f"{_PKG}.row_preserving.scalar_aggregate",
        base_class="ScalarAggregateFeatureGroup",
        subtype_label="agg type",
        subtypes=_scalar_aggregate_subtypes,
        probe=_scalar_aggregate_probe,
    ),
    _OperationSpec(
        name="scalar_arithmetic",
        package=f"{_PKG}.row_preserving.scalar_arithmetic",
        base_class="ScalarArithmeticFeatureGroup",
        subtype_label="op",
        subtypes=_arithmetic_subtypes,
        probe=_scalar_arithmetic_probe,
    ),
    _OperationSpec(
        name="sessionization",
        package=f"{_PKG}.row_preserving.sessionization",
        base_class="SessionizationFeatureGroup",
        subtype_label="op",
    ),
    _OperationSpec(
        name="string",
        package=f"{_PKG}.string",
        base_class="StringFeatureGroup",
        subtype_label="op",
        subtypes=_string_subtypes,
        probe=_string_probe,
    ),
    _OperationSpec(
        name="time_bucketization",
        package=f"{_PKG}.row_preserving.time_bucketization",
        base_class="TimeBucketizationFeatureGroup",
        subtype_label="op",
        subtypes=_time_bucketization_subtypes,
        probe=_time_bucketization_probe,
    ),
    _OperationSpec(
        name="window_aggregation",
        package=f"{_PKG}.row_preserving.window_aggregation",
        base_class="WindowAggregationFeatureGroup",
        subtype_label="agg type",
        subtypes=_window_aggregation_subtypes,
        probe=_window_aggregation_probe,
    ),
)


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


def _import_optional(module_name: str) -> ModuleType | None:
    """Import *module_name*, returning None when it (or an optional dependency) is missing."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _module_local_subclass(module: ModuleType, base_cls: type[Any]) -> type[Any] | None:
    """Return the concrete subclass of *base_cls* defined in *module*, or None."""
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, base_cls) and obj is not base_cls:
            return obj
    return None


def _supported_subtypes(
    concrete: type[Any],
    framework: type[Any],
    base_module: ModuleType,
    subtypes: tuple[str, ...],
    probe: _ProbeFn,
) -> frozenset[str]:
    """Subtypes the concrete class both matches and accepts for *framework* at match time."""
    supported: set[str] = set()
    for subtype in subtypes:
        feature_name, options = probe(base_module, subtype)
        if not concrete.match_feature_group_criteria(feature_name, options):
            continue
        if concrete.supports_compute_framework(feature_name, options, framework) is not True:
            continue
        supported.add(subtype)
    return frozenset(supported)


def _build_operation(spec: _OperationSpec) -> OperationInfo | None:
    """Build one OperationInfo, or None when the operation's package is not installed."""
    base_module = _import_optional(f"{spec.package}.base")
    if base_module is None:
        return None
    base_cls: type[Any] = getattr(base_module, spec.base_class)
    subtypes = spec.subtypes(base_module) if spec.subtypes is not None else None

    frameworks: dict[str, frozenset[str] | None] = {}
    op_dirname = spec.package.rsplit(".", 1)[-1]
    for prefix in _FRAMEWORK_MODULE_PREFIXES:
        backend_module = _import_optional(f"{spec.package}.{prefix}_{op_dirname}")
        if backend_module is None:
            continue
        concrete = _module_local_subclass(backend_module, base_cls)
        if concrete is None:
            continue
        for framework in concrete.compute_framework_definition():
            key = str(framework.__name__)
            if subtypes is None or spec.probe is None:
                frameworks[key] = None
            else:
                frameworks[key] = _supported_subtypes(concrete, framework, base_module, subtypes, spec.probe)

    return OperationInfo(
        name=spec.name,
        prefix_pattern=str(base_cls.PREFIX_PATTERN),
        subtype_label=spec.subtype_label,
        subtypes=subtypes,
        frameworks=MappingProxyType(frameworks),
    )


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[OperationInfo, ...]:
    """Build (once) and cache the catalog, sorted by operation name."""
    infos = [info for spec in _OPERATION_SPECS if (info := _build_operation(spec)) is not None]
    return tuple(sorted(infos, key=lambda info: info.name))


class DataOperationsCatalog:
    """Queryable catalog of the built-in data operations and their framework support."""

    @classmethod
    def list(cls) -> list[OperationInfo]:
        """Return every installed built-in operation, sorted by name."""
        return list(_load_catalog())

    @classmethod
    def get(cls, name: str) -> OperationInfo:
        """Return the OperationInfo for *name*; unknown names raise a ValueError listing all operations."""
        catalog = _load_catalog()
        for info in catalog:
            if info.name == name:
                return info
        raise unsupported_op_error(name, (info.name for info in catalog))

    @classmethod
    def is_supported(cls, operation: str, subtype: str | None = None, framework: str | None = None) -> bool:
        """Whether *operation* (optionally narrowed to a subtype and/or framework) is supported.

        Framework names match case-insensitively; unknown or absent frameworks
        return False. ``subtype=None`` asks whether the operation exists on the
        framework at all; ``framework=None`` asks whether any framework supports
        the subtype. Unknown operations and subtypes raise ValueError.
        """
        info = cls.get(operation)
        if subtype is not None and (info.subtypes is None or subtype not in info.subtypes):
            raise unsupported_subtype_error(subtype, info.subtypes or (), operation=info.name)
        if framework is None:
            if subtype is None:
                return bool(info.frameworks)
            return any(supported is not None and subtype in supported for supported in info.frameworks.values())
        for key, supported in info.frameworks.items():
            if key.lower() != framework.lower():
                continue
            if subtype is None:
                return True
            return supported is not None and subtype in supported
        return False
