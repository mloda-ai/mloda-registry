"""Runtime catalog of the built-in data operations and their per-framework capability.

:data:`FAMILY_BASE_MODULES` is the registry: one base module per built-in family, in
documentation order, each declaring exactly one self-describing
:class:`DataOperationFamily`, so the catalog never keeps its own copy of a family's
vocabulary. :data:`FRAMEWORKS` is the matching registry of compute frameworks and
names the ``<prefix>_<op>.py`` backend modules to look for.

Capability comes from the mloda match-time machinery: a subtype is supported on a
framework iff the concrete backend class both matches a probe feature
(``match_feature_group_criteria``) and accepts the framework
(``supports_compute_framework``). Nothing framework-heavy is imported at module
import time; family and backend modules are imported lazily and skipped when their
optional dependency (or their whole pip package) is missing.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType, ModuleType
from typing import Any

from mloda.community.feature_groups.data_operations.errors import (
    unsupported_op_error,
    unsupported_subtype_error,
)
from mloda.community.feature_groups.data_operations.family import DataOperationFamily

_PKG = "mloda.community.feature_groups.data_operations"

#: The built-in family base modules, in documentation order. The generated
#: framework-support-matrix block follows this order.
FAMILY_BASE_MODULES: tuple[str, ...] = (
    f"{_PKG}.aggregation.base",
    f"{_PKG}.row_preserving.binning.base",
    f"{_PKG}.row_preserving.datetime.base",
    f"{_PKG}.row_preserving.frame_aggregate.base",
    f"{_PKG}.row_preserving.offset.base",
    f"{_PKG}.row_preserving.percentile.base",
    f"{_PKG}.row_preserving.rank.base",
    f"{_PKG}.row_preserving.scalar_aggregate.base",
    f"{_PKG}.row_preserving.scalar_arithmetic.base",
    f"{_PKG}.row_preserving.point_arithmetic.base",
    f"{_PKG}.row_preserving.time_bucketization.base",
    f"{_PKG}.row_preserving.ffill.base",
    f"{_PKG}.row_preserving.ema.base",
    f"{_PKG}.row_preserving.sessionization.base",
    f"{_PKG}.row_preserving.window_aggregation.base",
    f"{_PKG}.string.base",
    f"{_PKG}.row_changing.resample.base",
)


@dataclass(frozen=True)
class FrameworkInfo:
    """One compute framework: its backend-filename prefix, its compute-framework class name, and its doc label."""

    module_prefix: str
    catalog_key: str
    label: str


#: Every compute framework a family can ship a backend for, in documentation column order.
FRAMEWORKS: tuple[FrameworkInfo, ...] = (
    FrameworkInfo("pyarrow", "PyArrowTable", "PyArrow"),
    FrameworkInfo("pandas", "PandasDataFrame", "Pandas"),
    FrameworkInfo("polars_lazy", "PolarsLazyDataFrame", "Polars lazy"),
    FrameworkInfo("duckdb", "DuckDBFramework", "DuckDB"),
    FrameworkInfo("sqlite", "SqliteFramework", "SQLite"),
)


@dataclass(frozen=True)
class OperationInfo:
    """Describes one built-in data operation and its per-framework capability."""

    name: str
    prefix_pattern: str
    subtype_label: str
    subtypes: tuple[str, ...] | None
    frameworks: Mapping[str, frozenset[str] | None]


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def _import_optional(module_name: str) -> ModuleType | None:
    """Import *module_name*, returning None when it (or an optional dependency) is missing."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _module_local_subclasses(module: ModuleType, base_cls: type[Any]) -> list[type[Any]]:
    """Every subclass of *base_cls* defined in *module* itself."""
    return [
        obj
        for _name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and issubclass(obj, base_cls) and obj is not base_cls
    ]


def _module_local_subclass(module: ModuleType, base_cls: type[Any]) -> type[Any] | None:
    """Return the concrete subclass of *base_cls* defined in *module*, or None."""
    found = _module_local_subclasses(module, base_cls)
    return found[0] if found else None


def _family_class(base_module: ModuleType) -> type[DataOperationFamily]:
    """The one family class a registry module declares."""
    found = _module_local_subclasses(base_module, DataOperationFamily)
    if len(found) != 1:
        raise RuntimeError(
            f"{base_module.__name__} declares {len(found)} module-local DataOperationFamily subclasses "
            f"({sorted(cls.__name__ for cls in found)}); a FAMILY_BASE_MODULES entry must declare exactly one."
        )
    return found[0]


@lru_cache(maxsize=1)
def installed_family_classes() -> tuple[type[DataOperationFamily], ...]:
    """The family class of every registry module, in registry order; uninstalled modules are skipped."""
    classes: list[type[DataOperationFamily]] = []
    for base_module_name in FAMILY_BASE_MODULES:
        base_module = _import_optional(base_module_name)
        if base_module is None:
            continue
        classes.append(_family_class(base_module))
    return tuple(classes)


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


def _supported_subtypes(
    concrete: type[Any],
    framework: type[Any],
    family: type[DataOperationFamily],
    subtypes: tuple[str, ...],
) -> frozenset[str]:
    """Subtypes the concrete class both matches and accepts for *framework* at match time."""
    supported: set[str] = set()
    for subtype in subtypes:
        feature_name, options = family.catalog_probe(subtype)
        if not concrete.match_feature_group_criteria(feature_name, options):
            continue
        if concrete.supports_compute_framework(feature_name, options, framework) is not True:
            continue
        supported.add(subtype)
    return frozenset(supported)


def _build_operation(family: type[DataOperationFamily]) -> OperationInfo:
    """Build one OperationInfo from a family class and the backends installed beside it."""
    package = family.__module__.rsplit(".", 1)[0]
    op_dirname = package.rsplit(".", 1)[-1]
    subtypes = family.catalog_subtypes()

    frameworks: dict[str, frozenset[str] | None] = {}
    for framework_info in FRAMEWORKS:
        backend_module = _import_optional(f"{package}.{framework_info.module_prefix}_{op_dirname}")
        if backend_module is None:
            continue
        concrete = _module_local_subclass(backend_module, family)
        if concrete is None:
            continue
        for framework in concrete.compute_framework_definition():
            key = str(framework.__name__)
            frameworks[key] = None if subtypes is None else _supported_subtypes(concrete, framework, family, subtypes)

    return OperationInfo(
        name=family.FAMILY_NAME,
        prefix_pattern=family.PREFIX_PATTERN,
        subtype_label=family.SUBTYPE_LABEL,
        subtypes=subtypes,
        frameworks=MappingProxyType(frameworks),
    )


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[OperationInfo, ...]:
    """Build (once) and cache the catalog in registry order."""
    return tuple(_build_operation(family) for family in installed_family_classes())


def operations_in_declaration_order() -> tuple[OperationInfo, ...]:
    """Every installed built-in operation, in FAMILY_BASE_MODULES order."""
    return _load_catalog()


class DataOperationsCatalog:
    """Queryable catalog of the built-in data operations and their framework support."""

    @classmethod
    def list(cls) -> list[OperationInfo]:
        """Return every installed built-in operation, sorted by name."""
        return sorted(_load_catalog(), key=lambda info: info.name)

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
