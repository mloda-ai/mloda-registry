"""Meta-tests guarding against silent test-method shadowing.

Two failure modes go unnoticed by pytest and silently drop coverage:

1. Two mixin classes define the same attribute (method) name. When both are
   mixed into the same concrete test class, one silently wins by MRO order
   and the other's assertions never run.
2. A concrete ``Test{Framework}{Op}`` class ends up with a ``test_*`` method
   defined on more than one ancestor in its MRO. The ancestor later in the
   MRO is shadowed, so any test it contributes is dropped without warning.

These checks are structural and cheap, so they run as part of the normal
test suite.
"""

from __future__ import annotations

import importlib
import pkgutil
from itertools import combinations
from types import ModuleType
from typing import Iterator

import mloda.community.feature_groups.data_operations as data_operations_pkg
import mloda.testing.feature_groups.data_operations.mixins as mixins_pkg


# Dunders that every class carries purely from being a class; they are not
# "methods the author added" and must be excluded from collision detection.
_CLASS_BOILERPLATE_NAMES: frozenset[str] = frozenset(
    {
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__annotations__",
        "__abstractmethods__",
        "_abc_impl",
        "__parameters__",
        "__orig_bases__",
    }
)


def _iter_submodules(package: ModuleType) -> Iterator[ModuleType]:
    """Yield every importable submodule under ``package`` (recursive)."""
    for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
        yield importlib.import_module(info.name)


def _own_attribute_names(cls: type) -> set[str]:
    """Return names defined directly on ``cls``, stripped of class boilerplate."""
    return set(vars(cls)) - _CLASS_BOILERPLATE_NAMES


def _own_test_method_names(cls: type) -> set[str]:
    """Return ``test_*`` callable names defined directly on ``cls``.

    Framework-adapter mixins (e.g. ``PandasTestMixin``) deliberately share
    non-test helper names (``create_test_data``, ``extract_column``, ...)
    because only one adapter is mixed in per concrete class. The silent-
    shadow hazard is exclusive to ``test_*`` methods, which pytest collects
    and which actually drop coverage when a later MRO entry wins.
    """
    return {name for name in _own_attribute_names(cls) if name.startswith("test_") and callable(vars(cls)[name])}


def _discover_mixin_classes() -> list[type]:
    """Find every ``*TestMixin`` class defined in the mixins package.

    Filters to classes whose ``__module__`` starts with the mixins package to
    avoid picking up re-exports from elsewhere.
    """
    mixin_prefix = mixins_pkg.__name__ + "."
    discovered: dict[str, type] = {}
    for module in _iter_submodules(mixins_pkg):
        for name, obj in vars(module).items():
            if not isinstance(obj, type):
                continue
            if not name.endswith("TestMixin"):
                continue
            if not obj.__module__.startswith(mixin_prefix):
                continue
            discovered[f"{obj.__module__}.{obj.__name__}"] = obj
    return sorted(discovered.values(), key=lambda c: (c.__module__, c.__name__))


def _iter_concrete_test_classes() -> Iterator[type]:
    """Yield concrete ``Test*`` classes defined in ``test_*`` modules.

    Restricted to classes whose ``__module__`` matches the module they were
    discovered in, so imported symbols (e.g. a base class imported into a
    test module) are not treated as concrete tests.
    """
    for module in _iter_submodules(data_operations_pkg):
        tail = module.__name__.rsplit(".", 1)[-1]
        if not tail.startswith("test_"):
            continue
        for name, obj in vars(module).items():
            if not isinstance(obj, type):
                continue
            if not name.startswith("Test"):
                continue
            if obj.__module__ != module.__name__:
                continue
            yield obj


class TestMixinIsolation:
    """Structural guards against silent mixin/MRO shadowing."""

    def test_mixin_classes_do_not_share_test_method_names(self) -> None:
        """Any two ``*TestMixin`` classes must define disjoint ``test_*`` methods.

        A shared test method means whichever mixin comes later in a concrete
        class's MRO silently wins, so the other mixin's assertions never run
        (coverage drops without any warning from pytest).
        """
        mixin_classes = _discover_mixin_classes()
        assert mixin_classes, "no *TestMixin classes discovered; discovery is broken"

        collisions: list[str] = []
        for left, right in combinations(mixin_classes, 2):
            shared = _own_test_method_names(left) & _own_test_method_names(right)
            if shared:
                collisions.append(
                    f"{left.__module__}.{left.__name__} <-> {right.__module__}.{right.__name__}: {sorted(shared)}"
                )
        assert not collisions, "mixin test-method collisions detected:\n" + "\n".join(collisions)

    def test_no_test_method_is_shadowed_in_concrete_class_mro(self) -> None:
        """No concrete ``Test*`` class may inherit the same ``test_*`` method twice.

        Walks the ancestors of each concrete class (excluding the class
        itself, since an explicit override in the concrete class is an
        intentional author decision, not a silent shadow). Any ``test_*``
        name that appears on two ancestors is a silent shadow: the lower-MRO
        definition is dropped and its assertions never run.
        """
        shadowed: list[str] = []
        concrete_classes = list(_iter_concrete_test_classes())
        assert concrete_classes, "no concrete Test* classes discovered; discovery is broken"

        for cls in concrete_classes:
            counts: dict[str, int] = {}
            # Skip cls itself: an override defined on the concrete class is
            # explicit, not a silent shadow. We care about hidden collisions
            # between ancestors (mixins, base classes).
            for ancestor in cls.__mro__[1:]:
                for name, value in vars(ancestor).items():
                    if not name.startswith("test_"):
                        continue
                    if not callable(value):
                        continue
                    counts[name] = counts.get(name, 0) + 1
            duplicated = sorted(name for name, count in counts.items() if count > 1)
            if duplicated:
                shadowed.append(f"{cls.__module__}.{cls.__name__}: {duplicated}")
        assert not shadowed, "test_* methods shadowed in concrete class MRO:\n" + "\n".join(shadowed)
