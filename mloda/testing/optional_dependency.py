"""Contract for a community extender package whose runtime dependency is optional."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest
from mloda.user import PluginLoader

from mloda.testing.import_isolation import block_root, evict_package


class OptionalDependencyPackageTestMixin:
    """Host declares package, root, extender_name, extender_module. The manifest never swallows a missing
    ``root`` (PluginLoader is the sole guard), and the package re-exports its extender lazily, so importing
    the package itself needs no dependency."""

    package: str
    root: str
    extender_name: str
    extender_module: str

    def test_manifest_lists_the_extender_when_installed(self) -> None:
        manifest = importlib.import_module(f"{self.package}.manifest")
        extender_module = importlib.import_module(f"{self.package}.{self.extender_module}")
        extender = getattr(extender_module, self.extender_name)

        assert manifest.EXTENDERS == [extender]

    def test_manifest_raises_when_dependency_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A direct manifest import with ``root`` missing raises; the skip-and-warn lives in PluginLoader."""
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"{self.package}.manifest")

    def test_package_imports_without_dependency_and_hides_the_extender(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(self.package)

        assert self.extender_name not in vars(module)

        with pytest.raises(ModuleNotFoundError):
            getattr(module, self.extender_name)

        assert getattr(module, "some_other_name", None) is None

    def test_package_import_does_not_import_the_extender_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The package re-exports lazily: only the first attribute access imports the extender module."""
        evict_package(monkeypatch, self.package)
        extender_module_name = f"{self.package}.{self.extender_module}"

        module = importlib.import_module(self.package)

        assert extender_module_name not in sys.modules

        extender = getattr(module, self.extender_name)

        assert extender_module_name in sys.modules
        assert extender is getattr(sys.modules[extender_module_name], self.extender_name)

    def test_star_import_without_dependency_succeeds_and_binds_no_extender(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A star import must not blow up just because the optional dependency is missing."""
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)
        namespace: dict[str, Any] = {}

        exec(f"from {self.package} import *", namespace)  # nosec

        assert self.extender_name not in namespace

    def test_star_import_all_lists_extender_when_dependency_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """__all__ must only promise the extender name when the dependency import can actually succeed."""
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(self.package)

        assert self.extender_name in module.__all__

    def test_plugin_loader_reraises_unrelated_import_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A poisoned extender submodule, unrelated to the optional dependency, is never swallowed."""
        evict_package(monkeypatch, self.package)
        monkeypatch.setitem(sys.modules, f"{self.package}.{self.extender_module}", None)

        with pytest.raises(ImportError) as excinfo:
            PluginLoader().load_entry_points(group="mloda.extenders")

        assert excinfo.value.name == f"{self.package}.{self.extender_module}"
