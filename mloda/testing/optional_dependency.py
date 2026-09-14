"""Contract for a community extender package whose runtime dependency is optional."""

from __future__ import annotations

import importlib
import sys

import pytest

from mloda.testing.import_isolation import block_root, evict_package


class OptionalDependencyPackageTestMixin:
    """Manifest resilience contract. Host declares package, root, distribution, extra, extender_name,
    extender_module: mloda's entry-point loader (``PluginLoader``, see
    ``tests/test_end2end/test_bundle_optional_dependencies.py`` for that contract) is now the sole
    guard tolerating a missing ``root``, so ``<package>.manifest`` itself no longer swallows anything:
    importing it directly always either succeeds or raises, whatever the reason. ``<package>``'s own
    ``__init__.py`` keeps a local guard (a real user doing ``import <package>`` outside of entry-point
    discovery still needs the friendly, deferred error), widened from ``ModuleNotFoundError`` to
    ``ImportError`` and chaining ``__cause__`` to the real underlying failure.

    ``broken_module``/``broken_name`` (default ``None``, tests skip when unset) name a real submodule of
    ``root`` and a real name imported from it by the extender module, for simulating an installed-but-
    too-old ``root`` whose failure is a plain ``ImportError`` (not a ``ModuleNotFoundError``): the widened
    ``__init__.py`` guard must catch that too, not just a fully-missing root.
    """

    package: str
    root: str
    distribution: str
    extra: str
    extender_name: str
    extender_module: str
    broken_module: str | None = None
    broken_name: str | None = None

    def test_manifest_lists_the_extender_when_installed(self) -> None:
        manifest = importlib.import_module(f"{self.package}.manifest")
        extender_module = importlib.import_module(f"{self.package}.{self.extender_module}")
        extender = getattr(extender_module, self.extender_name)

        assert manifest.EXTENDERS == [extender]

    def test_manifest_raises_when_dependency_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``<package>.manifest`` carries no try/except of its own any more: PluginLoader is the sole
        guard (declared via the ``mloda.optional_dependencies`` entry-point group), so a direct import
        of the manifest with ``root`` missing must raise, not degrade to an empty EXTENDERS list. This
        replaces the old ``test_manifest_is_empty_when_missing`` /
        ``test_manifest_reraises_unrelated_import_errors`` / ``test_manifest_logs_when_dependency_is_missing``,
        which pinned the now-deleted local guard; the skip-and-warn contract they used to cover for a
        missing/unusable ``root`` now lives at the bundle level against PluginLoader (see
        ``tests/test_end2end/test_bundle_optional_dependencies.py``), not here.
        """
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"{self.package}.manifest")

    def test_manifest_reraises_unrelated_import_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Still meaningful post-rewrite: proves the manifest has no special-casing left at all, so an
        import error unrelated to ``root`` propagates exactly like a missing ``root`` does above."""
        monkeypatch.setitem(sys.modules, f"{self.package}.{self.extender_module}", None)
        monkeypatch.delitem(sys.modules, f"{self.package}.manifest", raising=False)

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"{self.package}.manifest")

    def test_manifest_does_not_leave_a_partial_module_behind_after_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prior test's blocked import must not leave a broken half-imported manifest module cached
        under its name; a later, unblocked import must see a clean, fully-populated module again."""
        extender_module = importlib.import_module(f"{self.package}.{self.extender_module}")
        extender = getattr(extender_module, self.extender_name)
        manifest_name = f"{self.package}.manifest"

        # An earlier test in this class (test_manifest_lists_the_extender_when_installed) already
        # cached a real manifest module under this name; without evicting it here first, the nested
        # context's restore-on-exit below would reintroduce that stale reference instead of leaving
        # the name absent, making the assertion below order-dependent.
        monkeypatch.delitem(sys.modules, manifest_name, raising=False)

        with pytest.MonkeyPatch.context() as mp:
            block_root(mp, self.root)
            evict_package(mp, self.package)
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(manifest_name)

        assert manifest_name not in sys.modules, "a failed import must not leave a stale module cached"

        module = importlib.import_module(manifest_name)
        assert module.EXTENDERS == [extender]

    def test_package_imports_without_dependency_and_hides_the_extender(self, monkeypatch: pytest.MonkeyPatch) -> None:
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(self.package)

        assert self.extender_name not in vars(module)

        with pytest.raises(ImportError):
            getattr(module, self.extender_name)

    def test_missing_dependency_import_error_names_distribution_and_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        with pytest.raises(ImportError) as info:
            getattr(importlib.import_module(self.package), self.extender_name)

        assert self.distribution in str(info.value)
        assert self.extra in str(info.value)

        module = importlib.import_module(self.package)
        assert getattr(module, "some_other_name", None) is None

    def test_missing_dependency_import_error_chains_cause_to_the_original_import_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deferred ``__getattr__`` error must chain ``__cause__`` to the real underlying import
        failure (today it raises a bare, unchained ``ImportError`` with only a static message)."""
        block_root(monkeypatch, self.root)
        evict_package(monkeypatch, self.package)

        with pytest.raises(ImportError) as info:
            getattr(importlib.import_module(self.package), self.extender_name)

        cause = info.value.__cause__
        assert cause is not None, f"{self.extender_name} ImportError.__cause__ must chain the real import failure"
        assert isinstance(cause, ImportError)

    def test_package_hides_extender_and_chains_cause_when_dependency_is_broken(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``root`` installed but too old to expose ``broken_name`` inside ``broken_module``: this raises
        a plain ``ImportError`` (not a ``ModuleNotFoundError``), from the extender module's own
        ``from ... import ...`` statement. Today's ``__init__.py`` guard only catches
        ``ModuleNotFoundError``, so this plain ``ImportError`` is not caught at all and crashes the
        package import outright instead of degrading gracefully like a fully-missing root does.
        """
        if self.broken_module is None or self.broken_name is None:
            pytest.skip("no broken_module/broken_name declared")

        monkeypatch.delattr(importlib.import_module(self.broken_module), self.broken_name)
        evict_package(monkeypatch, self.package)

        module = importlib.import_module(self.package)

        assert self.extender_name not in vars(module)

        with pytest.raises(ImportError) as info:
            getattr(module, self.extender_name)

        assert self.distribution in str(info.value)
        assert self.extra in str(info.value)
        cause = info.value.__cause__
        assert isinstance(cause, ImportError)
        assert not isinstance(cause, ModuleNotFoundError), (
            "broken_name simulates an installed-but-too-old root: the real cause must be a plain "
            "ImportError, not a ModuleNotFoundError, or this test isn't exercising the widened except clause"
        )
        assert self.broken_name in str(cause)
