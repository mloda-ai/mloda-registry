"""Test-only helpers to poison and evict sys.modules entries, and to report what an import loads."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import pkgutil
import subprocess  # nosec
import sys
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from types import FrameType
from typing import Any

import pytest


def block_root(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Poison sys.modules so any import of ``root`` (or a cached submodule) raises ModuleNotFoundError."""
    names = {root} | {name for name in sys.modules if name == root or name.startswith(f"{root}.")}
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def _sibling_submodule_names(parent_name: str) -> set[str]:
    """Every submodule name findable under ``parent_name`` on disk, without importing it.

    Only names not yet in sys.modules are returned: already-loaded siblings are left alone,
    while names that could still be cold-imported during the test are pre-registered for cleanup.
    """
    if not parent_name:
        return set()

    spec = importlib.util.find_spec(parent_name)
    if spec is None or spec.submodule_search_locations is None:
        return set()

    names = set()
    for module_info in pkgutil.iter_modules(spec.submodule_search_locations):
        full_name = f"{parent_name}.{module_info.name}"
        if full_name not in sys.modules:
            names.add(full_name)
    return names


class _RootSnapshot(dict[str, None]):
    """Restore a module namespace when MonkeyPatch removes its teardown sentinel.

    Relies on ``MonkeyPatch.undo`` deleting a key that ``setitem`` added (``del d[key]``); the import
    isolation tests fail if that changes.
    """

    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root
        self.modules = {
            name: module for name, module in sys.modules.copy().items() if name == root or name.startswith(f"{root}.")
        }

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        for name in list(sys.modules):
            if name == self.root or name.startswith(f"{self.root}."):
                del sys.modules[name]
        sys.modules.update(self.modules)


def evict_root(monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    """Cold-evict every sys.modules entry at or under ``root``, forcing a genuine cold re-import
    (unlike ``block_root``, which poisons entries to raise ModuleNotFoundError instead).

    Used to simulate ``root`` being installed but missing one of its own transitive dependencies:
    the caller poisons that dependency's sys.modules entry separately, then this forces ``root`` to
    re-execute its own import machinery and hit that poison. Teardown restores pre-test module objects
    and removes modules first imported during the test, including nested and namespace packages. It does
    not touch the parent's attribute of a dotted ``root``; ``evict_package`` also detaches that.
    """
    snapshot = _RootSnapshot(root)
    # Queue restoration before later patches, so undo restores snapshots in reverse eviction order.
    monkeypatch.setitem(snapshot, root, None)
    for name in snapshot.modules:
        del sys.modules[name]


def _detach_from_parent(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    """Queue teardown removal of the ``dotted`` leaf attribute from its parent package, if loaded."""
    parent_name, _, leaf = dotted.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        # setattr then delattr queues two undo entries against the same (parent, leaf) pair, so
        # teardown overwrites whatever a cold import inside the test re-binds on the parent.
        monkeypatch.setattr(parent, leaf, None, raising=False)
        monkeypatch.delattr(parent, leaf, raising=False)


def evict_entry_points(monkeypatch: pytest.MonkeyPatch, group: str) -> None:
    """Pre-register teardown removal of every not-yet-loaded entry-point module in ``group``, plus its
    not-yet-loaded parent packages."""
    for entry_point in importlib.metadata.entry_points(group=group):
        name = entry_point.module
        if name in sys.modules:
            continue
        parts = name.split(".")
        unloaded = [n for n in (".".join(parts[: i + 1]) for i in range(len(parts))) if n not in sys.modules]
        for unloaded_name in unloaded:
            monkeypatch.setitem(sys.modules, unloaded_name, None)
            monkeypatch.delitem(sys.modules, unloaded_name, raising=False)
        _detach_from_parent(monkeypatch, unloaded[0])


def evict_package(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    """Cold-evict ``dotted`` (and its manifest) from sys.modules and detach it from its parent package.

    Also pre-registers removal for every sibling submodule the parent package could still cold-import,
    and every submodule ``dotted`` itself could cold-import, during the test, so none of them leak.
    """
    parent_name = dotted.rpartition(".")[0]
    _detach_from_parent(monkeypatch, dotted)

    evict_root(monkeypatch, dotted)

    candidates = (
        {dotted, f"{dotted}.manifest"} | _sibling_submodule_names(parent_name) | _sibling_submodule_names(dotted)
    )
    for name in candidates:
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name, raising=False)


def require_pyarrow() -> None:
    """Skip while pytest collects or runs a test; elsewhere raise ModuleNotFoundError naming the extra."""
    try:
        importlib.import_module("pyarrow")
    except ModuleNotFoundError as error:
        if error.name != "pyarrow" and not (error.name or "").startswith("pyarrow."):
            raise
        reason = "requires mloda-testing[pyarrow]"
        # pytest turns Skipped into a skip only in its collection and test phases (_pytest.runner frames);
        # in-process callers there skip too, so a probe that must see the error runs outside pytest.
        frame: FrameType | None = sys._getframe()
        while frame is not None:
            if frame.f_globals.get("__name__") == "_pytest.runner":
                pytest.skip(reason, allow_module_level=True)
            frame = frame.f_back
        raise ModuleNotFoundError(reason, name="pyarrow") from error
    else:
        return


_FRESH_IMPORT_SCRIPT = """
import importlib
import json
import sys

module, root, caller_path = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path[:] = caller_path
importlib.import_module(module)
mloda_core = "mloda.core" in sys.modules
modules = sorted(m for m in sys.modules if m == root or m.startswith(root + "."))
groups = []
if mloda_core:
    from mloda.provider import FeatureGroup, get_all_subclasses

    groups = sorted(
        f"{g.__module__}.{g.__qualname__}"
        for g in get_all_subclasses(FeatureGroup)
        if g.__module__ == root or g.__module__.startswith(root + ".")
    )
print("\\n" + json.dumps({"mloda_core": mloda_core, "modules": modules, "groups": groups}))
"""


def _run_probe(
    script: str, args: Sequence[str], env: Mapping[str, str] | None = None, *, timeout: float = 60
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run ``script`` in a fresh interpreter; return the result and its last stdout line parsed as JSON."""
    completed = subprocess.run(  # nosec B603
        [sys.executable, "-P", "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=None if env is None else dict(env),
    )
    if completed.returncode != 0:
        raise AssertionError(f"probe subprocess failed: {completed.stderr}")
    lines = completed.stdout.splitlines()
    try:
        summary: dict[str, Any] = json.loads(lines[-1])
    except (IndexError, ValueError) as error:
        raise AssertionError(
            f"no JSON summary on the last stdout line; stdout: {completed.stdout!r}, stderr: {completed.stderr!r}"
        ) from error
    return completed, summary


@dataclass(frozen=True)
class ImportReport:
    """What importing a module loaded: whether mloda core, the modules under a root, and the FeatureGroups there."""

    mloda_core: bool
    modules: tuple[str, ...]
    groups: tuple[str, ...]


def fresh_import_report(module: str, root: str, *, timeout: float = 60) -> ImportReport:
    """Import ``module`` in a fresh interpreter on the caller's sys.path and report what lands under ``root``."""
    path = json.dumps([p for p in sys.path if isinstance(p, str)])
    _, summary = _run_probe(_FRESH_IMPORT_SCRIPT, [module, root, path], timeout=timeout)
    return ImportReport(
        mloda_core=bool(summary["mloda_core"]), modules=tuple(summary["modules"]), groups=tuple(summary["groups"])
    )


_BASE_INSTALL_SCRIPT = """
import importlib
import json
import os
import pkgutil
import sys
import traceback

package = sys.argv[1]
allowed, exclude, caller_path = (json.loads(arg) for arg in sys.argv[2:5])
sys.path[:] = caller_path
entries = set(allowed) | {package} | set(sys.modules)


def covers(entry, name):
    return name == entry or name.startswith(entry + ".") or entry.startswith(name + ".")


def permitted(name):
    return name.split(".")[0] in sys.stdlib_module_names or any(covers(entry, name) for entry in entries)


class Guard:
    def __init__(self, finder):
        self._finder = finder

    def find_spec(self, name, *args, **kwargs):
        if not permitted(name):
            return None
        return self._finder.find_spec(name, *args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._finder, attr)


sys.meta_path[:] = [Guard(finder) for finder in sys.meta_path]
failures = {}


def skipped(name):
    return "tests" in name.split(".") or any(name == e or name.startswith(e + ".") for e in exclude)


def visit(name):
    if skipped(name):
        return
    try:
        module = importlib.import_module(name)
    except Exception:
        failures[name] = traceback.format_exc()
        return
    paths = getattr(module, "__path__", None)
    if paths is None:
        return
    children = set()
    for path in list(paths):
        for info in pkgutil.iter_modules([path]):
            children.add(info.name)
        if not os.path.isdir(path):
            continue
        for entry in os.listdir(path):
            if entry.isidentifier() and entry != "__pycache__" and os.path.isdir(os.path.join(path, entry)):
                children.add(entry)
    for child in sorted(children):
        visit(name + "." + child)


visit(package)
print("\\n" + json.dumps(failures))
"""


def assert_imports_with_base_deps(
    package: str, allowed: Set[str], *, exclude: Iterable[str] = (), timeout: float = 120
) -> None:
    """Import every module of ``package`` in a fresh interpreter that refuses non-stdlib names outside ``allowed``."""
    path = json.dumps([p for p in sys.path if isinstance(p, str)])
    args = [package, json.dumps(sorted(allowed)), json.dumps(sorted(exclude)), path]
    _, summary = _run_probe(_BASE_INSTALL_SCRIPT, args, timeout=timeout)
    if summary:
        details = "\n".join(f"{module}:\n{trace}" for module, trace in sorted(summary.items()))
        raise AssertionError(f"imports failed with only {sorted(allowed)} allowed:\n{details}")
