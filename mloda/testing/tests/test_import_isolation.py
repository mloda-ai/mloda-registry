"""Direct unit tests for mloda.testing.import_isolation."""

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess  # nosec
import sys
import uuid
from pathlib import Path

import pytest

from mloda.testing.import_isolation import (
    ImportReport,
    evict_entry_points,
    evict_package,
    evict_root,
    fresh_import_report,
)


def _write_throwaway_package(base: Path, name: str) -> None:
    package_dir = base / name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    (package_dir / "leaf.py").write_text("MARKER = 'loaded'\n")


def test_cold_imported_leaf_and_parent_attribute_are_gone_after_context_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_throwaway_package(tmp_path, package_name)
    monkeypatch.syspath_prepend(str(tmp_path))

    parent = importlib.import_module(package_name)
    assert not hasattr(parent, "leaf")

    leaf_name = f"{package_name}.leaf"
    with pytest.MonkeyPatch.context() as mp:
        evict_package(mp, leaf_name)
        leaf = importlib.import_module(leaf_name)

        assert leaf.MARKER == "loaded"
        assert leaf_name in sys.modules
        assert getattr(parent, "leaf", None) is leaf

    assert leaf_name not in sys.modules
    assert not hasattr(parent, "leaf")


def test_cold_imported_sibling_submodule_is_gone_after_context_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling of the evicted target (not `dotted` itself, not its `.manifest`) cold-imported
    inside the same context must not leak into sys.modules once the context exits."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_throwaway_package(tmp_path, package_name)
    (tmp_path / package_name / "extra.py").write_text("MARKER = 'extra'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    leaf_name = f"{package_name}.leaf"
    extra_name = f"{package_name}.extra"
    with pytest.MonkeyPatch.context() as mp:
        evict_package(mp, leaf_name)
        importlib.import_module(leaf_name)
        extra = importlib.import_module(extra_name)

        assert extra.MARKER == "extra"
        assert extra_name in sys.modules

    assert extra_name not in sys.modules


def test_cold_imported_submodule_under_the_evicted_package_is_gone_after_context_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `dotted` is itself a package, a submodule under it cold-imported inside the same
    context must not leak into sys.modules once the context exits."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    sub_dir = package_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inner.py").write_text("MARKER = 'inner'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    sub_name = f"{package_name}.sub"
    inner_name = f"{sub_name}.inner"
    with pytest.MonkeyPatch.context() as mp:
        evict_package(mp, sub_name)
        inner = importlib.import_module(inner_name)

        assert inner.MARKER == "inner"
        assert inner_name in sys.modules

    assert inner_name not in sys.modules


def test_evict_root_forces_a_genuine_cold_reimport_and_restores_the_original_module_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`evict_root` (unlike `block_root`, which poisons sys.modules to raise) removes the module so the
    next import re-executes it from scratch, then restores the original module object on teardown."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_throwaway_package(tmp_path, package_name)
    monkeypatch.syspath_prepend(str(tmp_path))

    original = importlib.import_module(package_name)

    with pytest.MonkeyPatch.context() as mp:
        evict_root(mp, package_name)
        assert package_name not in sys.modules

        reimported = importlib.import_module(package_name)
        assert reimported is not original

    assert sys.modules[package_name] is original


@pytest.mark.parametrize("preloaded", [True, False], ids=["warm-root", "cold-root"])
def test_evict_root_removes_first_imports_at_every_depth_and_restores_preloaded_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preloaded: bool
) -> None:
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    sibling_name = f"{package_name}_other"
    for name in (package_name, sibling_name):
        _write_throwaway_package(tmp_path, name)
    for relative in ("branch", "branch/nested"):
        directory = tmp_path / package_name / relative
        directory.mkdir()
        (directory / "__init__.py").write_text("")
    (tmp_path / package_name / "branch" / "nested" / "deep.py").write_text("MARKER = 'deep'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    root_names = [package_name, f"{package_name}.leaf"]
    cold_names = [f"{package_name}.{suffix}" for suffix in ("branch", "branch.nested", "branch.nested.deep")]
    for name in (*root_names, *cold_names, sibling_name, f"{sibling_name}.leaf"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)

    originals = {name: importlib.import_module(name) for name in root_names} if preloaded else {}
    sibling = importlib.import_module(sibling_name)
    with pytest.MonkeyPatch.context() as mp:
        evict_root(mp, package_name)
        assert not any(name in sys.modules for name in root_names)
        for name in root_names:
            assert importlib.import_module(name) is not originals.get(name)
        deep = importlib.import_module(cold_names[-1])
        sibling_leaf = importlib.import_module(f"{sibling_name}.leaf")
        assert deep.MARKER == "deep"

    assert sys.modules[sibling_name] is sibling
    assert sys.modules[f"{sibling_name}.leaf"] is sibling_leaf
    for name in root_names:
        if preloaded:
            assert sys.modules[name] is originals[name]
        else:
            assert name not in sys.modules
    assert not any(name in sys.modules for name in cold_names)


def test_repeated_root_eviction_restores_a_namespace_package_without_new_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    namespace_dir = tmp_path / package_name / "branch"
    namespace_dir.mkdir(parents=True)
    (namespace_dir / "leaf.py").write_text("MARKER = 'namespace leaf'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    names = [package_name, f"{package_name}.branch", f"{package_name}.branch.leaf"]
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    original = importlib.import_module(package_name)

    with pytest.MonkeyPatch.context() as mp:
        for _ in range(2):
            evict_root(mp, package_name)
            assert not any(name in sys.modules for name in names)
            leaf = importlib.import_module(names[-1])
            assert leaf.MARKER == "namespace leaf"
            assert sys.modules[package_name] is not original

    assert sys.modules[package_name] is original
    assert not any(name in sys.modules for name in names[1:])
    assert not hasattr(original, "branch")


@pytest.mark.parametrize("cold_parent_preloaded", [True, False], ids=["cold-parent-preloaded", "cold-parent-cold"])
def test_evict_entry_points_removes_cold_loaded_entry_point_modules_and_keeps_already_loaded_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cold_parent_preloaded: bool
) -> None:
    """Cold-loaded entry point modules and their cold parents are evicted on teardown; loaded ones are kept."""
    suffix = uuid.uuid4().hex
    group = f"ii_throwaway_group_{suffix}"
    cold_name = f"ii_throwaway_cold_{suffix}"
    warm_name = f"ii_throwaway_warm_{suffix}"
    for name in (cold_name, warm_name):
        _write_throwaway_package(tmp_path, name)
    dist_info = tmp_path / f"ii_throwaway_dist_{suffix}-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: ii-throwaway-dist-{suffix}\nVersion: 0.0.0\n")
    (dist_info / "entry_points.txt").write_text(f"[{group}]\ncold = {cold_name}.leaf\nwarm = {warm_name}.leaf\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    cold_leaf = f"{cold_name}.leaf"
    warm_leaf = f"{warm_name}.leaf"
    warm_module = importlib.import_module(warm_leaf)
    assert cold_leaf not in sys.modules
    parent = importlib.import_module(cold_name) if cold_parent_preloaded else None

    with pytest.MonkeyPatch.context() as mp:
        evict_entry_points(mp, group)
        for entry_point in importlib.metadata.entry_points(group=group):
            entry_point.load()

        assert cold_leaf in sys.modules
        assert sys.modules[warm_leaf] is warm_module

    assert cold_leaf not in sys.modules
    assert sys.modules[warm_leaf] is warm_module
    if cold_parent_preloaded:
        assert not hasattr(parent, "leaf")
    else:
        assert cold_name not in sys.modules


def test_fresh_import_report_lists_root_modules_without_mloda_or_the_twin_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only modules under the root are reported; a sibling `<root>_twin` package is outside it."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    twin_name = f"{package_name}_twin"
    _write_throwaway_package(tmp_path, package_name)
    _write_throwaway_package(tmp_path, twin_name)
    (tmp_path / package_name / "user.py").write_text(
        f"import {twin_name}.leaf\nimport {package_name}.leaf\nprint('loaded', end='')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "path", [*sys.path, tmp_path / "not_a_str_entry"])
    # In-process state does not reach the report.
    importlib.import_module(f"{package_name}.leaf")

    report = fresh_import_report(f"{package_name}.user", package_name)

    assert report == ImportReport(
        mloda_core=False,
        modules=(package_name, f"{package_name}.leaf", f"{package_name}.user"),
        groups=(),
    )


def test_fresh_import_report_lists_exactly_the_feature_groups_defined_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Groups under root are listed (abstract base included); a twin's group is not."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    twin_name = f"{package_name}_twin"
    for name in (package_name, twin_name):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("")
    (tmp_path / twin_name / "groups.py").write_text(
        "from mloda.provider import FeatureGroup\n\n\nclass TwinGroup(FeatureGroup):\n    pass\n"
    )
    (tmp_path / package_name / "groups.py").write_text(
        f"import {twin_name}.groups\n"
        "from mloda.provider import FeatureGroup\n\n\n"
        "class Base(FeatureGroup):\n    pass\n\n\n"
        "class Concrete(Base):\n    pass\n"
    )
    (tmp_path / package_name / "only_twin.py").write_text(f"import {twin_name}.groups\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with_groups = fresh_import_report(f"{package_name}.groups", package_name)
    only_twin = fresh_import_report(f"{package_name}.only_twin", package_name)

    assert with_groups.mloda_core is True
    expected = (f"{package_name}.groups.Base", f"{package_name}.groups.Concrete")
    assert with_groups.groups == expected
    assert fresh_import_report(f"{package_name}.groups", f"{package_name}.groups").groups == expected
    assert only_twin == ImportReport(mloda_core=True, modules=(package_name, f"{package_name}.only_twin"), groups=())


def test_fresh_import_report_says_no_mloda_core_for_a_root_in_the_mloda_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leaf under the `mloda.*` namespace that imports no core reports mloda_core=False."""
    name = f"ii_throwaway_{uuid.uuid4().hex}"
    leaf_dir = tmp_path / "mloda" / name
    leaf_dir.mkdir(parents=True)
    (leaf_dir / "leaf.py").write_text("MARKER = 'loaded'\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    report = fresh_import_report(f"mloda.{name}.leaf", f"mloda.{name}")

    assert report.mloda_core is False
    assert report.groups == ()
    assert f"mloda.{name}.leaf" in report.modules


@pytest.mark.parametrize(
    ("module_source", "expected_text"),
    [
        ("raise RuntimeError('boom from the child')\n", "boom from the child"),
        ("import sys\nsys.exit(0)\n", "no JSON summary"),
    ],
    ids=["import-raises", "exits-zero-without-summary"],
)
def test_fresh_import_report_fails_with_a_clear_message_when_the_child_does_not_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_source: str, expected_text: str
) -> None:
    """A raising or silent-exit import fails the report with an informative message."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_throwaway_package(tmp_path, package_name)
    (tmp_path / package_name / "broken.py").write_text(module_source)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AssertionError) as excinfo:
        fresh_import_report(f"{package_name}.broken", package_name)

    assert expected_text in str(excinfo.value)


def test_fresh_import_report_ignores_a_json_py_in_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child runs with -P, so a cwd json.py does not shadow the stdlib."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    pkg_base = tmp_path / "pkgs"
    pkg_base.mkdir()
    _write_throwaway_package(pkg_base, package_name)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "json.py").write_text("raise RuntimeError('shadowed json')\n")
    monkeypatch.syspath_prepend(str(pkg_base))
    monkeypatch.chdir(cwd)

    report = fresh_import_report(package_name, package_name)

    assert report.modules == (package_name,)


def test_fresh_import_report_raises_timeout_expired_when_the_child_exceeds_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child slower than `timeout` raises subprocess.TimeoutExpired."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_throwaway_package(tmp_path, package_name)
    (tmp_path / package_name / "slow.py").write_text("import time\ntime.sleep(30)\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(subprocess.TimeoutExpired):
        fresh_import_report(f"{package_name}.slow", package_name, timeout=1)
