"""Direct unit tests for mloda.testing.import_isolation."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import subprocess  # nosec
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest

from mloda.testing.import_isolation import (
    ImportReport,
    assert_imports_with_base_deps,
    base_import_names,
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


def _write_module(path: Path, source: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _write_dep(base: Path, name: str) -> None:
    _write_module(base / name / "__init__.py", "MARKER = 'dep'\n")


def _write_dist_info(base: Path, name: str, requires: Sequence[str] = (), top_level: Sequence[str] = ()) -> None:
    dist_info = base / f"{re.sub(r'[-_.]+', '_', name)}-0.0.0.dist-info"
    dist_info.mkdir()
    lines = ["Metadata-Version: 2.1", f"Name: {name}", "Version: 0.0.0", *(f"Requires-Dist: {r}" for r in requires)]
    (dist_info / "METADATA").write_text("\n".join(lines) + "\n")
    if top_level:
        (dist_info / "top_level.txt").write_text("\n".join(top_level) + "\n")


def test_assert_imports_with_base_deps_passes_a_clean_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowed deps, guarded blocked imports, `tests` subpackages and `__main__` do not fail."""
    suffix = uuid.uuid4().hex
    package_name = f"ii_throwaway_{suffix}"
    allowed_dep = f"ii_allowed_{suffix}"
    blocked_dep = f"ii_blocked_{suffix}"
    _write_dep(tmp_path, allowed_dep)
    _write_dep(tmp_path, blocked_dep)
    root = tmp_path / package_name
    _write_module(root / "__init__.py")
    _write_module(root / "plain.py", "MARKER = 'plain'\n")
    _write_module(root / "sub" / "__init__.py")
    _write_module(root / "sub" / "inner.py", f"from {package_name} import plain\n")
    _write_module(root / "ns" / "mod.py", "MARKER = 'namespace'\n")
    _write_module(root / "uses_allowed.py", f"import {allowed_dep}\n")
    _write_module(
        root / "guarded.py",
        f"try:\n    import {blocked_dep}\nexcept ImportError:\n    {blocked_dep} = None\n",
    )
    _write_module(
        root / "spec_guard.py",
        f"import importlib.util\n\nassert importlib.util.find_spec('{blocked_dep}') is None\n",
    )
    _write_module(root / "tests" / "__init__.py", f"import {blocked_dep}\n")
    _write_module(root / "sub" / "tests" / "test_x.py", f"import {blocked_dep}\n")
    _write_module(root / "__main__.py", "raise SystemExit(2)\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert_imports_with_base_deps(package_name, {allowed_dep})


def test_assert_imports_with_base_deps_reports_every_failing_module_with_its_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing modules are reported with tracebacks, a clean sibling is not."""
    suffix = uuid.uuid4().hex
    package_name = f"ii_throwaway_{suffix}"
    blocked_dep = f"ii_blocked_{suffix}"
    allowed_dep = f"ii_allowed_{suffix}"
    _write_dep(tmp_path, blocked_dep)
    _write_dep(tmp_path, allowed_dep)
    root = tmp_path / package_name
    _write_module(root / "__init__.py")
    _write_module(root / "bad_plain.py", f"import {blocked_dep}\n")
    _write_module(root / "bad_sub" / "__init__.py", f"import {blocked_dep}\n")
    _write_module(root / "ns" / "bad_ns.py", f"import {blocked_dep}\n")
    _write_module(root / "raises.py", "raise RuntimeError('boom from the module')\n")
    _write_module(root / "clean.py", "MARKER = 'clean'\n")
    _write_module(root / "exits.py", "import sys\nsys.exit(3)\n")
    _write_module(root / "per_extra" / "__init__.py", "raise RuntimeError('excluded')\n")
    _write_module(root / "per_extra_other" / "__init__.py", "raise RuntimeError('not excluded')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AssertionError) as excinfo:
        assert_imports_with_base_deps(package_name, {allowed_dep}, exclude={f"{package_name}.per_extra"})

    message = str(excinfo.value)
    assert f"{package_name}.per_extra_other:" in message
    assert f"{package_name}.per_extra:" not in message
    for failing in ("bad_plain", "bad_sub", "ns.bad_ns", "raises"):
        assert f"{package_name}.{failing}" in message
    assert f"{package_name}.exits:" in message
    assert "SystemExit" in message
    assert "ModuleNotFoundError" in message
    assert blocked_dep in message
    assert "boom from the module" in message
    assert f"{package_name}.clean" not in message
    assert allowed_dep in message


@pytest.mark.parametrize(
    ("source", "expected_failure"),
    [("raise RuntimeError('root boom')\n", True), ("MARKER = 'ok'\n", False)],
    ids=["fails", "clean"],
)
@pytest.mark.parametrize("is_package", [True, False], ids=["package", "plain-module"])
def test_assert_imports_with_base_deps_handles_a_failing_root_and_a_plain_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_package: bool, source: str, expected_failure: bool
) -> None:
    """A failing root is reported under its own name; a plain module is imported alone."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    if is_package:
        _write_module(tmp_path / package_name / "__init__.py", source)
        _write_module(tmp_path / package_name / "leaf.py")
    else:
        _write_module(tmp_path / f"{package_name}.py", source)
    monkeypatch.syspath_prepend(str(tmp_path))

    if expected_failure:
        with pytest.raises(AssertionError) as excinfo:
            assert_imports_with_base_deps(package_name, set())
        assert package_name in str(excinfo.value)
        assert "root boom" in str(excinfo.value)
    else:
        assert_imports_with_base_deps(package_name, set())


def test_assert_imports_with_base_deps_reports_a_failing_module_when_package_itself_is_a_tests_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tests` components are skipped only below the root, so a `tests` root is walked."""
    package_name = f"ii_throwaway_{uuid.uuid4().hex}"
    _write_module(tmp_path / package_name / "__init__.py")
    _write_module(tmp_path / package_name / "tests" / "__init__.py")
    _write_module(tmp_path / package_name / "tests" / "bad.py", "raise RuntimeError('tests root boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AssertionError) as excinfo:
        assert_imports_with_base_deps(f"{package_name}.tests", set())

    assert f"{package_name}.tests.bad:" in str(excinfo.value)
    assert "tests root boom" in str(excinfo.value)


def test_assert_imports_with_base_deps_does_not_admit_submodules_of_a_namespace_stub_preloaded_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stub module installed by `sitecustomize` is preloaded, yet its submodules stay blocked."""
    suffix = uuid.uuid4().hex
    package_name = f"ii_throwaway_{suffix}"
    stub_name = f"ii_stub_{suffix}"
    real_dir = tmp_path / "real" / stub_name
    _write_module(real_dir / "sub.py", "MARKER = 'sub'\n")
    startup_dir = tmp_path / "startup"
    _write_module(
        startup_dir / "sitecustomize.py",
        "import sys, types\n"
        f"m = types.ModuleType({stub_name!r})\n"
        f"m.__path__ = [{str(real_dir)!r}]\n"
        f"sys.modules.setdefault({stub_name!r}, m)\n",
    )
    code = tmp_path / "code"
    _write_module(code / package_name / "__init__.py")
    _write_module(code / package_name / "uses_stub.py", f"import {stub_name}.sub\n")
    monkeypatch.syspath_prepend(str(code))
    monkeypatch.setenv("PYTHONPATH", str(startup_dir))

    with pytest.raises(AssertionError) as excinfo:
        assert_imports_with_base_deps(package_name, set())

    assert f"{package_name}.uses_stub:" in str(excinfo.value)


def test_assert_imports_with_base_deps_dotted_allowed_entry_admits_its_subpackage_but_not_a_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`dep.sub` admits `dep.sub` and its ancestor `dep`, not `dep.other`."""
    suffix = uuid.uuid4().hex
    package_name = f"ii_throwaway_{suffix}"
    dep = f"ii_dep_{suffix}"
    _write_module(tmp_path / dep / "sub" / "__init__.py", "MARKER = 'sub'\n")
    _write_module(tmp_path / dep / "other" / "__init__.py", "MARKER = 'other'\n")
    root = tmp_path / package_name
    _write_module(root / "__init__.py")
    _write_module(root / "uses_sub.py", f"import {dep}.sub\n")
    _write_module(root / "uses_other.py", f"import {dep}.other\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AssertionError) as excinfo:
        assert_imports_with_base_deps(package_name, {f"{dep}.sub"})

    message = str(excinfo.value)
    assert f"{package_name}.uses_other" in message
    assert f"{package_name}.uses_sub:" not in message


@pytest.mark.parametrize(
    ("package", "exclude"),
    [
        ("mloda.community.extenders.shared", frozenset()),
        (
            "mloda.community.extenders",
            {"mloda.community.extenders.otel", "mloda.community.extenders.openlineage"},
        ),
    ],
    ids=["regular-package", "namespace-parent"],
)
def test_assert_imports_with_base_deps_passes_a_real_registry_package_with_dotted_core_entries(
    package: str, exclude: frozenset[str] | set[str]
) -> None:
    """Real registry packages pass with dotted core entries; a namespace `__path__` holds editable placeholders."""
    assert_imports_with_base_deps(package, {"mloda.core", "mloda.provider", "mloda.steward"}, exclude=exclude)


def test_base_import_names_resolves_a_real_registry_package_to_core_import_names() -> None:
    """A real registry package resolves to the core import names."""
    assert {"mloda", "mloda_plugins"} <= base_import_names("mloda-community-extenders-shared")


def test_base_import_names_collects_the_base_closure_as_import_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closure follows base requirements by canonical name, evaluates markers and extras, and terminates on cycles; nameless dists are skipped."""
    s = uuid.uuid4().hex
    _write_dist_info(
        tmp_path,
        f"ii-root-{s}",
        requires=[
            f"ii-mid-dep-{s}",
            f"ii-extra-{s}; extra == 'x'",
            f"ii-nothere-{s}; python_version < '3'",
            f"ii-marker-{s}; python_version >= '3'",
            f"ii-opt-{s}[opt]",
        ],
        top_level=[f"ii_root_pkg_{s}"],
    )
    _write_dist_info(tmp_path, f"Ii_Mid.Dep_{s}", requires=[f"ii-deep-{s}"], top_level=[f"ii_mid_{s}"])
    _write_dist_info(tmp_path, f"ii-deep-{s}", requires=[f"ii-root-{s}"], top_level=[f"ii_deep_{s}"])
    _write_dist_info(tmp_path, f"ii-extra-{s}", top_level=[f"ii_extra_{s}"])
    _write_dist_info(tmp_path, f"ii-marker-{s}", top_level=[f"ii_marker_{s}"])
    _write_dist_info(tmp_path, f"ii-opt-{s}", requires=[f"ii-optdep-{s}; extra == 'opt'"])
    _write_dist_info(tmp_path, f"ii-optdep-{s}", top_level=[f"ii_optdep_{s}"])
    _write_dist_info(tmp_path, f"ii-root2-{s}", requires=[f"ii-missing-{s}"], top_level=[f"ii_root2_{s}"])
    nameless = tmp_path / f"ii_nameless_{s}-0.0.0.dist-info"
    nameless.mkdir()
    (nameless / "METADATA").write_text("Metadata-Version: 2.1\nVersion: 0.0.0\n")
    (nameless / "top_level.txt").write_text(f"ii_nameless_{s}\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert base_import_names(f"ii-root-{s}") == frozenset(
        {f"ii_root_pkg_{s}", f"ii_mid_{s}", f"ii_deep_{s}", f"ii_marker_{s}", f"ii_optdep_{s}"}
    )
    with pytest.raises(importlib.metadata.PackageNotFoundError, match=f"ii-missing-{s}"):
        base_import_names(f"ii-root2-{s}")
