"""Guards in scripts/check_minor_release_type.py: seam table or core_dependency changes need ``minor:``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_minor_release_type.py"
_SHARED_PATH = _REPO_ROOT / "config" / "shared.toml"
_SEAM_SOURCE_PATH = _REPO_ROOT / "mloda" / "testing" / "extenders" / "openlineage.py"

chk = load_script("check_minor_release_type", _SCRIPT_PATH)

_BASE = 'OPENLINEAGE_EXTENDER_SEAMS = {"_dispatch": ("self",)}\nOPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS = ("producer",)\n'
_CHANGED = (
    'OPENLINEAGE_EXTENDER_SEAMS = {"_dispatch": ("self", "x")}\nOPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS = ("producer",)\n'
)
_NO_TABLES = "X = 1\n"

_CORE_BASE = '[project]\nversion = "0.13.0"\n\n[defaults]\ncore_dependency = "mloda>=0.20.0,<0.21.0"\n'
_CORE_CHANGED = _CORE_BASE.replace("<0.21.0", "<0.22.0")
_CORE_NO_KEY = '[project]\nversion = "0.13.0"\n\n[defaults]\nother = 1\n'

_DISPATCH_OLD = '"_dispatch": (_p("self"), _p("context"), _p("func"), _p("args"), _p("kwargs")),'
_DISPATCH_REFORMATTED = (
    '"_dispatch": (\n        _p("self"),\n        _p("context"),\n        _p("func"),\n        _p("args"),\n'
    '        _p("kwargs"),\n    ),'
)
_DISPATCH_EXTENDED = '"_dispatch": (_p("self"), _p("context"), _p("func"), _p("args"), _p("kwargs"), _p("extra")),'


def _real_source() -> str:
    return _SEAM_SOURCE_PATH.read_text()


def _edit(source: str, old: str, new: str) -> str:
    assert old in source, f"fixture text not found in real source: {old!r}"
    return source.replace(old, new, 1)


class TestSeamTables:
    def test_real_source_has_tables(self) -> None:
        assert chk.seam_tables(_real_source()) is not None

    def test_missing_table_returns_none(self) -> None:
        assert chk.seam_tables(_NO_TABLES) is None
        assert chk.seam_tables('OPENLINEAGE_EXTENDER_SEAMS = {"a": ()}\n') is None
        assert chk.seam_tables('OPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS = ("a",)\n') is None

    def test_annotated_assignment_is_found(self) -> None:
        source = (
            'OPENLINEAGE_EXTENDER_SEAMS: dict[str, tuple[str, ...]] = {"a": ()}\n'
            'OPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS: tuple[str, ...] = ("p",)\n'
        )
        assert chk.seam_tables(source) is not None

    def test_comment_edit_is_unchanged(self) -> None:
        source = _real_source()
        edited = _edit(source, "OPENLINEAGE_EXTENDER_SEAMS: ", "# a new comment\nOPENLINEAGE_EXTENDER_SEAMS: ")
        assert chk.seam_tables(edited) == chk.seam_tables(source)

    def test_reformat_is_unchanged(self) -> None:
        source = _real_source()
        assert chk.seam_tables(_edit(source, _DISPATCH_OLD, _DISPATCH_REFORMATTED)) == chk.seam_tables(source)

    def test_signature_edit_is_changed(self) -> None:
        source = _real_source()
        assert chk.seam_tables(_edit(source, _DISPATCH_OLD, _DISPATCH_EXTENDED)) != chk.seam_tables(source)

    def test_attribute_seam_edit_is_changed(self) -> None:
        source = _real_source()
        old = '("producer", "job_namespace", "dataset_namespace")'
        new = '("producer", "job_namespace", "dataset_namespace", "extra")'
        assert chk.seam_tables(_edit(source, old, new)) != chk.seam_tables(source)


class TestIsMinor:
    @pytest.mark.parametrize("subject", ["minor: x", "minor(lineage): x", "minor!: x", "minor(lineage)!: x"])
    def test_accepted(self, subject: str) -> None:
        assert chk.is_minor(subject) is True

    @pytest.mark.parametrize("subject", ["fix!: x", "fix: x", "feat: x", "minorx: x", "Minor: x"])
    def test_rejected(self, subject: str) -> None:
        assert chk.is_minor(subject) is False


class TestCoreDependency:
    def test_real_shared_toml_has_value(self) -> None:
        assert chk.core_dependency(_SHARED_PATH.read_text()) is not None

    @pytest.mark.parametrize(
        "source",
        [
            '[project]\nversion = "1"\n',
            "[defaults]\nother = 1\n",
            "[defaults]\ncore_dependency = 5\n",
            "defaults = 1\n",
            'defaults = "x"\n',
            '[defaults]\ncore_dependency = ["mloda>=0.20.0"]\n',
            "[defaults\ncore_dependency = ",
        ],
    )
    def test_unusable_source_returns_none(self, source: str) -> None:
        assert chk.core_dependency(source) is None

    def test_value_has_whitespace_removed(self) -> None:
        source = '[defaults]\ncore_dependency = "mloda >= 0.20.0, < 0.21.0"\n'
        assert chk.core_dependency(source) == "mloda>=0.20.0,<0.21.0"

    def test_whitespace_change_is_unchanged(self) -> None:
        spaced = _CORE_BASE.replace("mloda>=0.20.0,<0.21.0", "mloda >= 0.20.0, < 0.21.0")
        assert chk.core_dependency(spaced) == chk.core_dependency(_CORE_BASE)

    def test_bound_change_is_changed(self) -> None:
        assert chk.core_dependency(_CORE_CHANGED) != chk.core_dependency(_CORE_BASE)

    def test_project_version_edit_is_unchanged(self) -> None:
        bumped = _CORE_BASE.replace('version = "0.13.0"', 'version = "0.14.0"')
        assert chk.core_dependency(bumped) == chk.core_dependency(_CORE_BASE)


class TestGuards:
    def test_guard_definitions(self) -> None:
        assert chk.SEAM_GUARD.name == "seam tables"
        assert chk.CORE_DEPENDENCY_GUARD.name == "core_dependency"
        assert chk.SEAM_GUARD.path == "mloda/testing/extenders/openlineage.py"
        assert chk.SEAM_GUARD.extract is chk.seam_tables
        assert chk.CORE_DEPENDENCY_GUARD.path == "config/shared.toml"
        assert chk.CORE_DEPENDENCY_GUARD.extract is chk.core_dependency
        assert chk.GUARDS == (chk.SEAM_GUARD, chk.CORE_DEPENDENCY_GUARD)

    def test_guards_have_reasons(self) -> None:
        for guard in chk.GUARDS:
            assert isinstance(guard.reason, str) and guard.reason

    def test_seam_reason_keeps_literal_version_placeholder(self) -> None:
        error = chk.evaluate(chk.SEAM_GUARD, _BASE, _CHANGED, "fix: seam", ["fix: seam"])
        assert error is not None
        assert "~={version}" in error


# (guard, base source, changed source, source without the guarded value)
_CASES: list[Any] = [
    pytest.param(lambda: chk.SEAM_GUARD, _BASE, _CHANGED, _NO_TABLES, id="seam"),
    pytest.param(lambda: chk.CORE_DEPENDENCY_GUARD, _CORE_BASE, _CORE_CHANGED, _CORE_NO_KEY, id="core_dependency"),
]


@pytest.mark.parametrize("guard_factory, base, changed, missing", _CASES)
class TestEvaluate:
    def test_change_with_fix_title_and_commits_errors(
        self, guard_factory: Any, base: str, changed: str, missing: str
    ) -> None:
        guard = guard_factory()
        error = chk.evaluate(guard, base, changed, "fix: change", ["fix: change"])
        assert error is not None
        assert "minor:" in error
        assert guard.name in error
        assert guard.path in error

    def test_change_with_minor_title_and_commit_passes(
        self, guard_factory: Any, base: str, changed: str, missing: str
    ) -> None:
        assert chk.evaluate(guard_factory(), base, changed, "minor: change", ["fix: other", "minor: change"]) is None

    def test_change_with_minor_commit_but_fix_title_errors(
        self, guard_factory: Any, base: str, changed: str, missing: str
    ) -> None:
        assert chk.evaluate(guard_factory(), base, changed, "fix: change", ["minor: change"]) is not None

    def test_change_with_minor_title_but_no_minor_commit_errors(
        self, guard_factory: Any, base: str, changed: str, missing: str
    ) -> None:
        assert chk.evaluate(guard_factory(), base, changed, "minor: change", ["fix: change"]) is not None

    def test_no_change_with_fix_passes(self, guard_factory: Any, base: str, changed: str, missing: str) -> None:
        assert chk.evaluate(guard_factory(), base, base, "fix: other", ["fix: other"]) is None

    @pytest.mark.parametrize("absent", ["none", "missing"])
    def test_head_without_value_errors(
        self, guard_factory: Any, base: str, changed: str, missing: str, absent: str
    ) -> None:
        guard = guard_factory()
        head = None if absent == "none" else missing
        error = chk.evaluate(guard, base, head, "minor: change", ["minor: change"])
        assert error is not None
        assert "not found" in error
        assert guard.name in error
        assert guard.path in error
        assert "scripts/check_minor_release_type.py" in error

    @pytest.mark.parametrize("absent", ["none", "missing"])
    def test_base_without_value_is_a_change(
        self, guard_factory: Any, base: str, changed: str, missing: str, absent: str
    ) -> None:
        guard = guard_factory()
        old = None if absent == "none" else missing
        assert chk.evaluate(guard, old, base, "fix: change", ["fix: change"]) is not None
        assert chk.evaluate(guard, old, base, "minor: change", ["minor: change"]) is None
