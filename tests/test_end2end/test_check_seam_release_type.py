"""Release-type guard for scripts/check_seam_release_type.py. A change to the OpenLineage extender seam
tables must ship as a ``minor:`` release, because mloda-enterprise pins mloda-community-openlineage~={version}."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_seam_release_type.py"
_SEAM_SOURCE_PATH = _REPO_ROOT / "mloda" / "testing" / "extenders" / "openlineage.py"

_BASE = 'OPENLINEAGE_EXTENDER_SEAMS = {"_dispatch": ("self",)}\nOPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS = ("producer",)\n'
_CHANGED = (
    'OPENLINEAGE_EXTENDER_SEAMS = {"_dispatch": ("self", "x")}\nOPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS = ("producer",)\n'
)
_NO_TABLES = "X = 1\n"


def _script() -> ModuleType:
    if not _SCRIPT_PATH.is_file():
        pytest.fail(f"{_SCRIPT_PATH} does not exist yet")
    return load_script("check_seam_release_type", _SCRIPT_PATH)


def _fn(name: str) -> Callable[..., object]:
    func: Callable[..., object] | None = getattr(_script(), name, None)
    assert callable(func), f"check_seam_release_type.{name} must be a callable"
    return func


def _seam_tables(source: str) -> str | None:
    result = _fn("seam_tables")(source)
    assert result is None or isinstance(result, str)
    return result


def _is_minor(subject: str) -> bool:
    return bool(_fn("is_minor")(subject))


def _evaluate(base: str | None, head: str | None, title: str, subjects: list[str]) -> str | None:
    result = _fn("evaluate")(base, head, title, subjects)
    assert result is None or isinstance(result, str)
    return result


def _real_source() -> str:
    return _SEAM_SOURCE_PATH.read_text()


def _edit(source: str, old: str, new: str) -> str:
    assert old in source, f"fixture text not found in real source: {old!r}"
    return source.replace(old, new, 1)


class TestSeamTables:
    def test_real_source_has_tables(self) -> None:
        assert _seam_tables(_real_source()) is not None

    def test_missing_table_returns_none(self) -> None:
        assert _seam_tables(_NO_TABLES) is None
        assert _seam_tables('OPENLINEAGE_EXTENDER_SEAMS = {"a": ()}\n') is None
        assert _seam_tables('OPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS = ("a",)\n') is None

    def test_annotated_assignment_is_found(self) -> None:
        source = (
            'OPENLINEAGE_EXTENDER_SEAMS: dict[str, tuple[str, ...]] = {"a": ()}\n'
            'OPENLINEAGE_EXTENDER_ATTRIBUTE_SEAMS: tuple[str, ...] = ("p",)\n'
        )
        assert _seam_tables(source) is not None

    def test_comment_edit_is_unchanged(self) -> None:
        source = _real_source()
        edited = _edit(source, "OPENLINEAGE_EXTENDER_SEAMS: ", "# a new comment\nOPENLINEAGE_EXTENDER_SEAMS: ")
        assert _seam_tables(edited) == _seam_tables(source)

    def test_reformat_is_unchanged(self) -> None:
        source = _real_source()
        old = '"_dispatch": (_p("self"), _p("context"), _p("func"), _p("args"), _p("kwargs")),'
        new = '"_dispatch": (\n        _p("self"),\n        _p("context"),\n        _p("func"),\n        _p("args"),\n        _p("kwargs"),\n    ),'
        assert _seam_tables(_edit(source, old, new)) == _seam_tables(source)

    def test_signature_edit_is_changed(self) -> None:
        source = _real_source()
        old = '"_dispatch": (_p("self"), _p("context"), _p("func"), _p("args"), _p("kwargs")),'
        new = '"_dispatch": (_p("self"), _p("context"), _p("func"), _p("args"), _p("kwargs"), _p("extra")),'
        assert _seam_tables(_edit(source, old, new)) != _seam_tables(source)

    def test_attribute_seam_edit_is_changed(self) -> None:
        source = _real_source()
        old = '("producer", "job_namespace", "dataset_namespace")'
        new = '("producer", "job_namespace", "dataset_namespace", "extra")'
        assert _seam_tables(_edit(source, old, new)) != _seam_tables(source)


class TestIsMinor:
    @pytest.mark.parametrize("subject", ["minor: x", "minor(lineage): x"])
    def test_accepted(self, subject: str) -> None:
        assert _is_minor(subject) is True

    @pytest.mark.parametrize("subject", ["minor!: x", "fix: x", "feat: x", "minorx: x", "Minor: x"])
    def test_rejected(self, subject: str) -> None:
        assert _is_minor(subject) is False


class TestEvaluate:
    def test_change_with_fix_title_and_commits_errors(self) -> None:
        error = _evaluate(_BASE, _CHANGED, "fix: seam", ["fix: seam"])
        assert error is not None
        assert "minor:" in error

    def test_change_with_minor_title_and_commit_passes(self) -> None:
        assert _evaluate(_BASE, _CHANGED, "minor: seam", ["fix: other", "minor: seam"]) is None

    def test_change_with_minor_commit_but_fix_title_errors(self) -> None:
        assert _evaluate(_BASE, _CHANGED, "fix: seam", ["minor: seam"]) is not None

    def test_change_with_minor_title_but_no_minor_commit_errors(self) -> None:
        assert _evaluate(_BASE, _CHANGED, "minor: seam", ["fix: seam"]) is not None

    def test_no_change_with_fix_passes(self) -> None:
        assert _evaluate(_BASE, _BASE, "fix: other", ["fix: other"]) is None

    @pytest.mark.parametrize("head", [None, _NO_TABLES])
    def test_head_without_tables_errors(self, head: str | None) -> None:
        error = _evaluate(_BASE, head, "minor: seam", ["minor: seam"])
        assert error is not None
        assert "not found" in error

    @pytest.mark.parametrize("base", [None, _NO_TABLES])
    def test_base_without_tables_is_a_change(self, base: str | None) -> None:
        assert _evaluate(base, _BASE, "fix: seam", ["fix: seam"]) is not None
        assert _evaluate(base, _BASE, "minor: seam", ["minor: seam"]) is None
