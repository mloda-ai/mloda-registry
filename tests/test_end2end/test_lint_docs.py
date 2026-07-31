"""Tests for scripts/lint_docs.py.

The lint script is not a packaged module, so it's loaded via a sys.path
insert. ``tests/**`` has ``E402`` ignored in ruff config so the post-path
import does not trip the linter.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import lint_docs


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_empty_tree_only_root_index(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n")
    assert lint_docs.find_orphan_guides(tmp_path) == []


def test_linked_guide_not_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n\n[Guide](guide.md)\n")
    _write(tmp_path / "guide.md", "# Guide\n")
    assert lint_docs.find_orphan_guides(tmp_path) == []


def test_unlinked_guide_flagged(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n")
    _write(tmp_path / "orphan.md", "# Orphan\n")
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert len(errors) == 1
    assert "orphan.md" in errors[0]


def test_transitive_reach_via_subdir_index(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n\n[Sub](sub/index.md)\n")
    _write(tmp_path / "sub" / "index.md", "# Sub\n\n[Leaf](leaf.md)\n")
    _write(tmp_path / "sub" / "leaf.md", "# Leaf\n")
    assert lint_docs.find_orphan_guides(tmp_path) == []


def test_unlinked_subdir_index_is_flagged(tmp_path: Path) -> None:
    """Regression for the fixed CONFIRMED hole: subdir index.md is no longer exempt."""
    _write(tmp_path / "index.md", "# Root\n")
    _write(tmp_path / "sub" / "index.md", "# Sub\n\n[Leaf](leaf.md)\n")
    _write(tmp_path / "sub" / "leaf.md", "# Leaf\n")
    errors = lint_docs.find_orphan_guides(tmp_path)
    flagged = {err.split(":")[0] for err in errors}
    assert "sub/index.md" in flagged
    assert "sub/leaf.md" in flagged


def test_link_inside_code_fence_is_ignored(tmp_path: Path) -> None:
    """Regression for the fenced-code fix: links inside ``` blocks don't fabricate edges."""
    _write(
        tmp_path / "index.md",
        "# Root\n\n```markdown\n[fake](only-in-fence.md)\n```\n",
    )
    _write(tmp_path / "only-in-fence.md", "# OnlyInFence\n")
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert any("only-in-fence.md" in err for err in errors)


def test_anchor_bearing_link_reaches_target(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n\n[Guide](guide.md#section)\n")
    _write(tmp_path / "guide.md", "# Guide\n\n## Section\n")
    assert lint_docs.find_orphan_guides(tmp_path) == []


def test_http_link_does_not_count_as_local(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n\n[Ext](https://example.com/foo.md)\n")
    _write(tmp_path / "foo.md", "# Local foo\n")
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert any("foo.md" in err for err in errors)


def test_nested_bracket_link_text_is_reachable(tmp_path: Path) -> None:
    """A link with nested brackets in its text must still resolve to the target."""
    _write(tmp_path / "index.md", "# Root\n\n[a [nested] label](foo.md)\n")
    _write(tmp_path / "foo.md", "# Foo\n")
    assert lint_docs.find_orphan_guides(tmp_path) == []


def test_missing_root_index_returns_sentinel(tmp_path: Path) -> None:
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert len(errors) == 1
    assert "missing root index" in errors[0]


def test_link_outside_docs_dir_does_not_crash(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("# Outside\n")
    docs = tmp_path / "docs"
    _write(docs / "index.md", "# Root\n\n[Outside](../outside.md)\n")
    assert lint_docs.find_orphan_guides(docs) == []


def test_broken_link_suppresses_orphan_cascade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression for fix 3: broken-link errors suppress the orphan check in main()."""
    # "plgins" is an intentional typo: it's what makes the link broken for this test.
    _write(tmp_path / "index.md", "# Root\n\n[Plugins](plgins/index.md)\n")
    _write(tmp_path / "plugins" / "index.md", "# Plugins\n\n[A](a.md)\n")
    _write(tmp_path / "plugins" / "a.md", "# A\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "broken link" in out
    assert "orphan guide" not in out


def test_clean_tree_runs_orphan_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(tmp_path / "index.md", "# Root\n")
    _write(tmp_path / "orphan.md", "# Orphan\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "orphan guide" in out
    assert "orphan.md" in out


def test_fenced_broken_link_is_ignored_by_link_check(tmp_path: Path) -> None:
    """Broken links inside ``` fences must not be flagged by the relative-link check."""
    md_file = tmp_path / "a.md"
    content = "# A\n\n```markdown\n[fake](missing.md)\n```\n"
    _write(md_file, content)
    assert lint_docs.check_relative_links_and_anchors(md_file, content) == []


def test_check_relative_links_flags_broken_link(tmp_path: Path) -> None:
    md_file = tmp_path / "a.md"
    content = "# A\n\n[x](missing.md)\n"
    _write(md_file, content)
    errors = lint_docs.check_relative_links_and_anchors(md_file, content)
    assert len(errors) == 1
    assert "broken link" in errors[0]
    assert "missing.md" in errors[0]


def test_check_relative_links_flags_broken_anchor(tmp_path: Path) -> None:
    a_file = tmp_path / "a.md"
    content = "# A\n\n[link](b.md#nope)\n"
    _write(a_file, content)
    _write(tmp_path / "b.md", "# B\n\n## real\n")
    errors = lint_docs.check_relative_links_and_anchors(a_file, content)
    assert len(errors) == 1
    assert "broken anchor" in errors[0]
    assert "#nope" in errors[0]


def test_check_relative_links_accepts_valid_anchor(tmp_path: Path) -> None:
    a_file = tmp_path / "a.md"
    content = "# A\n\n[link](b.md#real)\n"
    _write(a_file, content)
    _write(tmp_path / "b.md", "# B\n\n## real\n")
    assert lint_docs.check_relative_links_and_anchors(a_file, content) == []


def test_link_cycle_terminates(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n\n[A](a.md)\n")
    _write(tmp_path / "a.md", "# A\n\n[B](b.md)\n")
    _write(tmp_path / "b.md", "# B\n\n[A](a.md)\n")
    assert lint_docs.find_orphan_guides(tmp_path) == []


def test_multiple_orphans_reported_sorted(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n")
    _write(tmp_path / "z_orphan.md", "# Z\n")
    _write(tmp_path / "a_orphan.md", "# A\n")
    _write(tmp_path / "m_orphan.md", "# M\n")
    errors = lint_docs.find_orphan_guides(tmp_path)
    orphan_errors = [err for err in errors if "orphan.md" in err]
    assert len(orphan_errors) == 3
    names = [err.split(":")[0] for err in orphan_errors]
    assert names == ["a_orphan.md", "m_orphan.md", "z_orphan.md"]


def test_check_internal_imports_flags_mloda_core_import(tmp_path: Path) -> None:
    md_file = tmp_path / "a.md"
    content = "# A\n\n```python\nfrom mloda.core.something import X\n```\n"
    _write(md_file, content)
    errors = lint_docs.check_internal_imports(md_file, content)
    assert len(errors) == 1
    assert "internal import" in errors[0]
    assert "mloda.core" in errors[0]


def test_check_internal_imports_ignores_prose_match(tmp_path: Path) -> None:
    """The internal-import check must only scan inside fenced code blocks."""
    md_file = tmp_path / "a.md"
    content = "# A\n\nDo not write `from mloda.core.` in prose, but here we just mention it.\n"
    _write(md_file, content)
    assert lint_docs.check_internal_imports(md_file, content) == []


# Spec fields that are no longer DefaultOptionKeys members: the attribute access raises AttributeError.
RETIRED_SPEC_FIELDS = [
    "explanation",
    "allowed_values",
    "default",
    "strict_validation",
    "element_validator",
    "required_when",
    "match_guard",
]

# Members that survive the retirement and must never be flagged.
SURVIVING_OPTION_KEYS = ["context", "group", "in_features"]


@pytest.mark.parametrize("field", RETIRED_SPEC_FIELDS)
def test_check_retired_property_spec_spellings_flags_retired_option_key(field: str, tmp_path: Path) -> None:
    md_file = tmp_path / "a.md"
    content = f"# A\n\n```python\nkey = DefaultOptionKeys.{field}\n```\n"
    _write(md_file, content)
    errors = lint_docs.check_retired_property_spec_spellings(md_file, content)
    assert len(errors) == 1
    assert f"DefaultOptionKeys.{field}" in errors[0]
    assert "retired" in errors[0].lower()


@pytest.mark.parametrize("field", SURVIVING_OPTION_KEYS)
def test_check_retired_property_spec_spellings_allows_surviving_option_key(field: str, tmp_path: Path) -> None:
    """``context``, ``group`` and ``in_features`` still exist on DefaultOptionKeys."""
    md_file = tmp_path / "a.md"
    content = f"# A\n\n```python\nkey = DefaultOptionKeys.{field}\n```\n"
    _write(md_file, content)
    assert lint_docs.check_retired_property_spec_spellings(md_file, content) == []


def test_check_retired_property_spec_spellings_ignores_prose_match(tmp_path: Path) -> None:
    """Like the internal-import check, only fenced code blocks are scanned."""
    md_file = tmp_path / "a.md"
    content = "# A\n\nThe retired `DefaultOptionKeys.allowed_values` spelling is gone; use `property_spec` instead.\n"
    _write(md_file, content)
    assert lint_docs.check_retired_property_spec_spellings(md_file, content) == []


def test_check_retired_property_spec_spellings_reports_file_and_line(tmp_path: Path) -> None:
    md_file = tmp_path / "a.md"
    content = "# A\n\nIntro prose.\n\n```python\nSPEC = {\n    DefaultOptionKeys.strict_validation: True,\n}\n```\n"
    _write(md_file, content)
    errors = lint_docs.check_retired_property_spec_spellings(md_file, content)
    assert len(errors) == 1
    assert errors[0].startswith(f"{md_file}:7:")


def test_check_retired_property_spec_spellings_flags_raw_dict_spec(tmp_path: Path) -> None:
    """A raw dict as a PROPERTY_MAPPING value now raises ValueError at class definition; flag where it opens."""
    md_file = tmp_path / "a.md"
    content = (
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "operation_type": {\n'
        '        "explanation": "Arithmetic operation",\n'
        "    },\n"
        "}\n"
        "```\n"
    )
    _write(md_file, content)
    errors = lint_docs.check_retired_property_spec_spellings(md_file, content)
    assert len(errors) == 1
    assert errors[0].startswith(f"{md_file}:5:")
    assert "property_spec" in errors[0]


def test_check_retired_property_spec_spellings_accepts_property_spec_value(tmp_path: Path) -> None:
    """The builder form is the supported spelling; its ``allowed_values={...}`` kwarg is not a spec value."""
    md_file = tmp_path / "a.md"
    content = (
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "operation_type": property_spec(\n'
        '        "Arithmetic operation",\n'
        "        strict=True,\n"
        '        allowed_values={"add": "Addition", "sub": "Subtraction"},\n'
        '        default="add",\n'
        "    ),\n"
        "}\n"
        "```\n"
    )
    _write(md_file, content)
    assert lint_docs.check_retired_property_spec_spellings(md_file, content) == []


def test_check_retired_property_spec_spellings_accepts_in_features_mapping_key(tmp_path: Path) -> None:
    """``DefaultOptionKeys.in_features`` as the mapping KEY stays valid when the value is a ``property_spec``."""
    md_file = tmp_path / "a.md"
    content = (
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        "    DefaultOptionKeys.in_features: property_spec(\n"
        '        "Source feature",\n'
        "        context=True,\n"
        "    ),\n"
        "}\n"
        "```\n"
    )
    _write(md_file, content)
    assert lint_docs.check_retired_property_spec_spellings(md_file, content) == []


def test_main_reports_retired_property_spec_spellings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(
        tmp_path / "index.md",
        "# Root\n\n```python\nkey = DefaultOptionKeys.allowed_values\n```\n",
    )
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "DefaultOptionKeys.allowed_values" in out


def test_missing_root_index_uses_relative_path(tmp_path: Path) -> None:
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert len(errors) == 1
    assert str(tmp_path) not in errors[0]
