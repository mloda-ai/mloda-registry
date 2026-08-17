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
    path.write_text(body, encoding="utf-8")


def test_empty_tree_only_root_index(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n")
    assert lint_docs.find_orphan_guides(tmp_path) == []


# (root index body, target file name, target body) triples whose link form must reach the target.
REACHING_LINK_FORMS = [
    pytest.param("# Root\n\n[Guide](guide.md)\n", "guide.md", "# Guide\n", id="linked_guide"),
    pytest.param(
        "# Root\n\n[Guide](guide.md#section)\n", "guide.md", "# Guide\n\n## Section\n", id="anchor_bearing_link"
    ),
    # A link with nested brackets in its text must still resolve to the target.
    pytest.param("# Root\n\n[a [nested] label](foo.md)\n", "foo.md", "# Foo\n", id="nested_bracket_link_text"),
]


@pytest.mark.parametrize(("index_body", "target_name", "target_body"), REACHING_LINK_FORMS)
def test_link_form_reaches_target(index_body: str, target_name: str, target_body: str, tmp_path: Path) -> None:
    _write(tmp_path / "index.md", index_body)
    _write(tmp_path / target_name, target_body)
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
    flagged = {Path(err.split(":")[0]).as_posix() for err in errors}
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


def test_http_link_does_not_count_as_local(tmp_path: Path) -> None:
    _write(tmp_path / "index.md", "# Root\n\n[Ext](https://example.com/foo.md)\n")
    _write(tmp_path / "foo.md", "# Local foo\n")
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert any("foo.md" in err for err in errors)


def test_missing_root_index_returns_sentinel(tmp_path: Path) -> None:
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert len(errors) == 1
    assert "missing root index" in errors[0]


def test_missing_root_index_names_scope_directory(tmp_path: Path) -> None:
    """The sentinel names the orphan-scope dir, which is no longer the same dir the linter walks."""
    guides = tmp_path / "guides"
    guides.mkdir()
    errors = lint_docs.find_orphan_guides(guides)
    assert len(errors) == 1
    assert errors[0] == "guides/index.md: missing root index for orphan check"
    assert str(tmp_path) not in errors[0]


def test_link_outside_docs_dir_does_not_crash(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.md"
    outside.write_text("# Outside\n")
    docs = tmp_path / "docs"
    _write(docs / "index.md", "# Root\n\n[Outside](../outside.md)\n")
    assert lint_docs.find_orphan_guides(docs) == []


def test_docs_dir_is_repo_docs_root() -> None:
    """DOCS_DIR is the docs root and GUIDES_DIR its guides subdir; re-narrowing DOCS_DIR breaks this."""
    assert lint_docs.DOCS_DIR == _REPO_ROOT / "docs"
    assert lint_docs.GUIDES_DIR == _REPO_ROOT / "docs" / "guides"
    assert lint_docs.GUIDES_DIR.is_dir()
    outside_guides = [p for p in lint_docs.DOCS_DIR.rglob("*.md") if not p.is_relative_to(lint_docs.GUIDES_DIR)]
    assert outside_guides, "markdown outside docs/guides must exist and be in lint scope"


def test_broken_link_suppresses_orphan_cascade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A link error inside guides can corrupt the reachability BFS, so it still suppresses the orphan check."""
    # "plgins" is an intentional typo: it's what makes the link broken for this test.
    _write(tmp_path / "guides" / "index.md", "# Guides\n\n[Plugins](plgins/index.md)\n")
    _write(tmp_path / "guides" / "plugins" / "index.md", "# Plugins\n\n[A](a.md)\n")
    _write(tmp_path / "guides" / "plugins" / "a.md", "# A\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path / "guides")
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "broken link" in out
    assert "orphan guide" not in out


def test_docs_root_link_error_does_not_suppress_guides_orphan_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A broken link outside guides cannot corrupt the guides BFS, so the orphan check must still run."""
    _write(tmp_path / "guides" / "index.md", "# Guides\n")
    _write(tmp_path / "guides" / "stray.md", "# Stray\n")
    _write(tmp_path / "packaging.md", "# Packaging\n\n[Missing](missing.md)\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path / "guides")
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "broken link" in out
    assert "packaging.md" in out
    assert "stray.md: orphan guide" in out


def test_clean_tree_runs_orphan_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(tmp_path / "index.md", "# Root\n")
    _write(tmp_path / "orphan.md", "# Orphan\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path)
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "orphan guide" in out
    assert "orphan.md" in out


def test_main_flags_broken_link_outside_guides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Markdown at the docs root (packaging.md, releasing.md) is link-checked, not only docs/guides."""
    _write(tmp_path / "guides" / "index.md", "# Guides\n")
    _write(tmp_path / "packaging.md", "# Packaging\n\n[Missing](missing.md)\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path / "guides")
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "broken link" in out
    assert "packaging.md" in out


def test_main_flags_bare_fence_outside_guides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The fence check follows the widened walk: a bare ``` opener at the docs root is flagged."""
    _write(tmp_path / "guides" / "index.md", "# Guides\n")
    _write(tmp_path / "releasing.md", "# Releasing\n\n```\nuv build\n```\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path / "guides")
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "bare fenced-code opener" in out
    assert "releasing.md" in out


def test_orphan_check_stays_scoped_to_guides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Docs-root files are not guides, so they are never orphans; unlinked files under guides still are."""
    _write(tmp_path / "guides" / "index.md", "# Guides\n")
    _write(tmp_path / "guides" / "stray.md", "# Stray\n")
    _write(tmp_path / "packaging.md", "# Packaging\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path / "guides")
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "stray.md: orphan guide" in out
    assert "packaging.md" not in out


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
    assert errors[0].startswith(f"{md_file}:4:")


def test_check_internal_imports_reports_own_line_for_repeated_import(tmp_path: Path) -> None:
    """Identical imports in separate fences must report their own line numbers."""
    md_file = tmp_path / "a.md"
    content = (
        "# Guide\n"
        "\n"
        "```python\n"
        "from mloda.core.abstract_plugins.components.options import Options\n"
        "```\n"
        "\n"
        "Some prose in between.\n"
        "Padding line.\n"
        "Padding line.\n"
        "Padding line.\n"
        "\n"
        "```python\n"
        "from mloda.core.abstract_plugins.components.options import Options\n"
        "```\n"
    )
    _write(md_file, content)
    errors = lint_docs.check_internal_imports(md_file, content)
    assert len(errors) == 2
    assert errors[0].startswith(f"{md_file}:4:")
    assert errors[1].startswith(f"{md_file}:13:")


def test_check_internal_imports_ignores_prose_match(tmp_path: Path) -> None:
    """The internal-import check must only scan inside fenced code blocks."""
    md_file = tmp_path / "a.md"
    content = "# A\n\nDo not write `from mloda.core.` in prose, but here we just mention it.\n"
    _write(md_file, content)
    assert lint_docs.check_internal_imports(md_file, content) == []


# Spec-field spellings that DefaultOptionKeys must not be used for: none of them are members of the
# current DefaultOptionKeys, and ``explanation`` was never a member at all.
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


# Snippets the retired-spelling check must leave alone, each with the reason it is legal.
ACCEPTED_SPEC_SNIPPETS = [
    # Like the internal-import check, only fenced code blocks are scanned.
    pytest.param(
        "# A\n\nThe retired `DefaultOptionKeys.allowed_values` spelling is gone; use `property_spec` instead.\n",
        id="prose_match",
    ),
    # The builder form is the supported spelling; its ``allowed_values={...}`` kwarg is not a spec value.
    pytest.param(
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
        "```\n",
        id="property_spec_value",
    ),
    # ``DefaultOptionKeys.in_features`` as the mapping KEY stays valid when the value is a ``property_spec``.
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        "    DefaultOptionKeys.in_features: property_spec(\n"
        '        "Source feature",\n'
        "        context=True,\n"
        "    ),\n"
        "}\n"
        "```\n",
        id="in_features_mapping_key",
    ),
    # ``: {`` inside an explanation string is prose, not a raw dict value.
    pytest.param(
        '# A\n\n```python\nPROPERTY_MAPPING = {\n    "fmt": property_spec("Template, e.g. {col}: {value}"),\n}\n```\n',
        id="colon_brace_inside_string",
    ),
    # ``: {`` inside a trailing comment is not a raw dict value either.
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "op": property_spec("Arithmetic"),  # shape: {value: doc}\n'
        "}\n"
        "```\n",
        id="colon_brace_in_comment",
    ),
    # A dict nested inside a ``property_spec(...)`` kwarg is not a PROPERTY_MAPPING value.
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "op": property_spec("Arithmetic", allowed_values={"add": {"alias": "plus"}}),\n'
        "}\n"
        "```\n",
        id="nested_dict_in_kwarg",
    ),
    # A ```text fence quoting a migration message is documentation, not code: skip both checks.
    pytest.param(
        "# A\n"
        "\n"
        "```text\n"
        "AttributeError: DefaultOptionKeys.allowed_values does not exist on the unreleased core.\n"
        "The old shape was: PROPERTY_MAPPING = {\n"
        '    "operation_type": {"explanation": "Arithmetic operation"},\n'
        "}\n"
        "```\n",
        id="non_python_fence",
    ),
    # Guide snippets are often fragments: a bare dict body must not raise and must not be read as a raw dict.
    pytest.param(
        '# A\n\n```python\n    "aggregation_type": ...,\n    "order_by": property_spec("x"),\n```\n',
        id="unparseable_fence",
    ),
    # A mapping cut off before its closing brace must not fall back to a scan that misreads the entries.
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "fmt": property_spec("Template, e.g. {col}: {value}"),\n'
        "    ...\n"
        "```\n",
        id="truncated_mapping_fence",
    ),
]


@pytest.mark.parametrize("content", ACCEPTED_SPEC_SNIPPETS)
def test_check_retired_property_spec_spellings_accepts_snippet(content: str, tmp_path: Path) -> None:
    md_file = tmp_path / "a.md"
    _write(md_file, content)
    assert lint_docs.check_retired_property_spec_spellings(md_file, content) == []


def test_check_retired_property_spec_spellings_reports_file_and_line(tmp_path: Path) -> None:
    md_file = tmp_path / "a.md"
    content = "# A\n\nIntro prose.\n\n```python\nSPEC = {\n    DefaultOptionKeys.strict_validation: True,\n}\n```\n"
    _write(md_file, content)
    errors = lint_docs.check_retired_property_spec_spellings(md_file, content)
    assert len(errors) == 1
    assert errors[0].startswith(f"{md_file}:7:")


# Raw dict PROPERTY_MAPPING values, paired with the 1-based line the single error must point at.
# A raw dict value now raises ValueError at class definition, so the error is reported where the dict opens.
FLAGGED_RAW_DICT_SNIPPETS = [
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "operation_type": {\n'
        '        "explanation": "Arithmetic operation",\n'
        "    },\n"
        "}\n"
        "```\n",
        5,
        id="raw_dict_spec",
    ),
    # A raw dict value written entirely on the PROPERTY_MAPPING opener line is still a raw dict.
    pytest.param(
        '# A\n\n```python\nPROPERTY_MAPPING = {"operation_type": {"explanation": "Arithmetic operation"}}\n```\n',
        4,
        id="single_line_raw_dict",
    ),
    # The first entry may open on the opener line and close later; it must still be checked.
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        'PROPERTY_MAPPING = {"operation_type": {\n'
        '    "explanation": "Arithmetic operation",\n'
        "}}\n"
        "```\n",
        4,
        id="raw_dict_opened_on_mapping_line",
    ),
    # An unbalanced paren inside a string literal must not hide the following raw dict entry.
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "a": property_spec("mismatched ( paren in text"),\n'
        '    "b": {"explanation": "raw"},\n'
        "}\n"
        "```\n",
        6,
        id="raw_dict_after_paren_in_string",
    ),
    # An unbalanced brace inside a string literal must not hide the following raw dict entry.
    pytest.param(
        "# A\n"
        "\n"
        "```python\n"
        "PROPERTY_MAPPING = {\n"
        '    "a": property_spec(r"pattern with a { brace"),\n'
        '    "b": {"explanation": "raw"},\n'
        "}\n"
        "```\n",
        6,
        id="raw_dict_after_brace_in_string",
    ),
]


@pytest.mark.parametrize(("content", "expected_line"), FLAGGED_RAW_DICT_SNIPPETS)
def test_check_retired_property_spec_spellings_flags_raw_dict_value(
    content: str, expected_line: int, tmp_path: Path
) -> None:
    md_file = tmp_path / "a.md"
    _write(md_file, content)
    errors = lint_docs.check_retired_property_spec_spellings(md_file, content)
    assert len(errors) == 1
    assert errors[0].startswith(f"{md_file}:{expected_line}:")
    assert "property_spec" in errors[0]


def test_check_retired_property_spec_spellings_scans_py_alias_fence(tmp_path: Path) -> None:
    """```py is a python fence too, so both checks still apply inside it."""
    md_file = tmp_path / "a.md"
    content = (
        "# A\n"
        "\n"
        "```py\n"
        "PROPERTY_MAPPING = {\n"
        '    "operation_type": {"explanation": "Arithmetic operation"},\n'
        "}\n"
        "key = DefaultOptionKeys.allowed_values\n"
        "```\n"
    )
    _write(md_file, content)
    errors = lint_docs.check_retired_property_spec_spellings(md_file, content)
    assert len(errors) == 2
    assert any(err.startswith(f"{md_file}:5:") and "property_spec" in err for err in errors)
    assert any("DefaultOptionKeys.allowed_values" in err for err in errors)


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
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path)
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "DefaultOptionKeys.allowed_values" in out


def test_main_flags_broken_link_in_top_level_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A broken relative .md link in a repo-root file must fail the lint gate."""
    docs = tmp_path / "docs"
    guides = docs / "guides"
    _write(guides / "index.md", "# Root\n")
    _write(
        tmp_path / "README.md",
        "# README\n\n[Guide](docs/guides/index.md)\n\n[Missing](missing.md)\n",
    )
    monkeypatch.setattr(lint_docs, "DOCS_DIR", docs)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", guides)
    monkeypatch.setattr(lint_docs, "REPO_ROOT", tmp_path)
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "broken link" in out
    assert "missing.md" in out


def test_main_flags_bare_fence_in_top_level_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare fenced-code opener in a repo-root file must fail the lint gate."""
    docs = tmp_path / "docs"
    guides = docs / "guides"
    _write(guides / "index.md", "# Root\n")
    _write(tmp_path / "README.md", "# README\n\n```\nplain\n```\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", docs)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", guides)
    monkeypatch.setattr(lint_docs, "REPO_ROOT", tmp_path)
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "bare fenced-code opener" in out


def test_main_orphan_check_ignores_top_level_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Orphan reachability is guides-only; root files need no inbound links."""
    docs = tmp_path / "docs"
    guides = docs / "guides"
    _write(guides / "index.md", "# Root\n")
    _write(tmp_path / "notes.md", "# Notes\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", docs)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", guides)
    monkeypatch.setattr(lint_docs, "REPO_ROOT", tmp_path)
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "orphan" not in out


def test_main_reads_markdown_as_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-ASCII UTF-8 in guides must not crash on Windows' GBK locale."""
    docs = tmp_path / "docs"
    guides = docs / "guides"
    _write(guides / "index.md", "# Root\n\n[Guide](guide.md)\n")
    _write(guides / "guide.md", "# Guide\n\n```text\n\u251c\u2500\u2500 plugin\n```\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", docs)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", guides)
    monkeypatch.setattr(lint_docs, "REPO_ROOT", tmp_path)
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    assert rc == 0


def test_missing_root_index_uses_relative_path(tmp_path: Path) -> None:
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert len(errors) == 1
    assert str(tmp_path) not in errors[0]


# check_source_property_mapping: same checks as check_retired_property_spec_spellings, applied
# directly to a whole .py file's content (no markdown fence extraction, offset always 0).


def test_plugin_source_dir_is_repo_mloda_root() -> None:
    """PLUGIN_SOURCE_DIR is the mloda/ package root scanned for source-level checks."""
    assert lint_docs.PLUGIN_SOURCE_DIR == _REPO_ROOT / "mloda"
    assert lint_docs.PLUGIN_SOURCE_DIR.is_dir()


def test_check_source_property_mapping_flags_raw_dict_value(tmp_path: Path) -> None:
    py_file = tmp_path / "base.py"
    content = (
        'PROPERTY_MAPPING = {\n    "operation_type": {\n        "explanation": "Arithmetic operation",\n    },\n}\n'
    )
    _write(py_file, content)
    errors = lint_docs.check_source_property_mapping(py_file, content)
    assert len(errors) == 1
    # The nested dict literal opens on line 2; the error must point there, not at the
    # PROPERTY_MAPPING assignment line.
    assert errors[0].startswith(f"{py_file}:2:")
    assert "property_spec" in errors[0]


def test_check_source_property_mapping_accepts_property_spec_values(tmp_path: Path) -> None:
    """No false positive when every PROPERTY_MAPPING value is a property_spec(...) call."""
    py_file = tmp_path / "base.py"
    content = (
        "PROPERTY_MAPPING = {\n"
        '    "operation_type": property_spec(\n'
        '        "Arithmetic operation",\n'
        "        strict=True,\n"
        '        allowed_values={"add": "Addition", "sub": "Subtraction"},\n'
        '        default="add",\n'
        "    ),\n"
        "}\n"
    )
    _write(py_file, content)
    assert lint_docs.check_source_property_mapping(py_file, content) == []


def test_check_source_property_mapping_flags_retired_option_key(tmp_path: Path) -> None:
    """The retired-spelling check is exhaustively field-tested against RETIRED_SPEC_FIELDS for
    check_retired_property_spec_spellings above; check_source_property_mapping delegates to the
    same underlying walk, so one representative field is enough to prove the delegation works."""
    py_file = tmp_path / "base.py"
    content = "key = DefaultOptionKeys.allowed_values\n"
    _write(py_file, content)
    errors = lint_docs.check_source_property_mapping(py_file, content)
    assert len(errors) == 1
    assert "DefaultOptionKeys.allowed_values" in errors[0]
    assert "retired" in errors[0].lower()
    assert errors[0].startswith(f"{py_file}:1:")


def test_check_source_property_mapping_accepts_surviving_option_key(tmp_path: Path) -> None:
    """``context`` still exists on DefaultOptionKeys; see the note on the retired-key test above
    for why one representative field suffices here."""
    py_file = tmp_path / "base.py"
    content = "key = DefaultOptionKeys.context\n"
    _write(py_file, content)
    assert lint_docs.check_source_property_mapping(py_file, content) == []


def test_check_source_property_mapping_returns_empty_for_syntax_error(tmp_path: Path) -> None:
    """A .py file that fails to parse yields no findings; unlike the markdown-fence path, there is no
    textual fallback scan. The fixture carries a retired DefaultOptionKeys spelling that a textual
    fallback (``_textual_retired_option_keys``) would catch if it were wired in, so this actually
    distinguishes the two behaviors rather than passing regardless."""
    py_file = tmp_path / "broken.py"
    content = "if True\n    x = DefaultOptionKeys.explanation\n"
    _write(py_file, content)
    assert lint_docs.check_source_property_mapping(py_file, content) == []


def test_main_reports_raw_dict_in_plugin_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    _write(docs / "index.md", "# Root\n")
    plugin_src = tmp_path / "plugin_src"
    _write(
        plugin_src / "a" / "b" / "base.py",
        'PROPERTY_MAPPING = {\n    "operation_type": {\n        "explanation": "Arithmetic operation",\n    },\n}\n',
    )
    monkeypatch.setattr(lint_docs, "DOCS_DIR", docs)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", docs)
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", plugin_src)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "raw dict PROPERTY_MAPPING" in out
    assert "base.py" in out


def test_check_source_property_mapping_has_no_findings_in_real_plugin_source() -> None:
    """Regression guard: production PROPERTY_MAPPING declarations are property_spec(...) calls,
    not raw dicts, and plugin source uses no retired DefaultOptionKeys spelling."""
    findings = [
        error
        for py_file in sorted(lint_docs.PLUGIN_SOURCE_DIR.rglob("*.py"))
        for error in lint_docs.check_source_property_mapping(py_file, py_file.read_text(encoding="utf-8"))
    ]
    assert findings == []


# check_readme_plugin_table: published data-operation packages must appear in the README Plugins table.

_DATA_OP_PACKAGES_TOML = """
[packages.mloda-community-data-operations]
path = "mloda/community/feature_groups/data_operations"
published = true

[packages.mloda-community-aggregation]
path = "mloda/community/feature_groups/data_operations/aggregation"
published = true

[packages.mloda-community-rank]
path = "mloda/community/feature_groups/data_operations/row_preserving/rank"
published = true

[packages.mloda-community-unreleased]
path = "mloda/community/feature_groups/data_operations/row_preserving/unreleased"
"""


def test_check_readme_plugin_table_accepts_complete_table(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    _write(readme, "## Plugins\n\n`mloda-community-aggregation`\n`mloda-community-rank`\n\n## PyPI packages\n")
    packages_config = tmp_path / "packages.toml"
    _write(packages_config, _DATA_OP_PACKAGES_TOML)
    assert lint_docs.check_readme_plugin_table(readme, packages_config) == []


def test_check_readme_plugin_table_flags_missing_package(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    _write(readme, "## Plugins\n\n`mloda-community-aggregation`\n\n## PyPI packages\n")
    packages_config = tmp_path / "packages.toml"
    _write(packages_config, _DATA_OP_PACKAGES_TOML)
    errors = lint_docs.check_readme_plugin_table(readme, packages_config)
    assert len(errors) == 1
    assert "mloda-community-rank" in errors[0]


def test_check_readme_plugin_table_excludes_base_and_unpublished_packages(tmp_path: Path) -> None:
    """The shared data-operations base and an unpublished package must never be required in the table."""
    readme = tmp_path / "README.md"
    _write(readme, "## Plugins\n\n`mloda-community-aggregation`\n`mloda-community-rank`\n\n## PyPI packages\n")
    packages_config = tmp_path / "packages.toml"
    _write(packages_config, _DATA_OP_PACKAGES_TOML)
    assert lint_docs.check_readme_plugin_table(readme, packages_config) == []


def test_check_readme_plugin_table_ignores_matches_outside_plugins_section(tmp_path: Path) -> None:
    """A package name mentioned elsewhere in the README does not satisfy the Plugins table requirement."""
    readme = tmp_path / "README.md"
    _write(
        readme,
        "## Plugins\n\n`mloda-community-aggregation`\n\n## PyPI packages\n\n`mloda-community-rank`\n",
    )
    packages_config = tmp_path / "packages.toml"
    _write(packages_config, _DATA_OP_PACKAGES_TOML)
    errors = lint_docs.check_readme_plugin_table(readme, packages_config)
    assert len(errors) == 1
    assert "mloda-community-rank" in errors[0]


def test_check_readme_plugin_table_real_repo_readme_is_complete() -> None:
    """Regression guard: the real README Plugins table stays in sync with published data-operation packages."""
    assert lint_docs.check_readme_plugin_table(lint_docs.README_PATH, lint_docs.PACKAGES_CONFIG) == []


def test_main_flags_readme_missing_published_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(tmp_path / "guides" / "index.md", "# Guides\n")
    _write(tmp_path / "README.md", "## Plugins\n\n`mloda-community-aggregation`\n\n## PyPI packages\n")
    _write(tmp_path / "packages.toml", _DATA_OP_PACKAGES_TOML)
    monkeypatch.setattr(lint_docs, "DOCS_DIR", tmp_path / "guides")
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", tmp_path / "guides")
    monkeypatch.setattr(lint_docs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lint_docs, "README_PATH", tmp_path / "README.md")
    monkeypatch.setattr(lint_docs, "PACKAGES_CONFIG", tmp_path / "packages.toml")
    empty_plugin_source = tmp_path / "empty_plugin_source"
    empty_plugin_source.mkdir()
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", empty_plugin_source)
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "mloda-community-rank" in out


def test_main_reports_missing_plugin_source_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    _write(docs / "guides" / "index.md", "# Root\n")
    monkeypatch.setattr(lint_docs, "DOCS_DIR", docs)
    monkeypatch.setattr(lint_docs, "GUIDES_DIR", docs / "guides")
    monkeypatch.setattr(lint_docs, "PLUGIN_SOURCE_DIR", tmp_path / "does_not_exist")
    rc = lint_docs.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "Plugin source directory not found" in out
