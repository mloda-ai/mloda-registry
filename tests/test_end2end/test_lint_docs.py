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


def test_link_text_with_brackets_still_parsed(tmp_path: Path) -> None:
    """Lock in the grammar shift: text uses [^]]* so a nested ] in text breaks the match.

    A link like ``[a [nested] label](foo.md)`` is parsed as if the link target is whatever
    follows the last ``]`` in the prefix. With our regex, ``[a [nested]`` matches as the
    text portion, then the body must immediately be ``(``, but the next char is ` ` so
    the match fails — foo.md ends up unreachable. Document the limitation in a test.
    """
    _write(tmp_path / "index.md", "# Root\n\n[a [nested] label](foo.md)\n")
    _write(tmp_path / "foo.md", "# Foo\n")
    errors = lint_docs.find_orphan_guides(tmp_path)
    assert any("foo.md" in err for err in errors)


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
