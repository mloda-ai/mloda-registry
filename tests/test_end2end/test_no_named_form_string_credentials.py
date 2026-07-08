"""Repository guard for GitHub issue #267.

mloda 0.9.0 made "named-form non-mapping credential values" raise
``ValueError`` at construction: a ``credentials=`` (or ``add_credentials(...)``)
mapping whose VALUE is a bare scalar/string literal instead of a nested mapping
``{...}`` or a ``Credential(...)``. So ``credentials={'prod': 'dsn-string'}``
now raises, while ``credentials={'prod': {'dsn': 'dsn-string'}}``,
``credentials={'pg-prod': Credential(host='h')}``,
``credentials=Credential(dsn='dsn-string')`` and list forms stay valid. This
guard flags the offending named-string shape. The registry is currently clean;
this trips loudly if such a usage ever reappears.

This is a best-effort text/proximity heuristic and can miss a reintroduction
split across lines or bound to an intermediate variable.
"""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


_SKIP_DIRS = {"__pycache__", "site-packages", "node_modules"}

# Named-form mapping opened right after a credentials marker whose first value is a
# bare quoted literal: ``credentials={'k': '...`` or ``add_credentials({'k': '...``.
# Anchoring to the marker avoids matching a nested inner mapping value.
_NAMED_FORM_STRING_RE = re.compile(r"(?:credentials\s*=|add_credentials\s*\()\s*\{\s*['\"][^'\"]*['\"]\s*:\s*['\"]")


def _in_scope_indices(path: Path, lines: list[str]) -> set[int]:
    """Return the set of line indices whose content is in scope for marker matching.

    For ``.py`` files every line is in scope. For ``.md`` files only lines inside
    triple-backtick fenced code blocks count (prose is ignored); the fence
    delimiter lines themselves are excluded. Mirrors scripts/lint_docs.py.
    """
    if path.suffix != ".md":
        return set(range(len(lines)))
    scope: set[int] = set()
    in_block = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            scope.add(i)
    return scope


def find_named_form_string_credential_usages(root: Path) -> list[str]:
    """Return "relpath:lineno: line" for every named-form string credential value under root."""
    hits: list[str] = []
    for path in list(root.rglob("*.py")) + list(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts):
            continue
        if path.name == "test_no_named_form_string_credentials.py":
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        scope = _in_scope_indices(path, lines)
        for i, line in enumerate(lines):
            if i not in scope:
                continue
            if _NAMED_FORM_STRING_RE.search(line):
                hits.append(f"{rel.as_posix()}:{i + 1}: {line.strip()}")
    return sorted(hits)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_named_form_single_quoted_string_flagged(tmp_path: Path) -> None:
    """A single-quoted named-form string credential value is flagged."""
    _write(tmp_path / "m.py", "credentials={'prod': 'dsn-string'}\n")
    hits = find_named_form_string_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "m.py" in hits[0]


def test_named_form_double_quoted_string_flagged(tmp_path: Path) -> None:
    """A double-quoted named-form string credential value is flagged."""
    _write(tmp_path / "m.py", 'credentials={"prod": "dsn-string"}\n')
    hits = find_named_form_string_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "m.py" in hits[0]


def test_add_credentials_named_form_string_flagged(tmp_path: Path) -> None:
    """A named-form string value passed to add_credentials is flagged."""
    _write(tmp_path / "m.py", "collection.add_credentials({'prod': 'dsn-string'})\n")
    hits = find_named_form_string_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "m.py" in hits[0]


def test_nested_mapping_value_not_flagged(tmp_path: Path) -> None:
    """A named-form value that is a nested mapping is not flagged."""
    _write(tmp_path / "m.py", "credentials={'prod': {'dsn': 'dsn-string'}}\n")
    assert find_named_form_string_credential_usages(tmp_path) == []


def test_named_form_credential_object_value_not_flagged(tmp_path: Path) -> None:
    """A named-form value that is a Credential(...) is not flagged."""
    _write(tmp_path / "m.py", "credentials={'pg-prod': Credential(host='h')}\n")
    assert find_named_form_string_credential_usages(tmp_path) == []


def test_bare_credential_object_not_flagged(tmp_path: Path) -> None:
    """A bare auto-named Credential(...) with a string kwarg is not flagged."""
    _write(tmp_path / "m.py", "credentials=Credential(dsn='dsn-string')\n")
    assert find_named_form_string_credential_usages(tmp_path) == []


def test_list_form_not_flagged(tmp_path: Path) -> None:
    """A list-form credentials value is not flagged."""
    _write(tmp_path / "m.py", "credentials=[Credential(host='h'), {'host': 'h2'}]\n")
    assert find_named_form_string_credential_usages(tmp_path) == []


def test_guard_module_self_excluded(tmp_path: Path) -> None:
    """A file named like this guard module is skipped even with a flaggable pattern."""
    _write(tmp_path / "test_no_named_form_string_credentials.py", "credentials={'prod': 'dsn-string'}\n")
    assert find_named_form_string_credential_usages(tmp_path) == []


def test_md_fenced_named_form_string_flagged(tmp_path: Path) -> None:
    """A flaggable named-form string credential line inside a fenced block in .md is flagged."""
    body = "Migration example:\n\n```python\ncredentials={'prod': 'dsn-string'}\n```\n"
    _write(tmp_path / "guide.md", body)
    hits = find_named_form_string_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "guide.md" in hits[0]


def test_md_prose_named_form_string_not_flagged(tmp_path: Path) -> None:
    """A named-form string credential mention only in .md prose (no fenced block) is not flagged."""
    body = "Previously credentials={'prod': 'dsn-string'} was allowed; now use a nested mapping.\n"
    _write(tmp_path / "guide.md", body)
    assert find_named_form_string_credential_usages(tmp_path) == []


def test_repo_root_is_clean() -> None:
    """The real repository must currently have zero named-form string credential usages."""
    offenders = find_named_form_string_credential_usages(_REPO_ROOT)
    assert offenders == [], "Named-form string credential usage reappeared:\n" + "\n".join(offenders)
