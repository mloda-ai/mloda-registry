"""Repository guard for GitHub issue #267.

mloda 0.9.0 made "named-form non-mapping credential values" raise
``ValueError`` at construction: a ``credentials=`` (or ``add_credentials(...)``)
mapping whose VALUE is a bare quoted-string literal instead of a nested mapping
``{...}`` or a ``Credential(...)``. So ``credentials={'prod': 'dsn-string'}``
now raises, while ``credentials={'prod': {'dsn': 'dsn-string'}}``,
``credentials={'pg-prod': Credential(host='h')}``,
``credentials=Credential(dsn='dsn-string')`` and list forms stay valid. This
guard flags a bare quoted-string value at any position of the credentials
mapping (non-string scalar values such as ints are out of scope). The registry
is currently clean; this trips loudly if such a usage ever reappears.

This is a best-effort single-line heuristic and can miss a reintroduction split
across lines or bound to an intermediate variable.
"""

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


_SKIP_DIRS = {"__pycache__", "site-packages", "node_modules", "build", "dist"}

# A credentials mapping opened right after a marker: ``credentials={`` or
# ``add_credentials({``. The captured ``{`` opens the mapping at depth 1; a
# top-level-aware scan then inspects each value for a bare quoted string.
_MARKER_RE = re.compile(r"(?:credentials\s*=|add_credentials\s*\()\s*\{")


def _mapping_has_top_level_string_value(line: str, open_idx: int) -> bool:
    """Scan the credentials mapping opened at ``line[open_idx] == '{'``.

    Returns True when any top-level (depth-1) value in the mapping is a bare
    quoted-string literal. Tracks brace/bracket/paren depth and quoted strings
    so nested ``{...}`` pairs, ``Credential(...)`` and list-form values are
    ignored, and stops at the matching closing ``}`` of the mapping.
    """
    depth = 0
    quote: str | None = None
    expect_value = False  # True right after a top-level ':' until its value starts
    i = open_idx
    n = len(line)
    while i < n:
        ch = line[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            if expect_value and depth == 1:
                return True
            quote = ch
            expect_value = False
        elif ch in "{[(":
            if expect_value and depth == 1:
                expect_value = False
            depth += 1
        elif ch in "}])":
            depth -= 1
            if depth == 0:
                return False
        elif ch == ":" and depth == 1:
            expect_value = True
        elif ch == "," and depth == 1:
            expect_value = False
        elif expect_value and depth == 1 and not ch.isspace():
            expect_value = False  # value starts with an identifier/number, not a string
        i += 1
    return False


def _line_flags_named_form_string(line: str) -> bool:
    """Return True when the line opens a credentials mapping with a bare top-level string value."""
    for m in _MARKER_RE.finditer(line):
        if _mapping_has_top_level_string_value(line, m.end() - 1):
            return True
    return False


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
        if any(part.startswith(".") or part in _SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
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
            if _line_flags_named_form_string(line):
                hits.append(f"{rel.as_posix()}:{i + 1}: {line.strip()}")
    return sorted(hits)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


# Single lines of ``m.py`` whose named-form credential value is a bare string.
_FLAGGED_LINES = [
    pytest.param("credentials={'prod': 'dsn-string'}\n", id="single_quoted_string"),
    pytest.param('credentials={"prod": "dsn-string"}\n', id="double_quoted_string"),
    pytest.param("collection.add_credentials({'prod': 'dsn-string'})\n", id="add_credentials_string"),
    # Every entry is inspected, not only the first: here the first value is a legal nested mapping.
    pytest.param("credentials={'prod': {'dsn': 'ok'}, 'staging': 'dsn-string'}\n", id="multi_entry_later_string"),
]

# Single lines of ``m.py`` whose credential form is legal, so the guard must leave them alone.
_ACCEPTED_LINES = [
    pytest.param("credentials={'prod': {'dsn': 'dsn-string'}}\n", id="nested_mapping_value"),
    pytest.param("credentials={'pg-prod': Credential(host='h')}\n", id="credential_object_value"),
    pytest.param("credentials={'prod': {'dsn': 'x'}, 'staging': {'dsn': 'y'}}\n", id="multi_entry_all_mappings"),
    pytest.param(
        "credentials={'prod': Credential(host='h'), 'staging': Credential(host='h2')}\n",
        id="multi_entry_all_credentials",
    ),
    # Auto-named form: the string is a Credential kwarg, not a mapping value.
    pytest.param("credentials=Credential(dsn='dsn-string')\n", id="bare_credential_object"),
    pytest.param("credentials=[Credential(host='h'), {'host': 'h2'}]\n", id="list_form"),
]


@pytest.mark.parametrize("line", _FLAGGED_LINES)
def test_named_form_string_value_flagged(line: str, tmp_path: Path) -> None:
    _write(tmp_path / "m.py", line)
    hits = find_named_form_string_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "m.py" in hits[0]


@pytest.mark.parametrize("line", _ACCEPTED_LINES)
def test_valid_credential_form_not_flagged(line: str, tmp_path: Path) -> None:
    _write(tmp_path / "m.py", line)
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


def test_build_artifact_dirs_not_scanned(tmp_path: Path) -> None:
    """A flaggable file under a build artifact directory is not scanned."""
    _write(tmp_path / "build" / "lib" / "m.py", "credentials={'prod': 'dsn-string'}\n")
    _write(tmp_path / "dist" / "m.py", "credentials={'prod': 'dsn-string'}\n")
    _write(tmp_path / "pkg.egg-info" / "m.py", "credentials={'prod': 'dsn-string'}\n")
    assert find_named_form_string_credential_usages(tmp_path) == []


def test_repo_root_is_clean() -> None:
    """The real repository must currently have zero named-form string credential usages."""
    offenders = find_named_form_string_credential_usages(_REPO_ROOT)
    assert offenders == [], "Named-form string credential usage reappeared:\n" + "\n".join(offenders)
