"""Lint documentation (docs/ and top-level markdown) for broken relative links and internal imports.

Run: python scripts/lint_docs.py
Exit code: 1 if any issues found, 0 otherwise.
"""

import ast
import functools
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# The orphan BFS is rooted at the guides index; docs-root files are linted but not part of that graph.
GUIDES_DIR = DOCS_DIR / "guides"

PLUGIN_SOURCE_DIR = REPO_ROOT / "mloda"

README_PATH = REPO_ROOT / "README.md"

PACKAGES_CONFIG = REPO_ROOT / "config" / "packages.toml"

# Trailing slash excludes the shared base package itself, whose path has no nested segment.
DATA_OPERATIONS_PATH_PREFIX = "mloda/community/feature_groups/data_operations/"

PLUGINS_SECTION_RE = re.compile(r"^## Plugins\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)

INTERNAL_IMPORT_RE = re.compile(r"from mloda\.core\.")

MD_LINK_RE = re.compile(r"\[.*?\]\((?!https?://|mailto:)([^)#\s]+\.md)(?:#([^)\s]+))?\)")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

CODE_BLOCK_RE = re.compile(r"^```", re.MULTILINE)

# Spec fields DefaultOptionKeys must not be used for: six are gone from the current core, and ``explanation``
# was never a member at all. ``context``, ``group`` and ``in_features`` survive and must never be flagged.
RETIRED_SPEC_FIELDS = frozenset(
    {
        "explanation",
        "allowed_values",
        "default",
        "strict_validation",
        "element_validator",
        "required_when",
        "match_guard",
    }
)

RETIRED_OPTION_KEY_RE = re.compile(rf"\bDefaultOptionKeys\.(?:{'|'.join(sorted(RETIRED_SPEC_FIELDS))})\b")

PROPERTY_MAPPING_NAME = "PROPERTY_MAPPING"

OPTION_KEYS_NAME = "DefaultOptionKeys"

PYTHON_FENCE_TAGS = frozenset({"python", "py"})

INDEX_FILENAME = "index.md"


def _strip_code_blocks(content: str) -> str:
    """Remove fenced code blocks from markdown content."""
    return "".join(CODE_BLOCK_RE.split(content)[::2])


def _fenced_ranges(content: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of fenced code blocks in content."""
    fences = [m.start() for m in CODE_BLOCK_RE.finditer(content)]
    ranges: list[tuple[int, int]] = []
    for i in range(0, len(fences) - 1, 2):
        ranges.append((fences[i], fences[i + 1]))
    return ranges


def _fence_bodies(content: str) -> list[tuple[int, str]]:
    """Return (opener line number, body) for every fenced code block.

    The opener line number doubles as the offset: a body line ``n`` is file line
    ``offset + n``.
    """
    blocks: list[tuple[int, str]] = []
    opener_lineno = 0
    body: list[str] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if line.startswith("```"):
            if opener_lineno:
                blocks.append((opener_lineno, "\n".join(body)))
                opener_lineno = 0
                body = []
            else:
                opener_lineno = lineno
            continue
        if opener_lineno:
            body.append(line)
    if opener_lineno:
        blocks.append((opener_lineno, "\n".join(body)))
    return blocks


def _slugify_heading(text: str) -> str:
    """GFM-style slug; ignores duplicate-heading disambiguation and inline formatting beyond backticks."""
    text = text.lower().replace("`", "")
    text = text.replace(" ", "-")
    return re.sub(r"[^\w-]", "", text)


@functools.lru_cache(maxsize=None)
def _heading_slugs(md_file: Path) -> frozenset[str]:
    slugs: set[str] = set()
    content = _strip_code_blocks(md_file.read_text(encoding="utf-8"))
    for match in HEADING_RE.finditer(content):
        slugs.add(_slugify_heading(match.group(2)))
    return frozenset(slugs)


def check_relative_links_and_anchors(md_file: Path, content: str) -> list[str]:
    """Validate relative markdown links and their optional anchor fragments."""
    errors = []
    fenced = _fenced_ranges(content)
    for match in MD_LINK_RE.finditer(content):
        start = match.start()
        if any(lo <= start < hi for lo, hi in fenced):
            continue
        rel_path = match.group(1)
        anchor = match.group(2)
        target = (md_file.parent / rel_path).resolve()
        line_num = content[:start].count("\n") + 1
        if not target.exists():
            errors.append(f"{md_file}:{line_num}: broken link -> {rel_path}")
            continue
        if anchor:
            slug = _slugify_heading(anchor)
            if slug not in _heading_slugs(target):
                errors.append(f"{md_file}:{line_num}: broken anchor -> {rel_path}#{anchor}")
    return errors


def check_internal_imports(md_file: Path, content: str) -> list[str]:
    errors = []
    for opener_lineno, body in _fence_bodies(content):
        for lineno, line in enumerate(body.splitlines(), start=1):
            stripped = line.strip()
            if INTERNAL_IMPORT_RE.search(stripped):
                errors.append(f"{md_file}:{opener_lineno + lineno}: internal import in code snippet -> {stripped}")
    return errors


def _load_packages_config(packages_config: Path) -> dict[str, dict[str, Any]]:
    with open(packages_config, "rb") as f:
        data = tomllib.load(f)
    packages: dict[str, dict[str, Any]] = data.get("packages", {})
    return packages


def _published_data_operation_packages(packages: dict[str, dict[str, Any]]) -> list[str]:
    """Published data-operation plugin packages, excluding the shared base package itself."""
    names = [
        name
        for name, cfg in packages.items()
        if cfg.get("published") and cfg.get("path", "").startswith(DATA_OPERATIONS_PATH_PREFIX)
    ]
    return sorted(names)


def _table_rows(section: str) -> str:
    """Join the section's table-row lines (those starting with ``|``), so prose mentions don't count."""
    return "\n".join(line for line in section.splitlines() if line.lstrip().startswith("|"))


def check_readme_plugin_table(readme_path: Path, packages_config: Path) -> list[str]:
    """Flag published data-operation packages missing from the README Plugins table."""
    if not readme_path.is_file():
        return [f"{readme_path}: README not found for Plugins table check"]
    if not packages_config.is_file():
        return [f"{packages_config}: packages config not found for Plugins table check"]
    names = _published_data_operation_packages(_load_packages_config(packages_config))
    match = PLUGINS_SECTION_RE.search(readme_path.read_text(encoding="utf-8"))
    rows = _table_rows(match.group(1)) if match else ""
    return [
        f"{readme_path}: Plugins table missing published package `{name}`" for name in names if f"`{name}`" not in rows
    ]


def _collect_linked_md(md_file: Path, content: str) -> set[Path]:
    """Return the set of .md files linked from md_file (resolved absolute paths).

    Links inside fenced code blocks are ignored so illustrative snippets do not
    fabricate reachability edges.
    """
    linked: set[Path] = set()
    stripped = _strip_code_blocks(content)
    for match in MD_LINK_RE.finditer(stripped):
        target = match.group(1)
        resolved = (md_file.parent / target).resolve()
        linked.add(resolved)
    return linked


def find_orphan_guides(docs_dir: Path, contents: dict[Path, str] | None = None) -> list[str]:
    """Flag any .md file under docs_dir that is unreachable from docs_dir/index.md.

    Reachability is transitive via inline markdown links. Only the root ``index.md``
    is exempt from the inbound-link check (it is the BFS source); subdirectory
    ``index.md`` files must themselves be linked from somewhere reachable.

    Links to files outside ``docs_dir`` or to missing targets are silently skipped
    here; the latter are surfaced by ``check_relative_links_and_anchors``.
    """
    errors = []
    docs_root = docs_dir.resolve()
    root_index = docs_dir / INDEX_FILENAME
    if not root_index.is_file():
        return [f"{docs_dir.name}/{INDEX_FILENAME}: missing root index for orphan check"]

    def _read(path: Path) -> str:
        if contents is not None and path in contents:
            return contents[path]
        return path.read_text(encoding="utf-8")

    root_resolved = root_index.resolve()
    reachable: set[Path] = {root_resolved}
    frontier: deque[Path] = deque([root_resolved])
    while frontier:
        current = frontier.popleft()
        for linked in _collect_linked_md(current, _read(current)):
            if linked in reachable:
                continue
            if not linked.is_file():
                continue
            try:
                linked.relative_to(docs_root)
            except ValueError:
                continue
            reachable.add(linked)
            frontier.append(linked)

    for md_file in sorted(docs_dir.rglob("*.md")):
        resolved = md_file.resolve()
        if resolved == root_resolved:
            continue
        if resolved not in reachable:
            rel = md_file.relative_to(docs_dir)
            errors.append(f"{rel}: orphan guide not reachable from {INDEX_FILENAME}")
    return errors


def check_bare_fence_openers(md_file: Path, content: str) -> list[str]:
    """Flag fenced-code openers (```) that lack a language tag.

    GitHub renders bare openers as plain monospace with no syntax highlighting.
    Use `text` for ASCII trees/diagrams when no real language applies.

    Known limitations: does not handle indented fences (CommonMark allows up to
    3 leading spaces), tilde fences (``~~~``), or nested-fence weirdness inside
    the same delimiter.
    """
    errors = []
    in_block = False
    for lineno, line in enumerate(content.splitlines(), start=1):
        if line.startswith("```"):
            opening = not in_block
            in_block = not in_block
            # Trailing whitespace alone does not qualify as a language tag, so rstrip before comparing.
            if opening and line.rstrip() == "```":
                errors.append(
                    f"{md_file}:{lineno}:1: bare fenced-code opener (missing language tag)"
                    " - use ```text for plain blocks"
                )
    return errors


def _python_fences(content: str) -> list[tuple[int, str]]:
    """Return (opener line number, body) for every ```python / ```py fence.

    The opener line number doubles as the offset: a body line ``n`` is file line ``offset + n``.
    """
    fences: list[tuple[int, str]] = []
    opener_lineno = 0
    is_python = False
    body: list[str] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if line.startswith("```"):
            if opener_lineno:
                if is_python:
                    fences.append((opener_lineno, "\n".join(body)))
                opener_lineno = 0
                body = []
            else:
                opener_lineno = lineno
                info = line[3:].strip().lower().split()
                is_python = bool(info) and info[0] in PYTHON_FENCE_TAGS
            continue
        if opener_lineno:
            body.append(line)
    if opener_lineno and is_python:
        fences.append((opener_lineno, "\n".join(body)))
    return fences


def _parse_python(source: str) -> ast.Module | None:
    """Parse a fence body; None when it is a fragment rather than a whole module."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _property_mapping_value(node: ast.AST) -> ast.Dict | None:
    """Return the dict literal assigned to PROPERTY_MAPPING by node, if that is what node does."""
    targets: list[ast.expr]
    value: ast.expr | None
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign):
        targets, value = [node.target], node.value
    else:
        return None
    if not isinstance(value, ast.Dict):
        return None
    if any(isinstance(target, ast.Name) and target.id == PROPERTY_MAPPING_NAME for target in targets):
        return value
    return None


def _retired_key_error(path: Path, line: int, spelling: str) -> tuple[int, str]:
    return line, f"{path}:{line}: retired spelling {spelling} (use property_spec(...) instead)"


def _raw_dict_error(path: Path, line: int) -> tuple[int, str]:
    return line, f"{path}:{line}: raw dict PROPERTY_MAPPING value (use property_spec(...) instead)"


def _parsed_fence_errors(path: Path, offset: int, tree: ast.Module) -> list[tuple[int, str]]:
    """Collect (file line, message) pairs from a parsed fence."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in RETIRED_SPEC_FIELDS
            and isinstance(node.value, ast.Name)
            and node.value.id == OPTION_KEYS_NAME
        ):
            spelling = f"{OPTION_KEYS_NAME}.{node.attr}"
            found.append(_retired_key_error(path, offset + node.lineno, spelling))
        mapping = _property_mapping_value(node)
        if mapping is None:
            continue
        for value in mapping.values:
            if isinstance(value, ast.Dict):
                found.append(_raw_dict_error(path, offset + value.lineno))
    return found


def _textual_retired_option_keys(md_file: Path, offset: int, source: str) -> list[tuple[int, str]]:
    """Fallback for fences that do not parse: the attribute spelling only, never the dict shape."""
    found: list[tuple[int, str]] = []
    for lineno, body_line in enumerate(source.splitlines(), start=1):
        for match in RETIRED_OPTION_KEY_RE.finditer(body_line):
            found.append(_retired_key_error(md_file, offset + lineno, match.group(0)))
    return found


def check_retired_property_spec_spellings(md_file: Path, content: str) -> list[str]:
    """Flag retired PROPERTY_MAPPING spellings (removed DefaultOptionKeys fields, raw dict values) in python fences.

    Each ```python / ```py fence is parsed with ``ast``, so string literals, comments and
    line breaks cannot fool either check. Other languages are documentation, not code, and
    are skipped.

    Known limitations:
    - tilde fences (``~~~``) and indented fences are not recognized, the same blind spot
      ``check_bare_fence_openers`` documents.
    - indirection hides a spec: ``PROPERTY_MAPPING = {"a": OP_SPEC}``, ``dict(explanation=...)``,
      or a mapping built under another name and only later assigned.
    - a fence that does not parse (a fragment) keeps the DefaultOptionKeys scan textually, so
      there the spelling is caught inside strings and comments too, and the dict shape not at all.
    """
    found: list[tuple[int, str]] = []
    for offset, source in _python_fences(content):
        tree = _parse_python(source)
        if tree is None:
            found.extend(_textual_retired_option_keys(md_file, offset, source))
            continue
        found.extend(_parsed_fence_errors(md_file, offset, tree))
    return [message for _, message in sorted(found, key=lambda item: item[0])]


def check_source_property_mapping(py_file: Path, content: str) -> list[str]:
    """Same checks as ``check_retired_property_spec_spellings``, applied directly to a whole .py file;
    unlike the markdown-fence path, a file that fails to parse yields no findings (no textual fallback)."""
    tree = _parse_python(content)
    if tree is None:
        return []
    found = _parsed_fence_errors(py_file, 0, tree)
    return [message for _, message in sorted(found, key=lambda item: item[0])]


def main() -> int:
    if not DOCS_DIR.is_dir():
        print(f"Docs directory not found: {DOCS_DIR}")
        return 1

    if not PLUGIN_SOURCE_DIR.is_dir():
        print(f"Plugin source directory not found: {PLUGIN_SOURCE_DIR}")
        return 1

    all_errors: list[str] = []
    link_errors: list[str] = []
    guides_link_errors: list[str] = []
    contents: dict[Path, str] = {}
    guides_root = GUIDES_DIR.resolve()

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        resolved = md_file.resolve()
        contents[resolved] = content
        file_link_errors = check_relative_links_and_anchors(md_file, content)
        link_errors.extend(file_link_errors)
        if resolved.is_relative_to(guides_root):
            guides_link_errors.extend(file_link_errors)
        all_errors.extend(check_internal_imports(md_file, content))
        all_errors.extend(check_bare_fence_openers(md_file, content))
        all_errors.extend(check_retired_property_spec_spellings(md_file, content))

    # Root markdown (README.md, CONTRIBUTING.md, ...) links into docs/ and at the
    # other root files, so it needs the same guard. Orphan reachability stays a
    # guides-only rule: root files need not be linked from docs/guides/index.md.
    for md_file in sorted(REPO_ROOT.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        link_errors.extend(check_relative_links_and_anchors(md_file, content))
        all_errors.extend(check_bare_fence_openers(md_file, content))

    for py_file in sorted(PLUGIN_SOURCE_DIR.rglob("*.py")):
        content = py_file.read_text(encoding="utf-8")
        all_errors.extend(check_source_property_mapping(py_file, content))

    all_errors.extend(check_readme_plugin_table(README_PATH, PACKAGES_CONFIG))
    all_errors.extend(link_errors)

    # Only link errors inside guides can corrupt the reachability BFS.
    if not guides_link_errors:
        all_errors.extend(find_orphan_guides(GUIDES_DIR, contents))

    if all_errors:
        print(f"Found {len(all_errors)} doc issue(s):\n")
        for error in all_errors:
            print(f"  {error}")
        return 1

    print("All docs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
