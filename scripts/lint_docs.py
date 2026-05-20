"""Lint documentation guides for broken relative links and internal imports in code snippets.

Run: python scripts/lint_docs.py
Exit code: 1 if any issues found, 0 otherwise.
"""

import functools
import re
import sys
from collections import deque
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "guides"

INTERNAL_IMPORT_RE = re.compile(r"from mloda\.core\.")

MD_LINK_RE = re.compile(r"\[.*?\]\((?!https?://|mailto:)([^)#\s]+\.md)(?:#([^)\s]+))?\)")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

CODE_BLOCK_RE = re.compile(r"^```", re.MULTILINE)

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


def find_code_blocks(content: str) -> list[str]:
    """Extract code block contents from markdown."""
    blocks = []
    parts = CODE_BLOCK_RE.split(content)
    for i in range(1, len(parts), 2):
        blocks.append(parts[i])
    return blocks


def _slugify_heading(text: str) -> str:
    """GFM-style slug; ignores duplicate-heading disambiguation and inline formatting beyond backticks."""
    text = text.lower().replace("`", "")
    text = text.replace(" ", "-")
    return re.sub(r"[^\w-]", "", text)


@functools.lru_cache(maxsize=None)
def _heading_slugs(md_file: Path) -> frozenset[str]:
    slugs: set[str] = set()
    content = _strip_code_blocks(md_file.read_text())
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
    code_blocks = find_code_blocks(content)
    for block in code_blocks:
        for line in block.splitlines():
            stripped = line.strip()
            if INTERNAL_IMPORT_RE.search(stripped):
                line_num = content.find(stripped)
                if line_num >= 0:
                    line_num = content[:line_num].count("\n") + 1
                errors.append(f"{md_file}:{line_num}: internal import in code snippet -> {stripped}")
    return errors


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
        return [f"{INDEX_FILENAME}: missing root index for orphan check"]

    def _read(path: Path) -> str:
        if contents is not None and path in contents:
            return contents[path]
        return path.read_text()

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


def main() -> int:
    if not DOCS_DIR.is_dir():
        print(f"Docs directory not found: {DOCS_DIR}")
        return 1

    all_errors: list[str] = []
    link_errors: list[str] = []
    contents: dict[Path, str] = {}

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        content = md_file.read_text()
        contents[md_file.resolve()] = content
        link_errors.extend(check_relative_links_and_anchors(md_file, content))
        all_errors.extend(check_internal_imports(md_file, content))
        all_errors.extend(check_bare_fence_openers(md_file, content))

    all_errors.extend(link_errors)

    if not link_errors:
        all_errors.extend(find_orphan_guides(DOCS_DIR, contents))

    if all_errors:
        print(f"Found {len(all_errors)} doc issue(s):\n")
        for error in all_errors:
            print(f"  {error}")
        return 1

    print("All docs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
