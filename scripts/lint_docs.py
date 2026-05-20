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

RELATIVE_LINK_RE = re.compile(r"\[.*?\]\((?!https?://|mailto:)([^)#\s]+\.md)(?:#([^)\s]+))?\)")

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")

CODE_BLOCK_RE = re.compile(r"^```", re.MULTILINE)

INDEX_FILENAME = "index.md"


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
    content = "".join(CODE_BLOCK_RE.split(md_file.read_text())[::2])
    for match in HEADING_RE.finditer(content):
        slugs.add(_slugify_heading(match.group(2)))
    return frozenset(slugs)


def check_relative_links_and_anchors(md_file: Path, content: str) -> list[str]:
    """Validate relative markdown links and their optional anchor fragments."""
    errors = []
    for match in RELATIVE_LINK_RE.finditer(content):
        rel_path = match.group(1)
        anchor = match.group(2)
        target = (md_file.parent / rel_path).resolve()
        line_num = content[: match.start()].count("\n") + 1
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


def _collect_linked_md(md_file: Path) -> set[Path]:
    """Return the set of .md files linked from md_file (resolved absolute paths)."""
    linked: set[Path] = set()
    content = md_file.read_text()
    for match in MARKDOWN_LINK_RE.finditer(content):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (md_file.parent / target).resolve()
        linked.add(resolved)
    return linked


def find_orphan_guides(docs_dir: Path) -> list[str]:
    """Flag any .md file under docs_dir that is unreachable from docs_dir/index.md.

    Reachability is transitive via inline markdown links. Files named ``index.md`` are
    exempt from the "must be linked" requirement (a section index does not need an
    inbound link from itself), but they do participate as relay hops in the walk.

    Caveat: a subdirectory whose only file is ``index.md`` and which has no inbound
    link from any parent is still considered reachable, because ``index.md`` files
    are exempt from the inbound-link check.
    """
    errors = []
    docs_root = docs_dir.resolve()
    root_index = docs_dir / INDEX_FILENAME
    if not root_index.is_file():
        return [f"{root_index}: missing root index for orphan check"]

    reachable: set[Path] = {root_index.resolve()}
    frontier: deque[Path] = deque([root_index.resolve()])
    while frontier:
        current = frontier.popleft()
        for linked in _collect_linked_md(current):
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
        if md_file.name == INDEX_FILENAME:
            continue
        if resolved not in reachable:
            rel = md_file.relative_to(docs_dir)
            errors.append(f"{rel}: orphan guide not reachable from {INDEX_FILENAME}")
    return errors


def main() -> int:
    if not DOCS_DIR.is_dir():
        print(f"Docs directory not found: {DOCS_DIR}")
        return 1

    all_errors: list[str] = []

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        content = md_file.read_text()
        all_errors.extend(check_relative_links_and_anchors(md_file, content))
        all_errors.extend(check_internal_imports(md_file, content))

    all_errors.extend(find_orphan_guides(DOCS_DIR))

    if all_errors:
        print(f"Found {len(all_errors)} doc issue(s):\n")
        for error in all_errors:
            print(f"  {error}")
        return 1

    print("All docs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
