"""Import-root helpers shared by the tests that scan runtime sources."""

from __future__ import annotations

import ast
import sys

# mloda and mloda_plugins are required core dependencies, not optional backends.
FIRST_PARTY_IMPORT_ROOTS = frozenset({"mloda", "mloda_plugins"})

# Bridge stdlib drift across the 3.11-3.14 support window (e.g. PEP 594 removals, tomllib addition).
STDLIB_ROOTS: frozenset[str] = frozenset(sys.stdlib_module_names) | frozenset(
    {
        "aifc",
        "audioop",
        "cgi",
        "cgitb",
        "chunk",
        "crypt",
        "imghdr",
        "mailcap",
        "msilib",
        "nis",
        "nntplib",
        "ossaudiodev",
        "pipes",
        "smtpd",
        "sndhdr",
        "spwd",
        "sunau",
        "telnetlib",
        "tomllib",
        "uu",
        "xdrlib",
    }
)


def trusted_type_checking_alias(tree: ast.Module) -> str | None:
    """The module-level name that genuinely aliases ``typing.TYPE_CHECKING``, or ``None``.

    Only a bare ``from typing import TYPE_CHECKING [as alias]`` at module level earns trust, and
    only if nothing else at module level rebinds that name. Without this, a module could spoof the
    guard's exemption with ``TYPE_CHECKING = True`` and hide a real, always-executed import from it.
    """
    alias_name: str | None = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "typing":
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    alias_name = alias.asname or alias.name
    if alias_name is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == "typing":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == alias_name and isinstance(child.ctx, ast.Store):
                return None
    return alias_name


def is_type_checking_block(node: ast.If, trusted_alias: str | None) -> bool:
    """Whether an ``if`` tests the trusted TYPE_CHECKING alias or ``typing.TYPE_CHECKING``."""
    return (isinstance(node.test, ast.Name) and trusted_alias is not None and node.test.id == trusted_alias) or (
        isinstance(node.test, ast.Attribute)
        and node.test.attr == "TYPE_CHECKING"
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "typing"
    )
