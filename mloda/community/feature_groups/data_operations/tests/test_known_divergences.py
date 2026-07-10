"""Drift check: every known-divergence entry must cite a regression test that exists.

The page at ``docs/guides/data-operation-patterns/known-divergences.md`` is the
authoritative record of every place a backend silently diverges from the PyArrow
reference. Each entry names a regression test that should fail if its mitigation
is removed, but historically those references were prose: if a cited test was
renamed, moved, or deleted, the doc rotted silently and a steward lost the
guarantee with no signal.

To make the link machine-checkable, every entry under ``## Entries`` carries a
``<!-- machine-checked ... -->`` HTML-comment block with these keys::

    operation: <comma-separated data operations>
    framework: <comma-separated frameworks>
    condition: <one-line description of the divergence>
    mitigation_location:
    - <repo-relative path to the file where the mitigation lives>
    regression_test:
    - <repo-relative path>::<Class>[::<method>]  (or <path>::<test_function>)

The ``operation`` / ``framework`` / ``condition`` values are single-line; the list
fields use ``- `` items, one per line. A wrapped value or an unrecognized line in
the block is a parse error, by design, so the block stays trivially machine-readable.

This module parses those blocks and asserts:

- ``test_every_entry_has_machine_block_with_required_keys`` -- every ``###``
  entry under ``## Entries`` carries a block with all required keys populated.
  This is the completeness guard: a new entry cannot land without a checkable
  guard.
- ``test_every_regression_test_reference_resolves`` -- every ``regression_test``
  reference resolves to a real, importable test class / method / function. The
  cited test methods live on ``*TestBase`` classes that pytest collects only
  through per-framework subclasses; resolving the symbol via ``importlib`` +
  ``getattr`` confirms it exists, which is what the sibling
  ``test_framework_support_matrix`` drift check does too. Beyond existence, for a
  concrete ``Test*`` class citing an inherited zero-fixture ``(self)`` method it
  also runs that method and fails if it skips for the class it is cited under,
  closing the gap where an inherited shared test (e.g. a capability mixin) is
  inert because the concrete class declared no config. The liveness check is
  scoped to inherited ``(self)``-only methods precisely because literal
  pytest-collection fidelity would be far more fragile.
- ``test_every_mitigation_location_exists`` -- every ``mitigation_location``
  path points at a file that still exists (DoD item 3).
- ``test_operation_and_framework_values_are_known`` -- every ``operation`` and
  ``framework`` token is a name the sibling support-matrix check recognizes, so
  those two fields cannot silently hold a typo or a stale name.

When this test fails, the doc references a test or file that no longer exists:
either restore the guard or update the entry's machine block to point at the
test that now guards the divergence.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import (
    FRAMEWORKS,
    OPERATIONS,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DOC_PATH = REPO_ROOT / "docs" / "guides" / "data-operation-patterns" / "known-divergences.md"

# Known vocabularies, reused from the sibling support-matrix drift check so the two
# checks cannot disagree about what a valid framework / operation name is. Validating
# the machine block's ``framework`` and ``operation`` fields against these turns a typo
# or a stale name into a failure instead of letting those fields hold arbitrary text.
KNOWN_FRAMEWORKS: frozenset[str] = frozenset(key for key, _label in FRAMEWORKS)
KNOWN_OPERATIONS: frozenset[str] = frozenset(OPERATIONS)

ENTRIES_HEADING = "## Entries"
BLOCK_BEGIN = "<!-- machine-checked"
BLOCK_END = "-->"

REQUIRED_KEYS: tuple[str, ...] = (
    "operation",
    "framework",
    "condition",
    "mitigation_location",
    "regression_test",
)
LIST_KEYS: tuple[str, ...] = ("mitigation_location", "regression_test")

_KEY_RE = re.compile(r"^([a-z_]+):\s*(.*)$")
_ITEM_RE = re.compile(r"^-\s+(.+)$")


def _entries_section(doc: str) -> str:
    """Return the text of the ``## Entries`` section, up to the next ``## `` heading.

    Bounding on the next level-2 heading keeps the ``## Categories``,
    ``## Audit coverage`` and ``## When to add`` sections out of the entry set:
    only ``### `` headings between ``## Entries`` and the following ``## `` count
    as divergence entries.
    """
    lines = doc.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == ENTRIES_HEADING:
            start = i + 1
            break
    if start is None:
        raise RuntimeError(f"{ENTRIES_HEADING!r} section not found in {DOC_PATH}")
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


def _split_entries(section: str) -> list[tuple[str, str]]:
    """Split the entries section into ``(title, body)`` pairs, one per ``### `` heading."""
    entries: list[tuple[str, str]] = []
    title: str | None = None
    buf: list[str] = []
    for line in section.splitlines():
        if line.startswith("### "):
            if title is not None:
                entries.append((title, "\n".join(buf)))
            title = line[len("### ") :].strip()
            buf = []
        elif title is not None:
            buf.append(line)
    if title is not None:
        entries.append((title, "\n".join(buf)))
    return entries


def _parse_block(title: str, body: str) -> dict[str, Any]:
    """Parse the ``<!-- machine-checked ... -->`` block from an entry body."""
    if BLOCK_BEGIN not in body:
        raise AssertionError(f"entry {title!r} has no {BLOCK_BEGIN!r} machine block")
    rest = body[body.index(BLOCK_BEGIN) + len(BLOCK_BEGIN) :]
    if BLOCK_END not in rest:
        raise AssertionError(f"entry {title!r} machine block is not closed with {BLOCK_END!r}")
    block = rest[: rest.index(BLOCK_END)]

    data: dict[str, Any] = {}
    current: str | None = None
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        key_match = _KEY_RE.match(line)
        item_match = _ITEM_RE.match(line)
        if key_match:
            current = key_match.group(1)
            value = key_match.group(2).strip()
            data[current] = value if value else []
        elif item_match:
            if current is None or not isinstance(data.get(current), list):
                raise AssertionError(f"entry {title!r}: list item with no preceding list key: {line!r}")
            data[current].append(item_match.group(1).strip())
        else:
            raise AssertionError(f"entry {title!r}: unparseable machine-block line: {line!r}")
    return data


def _collected_entries() -> list[tuple[str, dict[str, Any]]]:
    doc = DOC_PATH.read_text()
    entries = _split_entries(_entries_section(doc))
    assert entries, f"no '### ' entries found under {ENTRIES_HEADING!r} in {DOC_PATH}"
    return [(title, _parse_block(title, body)) for title, body in entries]


def _assert_leaf_is_live(cls: type, method_name: str, ref: str) -> None:
    """Fail if a cited inherited ``(self)``-only test method skips for its concrete class.

    Closes the hole where an inherited shared test guards nothing because the
    concrete class declared no per-class config. Limitation: methods taking
    fixtures/params (signature not exactly ``(self)``) cannot run outside pytest,
    so they fall back to the existence check only.
    """
    if method_name in cls.__dict__:
        return
    params = list(inspect.signature(getattr(cls, method_name)).parameters)
    if params != ["self"]:
        return
    try:
        getattr(cls(), method_name)()
    except pytest.skip.Exception as exc:
        raise AssertionError(
            f"{ref}: cited regression test {method_name!r} skips for {cls.__name__} and so does not "
            "guard the divergence (inherited shared test with no per-class config)"
        ) from exc


def _resolve_test_ref(ref: str) -> None:
    """Resolve a ``path.py::Class[::method]`` reference to a real test symbol.

    Raises ``AssertionError`` if the file, module, attribute chain, or leaf-test
    shape does not resolve. For a concrete ``Test*`` class citing an inherited
    ``(self)``-only ``test`` method, also asserts the method is live (does not
    skip) for that class via ``_assert_leaf_is_live``.
    """
    parts = ref.split("::")
    rel = parts[0]
    if not rel.endswith(".py"):
        raise AssertionError(f"reference path must be a .py file: {ref!r}")
    if not (REPO_ROOT / rel).exists():
        raise AssertionError(f"reference file does not exist: {rel!r}")
    if len(parts) < 2:
        raise AssertionError(f"reference must name a test class or function: {ref!r}")

    module_name = rel[: -len(".py")].replace("/", ".")
    try:
        module: Any = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - surface any import failure as a resolution failure
        raise AssertionError(f"could not import module {module_name!r} for {ref!r}: {exc}") from exc

    obj: Any = module
    for attr in parts[1:]:
        nxt = getattr(obj, attr, None)
        if nxt is None:
            raise AssertionError(f"attribute {attr!r} not found while resolving {ref!r}")
        obj = nxt

    leaf = parts[-1]
    if inspect.isclass(obj):
        if not leaf.startswith("Test"):
            raise AssertionError(
                f"class reference must be a pytest-collected Test* class: {ref!r}. "
                f"Base classes such as {leaf!r} are collected only via per-framework "
                "subclasses, so reference a specific '::method' on the base instead."
            )
    else:
        if not callable(obj):
            raise AssertionError(f"reference leaf is not callable: {ref!r}")
        if not leaf.startswith("test"):
            raise AssertionError(f"function/method reference must start with 'test': {ref!r}")

    # For ``path::TestClass::test_method`` refs whose class is a concrete
    # pytest-collected ``Test*`` (not a ``*TestBase`` / ``*Mixin``), also verify
    # the cited method is live for that class, not an inert inherited skip.
    if len(parts) == 3 and parts[1].startswith("Test") and leaf.startswith("test"):
        cls = getattr(module, parts[1])
        if inspect.isclass(cls):
            _assert_leaf_is_live(cls, leaf, ref)


def test_every_entry_has_machine_block_with_required_keys() -> None:
    for title, data in _collected_entries():
        for key in REQUIRED_KEYS:
            assert key in data, f"entry {title!r} missing required key {key!r}"
            assert data[key], f"entry {title!r} has an empty value for {key!r}"
        for key in LIST_KEYS:
            assert isinstance(data[key], list), f"entry {title!r}: {key!r} must be a list, got {type(data[key])}"


def test_every_regression_test_reference_resolves() -> None:
    """Every cited regression test resolves and, if an inherited ``(self)`` method, is live for its class."""
    for title, data in _collected_entries():
        for ref in data.get("regression_test", []):
            try:
                _resolve_test_ref(ref)
            except AssertionError as exc:
                raise AssertionError(f"entry {title!r}: {exc}") from exc


def test_every_mitigation_location_exists() -> None:
    for title, data in _collected_entries():
        for loc in data.get("mitigation_location", []):
            assert (REPO_ROOT / loc).exists(), f"entry {title!r}: mitigation_location does not exist: {loc!r}"


def test_operation_and_framework_values_are_known() -> None:
    """Each ``operation`` / ``framework`` token must be a name the suite recognizes.

    Without this, the two fields are write-only metadata: ``test_..._required_keys``
    only checks they are non-empty, so a typo (``polars_lzy``) or a stale operation
    name would stay green. Validating against the same vocabularies the sibling
    support-matrix check derives from keeps the fields meaningful and the two drift
    checks from disagreeing about valid names.
    """
    for title, data in _collected_entries():
        for op in (tok.strip() for tok in str(data["operation"]).split(",")):
            assert op in KNOWN_OPERATIONS, (
                f"entry {title!r}: unknown operation {op!r} (known: {sorted(KNOWN_OPERATIONS)})"
            )
        for fw in (tok.strip() for tok in str(data["framework"]).split(",")):
            assert fw in KNOWN_FRAMEWORKS, (
                f"entry {title!r}: unknown framework {fw!r} (known: {sorted(KNOWN_FRAMEWORKS)})"
            )


# --- _assert_leaf_is_live: inherited-skip hole in _resolve_test_ref ----------
#
# ``_resolve_test_ref`` confirms a cited symbol exists, but a method INHERITED
# from a mixin that ``pytest.skip(...)``s when the concrete class declares no
# config resolves fine while asserting nothing for the class it is cited under.
# The green helper ``_assert_leaf_is_live(cls, method_name, ref)`` closes that
# hole: it must raise ``AssertionError`` when an inherited cited method skips,
# and stay silent when it is live or defined on the concrete class itself.
#
# The probe classes below are named with a leading underscore so pytest does
# not collect them as test cases.


class _InertProbeMixin:
    """Mixin whose shared test skips when a concrete class declares no config."""

    def test_probe(self) -> None:
        """Skip: no per-class config declared."""
        pytest.skip("no per-class config declared")


class _ConcreteInertProbe(_InertProbeMixin):
    """Concrete class that inherits the skipping shared test unchanged."""


class _LiveProbeMixin:
    """Mixin whose shared test asserts for real."""

    def test_probe(self) -> None:
        """Live: asserts unconditionally."""
        assert True


class _ConcreteLiveProbe(_LiveProbeMixin):
    """Concrete class that inherits a live shared test."""


class _ConcreteOwnProbe:
    """Concrete class that defines the cited method directly in its own body."""

    def test_probe(self) -> None:
        """Skip that must be ignored: the method is defined on the concrete class itself."""
        pytest.skip("this skip must be ignored: method is defined on the concrete class itself")


def test_assert_leaf_is_live_rejects_inherited_skipping_method() -> None:
    """An inherited cited method that skips for the concrete class must fail the drift check."""
    with pytest.raises(AssertionError, match="(?i)skip"):
        _assert_leaf_is_live(_ConcreteInertProbe, "test_probe", "fake_ref::_ConcreteInertProbe::test_probe")


def test_assert_leaf_is_live_accepts_inherited_live_method() -> None:
    """An inherited cited method that asserts for real is live and must not fail the check."""
    _assert_leaf_is_live(_ConcreteLiveProbe, "test_probe", "fake_ref")


def test_assert_leaf_is_live_skips_directly_defined_method() -> None:
    """A method defined on the concrete class itself is out of scope for the inherited-skip check."""
    _assert_leaf_is_live(_ConcreteOwnProbe, "test_probe", "fake_ref")


def test_cited_capability_mixin_refs_are_live() -> None:
    """The frame_aggregate agg-type entry cites a capability-hook test that is live today."""
    from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.tests.test_sqlite import (
        TestSqliteFrameAggregate,
    )

    _assert_leaf_is_live(TestSqliteFrameAggregate, "test_mixin_capability_hook_rejects", "ref")
