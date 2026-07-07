"""Tests for scripts/lint_credentials.py (regression guard for GitHub issue #270).

The guard fails if ``HashableDict`` reappears near credential /
``BaseInputData`` data-access construction (which mloda 0.9.0 rejects), while
NOT flagging legitimate ``HashableDict`` usage (Options group values,
ApiInputDataCollection, provider export/import).

Like ``test_lint_docs.py``, the lint script is not a packaged module, so it's
loaded via a sys.path insert. ``tests/**`` has ``E402`` ignored in ruff config
so the post-path import does not trip the linter.

Contract notes for the Green agent implementing scripts/lint_credentials.py:
- Stable assertion phrase pinned below: every violation string must contain the
  substring ``"HashableDict"`` AND the substring ``"credential"`` (lowercase),
  plus the ``{rel_path}:{lineno}:`` locator.
- Detection is marker-anchored and bracket-span based (there is NO ``window``
  parameter): from each construction-shaped credential marker
  (``credentials=``, ``add_credentials(``, or a quoted ``BaseInputData`` group
  key), walk forward tracking bracket depth and flag any ``HashableDict`` token
  that falls inside that enclosing construct, reported at the token's 1-based
  line (deduped, one error per HashableDict line).
- ``main()`` must scan a module-level, rebindable root named ``SCAN_ROOT``
  (defaulting to ``REPO_ROOT``); these tests monkeypatch
  ``lint_credentials.SCAN_ROOT`` to point at a tmp tree.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import lint_credentials


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


# --------------------------------------------------------------------------
# scan_content — positive cases (MUST be flagged), one per issue example form
# --------------------------------------------------------------------------


def test_scan_flags_credentials_list_form() -> None:
    content = 'credentials=[HashableDict({"user": "x"})]\n'
    errors = lint_credentials.scan_content("mod.py", content)
    assert len(errors) == 1
    assert "mod.py:1:" in errors[0]
    assert "HashableDict" in errors[0]
    assert "credential" in errors[0]


def test_scan_flags_credentials_dict_value_form() -> None:
    content = 'credentials={"db": HashableDict({"user": "x"})}\n'
    errors = lint_credentials.scan_content("mod.py", content)
    assert len(errors) == 1
    assert "mod.py:1:" in errors[0]
    assert "HashableDict" in errors[0]
    assert "credential" in errors[0]


def test_scan_flags_add_credentials_call_form() -> None:
    content = 'add_credentials(HashableDict({"user": "x"}))\n'
    errors = lint_credentials.scan_content("mod.py", content)
    assert len(errors) == 1
    assert "mod.py:1:" in errors[0]
    assert "HashableDict" in errors[0]
    assert "credential" in errors[0]


def test_scan_flags_base_input_data_group_key_form() -> None:
    content = 'Options(group={"BaseInputData": (Reader, HashableDict({"user": "x"}))})\n'
    errors = lint_credentials.scan_content("mod.py", content)
    assert len(errors) == 1
    assert "mod.py:1:" in errors[0]
    assert "HashableDict" in errors[0]
    assert "credential" in errors[0]


def test_scan_flags_hashable_dict_on_later_line_of_same_construct() -> None:
    """Cross-line: marker and HashableDict on DIFFERENT lines, both inside the
    same bracket construct. Pins forward bracket-span detection (a per-line-only
    impl would miss it)."""
    content = "    credentials=[\n        HashableDict({}),\n    ]\n"
    errors = lint_credentials.scan_content("mod.py", content)
    assert len(errors) == 1
    # The HashableDict token is on line 2, inside the credentials=[...] span.
    assert "mod.py:2:" in errors[0]
    assert "HashableDict" in errors[0]
    assert "credential" in errors[0]


def test_scan_flags_wrapped_credential_construction() -> None:
    """Deeply wrapped: the HashableDict token sits several lines below the marker,
    still inside the enclosing brackets. Proves wrapping beyond 2 lines no longer
    slips through a fixed proximity window."""
    content = "credentials=[\n    some_factory(\n        extra_arg,\n        HashableDict({}),\n    ),\n]\n"
    errors = lint_credentials.scan_content("mod.py", content)
    assert len(errors) == 1
    # The HashableDict token is on line 4.
    assert "mod.py:4:" in errors[0]
    assert "HashableDict" in errors[0]
    assert "credential" in errors[0]


# --------------------------------------------------------------------------
# scan_content — negative cases (MUST NOT be flagged)
# --------------------------------------------------------------------------


def test_scan_ignores_legit_options_group_value() -> None:
    content = 'Options(group={"MyReader": HashableDict({"path": "/x"})})\n'
    assert lint_credentials.scan_content("mod.py", content) == []


def test_scan_ignores_hashable_dict_near_unrelated_marker_mention() -> None:
    """Bare prose ``credentials`` and a bare ``DataAccessCollection`` mention are
    NOT markers; a legit Options-group HashableDict nearby must NOT be flagged."""
    content = (
        "You can also pass credentials to the reader.\n"
        "from mloda.user import DataAccessCollection\n"
        'opts = Options(group={"MyReader": HashableDict({"path": "/x"})})\n'
    )
    assert lint_credentials.scan_content("mod.py", content) == []


def test_scan_ignores_hashable_dict_far_from_marker() -> None:
    """HashableDict outside the ``credentials = load_credentials()`` statement's
    span (the marker's construct already ended) is clean."""
    lines = [
        "credentials = load_credentials()",  # marker's statement ends on line 1
        "a = 1",
        "b = 2",
        "c = 3",
        "d = 4",
        'value = HashableDict({"path": "/x"})',  # line 6, outside the marker span
    ]
    content = "\n".join(lines) + "\n"
    assert lint_credentials.scan_content("mod.py", content) == []


def test_scan_ignores_typed_credential_without_hashable_dict() -> None:
    content = 'credentials=[Credential(user="x")]\n'
    assert lint_credentials.scan_content("mod.py", content) == []


def test_scan_ignores_plain_dict_credential_without_hashable_dict() -> None:
    content = 'credentials={"db": {"user": "x"}}\n'
    assert lint_credentials.scan_content("mod.py", content) == []


# --------------------------------------------------------------------------
# find_credential_hashable_dict — tree-level
# --------------------------------------------------------------------------


def test_real_repo_tree_is_clean() -> None:
    """The actual audit: the repo currently has zero credential-context usages."""
    assert lint_credentials.find_credential_hashable_dict(lint_credentials.REPO_ROOT) == []


def test_find_excludes_junk_dirs(tmp_path: Path) -> None:
    _write(tmp_path / ".venv" / "mod.py", "credentials=[HashableDict({})]\n")
    _write(tmp_path / "build" / "mod.py", "credentials=[HashableDict({})]\n")
    _write(tmp_path / "pkg.egg-info" / "mod.py", "credentials=[HashableDict({})]\n")
    assert lint_credentials.find_credential_hashable_dict(tmp_path) == []


def test_find_flags_planted_violation(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "mod.py", 'credentials=[HashableDict({"user": "x"})]\n')
    errors = lint_credentials.find_credential_hashable_dict(tmp_path)
    assert len(errors) == 1
    assert "src/mod.py:1:" in errors[0]


def test_guard_excludes_own_files(tmp_path: Path) -> None:
    """A planted violation inside the guard's own two files must NOT be returned,
    since those files legitimately carry the literal token as fixtures."""
    _write(tmp_path / "scripts" / "lint_credentials.py", "credentials=[HashableDict({})]\n")
    _write(
        tmp_path / "tests" / "test_end2end" / "test_lint_credentials.py",
        "credentials=[HashableDict({})]\n",
    )
    assert lint_credentials.find_credential_hashable_dict(tmp_path) == []


# --------------------------------------------------------------------------
# main() — return-code contract (monkeypatch module-level SCAN_ROOT)
# --------------------------------------------------------------------------


def test_main_clean_tree_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(tmp_path / "src" / "mod.py", 'credentials=[Credential(user="x")]\n')
    monkeypatch.setattr(lint_credentials, "SCAN_ROOT", tmp_path)
    rc = lint_credentials.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "No credential-context" in out


def test_main_flags_planted_violation_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write(tmp_path / "src" / "mod.py", 'credentials=[HashableDict({"user": "x"})]\n')
    monkeypatch.setattr(lint_credentials, "SCAN_ROOT", tmp_path)
    rc = lint_credentials.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "src/mod.py" in out
