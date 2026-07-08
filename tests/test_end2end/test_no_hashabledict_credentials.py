"""Repository guard for GitHub issue #270.

mloda 0.9.0 dropped ``HashableDict`` acceptance from the credentials /
DB-reader data-access path: passing a ``HashableDict`` as a credential value to
``DataAccessCollection`` / ``add_credentials``, or in a ``BaseInputData`` reader
tuple, now raises ``ValueError`` (use ``Credential(...)`` or a plain dict).
``HashableDict`` stays valid for other uses (e.g. ``Options`` group hashing),
so this guard only flags ``HashableDict`` usages near a credential /
data-access marker. The registry is currently clean; this trips loudly if such
a usage ever reappears.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


_SKIP_DIRS = {"__pycache__", "site-packages", "node_modules"}
_MARKERS = ("credentials", "add_credentials", "DataAccessCollection", "BaseInputData")


def find_hashabledict_credential_usages(root: Path) -> list[str]:
    """Return "relpath:lineno: line" for every HashableDict usage near a credential/data-access marker under root."""
    hits: list[str] = []
    for path in list(root.rglob("*.py")) + list(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part.startswith(".") or part in _SKIP_DIRS for part in rel.parts):
            continue
        if path.name == "test_no_hashabledict_credentials.py":
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines):
            if "HashableDict" not in line:
                continue
            window = lines[max(0, i - 3) : i + 4]
            if any(marker in w for w in window for marker in _MARKERS):
                hits.append(f"{rel.as_posix()}:{i + 1}: {line.strip()}")
    return sorted(hits)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_credentials_list_hashabledict_flagged(tmp_path: Path) -> None:
    """A HashableDict inside a credentials= list on one line is flagged."""
    _write(tmp_path / "m.py", "credentials=[HashableDict({'host': 'h'})]\n")
    hits = find_hashabledict_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "m.py" in hits[0]


def test_credentials_dict_value_hashabledict_flagged(tmp_path: Path) -> None:
    """A HashableDict as a credentials dict value is flagged."""
    _write(tmp_path / "m.py", "credentials={'pg-prod': HashableDict({'host': 'h'})}\n")
    hits = find_hashabledict_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "m.py" in hits[0]


def test_add_credentials_hashabledict_flagged(tmp_path: Path) -> None:
    """A HashableDict passed to add_credentials is flagged."""
    _write(tmp_path / "m.py", "collection.add_credentials(HashableDict({'host': 'h'}))\n")
    hits = find_hashabledict_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "m.py" in hits[0]


def test_multiline_baseinputdata_reader_tuple_flagged(tmp_path: Path) -> None:
    """A HashableDict a couple lines from a BaseInputData marker is flagged."""
    body = "reader = BaseInputData(\n    source='pg',\n    creds=HashableDict({'host': 'h'}),\n)\n"
    _write(tmp_path / "m.py", body)
    hits = find_hashabledict_credential_usages(tmp_path)
    assert len(hits) == 1
    assert "HashableDict" in hits[0]


def test_credential_object_not_flagged(tmp_path: Path) -> None:
    """A credentials= using Credential(...) with no HashableDict is not flagged."""
    _write(tmp_path / "m.py", "credentials=Credential(host='h')\n")
    assert find_hashabledict_credential_usages(tmp_path) == []


def test_hashabledict_far_from_marker_not_flagged(tmp_path: Path) -> None:
    """A legitimate HashableDict used for Options hashing (no marker nearby) is not flagged."""
    body = "options = Options(\n    group={'algorithm': HashableDict({'k': 1})},\n)\n"
    _write(tmp_path / "m.py", body)
    assert find_hashabledict_credential_usages(tmp_path) == []


def test_guard_module_self_excluded(tmp_path: Path) -> None:
    """A file named like this guard module is skipped even with a flaggable pattern."""
    _write(tmp_path / "test_no_hashabledict_credentials.py", "credentials=[HashableDict({'host': 'h'})]\n")
    assert find_hashabledict_credential_usages(tmp_path) == []


def test_repo_root_is_clean() -> None:
    """The real repository must currently have zero HashableDict credential usages."""
    offenders = find_hashabledict_credential_usages(_REPO_ROOT)
    assert offenders == [], "HashableDict credential/data-access usage reappeared:\n" + "\n".join(offenders)
