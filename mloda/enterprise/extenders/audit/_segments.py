"""Run manifest segments: archive helpers, segment rotation and segment verification."""

from __future__ import annotations

import os
import stat
import tempfile
from collections import Counter
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import IO, Any

from mloda.enterprise.extenders.audit._core import (
    HeadAnchor,
    LogCoverage,
    ManifestVerificationError,
    _check_log_id,
    _decode_line,
    _digest_runs,
    _flock,
    _fsync,
    _genesis_entry,
    _is_terminated,
    _LogState,
    _OversizedLine,
    _parse_ndjson_line,
    _read_lines,
    _read_manifests,
    _record_run_id,
    _reject_aliased_paths,
    _RunDigest,
    _scan_for_run,
    _Uncovered,
    _unlink_durably,
    _verify_log,
    manifest_hash,
)
from mloda.enterprise.extenders.audit._records import (
    _canonical_json,
    _is_blank,
    _open_locked,
)
from mloda.enterprise.extenders.audit._seal_index import (
    _indexed_archive,
    _sqlite,
)
from mloda.enterprise.extenders.audit._signers import (
    ManifestSigner,
    _signer_map,
)


def _segment_path(live: str | Path, number: int) -> Path:
    live = Path(live)
    return live.with_name(f"{live.name}.{number:06d}")


def _archived_segments(live: str | Path) -> list[tuple[int, Path]]:
    """The archived `<live>.<NNNNNN>` files next to `live`, oldest first."""
    live = Path(live)
    found: list[tuple[int, Path]] = []
    with suppress(FileNotFoundError):
        for name in os.listdir(live.parent):
            suffix = name[len(live.name) + 1 :]
            if name.startswith(live.name + ".") and len(suffix) == 6 and suffix.isascii() and suffix.isdigit():
                found.append((int(suffix), live.parent / name))
    return sorted(found)


def _archived_sealed_ids(manifest_path: str | Path) -> set[str]:
    """The run_ids sealed in the archived manifest segments (unverified decode, like `_scan_for_run`)."""
    ids: set[str] = set()
    for _, path in _archived_segments(manifest_path):
        try:
            with open(path, "rb") as file:
                for raw in _read_lines(file):
                    if isinstance(raw, _OversizedLine):
                        continue
                    with suppress(ManifestVerificationError):
                        manifest = _decode_line(str(path), raw.rstrip(b"\n"))
                        if (
                            isinstance(manifest, dict)
                            and "kind" not in manifest
                            and isinstance(manifest.get("run_id"), str)
                        ):
                            ids.add(manifest["run_id"])
        except FileNotFoundError:
            continue
    return ids


def _sealed_in_archives(manifest_path: str | Path, run_id: str, index_path: str | Path | None = None) -> bool:
    """Whether an archive seals run_id: from the index's `archived` table when usable, else a scan."""
    if index_path is not None:
        db = _sqlite()
        with suppress(Exception):
            if db is not None:
                return _indexed_archive(db, index_path, run_id)
    return any(_scan_for_run(path, run_id) for _, path in reversed(_archived_segments(manifest_path)))


def _refuse_interrupted_rotation(manifest_path: str | Path, audit_path: str | Path | None = None) -> None:
    """Raise ValueError when a live file is still the same file as its newest archive (an interrupted rotation)."""
    archived = _archived_segments(manifest_path)
    if not archived:
        return
    number, newest = archived[-1]
    pairs = [(manifest_path, newest)]
    if audit_path is not None:
        pairs.append((audit_path, _segment_path(audit_path, number)))
    for live, archive in pairs:
        if _same_file(live, archive):
            raise ValueError(
                f"{live} is linked to an archived segment: finish the interrupted rotation with rotate_ndjson_segment"
            )


def verify_ndjson_segments(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    previous_signers: Iterable[ManifestSigner] = (),
    log_id: str | None = None,
    expected_head: str | None = None,
    anchored_heads: Iterable[str] = (),
) -> LogCoverage:
    """Verify the retained archived segment pairs in order plus the live pair as one chain: each segment against its
    own audit file, genesis chaining, key continuity, no run sealed twice, no carried record stranded. `expected_head`
    pins the live head; an anchored head may be a line of any retained segment. The oldest retained segment may start
    with a non-null predecessor (only anchors detect retention). `unsealed_lines` is the live segment's."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path)
    _check_log_id("verify_ndjson_segments", log_id)
    sealed_runs = sealed_lines = 0
    uncovered = _Uncovered()
    first_log_id: str | None = None
    with _flock(manifest_path, exclusive=False):
        _refuse_interrupted_rotation(manifest_path, audit_path)
        pairs = [
            (_segment_path(manifest_path, n), _segment_path(audit_path, n))
            for n, _ in _archived_segments(manifest_path)
        ]
        pairs.append((Path(manifest_path), Path(audit_path)))
        state = _LogState(set(), None, None, frozenset(), set())
        digests: dict[str, _RunDigest] = {}
        for index, (manifests_file, audit_file) in enumerate(pairs):
            last = index == len(pairs) - 1
            manifests = _read_manifests(manifests_file)
            try:
                found = _digest_runs(audit_file, lambda _: True, uncovered)
            except FileNotFoundError:
                found = {}
            here = {m["run_id"] for m in manifests if isinstance(m.get("run_id"), str)}
            seed = replace(state, seen_v2=True, log_id=None, lines=0) if index else None
            earlier = state.sealed
            state = _verify_log(
                manifests,
                signer=signer,
                signers=signers,
                expected_head=expected_head if last else None,
                digests={run: found[run] for run in here if run in found},
                require_current=last,
                log_id=log_id,
                anchored_heads=anchored_heads if last else (),
                seed=seed,
            )
            if not index:
                first_log_id = state.log_id
            elif state.log_id != first_log_id:
                raise ManifestVerificationError(
                    f"{manifests_file} genesis log_id {state.log_id!r} is not the first segment's {first_log_id!r}"
                )
            for run in found.keys() & earlier:
                raise ManifestVerificationError(f"run_id {run!r} has records in {audit_file} after it was sealed")
            for run, old in digests.items():
                if run not in earlier and Counter(old.hashes) - Counter(found[run].hashes if run in found else ()):
                    raise ManifestVerificationError(f"run_id {run!r} has records stranded in {audit_file}")
            sealed_runs += len(here)
            sealed_lines += sum(len(found[run].hashes) for run in here if run in found)
            digests = found
    return LogCoverage(
        head=state.head,
        sealed_runs=sealed_runs,
        sealed_lines=sealed_lines,
        unattributed_lines=uncovered.unattributed,
        unsealed_lines={run: len(digest.hashes) for run, digest in digests.items() if run not in state.sealed},
    )


def _outgoing_state(
    manifest_src: str | Path,
    manifests: list[dict[str, Any]],
    digests: Mapping[str, _RunDigest] | None,
    *,
    signer: ManifestSigner,
    signers: Mapping[str, ManifestSigner],
    expected_head: str | None,
    log_id: str,
    anchors: Sequence[str],
) -> _LogState:
    """Verify a segment's `manifests` like verify_ndjson_log_coverage (against `digests` when given)."""
    state = _verify_log(
        manifests,
        signer=signer,
        signers=signers,
        expected_head=expected_head,
        digests=digests,
        require_current=True,
        log_id=log_id,
        anchored_heads=anchors,
    )
    if state.head is None:
        raise ValueError(f"{manifest_src} has no manifests to rotate")
    return state


def _stage_temp(live: str | Path) -> tuple[IO[bytes], str]:
    """A temp file beside `live`, opened for writing."""
    live = Path(live)
    fd, temp = tempfile.mkstemp(dir=live.parent, prefix=f".{live.name}.")
    return os.fdopen(fd, "wb"), temp


def _stage_genesis(manifest_path: str | Path, genesis: Mapping[str, Any]) -> str:
    """Write `genesis` to a durable temp file with the live manifest's mode; return its path."""
    out, temp = _stage_temp(manifest_path)
    try:
        with out:
            out.write(_canonical_json(genesis) + b"\n")
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temp, stat.S_IMODE(os.stat(manifest_path).st_mode))
    except BaseException:
        with suppress(OSError):
            os.unlink(temp)
        raise
    return temp


def _carry_pending(
    audit_path: str | Path,
    out: IO[bytes],
    start: int,
    skip: Container[str],
    refuse: Container[str],
    *,
    sealed: Container[str] = (),
    digests: dict[str, _RunDigest] | None = None,
    final: bool = False,
) -> int:
    """Copy the audit lines from `start` whose run_id is usable and not in `skip`; raise on one in `refuse`; digest
    the lines of runs in `sealed` into `digests`. An unterminated last line ends the pass before it, or raises when
    `final`. Returns the end offset of the last terminated line."""
    end = start
    with open(audit_path, "rb") as file:
        file.seek(start)
        for number, raw in enumerate(_read_lines(file), start=1):
            if not final and not _is_terminated(raw):
                break
            line, record = _parse_ndjson_line(audit_path, number, raw)
            run_id = _record_run_id(record)
            if run_id is not None and not _is_blank(run_id):
                if run_id in refuse:
                    raise ManifestVerificationError(f"run_id {run_id!r} gained records after it was sealed")
                if digests is not None and run_id in sealed:
                    digests.setdefault(run_id, _RunDigest()).add(line, record)
                if run_id not in skip:
                    out.write(line + b"\n")
            end += len(line) + 1
    return end


@contextmanager
def _lock_staged(temp: str) -> Iterator[None]:
    """Hold an exclusive flock on the staged file until the block ends: the lock follows the inode through
    os.replace, so a sealer opening the new live file waits for it. No lock without fcntl."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    fd = os.open(temp, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _fsync_parents(*paths: str | Path) -> None:
    for parent in {Path(path).parent for path in paths}:
        _fsync(parent)


def _same_file(first: str | Path, second: str | Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except FileNotFoundError:
        return False


def _swap_in(temp: str, live: str | Path) -> None:
    os.replace(temp, live)
    _fsync(Path(live).parent)


def rotate_ndjson_segment(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signer: ManifestSigner,
    log_id: str,
    previous_signers: Iterable[ManifestSigner] = (),
    expected_head: str | None = None,
    anchored_heads: Iterable[str] = (),
    head_anchor: HeadAnchor | None = None,
) -> dict[str, Any]:
    """Archive both files as `<path>.<NNNNNN>` and start a new segment; return its genesis (`head_anchor` gets its
    hash). Verifies the outgoing segment first like verify_ndjson_log and changes nothing on failure. Audit lines of
    unsealed runs are carried over. Safe against NdjsonAuditSink writers only; a crash mid-rotation blocks sealing
    until it runs again. Symlinked live paths are refused. Run it as the account that writes the logs: the new live
    files are created by the rotating process."""
    signers = _signer_map(signer, previous_signers)
    _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path)
    if log_id is None:
        raise ValueError("rotate_ndjson_segment requires a log_id")
    _check_log_id("rotate_ndjson_segment", log_id)
    for name, path in (("audit_path", audit_path), ("manifest_path", manifest_path)):
        if os.path.islink(path):
            raise ValueError(f"rotate_ndjson_segment {name} {path} is a symlink; rotate the real file")
    anchors = list(anchored_heads)
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        raise ValueError(f"{manifest_path} has no manifests to rotate")
    check: dict[str, Any] = {
        "signer": signer,
        "signers": signers,
        "expected_head": expected_head,
        "log_id": log_id,
        "anchors": anchors,
    }
    with _flock(manifest_path, exclusive=True):
        archived = _archived_segments(manifest_path)
        if archived and os.stat(manifest_path).st_nlink > 1 and _same_file(manifest_path, archived[-1][1]):
            number, newest = archived[-1]
            audit_archive = _segment_path(audit_path, number)
            if _same_file(audit_path, audit_archive) or not audit_archive.exists():
                # Died before the audit swap: drop the stale links and rotate again under the same number.
                with suppress(FileNotFoundError):
                    _unlink_durably(audit_archive)
                _unlink_durably(newest)
                archived.pop()
            else:
                state = _outgoing_state(newest, _read_manifests(newest), None, **check)
                genesis = _genesis_entry(signer, log_id, state.head)
                temp = _stage_genesis(manifest_path, genesis)
                try:
                    with _lock_staged(temp):
                        _swap_in(temp, manifest_path)
                        if head_anchor is not None:
                            head_anchor.write(manifest_hash(genesis))
                except BaseException:
                    with suppress(OSError):
                        os.unlink(temp)
                    raise
                return genesis
        manifests = _read_manifests(manifest_path)
        sealed_here = {m["run_id"] for m in manifests if isinstance(m.get("run_id"), str)}
        number = archived[-1][0] + 1 if archived else 1
        manifest_archive, audit_archive = _segment_path(manifest_path, number), _segment_path(audit_path, number)
        if os.path.lexists(manifest_archive) or os.path.lexists(audit_archive):
            raise ValueError(f"archive {manifest_archive} or {audit_archive} already exists")
        skip = sealed_here | _archived_sealed_ids(manifest_path)
        temps: list[str] = []
        with ExitStack() as held:
            try:
                out, audit_temp = _stage_temp(audit_path)
                temps.append(audit_temp)
                with out:
                    # One unlocked pass digests the runs sealed here and copies the pending lines.
                    digests: dict[str, _RunDigest] = {}
                    end = 0
                    with suppress(FileNotFoundError):
                        end = _carry_pending(audit_path, out, 0, skip, (), sealed=sealed_here, digests=digests)
                    state = _outgoing_state(manifest_path, manifests, digests, **check)
                    genesis = _genesis_entry(signer, log_id, state.head)
                    fd = _open_locked(audit_path, os.O_RDWR | os.O_CREAT, exclusive=True)
                    try:
                        # Writers now wait on the lock: copy only the tail appended meanwhile.
                        _carry_pending(audit_path, out, end, skip, skip, final=True)
                        out.flush()
                        os.fsync(out.fileno())
                        os.chmod(audit_temp, stat.S_IMODE(os.stat(audit_path).st_mode))
                        manifest_temp = _stage_genesis(manifest_path, genesis)
                        temps.append(manifest_temp)
                        # Held until the genesis is anchored: a sealer opening the new manifest waits for it.
                        held.enter_context(_lock_staged(manifest_temp))
                        os.link(manifest_path, manifest_archive)
                        try:
                            os.link(audit_path, audit_archive)
                        except BaseException:
                            with suppress(OSError):
                                os.unlink(manifest_archive)
                            raise
                        _fsync_parents(manifest_archive, audit_archive)
                        # The audit first: after a crash writers append to the new file, never to an archive.
                        _swap_in(audit_temp, audit_path)
                        temps.remove(audit_temp)
                        _swap_in(manifest_temp, manifest_path)
                        temps.remove(manifest_temp)
                    finally:
                        if fd is not None:
                            os.close(fd)
            except BaseException:
                for temp in temps:
                    with suppress(OSError):
                        os.unlink(temp)
                raise
            if head_anchor is not None:
                head_anchor.write(manifest_hash(genesis))
            return genesis
