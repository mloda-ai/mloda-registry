"""Resolving and probing a binary before it is ever run (contract: Invocation, Capabilities):
locate the executable, run ``--version``/``--capabilities`` up front, validate their shape, and
cache the result per process so a warm binary is never re-probed.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shutil
import subprocess  # nosec
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mloda.community.feature_groups.binary_model.errors import BinaryUnavailableError

logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
COLUMN_TYPE_VOCABULARY = frozenset({"int64", "float64", "utf8", "boolean"})
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+\-]+)?$")

_CacheKey = tuple[str, tuple[str, ...], int, int]


@dataclass(frozen=True)
class BinaryCapabilities:
    contract: int
    plugin_id: str
    version: str
    operations: frozenset[str]
    column_types: frozenset[str]


@dataclass(frozen=True)
class ResolvedBinary:
    argv: tuple[str, ...]
    capabilities: BinaryCapabilities


_capability_cache: dict[_CacheKey, BinaryCapabilities] = {}


def clear_capability_cache() -> None:
    """Drop every cached probe result, forcing the next ``resolve_binary`` call to re-probe."""
    _capability_cache.clear()


def _resolve_executable_path(candidate: str, path: str) -> Path:
    """Resolve ``argv[0]`` to an absolute, executable, regular file path (contract: Platform
    naming), or raise ``BinaryUnavailableError``. Made absolute with ``os.path.abspath``, not a
    symlink-following resolve: a symlinked interpreter (a venv's own launcher, used as a test
    stand-in binary) must stay invocable as the symlink, since dereferencing it would drop the
    venv context that makes it resolvable at all. A bare name (no path separator) is looked up
    only via ``path``, the resolved environment's own ``PATH``, never the parent process's."""
    if os.sep in candidate or (os.altsep is not None and os.altsep in candidate):
        resolved = Path(os.path.abspath(candidate))
    else:
        found = shutil.which(candidate, path=path)
        if found is None:
            raise BinaryUnavailableError(f"binary not found on PATH: {candidate!r}")
        resolved = Path(found)

    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise BinaryUnavailableError(f"binary is not an executable regular file: {resolved}")
    return resolved


def _build_argv(plugin_id: str, override: Sequence[str] | str | os.PathLike[str] | None) -> list[str]:
    if override is None:
        try:
            module = importlib.import_module(plugin_id)
        except ModuleNotFoundError as exc:
            raise BinaryUnavailableError(f"binary package for plugin_id {plugin_id!r} is not installed: {exc}") from exc
        try:
            path = module.binary_path()
        except FileNotFoundError as exc:
            raise BinaryUnavailableError(f"binary data file missing from the {plugin_id!r} wheel: {exc}") from exc
        return [str(path)]
    if isinstance(override, (str, os.PathLike)):
        return [str(override)]
    return list(override)


def _run_probe(argv: list[str], flag: str, env: Mapping[str, str], timeout: float | None) -> bytes:
    try:
        result = subprocess.run(  # nosec B603
            [*argv, flag], env=dict(env), capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise BinaryUnavailableError(f"binary {argv[0]!r} timed out probing {flag}") from exc
    except OSError as exc:
        raise BinaryUnavailableError(f"binary {argv[0]!r} could not be run for {flag}: {exc}") from exc
    if result.returncode != 0:
        raise BinaryUnavailableError(f"binary {argv[0]!r} exited {result.returncode} probing {flag}")
    return bytes(result.stdout)


def _parse_version(argv: list[str], plugin_id: str, stdout: bytes) -> str:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BinaryUnavailableError(f"binary {argv[0]!r} --version output is not valid UTF-8: {exc}") from exc
    lines = text.splitlines()
    if len(lines) != 1:
        raise BinaryUnavailableError(f"binary {argv[0]!r} --version must print exactly one line, got {len(lines)}")
    parts = lines[0].split(" ")
    if len(parts) != 2 or parts[0] != plugin_id or _SEMVER_PATTERN.match(parts[1]) is None:
        raise BinaryUnavailableError(
            f"binary {argv[0]!r} --version must print '{plugin_id} <semver>', got {lines[0]!r}"
        )
    return parts[1]


def _parse_capabilities(argv: list[str], plugin_id: str, stdout: bytes) -> BinaryCapabilities:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BinaryUnavailableError(f"binary {argv[0]!r} --capabilities output is not valid UTF-8: {exc}") from exc

    lines = text.splitlines()
    if len(lines) != 1:
        raise BinaryUnavailableError(
            f"binary {argv[0]!r} --capabilities must print exactly one JSON object line, got {len(lines)} lines"
        )
    try:
        payload: Any = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise BinaryUnavailableError(f"binary {argv[0]!r} --capabilities output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BinaryUnavailableError(f"binary {argv[0]!r} --capabilities must print a JSON object")

    contract = payload.get("contract")
    if not isinstance(contract, int) or isinstance(contract, bool):
        raise BinaryUnavailableError(f"binary {argv[0]!r} --capabilities is missing an integer 'contract' key")
    if contract != CONTRACT_VERSION:
        raise BinaryUnavailableError(
            f"binary {argv[0]!r} implements contract {contract}, but this mixin requires contract {CONTRACT_VERSION}"
        )

    reported_plugin_id = payload.get("plugin_id")
    if reported_plugin_id != plugin_id:
        raise BinaryUnavailableError(
            f"binary {argv[0]!r} reports plugin_id {reported_plugin_id!r}, expected {plugin_id!r}"
        )

    operations = payload.get("operations")
    if not isinstance(operations, list) or not all(isinstance(op, str) for op in operations):
        raise BinaryUnavailableError(f"binary {argv[0]!r} --capabilities 'operations' must be a list of strings")

    column_types = payload.get("column_types")
    if not isinstance(column_types, list) or not all(isinstance(ct, str) for ct in column_types):
        raise BinaryUnavailableError(f"binary {argv[0]!r} --capabilities 'column_types' must be a list of strings")
    column_types_set = frozenset(column_types)
    if not column_types_set <= COLUMN_TYPE_VOCABULARY:
        raise BinaryUnavailableError(
            f"binary {argv[0]!r} --capabilities 'column_types' {sorted(column_types_set)} "
            f"is not a subset of {sorted(COLUMN_TYPE_VOCABULARY)}"
        )

    return BinaryCapabilities(
        contract=contract,
        plugin_id=plugin_id,
        version="",
        operations=frozenset(operations),
        column_types=column_types_set,
    )


def resolve_binary(
    plugin_id: str,
    override: Sequence[str] | str | os.PathLike[str] | None,
    *,
    env: Mapping[str, str],
    timeout: float | None,
) -> ResolvedBinary:
    """Resolve ``plugin_id`` to an executable argv, probe its ``--version`` and
    ``--capabilities`` unless already cached, and return a ``ResolvedBinary`` (contract:
    Invocation, Capabilities, Platform naming)."""
    argv = _build_argv(plugin_id, override)
    resolved_path = _resolve_executable_path(argv[0], env.get("PATH", os.defpath))
    argv = [str(resolved_path), *argv[1:]]

    stat_result = resolved_path.stat()
    cache_key: _CacheKey = (plugin_id, tuple(argv), stat_result.st_size, stat_result.st_mtime_ns)
    cached = _capability_cache.get(cache_key)
    if cached is not None:
        return ResolvedBinary(argv=tuple(argv), capabilities=cached)

    probe_env = {key: value for key, value in env.items() if key not in ("MLODA_LICENSE_FILE", "MLODA_LICENSE_KEY")}

    version_stdout = _run_probe(argv, "--version", probe_env, timeout)
    version = _parse_version(argv, plugin_id, version_stdout)

    capabilities_stdout = _run_probe(argv, "--capabilities", probe_env, timeout)
    capabilities = _parse_capabilities(argv, plugin_id, capabilities_stdout)
    capabilities = BinaryCapabilities(
        contract=capabilities.contract,
        plugin_id=capabilities.plugin_id,
        version=version,
        operations=capabilities.operations,
        column_types=capabilities.column_types,
    )

    logger.debug("resolved binary %s version %s", plugin_id, version)
    _capability_cache[cache_key] = capabilities
    return ResolvedBinary(argv=tuple(argv), capabilities=capabilities)
