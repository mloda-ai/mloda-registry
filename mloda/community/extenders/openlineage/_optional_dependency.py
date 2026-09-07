"""Guard for an optional third-party dependency; vendored byte-identically because each leaf is
independently installable and shares no runtime package to import this from."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Callable


def _traceback_blames(exc: ImportError, root: str) -> bool:
    tb = exc.__traceback__
    while tb is not None:
        module_name = tb.tb_frame.f_globals.get("__name__")
        if isinstance(module_name, str) and (module_name == root or module_name.startswith(f"{root}.")):
            return True
        tb = tb.tb_next
    return False


def _blames(exc: ImportError, root: str) -> bool:
    name = exc.name
    if name is not None and (name == root or name.startswith(f"{root}.")):
        return True
    return _traceback_blames(exc, root)


def reraise_unless_optional(exc: ImportError, root: str) -> None:
    """Re-raise ``exc`` unchanged unless ``root`` (named directly, or blamed from one of its own
    frames, e.g. a missing transitive dependency) is responsible for it."""
    if not _blames(exc, root):
        raise exc


def _root_is_installed(root: str) -> bool:
    try:
        return importlib.util.find_spec(root) is not None
    except (ImportError, ValueError):
        return False


def log_unavailable(
    logger: logging.Logger,
    exc: ImportError,
    root: str,
    distribution: str,
    extra: str,
    subject: str,
) -> None:
    """Log why ``subject`` is unavailable: INFO when ``root`` is genuinely absent, WARNING (naming the
    real failure) when ``root`` is installed but unusable."""
    if _root_is_installed(root):
        logger.warning(
            "%s unavailable: install '%s' via '%s' (real cause, %s: %s).",
            subject,
            distribution,
            extra,
            type(exc).__name__,
            exc,
        )
    else:
        logger.info("%s unavailable: install '%s' via '%s'.", subject, distribution, extra)


def missing_attribute(subject: str, distribution: str, extra: str, cause: ImportError) -> Callable[[str], Any]:
    """Build a module-level ``__getattr__`` that raises for ``subject`` and chains ``cause``."""

    def __getattr__(name: str) -> Any:
        if name == subject:
            raise ImportError(f"{subject} requires '{distribution}'; install it via '{extra}'.") from cause
        raise AttributeError(name)

    return __getattr__
