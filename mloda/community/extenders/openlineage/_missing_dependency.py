"""Builds a module-level __getattr__ raised when openlineage-python is not installed."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_DISTRIBUTION = "openlineage-python"
_EXTRA = "mloda-community[openlineage]"
_EXTENDER_NAME = "OpenLineageExtender"


def blames_root(exc: ImportError, root: str) -> bool:
    """True if ``exc`` names ``root`` directly, or its innermost traceback frame runs inside
    ``root`` (a transitive dependency can fail under its own name, not ``root``'s)."""
    if (exc.name or "").split(".")[0] == root:
        return True
    tb = exc.__traceback__
    if tb is None:
        return False
    while tb.tb_next is not None:
        tb = tb.tb_next
    module_name = tb.tb_frame.f_globals.get("__name__")
    return isinstance(module_name, str) and (module_name == root or module_name.startswith(f"{root}."))


def build_getattr(cause: BaseException) -> Callable[[str], Any]:
    """Return a module __getattr__ chaining ``cause`` (the real import failure) onto the deferred error."""

    def __getattr__(name: str) -> Any:
        if name == _EXTENDER_NAME:
            raise ImportError(f"{_EXTENDER_NAME} requires '{_DISTRIBUTION}'; install it via '{_EXTRA}'.") from cause
        raise AttributeError(name)

    return __getattr__
