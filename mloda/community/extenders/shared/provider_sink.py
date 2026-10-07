"""Provider resolution and pickle-drop steps shared by sink-backed extenders; must not import opentelemetry."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from mloda.steward import WarnOncePerInstance, pickle_failure_reason

T = TypeVar("T")


def configured_provider(injected: T | None, use_sdk_defaults: bool, ambient: Callable[[], T]) -> T | None:
    """Injected provider wins, else the ambient one when use_sdk_defaults, else None."""
    if injected is not None:
        return injected
    if use_sdk_defaults:
        return ambient()
    return None


def drop_unpicklable_provider(
    state: dict[str, Any], noun: str, *, owner_name: str, warning: WarnOncePerInstance, log: logging.Logger
) -> None:
    """Replace an unpicklable state[f"_{noun}"] with None, warning once."""
    key = f"_{noun}"
    provider = state.get(key)
    reason = pickle_failure_reason(provider) if provider is not None else None
    if reason is None:
        return
    warning.warn_once(
        lambda: log.warning(
            f"{owner_name} drops an injected {noun} when pickled or copied because it "
            f"isn't picklable ({reason}); the copy is inert unless use_sdk_defaults=True, which lets it "
            "resolve a provider installed in its own process, e.g. via child_bootstrap under "
            "MULTIPROCESSING."
        )
    )
    state[key] = None
