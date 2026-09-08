"""Cross-process pickle-drop assertion shared by extender packages with a process-local sink registry."""

from __future__ import annotations

import multiprocessing
import pickle  # nosec

from mloda.steward import Extender


def _child_reports_attribute_is_none(
    payload: bytes, attribute_name: str, result_queue: "multiprocessing.Queue[bool]"
) -> None:
    """Module-level so the spawn context can import it by qualified name; unpickles payload in the child."""
    import pickle  # nosec

    copy = pickle.loads(payload)  # nosec
    result_queue.put(getattr(copy, attribute_name) is None)


def assert_spawned_child_drops(extender: Extender, attribute_name: str) -> None:
    """Pickle `extender`, unpickle it in a real spawned child process (not a monkeypatched
    simulation), and assert `attribute_name` resolved to None there: a process-local registry hit
    must never survive a real process boundary."""
    payload = pickle.dumps(extender)  # nosec

    ctx = multiprocessing.get_context("spawn")
    result_queue: "multiprocessing.Queue[bool]" = ctx.Queue()
    process = ctx.Process(target=_child_reports_attribute_is_none, args=(payload, attribute_name, result_queue))
    process.start()
    try:
        is_none = result_queue.get(timeout=30)
    finally:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert is_none is True
