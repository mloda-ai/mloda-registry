"""Pure classification for the "wheel is absent" precondition shared by
``binary_example/tests/test_binary_example_feature_group.py`` and ``test_licensing_invariants.py``:
an installed wheel is either a deliberate opt-in (skip) or unexpected (fail), never silently
either.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal


def classify_wheel_presence(spec_present: bool, opt_in: bool) -> Literal["absent", "opted_in", "unexpected"]:
    """``"absent"`` if no wheel is installed; otherwise ``"opted_in"`` when the
    ``MLODA_REAL_WHEEL=1`` opt-in was set deliberately, ``"unexpected"`` when it was not."""
    if not spec_present:
        return "absent"
    return "opted_in" if opt_in else "unexpected"


def unexpected_wheel_plugin_ids(entries: Iterable[tuple[str, bool]], opt_in: bool) -> list[str]:
    """Every ``plugin_id`` from ``entries`` (``(plugin_id, spec_present)`` pairs) classified
    ``"unexpected"``. Checks every entry, never stopping at the first installed one."""
    return [
        plugin_id
        for plugin_id, spec_present in entries
        if classify_wheel_presence(spec_present, opt_in) == "unexpected"
    ]
