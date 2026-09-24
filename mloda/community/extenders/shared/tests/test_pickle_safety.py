"""Tests for is_picklable: the trial-pickle helper shared by extenders that degrade gracefully
when an injected sink can't pickle.
"""

from __future__ import annotations

import threading

from mloda.community.extenders.shared import is_picklable


class _LockHolder:
    """A plain object holding a threading.Lock, which can never survive plain pickling."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


def test_is_picklable_true_for_plain_list() -> None:
    assert is_picklable([1, 2, 3]) is True


def test_is_picklable_true_for_plain_dict() -> None:
    assert is_picklable({"a": 1, "b": 2}) is True


def test_is_picklable_false_for_value_holding_a_lock() -> None:
    assert is_picklable(_LockHolder()) is False
