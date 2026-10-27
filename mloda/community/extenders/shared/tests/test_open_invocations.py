"""Direct tests for the shared invocation stack and exception type naming."""

from __future__ import annotations

import pytest

from mloda.community.extenders.shared.exception_type import qualified_type_name
from mloda.community.extenders.shared.open_invocations import OpenInvocationStack

_OWNER = object()
_OTHER_OWNER = object()
_STACK: OpenInvocationStack[str] = OpenInvocationStack("test_open_invocations")


def test_find_is_none_when_nothing_is_open() -> None:
    assert _STACK.find(_OWNER) is None


def test_open_is_visible_only_inside_its_block() -> None:
    with _STACK.open(_OWNER, "outer"):
        assert _STACK.find(_OWNER) == "outer"

    assert _STACK.find(_OWNER) is None


def test_nested_calls_return_the_newest_then_the_outer_invocation() -> None:
    with _STACK.open(_OWNER, "outer"):
        with _STACK.open(_OWNER, "inner"):
            assert _STACK.find(_OWNER) == "inner"
        assert _STACK.find(_OWNER) == "outer"


def test_owners_do_not_see_each_others_invocations() -> None:
    with _STACK.open(_OWNER, "mine"):
        assert _STACK.find(_OTHER_OWNER) is None
        with _STACK.open(_OTHER_OWNER, "theirs"):
            assert _STACK.find(_OWNER) == "mine"
            assert _STACK.find(_OTHER_OWNER) == "theirs"


def test_a_raising_block_still_removes_its_entry() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with _STACK.open(_OWNER, "raised"):
            raise RuntimeError("boom")

    assert _STACK.find(_OWNER) is None


def test_qualified_type_name_for_a_builtin_exception() -> None:
    assert qualified_type_name(ValueError("x")) == "builtins.ValueError"


def test_qualified_type_name_keeps_the_nested_class_qualname() -> None:
    class Outer:
        class NestedError(RuntimeError):
            pass

    assert qualified_type_name(Outer.NestedError("x")).endswith(".Outer.NestedError")
