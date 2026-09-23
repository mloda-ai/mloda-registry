"""Tests for bound_method and class_attribute: the owning-class helpers shared by extenders that read
class-level declarations off the wrapped call (moved here from lineage_extender's private copies)."""

from __future__ import annotations

import functools
from typing import Any

from mloda.community.extenders.shared.bound_method import bound_method, class_attribute


def _instrumented_wrapper(func: Any) -> Any:
    """A plain function standing in for core's instrument() wrapper: copies __self__ but not
    __func__, and carries __wrapped__ via functools.wraps."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    if hasattr(func, "__self__"):
        wrapper.__self__ = func.__self__  # type: ignore[attr-defined]
    return wrapper


class _Owner:
    marker = "owner-value"

    @classmethod
    def classmethod_call(cls) -> Any:
        return cls

    def instance_call(self) -> Any:
        return self


def test_bound_method_returns_plain_function_unchanged() -> None:
    def plain() -> None:
        return None

    assert bound_method(plain) is plain


def test_bound_method_unwraps_an_instrumented_wrapper_to_the_bound_classmethod() -> None:
    bound = _Owner.classmethod_call
    wrapper = _instrumented_wrapper(bound)

    assert bound_method(wrapper) is bound


def test_bound_method_unwraps_an_instrumented_wrapper_to_the_bound_instance_method() -> None:
    owner = _Owner()
    bound = owner.instance_call
    wrapper = _instrumented_wrapper(bound)

    assert bound_method(wrapper) is bound


def test_class_attribute_reads_class_level_value_via_classmethod() -> None:
    assert class_attribute(_Owner.classmethod_call, "marker") == "owner-value"


def test_class_attribute_reads_class_level_value_via_instance_method() -> None:
    assert class_attribute(_Owner().instance_call, "marker") == "owner-value"


def test_class_attribute_unwraps_an_instrumented_wrapper_first() -> None:
    wrapper = _instrumented_wrapper(_Owner.classmethod_call)

    assert class_attribute(wrapper, "marker") == "owner-value"


def test_class_attribute_returns_none_when_attribute_is_missing() -> None:
    assert class_attribute(_Owner.classmethod_call, "does_not_exist") is None


def test_class_attribute_returns_none_for_a_plain_function() -> None:
    def plain() -> None:
        return None

    assert class_attribute(plain, "marker") is None
