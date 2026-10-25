"""Tests for the declared_scalars, readable_declared and same_scalar helpers."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from mloda.community.extenders.shared.declared import declared_scalars, readable_declared, same_scalar
from mloda.testing.extenders.hook_context import make_hook_context


@pytest.mark.parametrize("declared", [None, {}], ids=["none", "empty"])
def test_declared_scalars_of_nothing_is_an_empty_dict(declared: Mapping[str, Any] | None) -> None:
    assert declared_scalars(declared) == {}


def test_declared_scalars_keeps_scalars_with_their_types() -> None:
    result = declared_scalars({"text": "x", "flag": True, "count": 1, "scale": 0.5})

    assert result == {"text": "x", "flag": True, "count": 1, "scale": 0.5}
    assert [type(v) for v in result.values()] == [str, bool, int, float]


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), [1], {"a": 1}, None, object(), 2**63, -(2**63) - 1, 10**5000],
    ids=["nan", "inf", "neg_inf", "list", "dict", "none", "object", "above_int64", "below_int64", "huge_int"],
)
def test_hook_context_drops_non_finite_floats_non_scalars_and_ints_outside_int64_before_declared_scalars(
    value: Any,
) -> None:
    context = make_hook_context(declared_attributes={"keep": "k", "drop": value})

    assert declared_scalars(context.declared_attributes) == {"keep": "k"}


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63)], ids=["int64_max", "int64_min"])
def test_hook_context_keeps_the_int64_bounds_for_declared_scalars(value: int) -> None:
    context = make_hook_context(declared_attributes={"n": value})
    result = declared_scalars(context.declared_attributes)

    assert result == {"n": value}
    assert type(result["n"]) is int


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (1, 1, True),
        ("a", "a", True),
        (1, True, False),
        (1, 1.0, False),
        (True, 1.0, False),
        ("a", "b", False),
        (0.5, 0.5, True),
    ],
)
def test_same_scalar_requires_the_same_type_and_an_equal_value(left: object, right: object, expected: bool) -> None:
    assert same_scalar(left, right) is expected


def test_declared_scalars_returns_a_fresh_plain_dict() -> None:
    source = {"a": 1}

    assert declared_scalars(source) is not source
    assert type(declared_scalars(MappingProxyType(source))) is dict


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        pytest.param(None, None, id="unreadable"),
        pytest.param({}, {}, id="empty"),
    ],
)
def test_readable_declared_is_none_when_unreadable_and_empty_when_empty(
    declared: Mapping[str, Any] | None, expected: dict[str, Any] | None
) -> None:
    assert readable_declared(declared) == expected


def test_readable_declared_returns_a_fresh_dict_of_the_filtered_scalars() -> None:
    declared: dict[str, Any] = {"keep": "k", "n": 1, "bad": float("nan"), "nested": {"a": 1}}
    context = make_hook_context(declared_attributes=declared)
    result = readable_declared(context.declared_attributes)

    assert result == {"keep": "k", "n": 1}
    assert type(result) is dict
    assert result is not context.declared_attributes
