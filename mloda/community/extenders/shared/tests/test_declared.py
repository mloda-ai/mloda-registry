"""Tests for the declared_scalars helper."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from mloda.community.extenders.shared.declared import declared_scalars, same_scalar


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
def test_declared_scalars_drops_non_finite_floats_non_scalars_and_ints_outside_int64(value: Any) -> None:
    assert declared_scalars({"keep": "k", "drop": value}) == {"keep": "k"}


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63)], ids=["int64_max", "int64_min"])
def test_declared_scalars_keeps_the_int64_bounds(value: int) -> None:
    result = declared_scalars({"n": value})

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
