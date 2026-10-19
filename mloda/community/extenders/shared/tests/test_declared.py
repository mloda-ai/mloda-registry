"""Tests for the declared_scalars helper."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from mloda.community.extenders.shared.declared import declared_scalars, readable_declared, same_scalar


@pytest.mark.parametrize("declared", [None, {}], ids=["none", "empty"])
def test_declared_scalars_of_nothing_is_an_empty_dict(declared: Mapping[str, Any] | None) -> None:
    assert declared_scalars(declared) == {}


def test_declared_scalars_keeps_scalars_with_their_types() -> None:
    result = declared_scalars({"text": "x", "flag": True, "count": 1, "scale": 0.5})

    assert result == {"text": "x", "flag": True, "count": 1, "scale": 0.5}
    assert [type(v) for v in result.values()] == [str, bool, int, float]


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
        pytest.param({"keep": "k", "n": 1}, {"keep": "k", "n": 1}, id="scalars"),
    ],
)
def test_readable_declared_is_none_when_unreadable_else_a_plain_copy(
    declared: Mapping[str, Any] | None, expected: dict[str, Any] | None
) -> None:
    result = readable_declared(declared)

    assert result == expected
    if expected is not None:
        assert type(result) is dict
        assert result is not declared
