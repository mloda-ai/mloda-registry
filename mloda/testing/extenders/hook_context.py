"""Builds a HookContext test fixture with sane FEATURE_GROUP_CALCULATE_FEATURE defaults."""

from __future__ import annotations

from typing import Any

from mloda.steward import ExtenderHook, HookContext


def make_hook_context(**overrides: Any) -> HookContext:
    """Build a HookContext from the kit defaults merged under the overrides; mypy does not check the keywords."""
    defaults: dict[str, Any] = {
        "hook": ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        "feature_group_class": "mloda.testing.DummyFeatureGroup",
        "feature_group_version": "1",
        "feature_names": ("value_int",),
        "compute_framework_name": "PyArrowTable",
    }
    return HookContext(**{**defaults, **overrides})
