"""Shared test re-exports for window_aggregation tests."""

from __future__ import annotations

from mloda.testing.feature_groups.data_operations.helpers import (
    make_feature_set,
)

# Re-export make_feature_set so existing imports from conftest still work.
__all__ = ["make_feature_set"]
