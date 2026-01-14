"""Base test utilities for mloda plugins."""

from typing import Any


class FeatureGroupTestBase:
    """Base class for testing FeatureGroup implementations."""

    feature_group_class: type[Any] | None = None
