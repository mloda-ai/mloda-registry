"""Community Example FeatureGroup implementation."""

from typing import Any

from mloda.provider import FeatureGroup


class CommunityExampleFeatureGroup(FeatureGroup):
    """Community Example FeatureGroup for demonstrating plugin structure."""

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        """Return dummy data."""
        return {"community_example": "data"}
