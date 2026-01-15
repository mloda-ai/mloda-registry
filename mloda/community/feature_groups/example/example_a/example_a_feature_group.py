"""Example A FeatureGroup implementation."""

from typing import Any

from mloda.community.feature_groups.example.community_example_feature_group import CommunityExampleFeatureGroup


class ExampleAFeatureGroup(CommunityExampleFeatureGroup):
    """Example A FeatureGroup extending the community example base."""

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        """Return example A specific data."""
        return {"example_a": "data", "source": "community_example_base"}
