"""Example B FeatureGroup implementation."""

from typing import Any

from mloda.community.feature_groups.example.community_example_feature_group import CommunityExampleFeatureGroup


class ExampleBFeatureGroup(CommunityExampleFeatureGroup):
    """Example B FeatureGroup extending the community example base."""

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        """Return example B specific data."""
        return {"example_b": "data", "source": "community_example_base"}
