"""Tests for ExampleBFeatureGroup."""

from mloda.community.feature_groups.example.example_b import ExampleBFeatureGroup
from mloda.community.feature_groups.example.community_example_feature_group import CommunityExampleFeatureGroup


def test_example_b_extends_base() -> None:
    """ExampleBFeatureGroup should extend CommunityExampleFeatureGroup."""
    assert issubclass(ExampleBFeatureGroup, CommunityExampleFeatureGroup)


def test_example_b_calculate_feature() -> None:
    """calculate_feature should return example B specific data."""
    result = ExampleBFeatureGroup.calculate_feature(None, None)
    assert result == {"example_b": "data", "source": "community_example_base"}
