"""Tests for ExampleAFeatureGroup."""

from mloda.community.feature_groups.example.example_a import ExampleAFeatureGroup
from mloda.community.feature_groups.example.community_example_feature_group import CommunityExampleFeatureGroup


def test_example_a_extends_base() -> None:
    """ExampleAFeatureGroup should extend CommunityExampleFeatureGroup."""
    assert issubclass(ExampleAFeatureGroup, CommunityExampleFeatureGroup)


def test_example_a_calculate_feature() -> None:
    """calculate_feature should return example A specific data."""
    result = ExampleAFeatureGroup.calculate_feature(None, None)
    assert result == {"example_a": "data", "source": "community_example_base"}
