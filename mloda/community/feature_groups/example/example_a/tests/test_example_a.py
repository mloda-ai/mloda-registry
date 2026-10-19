"""Tests for ExampleAFeatureGroup."""

from mloda.community.feature_groups.example.community_example_feature_group import CommunityExampleFeatureGroup
from mloda.community.feature_groups.example.example_a import ExampleAFeatureGroup
from mloda.testing.base import ALL_COMPUTE_FRAMEWORKS, FeatureGroupTestBase


class TestExampleAFeatureGroupContract(FeatureGroupTestBase):
    feature_group_class = ExampleAFeatureGroup
    expected_compute_frameworks = ALL_COMPUTE_FRAMEWORKS
    matching_feature_names = [ExampleAFeatureGroup.get_class_name()]
    non_matching_feature_names = [CommunityExampleFeatureGroup.get_class_name(), "unrelated_feature"]
    expected_input_features = {ExampleAFeatureGroup.get_class_name(): None}


def test_example_a_extends_base() -> None:
    """ExampleAFeatureGroup should extend CommunityExampleFeatureGroup."""
    assert issubclass(ExampleAFeatureGroup, CommunityExampleFeatureGroup)


def test_example_a_calculate_feature() -> None:
    """calculate_feature should return example A specific data."""
    result = ExampleAFeatureGroup.calculate_feature(None, None)
    assert result == {"example_a": "data", "source": "community_example_base"}
