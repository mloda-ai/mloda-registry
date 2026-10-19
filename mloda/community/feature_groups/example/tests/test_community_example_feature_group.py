"""Tests for CommunityExampleFeatureGroup."""

from mloda.community.feature_groups.example import CommunityExampleFeatureGroup
from mloda.testing.base import ALL_COMPUTE_FRAMEWORKS, FeatureGroupTestBase


class TestCommunityExampleFeatureGroup(FeatureGroupTestBase):
    """Contract check for CommunityExampleFeatureGroup (default class-name match, root group)."""

    feature_group_class = CommunityExampleFeatureGroup
    expected_compute_frameworks = ALL_COMPUTE_FRAMEWORKS
    matching_feature_names = [CommunityExampleFeatureGroup.get_class_name()]
    non_matching_feature_names = ["unrelated_feature"]
    expected_input_features = {CommunityExampleFeatureGroup.get_class_name(): None}
