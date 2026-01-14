"""Tests for CommunityExampleFeatureGroup."""

from mloda.community.feature_groups.example import CommunityExampleFeatureGroup
from mloda.testing.base import FeatureGroupTestBase


class TestCommunityExampleFeatureGroup(FeatureGroupTestBase):
    """Test CommunityExampleFeatureGroup using FeatureGroupTestBase."""

    feature_group_class = CommunityExampleFeatureGroup

    def test_feature_group_class_set(self) -> None:
        """Verify feature_group_class is set."""
        assert self.feature_group_class is CommunityExampleFeatureGroup
