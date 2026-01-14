"""Tests for EnterpriseExampleFeatureGroup."""

from mloda.enterprise.feature_groups.example import EnterpriseExampleFeatureGroup
from mloda.testing.base import FeatureGroupTestBase


class TestEnterpriseExampleFeatureGroup(FeatureGroupTestBase):
    """Test EnterpriseExampleFeatureGroup using FeatureGroupTestBase."""

    feature_group_class = EnterpriseExampleFeatureGroup

    def test_feature_group_class_set(self) -> None:
        """Verify feature_group_class is set."""
        assert self.feature_group_class is EnterpriseExampleFeatureGroup
