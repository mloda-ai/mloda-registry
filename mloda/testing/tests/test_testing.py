"""Tests for mloda.testing package."""

from mloda.testing.base import FeatureGroupTestBase


def test_feature_group_test_base_import() -> None:
    """Verify FeatureGroupTestBase can be imported."""
    assert FeatureGroupTestBase is not None
