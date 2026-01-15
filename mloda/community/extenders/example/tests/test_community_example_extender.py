"""Tests for CommunityExampleExtender."""

import pytest

from mloda.steward import Extender


class TestCommunityExampleExtenderImport:
    """Test that CommunityExampleExtender can be imported."""

    def test_import_from_package(self) -> None:
        """Test that CommunityExampleExtender can be imported from the package."""
        from mloda.community.extenders.example import CommunityExampleExtender

        assert CommunityExampleExtender is not None

    def test_class_is_accessible(self) -> None:
        """Test that the class is a proper class object."""
        from mloda.community.extenders.example import CommunityExampleExtender

        assert isinstance(CommunityExampleExtender, type)


class TestCommunityExampleExtenderInheritance:
    """Test that CommunityExampleExtender inherits from Extender."""

    def test_inherits_from_extender(self) -> None:
        """Test that CommunityExampleExtender is a subclass of Extender."""
        from mloda.community.extenders.example import CommunityExampleExtender

        assert issubclass(CommunityExampleExtender, Extender)

    def test_instance_is_extender(self) -> None:
        """Test that an instance is an instance of Extender."""
        from mloda.community.extenders.example import CommunityExampleExtender

        instance = CommunityExampleExtender()
        assert isinstance(instance, Extender)


class TestCommunityExampleExtenderBasicFunctionality:
    """Test basic functionality of CommunityExampleExtender."""

    def test_has_name_attribute(self) -> None:
        """Test that the extender has a name or identifier."""
        from mloda.community.extenders.example import CommunityExampleExtender

        # Extender implementations should have some form of identification
        instance = CommunityExampleExtender()
        assert hasattr(instance, "__class__")
        assert instance.__class__.__name__ == "CommunityExampleExtender"

    def test_can_instantiate(self) -> None:
        """Test that the extender can be instantiated."""
        from mloda.community.extenders.example import CommunityExampleExtender

        instance = CommunityExampleExtender()
        assert instance is not None
