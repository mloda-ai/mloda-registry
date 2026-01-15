"""Tests for CommunityExampleComputeFramework."""

import pytest

from mloda.provider import ComputeFramework


class TestCommunityExampleComputeFrameworkImport:
    """Test that CommunityExampleComputeFramework can be imported."""

    def test_import_from_package(self) -> None:
        """Test that CommunityExampleComputeFramework can be imported from the package."""
        from mloda.community.compute_frameworks.example import CommunityExampleComputeFramework

        assert CommunityExampleComputeFramework is not None

    def test_class_is_accessible(self) -> None:
        """Test that the class is a proper class object."""
        from mloda.community.compute_frameworks.example import CommunityExampleComputeFramework

        assert isinstance(CommunityExampleComputeFramework, type)


class TestCommunityExampleComputeFrameworkInheritance:
    """Test that CommunityExampleComputeFramework inherits from ComputeFramework."""

    def test_inherits_from_compute_framework(self) -> None:
        """Test that CommunityExampleComputeFramework is a subclass of ComputeFramework."""
        from mloda.community.compute_frameworks.example import CommunityExampleComputeFramework

        assert issubclass(CommunityExampleComputeFramework, ComputeFramework)

    def test_instance_is_compute_framework(self) -> None:
        """Test that an instance is an instance of ComputeFramework."""
        from mloda.community.compute_frameworks.example import CommunityExampleComputeFramework

        instance = CommunityExampleComputeFramework()
        assert isinstance(instance, ComputeFramework)


class TestCommunityExampleComputeFrameworkBasicFunctionality:
    """Test basic functionality of CommunityExampleComputeFramework."""

    def test_has_name_attribute(self) -> None:
        """Test that the compute framework has a name or identifier."""
        from mloda.community.compute_frameworks.example import CommunityExampleComputeFramework

        # ComputeFramework implementations should have some form of identification
        instance = CommunityExampleComputeFramework()
        assert hasattr(instance, "__class__")
        assert instance.__class__.__name__ == "CommunityExampleComputeFramework"

    def test_can_instantiate(self) -> None:
        """Test that the compute framework can be instantiated."""
        from mloda.community.compute_frameworks.example import CommunityExampleComputeFramework

        instance = CommunityExampleComputeFramework()
        assert instance is not None
