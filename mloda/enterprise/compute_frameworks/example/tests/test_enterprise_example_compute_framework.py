"""Tests for EnterpriseExampleComputeFramework."""

import pytest

from mloda.provider import ComputeFramework


class TestEnterpriseExampleComputeFrameworkImport:
    """Test that EnterpriseExampleComputeFramework can be imported."""

    def test_import_from_package(self) -> None:
        """Test that EnterpriseExampleComputeFramework can be imported from the package."""
        from mloda.enterprise.compute_frameworks.example import EnterpriseExampleComputeFramework

        assert EnterpriseExampleComputeFramework is not None

    def test_class_is_accessible(self) -> None:
        """Test that the class is a proper class object."""
        from mloda.enterprise.compute_frameworks.example import EnterpriseExampleComputeFramework

        assert isinstance(EnterpriseExampleComputeFramework, type)


class TestEnterpriseExampleComputeFrameworkInheritance:
    """Test that EnterpriseExampleComputeFramework inherits from ComputeFramework."""

    def test_inherits_from_compute_framework(self) -> None:
        """Test that EnterpriseExampleComputeFramework is a subclass of ComputeFramework."""
        from mloda.enterprise.compute_frameworks.example import EnterpriseExampleComputeFramework

        assert issubclass(EnterpriseExampleComputeFramework, ComputeFramework)

    def test_instance_is_compute_framework(self) -> None:
        """Test that an instance is an instance of ComputeFramework."""
        from mloda.enterprise.compute_frameworks.example import EnterpriseExampleComputeFramework

        instance = EnterpriseExampleComputeFramework()
        assert isinstance(instance, ComputeFramework)


class TestEnterpriseExampleComputeFrameworkBasicFunctionality:
    """Test basic functionality of EnterpriseExampleComputeFramework."""

    def test_has_name_attribute(self) -> None:
        """Test that the compute framework has a name or identifier."""
        from mloda.enterprise.compute_frameworks.example import EnterpriseExampleComputeFramework

        # ComputeFramework implementations should have some form of identification
        instance = EnterpriseExampleComputeFramework()
        assert hasattr(instance, "__class__")
        assert instance.__class__.__name__ == "EnterpriseExampleComputeFramework"

    def test_can_instantiate(self) -> None:
        """Test that the compute framework can be instantiated."""
        from mloda.enterprise.compute_frameworks.example import EnterpriseExampleComputeFramework

        instance = EnterpriseExampleComputeFramework()
        assert instance is not None
