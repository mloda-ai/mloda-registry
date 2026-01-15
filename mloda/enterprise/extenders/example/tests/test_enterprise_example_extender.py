"""Tests for EnterpriseExampleExtender."""

import pytest

from mloda.steward import Extender


class TestEnterpriseExampleExtenderImport:
    """Test that EnterpriseExampleExtender can be imported."""

    def test_import_from_package(self) -> None:
        """Test that EnterpriseExampleExtender can be imported from the package."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        assert EnterpriseExampleExtender is not None

    def test_class_is_accessible(self) -> None:
        """Test that the class is a proper class object."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        assert isinstance(EnterpriseExampleExtender, type)


class TestEnterpriseExampleExtenderInheritance:
    """Test that EnterpriseExampleExtender inherits from Extender."""

    def test_inherits_from_extender(self) -> None:
        """Test that EnterpriseExampleExtender is a subclass of Extender."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        assert issubclass(EnterpriseExampleExtender, Extender)

    def test_instance_is_extender(self) -> None:
        """Test that an instance is an instance of Extender."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        instance = EnterpriseExampleExtender()
        assert isinstance(instance, Extender)


class TestEnterpriseExampleExtenderBasicFunctionality:
    """Test basic functionality of EnterpriseExampleExtender."""

    def test_has_name_attribute(self) -> None:
        """Test that the extender has a name or identifier."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        # Extender implementations should have some form of identification
        instance = EnterpriseExampleExtender()
        assert hasattr(instance, "__class__")
        assert instance.__class__.__name__ == "EnterpriseExampleExtender"

    def test_can_instantiate(self) -> None:
        """Test that the extender can be instantiated."""
        from mloda.enterprise.extenders.example import EnterpriseExampleExtender

        instance = EnterpriseExampleExtender()
        assert instance is not None
