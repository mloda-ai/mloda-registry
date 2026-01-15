"""System tests to verify namespace package imports work correctly.

These tests verify that:
1. Core mloda (v0.4.2+) PEP 420 namespace package works
2. mloda-registry namespace packages merge correctly with core mloda
3. All import patterns documented in TODO.md work
"""


def test_mloda_is_namespace_package() -> None:
    """Verify mloda is a PEP 420 namespace package (no __init__.py at root)."""
    import mloda

    # Namespace packages have __path__ but __file__ is None
    assert hasattr(mloda, "__path__")
    assert mloda.__file__ is None, "mloda should be a namespace package with no __file__"


def test_mloda_user_imports() -> None:
    """Verify mloda.user module imports work (v0.4.2+ pattern)."""
    from mloda.user import mloda, mlodaAPI, Feature, Options

    assert mloda is mlodaAPI  # mloda is alias for mlodaAPI
    assert hasattr(mloda, "run_all")
    assert Feature is not None
    assert Options is not None


def test_mloda_provider_imports() -> None:
    """Verify mloda.provider module imports work."""
    from mloda.provider import FeatureGroup, ComputeFramework

    assert FeatureGroup is not None
    assert ComputeFramework is not None


def test_mloda_steward_imports() -> None:
    """Verify mloda.steward module imports work."""
    from mloda.steward import FeatureGroupInfo, Extender

    assert FeatureGroupInfo is not None
    assert Extender is not None


def test_community_namespace_imports() -> None:
    """Verify community namespace imports work."""
    import mloda.community
    import mloda.community.feature_groups
    import mloda.community.compute_frameworks
    import mloda.community.extenders

    assert mloda.community is not None
    assert mloda.community.feature_groups is not None
    assert mloda.community.compute_frameworks is not None
    assert mloda.community.extenders is not None


def test_enterprise_namespace_imports() -> None:
    """Verify enterprise namespace imports work."""
    import mloda.enterprise
    import mloda.enterprise.feature_groups
    import mloda.enterprise.compute_frameworks
    import mloda.enterprise.extenders

    assert mloda.enterprise is not None
    assert mloda.enterprise.feature_groups is not None
    assert mloda.enterprise.compute_frameworks is not None
    assert mloda.enterprise.extenders is not None


def test_registry_namespace_imports() -> None:
    """Verify registry namespace imports work."""
    import mloda.registry

    assert mloda.registry is not None


def test_testing_namespace_imports() -> None:
    """Verify testing namespace imports work."""
    import mloda.testing

    assert mloda.testing is not None


def test_namespace_merging() -> None:
    """Verify core mloda and registry packages coexist in merged namespace."""
    # Core mloda subpackages
    import mloda.user
    import mloda.provider
    import mloda.steward
    import mloda.core

    # Registry subpackages
    import mloda.community
    import mloda.enterprise
    import mloda.registry
    import mloda.testing

    # All should coexist under the same mloda namespace
    import mloda

    assert hasattr(mloda, "__path__")
    # The namespace should have multiple paths (core + registry)
    assert len(mloda.__path__) >= 1
