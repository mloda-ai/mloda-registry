"""Self-test: a probe host declaring the mixin's attributes satisfies its own installed-manifest test."""

from __future__ import annotations

from mloda.testing.optional_dependency import OptionalDependencyPackageTestMixin


class _ProbeOpenLineageManifest(OptionalDependencyPackageTestMixin):
    package = "mloda.community.extenders.openlineage"
    root = "openlineage"
    distribution = "openlineage-python"
    extra = "mloda-community[openlineage]"
    extender_name = "OpenLineageExtender"
    extender_module = "openlineage_extender"
    broken_module = "openlineage.client.event_v2"
    broken_name = "RunEvent"
    transitive_dependency = "attr"


class TestOptionalDependencyPackageTestMixinShape:
    def test_declared_attribute_names_are_all_consumed_by_the_installed_test(self) -> None:
        annotated = set(OptionalDependencyPackageTestMixin.__annotations__)
        declared = set(vars(_ProbeOpenLineageManifest))
        missing = annotated - declared
        assert not missing, f"probe is missing declarations for: {sorted(missing)}"

        _ProbeOpenLineageManifest().test_manifest_lists_the_extender_when_installed()
