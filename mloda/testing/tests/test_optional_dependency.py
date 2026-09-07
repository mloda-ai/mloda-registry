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


class TestOptionalDependencyPackageTestMixinShape:
    def test_declared_attribute_names_are_all_consumed_by_the_installed_test(self) -> None:
        _ProbeOpenLineageManifest().test_manifest_lists_the_extender_when_installed()
