"""Self-test: a probe host declaring the mixin's attributes satisfies its own installed-manifest test."""

from __future__ import annotations

from mloda.testing.optional_dependency import OptionalDependencyPackageTestMixin


class _ProbeOpenLineageManifest(OptionalDependencyPackageTestMixin):
    package = "mloda.community.extenders.openlineage"
    root = "openlineage"
    extender_name = "OpenLineageExtender"
    extender_module = "openlineage_extender"
    api_module = "openlineage.client"


class TestOptionalDependencyPackageTestMixinShape:
    def test_declared_attribute_names_are_all_consumed_by_the_installed_test(self) -> None:
        _ProbeOpenLineageManifest().test_manifest_lists_the_extender_when_installed()


class TestAdditionalExtenders:
    def test_default_exposes_only_the_primary_extender(self) -> None:
        assert _ProbeOpenLineageManifest.additional_extenders == {}
        assert _ProbeOpenLineageManifest()._extenders() == {"OpenLineageExtender": "openlineage_extender"}

    def test_additional_extenders_follow_the_primary_in_order(self) -> None:
        class _Probe(_ProbeOpenLineageManifest):
            additional_extenders = {"OtherExtender": "other_extender"}

        assert list(_Probe()._extenders().items()) == [
            ("OpenLineageExtender", "openlineage_extender"),
            ("OtherExtender", "other_extender"),
        ]
