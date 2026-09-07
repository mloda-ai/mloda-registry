"""Manifest resilience for mloda-community-openlineage; contract lives in OptionalDependencyPackageTestMixin."""

from __future__ import annotations

from mloda.testing.optional_dependency import OptionalDependencyPackageTestMixin


class TestOpenLineageManifest(OptionalDependencyPackageTestMixin):
    package = "mloda.community.extenders.openlineage"
    root = "openlineage"
    distribution = "openlineage-python"
    extra = "mloda-community[openlineage]"
    extender_name = "OpenLineageExtender"
    extender_module = "openlineage_extender"
