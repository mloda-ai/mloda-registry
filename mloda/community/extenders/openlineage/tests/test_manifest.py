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
    # A real submodule/name openlineage_extender.py imports from: simulates an installed-but-too-old
    # openlineage-python (the widened __init__.py guard must also catch a plain ImportError here).
    broken_module = "openlineage.client.client"
    broken_name = "OpenLineageClient"
