"""Manifest resilience for mloda-community-otel; contract lives in OptionalDependencyPackageTestMixin."""

from __future__ import annotations

from mloda.testing.optional_dependency import OptionalDependencyPackageTestMixin


class TestOtelManifest(OptionalDependencyPackageTestMixin):
    package = "mloda.community.extenders.otel"
    root = "opentelemetry"
    extender_name = "OtelExtender"
    extender_module = "otel_extender"
    api_module = "opentelemetry.trace"
