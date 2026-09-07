"""Manifest resilience for mloda-community-otel; contract lives in OptionalDependencyPackageTestMixin."""

from __future__ import annotations

from mloda.testing.optional_dependency import OptionalDependencyPackageTestMixin


class TestOtelManifest(OptionalDependencyPackageTestMixin):
    package = "mloda.community.extenders.otel"
    root = "opentelemetry"
    distribution = "opentelemetry-api"
    extra = "mloda-community[otel]"
    extender_name = "OtelExtender"
    extender_module = "otel_extender"
    broken_module = "opentelemetry.trace"
    broken_name = "NonRecordingSpan"
    transitive_dependency = "typing_extensions"
