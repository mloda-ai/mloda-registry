"""mloda-community-otel: OpenTelemetry spans for mloda pipelines."""

__all__: list[str] = []

# The mloda-community bundle ships this extender behind the mloda-community[otel]
# extra; core's loader would otherwise raise on the missing opentelemetry-api dependency.
try:
    from mloda.community.extenders.otel.otel_extender import OtelExtender
except ImportError as exc:
    from ._optional_dependency import missing_attribute, reraise_unless_optional

    reraise_unless_optional(exc, "opentelemetry")
    __getattr__ = missing_attribute("OtelExtender", "opentelemetry-api", "mloda-community[otel]", exc)
else:
    __all__ = ["OtelExtender"]
