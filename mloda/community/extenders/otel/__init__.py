"""mloda-community-otel: OpenTelemetry spans for mloda pipelines."""

__all__: list[str] = []

# The mloda-community bundle ships this extender behind the mloda-community[otel]
# extra; core's loader would otherwise raise on the missing opentelemetry-api dependency.
try:
    from mloda.community.extenders.otel.otel_extender import OtelExtender
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] != "opentelemetry":
        raise
    from ._missing_dependency import __getattr__  # noqa: F401
else:
    __all__ = ["OtelExtender"]
