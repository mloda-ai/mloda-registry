"""mloda-community-otel: OpenTelemetry spans for mloda pipelines."""

__all__: list[str] = []

# The mloda-community bundle ships this extender behind the mloda-community[otel]
# extra; core's loader would otherwise raise on the missing opentelemetry-api dependency.
# Uses ImportError (not just ModuleNotFoundError) and blames_root(): an installed-but-broken
# opentelemetry-api can fail directly or via one of its own transitive dependencies, whose
# failure is named after that dependency, not "opentelemetry" - both must still degrade.
try:
    from mloda.community.extenders.otel.otel_extender import OtelExtender
except ImportError as exc:
    from ._missing_dependency import blames_root, build_getattr

    if not blames_root(exc, "opentelemetry"):
        raise
    __getattr__ = build_getattr(exc)
else:
    __all__ = ["OtelExtender"]
