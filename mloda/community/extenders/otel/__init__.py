"""mloda-community-otel: OpenTelemetry spans for mloda pipelines."""

__all__: list[str] = []

# The mloda-community bundle ships this extender behind the mloda-community[otel]
# extra; core's loader would otherwise raise on the missing opentelemetry-api dependency.
# Widened from ModuleNotFoundError to ImportError, and from a bare exc.name prefix check to
# blames_root(): an installed-but-broken opentelemetry-api can fail either directly (a missing
# attribute inside opentelemetry itself) or via one of ITS OWN transitive dependencies (e.g.
# typing_extensions), whose failure is named after that dependency, not "opentelemetry" - both
# must still degrade.
try:
    from mloda.community.extenders.otel.otel_extender import OtelExtender
except ImportError as exc:
    from ._missing_dependency import blames_root, build_getattr

    if not blames_root(exc, "opentelemetry"):
        raise
    __getattr__ = build_getattr(exc)
else:
    __all__ = ["OtelExtender"]
