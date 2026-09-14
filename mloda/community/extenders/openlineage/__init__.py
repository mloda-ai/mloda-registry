"""mloda-community-openlineage: OpenLineage RunEvents for mloda pipelines."""

__all__: list[str] = []

# The mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra; core's loader would otherwise raise on the missing openlineage-python dependency.
# Widened from ModuleNotFoundError to ImportError, and from a bare exc.name prefix check to
# blames_root(): an installed-but-broken openlineage-python can fail either directly (a missing
# attribute inside openlineage itself) or via one of ITS OWN transitive dependencies (e.g. attr),
# whose failure is named after that dependency, not "openlineage" - both must still degrade.
try:
    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
except ImportError as exc:
    from ._missing_dependency import blames_root, build_getattr

    if not blames_root(exc, "openlineage"):
        raise
    __getattr__ = build_getattr(exc)
else:
    __all__ = ["OpenLineageExtender"]
