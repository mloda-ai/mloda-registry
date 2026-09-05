"""mloda-community-openlineage: OpenLineage RunEvents for mloda pipelines."""

__all__: list[str] = []

# The mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra; core's loader would otherwise raise on the missing openlineage-python dependency.
try:
    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] != "openlineage":
        raise
else:
    __all__ = ["OpenLineageExtender"]
