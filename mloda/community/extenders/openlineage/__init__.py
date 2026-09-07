"""mloda-community-openlineage: OpenLineage RunEvents for mloda pipelines."""

__all__: list[str] = []

# The mloda-community bundle ships this extender behind the mloda-community[openlineage]
# extra; core's loader would otherwise raise on the missing openlineage-python dependency.
try:
    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
except ImportError as exc:
    from ._optional_dependency import missing_attribute, reraise_unless_optional

    reraise_unless_optional(exc, "openlineage")
    __getattr__ = missing_attribute("OpenLineageExtender", "openlineage-python", "mloda-community[openlineage]", exc)
else:
    __all__ = ["OpenLineageExtender"]
