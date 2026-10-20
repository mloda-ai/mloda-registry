"""Custom OpenLineage facets. Imported lazily by the extender (see openlineage_extender.py)."""

from __future__ import annotations

import importlib.metadata

import attr

from openlineage.client.facet_v2 import DatasetFacet, RunFacet


def versioned_schema_url(distribution: str, module_path: str) -> str:
    """Return the GitHub URL of module_path at the installed release tag of distribution, or main if unknown."""
    try:
        ref = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        ref = ""
    return f"https://github.com/mloda-ai/mloda-registry/blob/{ref or 'main'}/{module_path}"


# The release tag pins the facet shape a stored event names.
_SCHEMA_URL = versioned_schema_url("mloda-community-openlineage", "mloda/community/extenders/openlineage/_facets.py")


@attr.define
class MlodaDataAccessFacet(DatasetFacet):
    identityIsFallback: bool = attr.field()

    @staticmethod
    def _get_schema() -> str:
        # The module in this repo, not a hosted JSON schema.
        return _SCHEMA_URL


@attr.define
class MlodaDeclaredDatasetFacet(DatasetFacet):
    attributes: dict[str, str | int | float | bool] = attr.field()
    conflicts: list[str] = attr.field(factory=list)

    @staticmethod
    def _get_schema() -> str:
        return _SCHEMA_URL


@attr.define
class MlodaPlanRunFacet(RunFacet):
    planId: str = attr.field()
    structureHash: str | None = attr.field(default=None)
    origin: str | None = attr.field(default=None)
    requestedFeatures: list[str] | None = attr.field(default=None)

    @staticmethod
    def _get_schema() -> str:
        return _SCHEMA_URL


@attr.define
class MlodaStepRunFacet(RunFacet):
    featureGroupVersion: str | None = attr.field(default=None)
    pluginVersion: str | None = attr.field(default=None)
    computeFramework: str | None = attr.field(default=None)
    featureNames: list[str] | None = attr.field(default=None)

    @staticmethod
    def _get_schema() -> str:
        return _SCHEMA_URL


@attr.define
class MlodaTraceRunFacet(RunFacet):
    traceId: str = attr.field()
    spanId: str = attr.field()

    @staticmethod
    def _get_schema() -> str:
        return _SCHEMA_URL
