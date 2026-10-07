"""Custom OpenLineage facets. Imported lazily by the extender (see openlineage_extender.py)."""

from __future__ import annotations

import attr

from openlineage.client.facet_v2 import DatasetFacet, RunFacet

_SCHEMA_URL = "https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/extenders/openlineage/_facets.py"


@attr.define
class MlodaDataAccessFacet(DatasetFacet):
    identityIsFallback: bool = attr.field()

    @staticmethod
    def _get_schema() -> str:
        # The module in this repo, not a hosted JSON schema.
        return _SCHEMA_URL


@attr.define
class MlodaPlanRunFacet(RunFacet):
    planId: str = attr.field()
    structureHash: str | None = attr.field(default=None)

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
