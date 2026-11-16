"""Test source group serving `alpha` from a credentialed URI. Kept out of `runners` so importing the runners does not
register a source group for every unrestricted PluginCollector run."""

from __future__ import annotations

from typing import Any, ClassVar

import pyarrow as pa
from mloda.provider import (
    FeatureSet,
    Implementation,
    ReadFileFG,
    SourceMatch,
    credential_handle_spec,
    read_credential,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


# Module-level so MULTIPROCESSING can pickle it by path. The unique suffix keeps it from claiming a host's files.
class MlodaTestingUriFileFG(ReadFileFG):
    """File group serving an `alpha` column from a URI pointer; used to read a source whose URI carries credentials."""

    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return (".mlodatesting",)

    @classmethod
    def column_names(cls, path: str) -> list[str]:
        return ["alpha"]

    @classmethod
    def columns(cls, match: SourceMatch) -> list[str]:
        return ["alpha"]

    @classmethod
    def has_column(cls, match: SourceMatch, column: str) -> bool:
        return column == "alpha"


class MlodaTestingUriFilePyArrowTable(Implementation[MlodaTestingUriFileFG]):
    feature_group = MlodaTestingUriFileFG
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[MlodaTestingUriFileFG], data: SourceMatch, features: FeatureSet, framework: Any
    ) -> Any:
        return pa.table({"alpha": [1, 3]})


URI_SECRET = "hunter2"  # nosec
CREDENTIALED_URI = f"memory://user:{URI_SECRET}@host/data.mlodatesting"
CREDENTIALED_URI_IDENTITY = "memory://host/data.mlodatesting"


CREDENTIAL_HANDLE = "mlodatesting-token"
CREDENTIAL_TOKEN = "s3cr3t-token-value"  # nosec


class MlodaTestingCredentialedUriFileFG(MlodaTestingUriFileFG):
    """URI file group whose load reads the credential its handle option names."""

    HANDLE_OPTION = "mlodatesting_token_handle"
    PROPERTY_MAPPING: ClassVar = {
        HANDLE_OPTION: credential_handle_spec(
            "Credential handle whose 'token' field the load reads", required_fields=("token",)
        )
    }


class MlodaTestingCredentialedUriFilePyArrowTable(Implementation[MlodaTestingCredentialedUriFileFG]):
    feature_group = MlodaTestingCredentialedUriFileFG
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[MlodaTestingCredentialedUriFileFG], data: SourceMatch, features: FeatureSet, framework: Any
    ) -> Any:
        token = read_credential(features.get_options_key(group.HANDLE_OPTION))["token"]
        return pa.table({"alpha": [1, 3] if token == CREDENTIAL_TOKEN else [-1]})
