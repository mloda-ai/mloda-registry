"""Enterprise FeatureGroup that pseudonymizes a utf8 column with keyed HMAC-SHA256 via ``anonymizer_binary``.

The key is the ``key`` field of the credential named by ``pii_key_handle``, never an option value.
"""

from __future__ import annotations

import re
from typing import ClassVar

import pyarrow as pa
from mloda.provider import (
    ComputeFramework,
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    credential_handle_spec,
    property_spec,
    read_credential,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.binary_model.errors import BinaryUsageError
from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin

_KEY_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


class AnonymizerFeatureGroup(BinaryModelMixin, FeatureChainParserMixin, FeatureGroup):
    """Keyed HMAC-SHA256 pseudonym of one utf8 column, computed by the ``anonymizer_binary`` wheel."""

    BINARY_PLUGIN_ID = "anonymizer_binary"
    BINARY_WHEEL_DISTRIBUTION = "mloda-anonymizer-binary"
    BINARY_INSTALL_EXTRA = "mloda-enterprise[anonymizer]"
    OUTPUT_KEY = "result"
    ALGORITHM = "pseudonymization_algorithm"
    KEY_HANDLE = "pii_key_handle"

    PREFIX_PATTERN = r".+__(hmac_sha256)_pseudonymized$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING: ClassVar = {
        ALGORITHM: property_spec(
            "Pseudonymization algorithm",
            strict=True,
            allowed_values={"hmac_sha256": "Keyed HMAC-SHA256, 64 lowercase hex characters"},
        ),
        KEY_HANDLE: credential_handle_spec(
            "Credential handle whose 'key' field holds the 64-hex-character key", required_fields=("key",)
        ),
        DefaultOptionKeys.in_features: property_spec("Single utf8 source column to pseudonymize"),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: pa.Table, features: FeatureSet) -> pa.Table:
        for feature in features.features:
            algorithm = cls._resolve_operation(feature, cls.ALGORITHM)
            if algorithm is None:
                raise BinaryUsageError(f"{cls.ALGORITHM} is required")
            source = cls._extract_single_source_feature(feature)
            key = cls._read_key(feature.options.get(cls.KEY_HANDLE))
            result = cls.run_binary_model(data, [source], algorithm, {"key": key}, {cls.OUTPUT_KEY: feature.name})
            data = data.append_column(feature.name, result.column(feature.name))
        return data

    @classmethod
    def _read_key(cls, handle: object) -> str:
        """The key field of the named credential; errors never echo the value."""
        if not isinstance(handle, str) or not handle:
            raise BinaryUsageError(f"{cls.KEY_HANDLE} must be a non-empty string")
        value = read_credential(handle).get("key")
        key = value.strip() if isinstance(value, str) else ""
        if not _KEY_PATTERN.fullmatch(key):
            raise BinaryUsageError(
                f"The 'key' field of the credential named by {cls.KEY_HANDLE} must be 64 hex characters"
            )
        return key
