"""Enterprise FeatureGroup that pseudonymizes a utf8 column with keyed HMAC-SHA256 via ``anonymizer_binary``.

The key is read from the environment variable named by ``pii_key_env``, never from an option value.
"""

from __future__ import annotations

import os
import re
from typing import ClassVar

import pyarrow as pa
from mloda.provider import (
    ComputeFramework,
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    property_spec,
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
    KEY_ENV = "pii_key_env"

    PREFIX_PATTERN = r".+__(hmac_sha256)_pseudonymized$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    # KEY_ENV is neither strict nor guarded, so no validator can echo a mistaken value (which may be the key).
    PROPERTY_MAPPING: ClassVar = {
        ALGORITHM: property_spec(
            "Pseudonymization algorithm",
            strict=True,
            allowed_values={"hmac_sha256": "Keyed HMAC-SHA256, 64 lowercase hex characters"},
        ),
        KEY_ENV: property_spec("Name of the environment variable holding the 64-hex-character key"),
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
            key = cls._read_key(feature.options.get(cls.KEY_ENV))
            result = cls.run_binary_model(data, [source], algorithm, {"key": key}, {cls.OUTPUT_KEY: feature.name})
            data = data.append_column(feature.name, result.column(feature.name))
        return data

    @classmethod
    def _read_key(cls, env_name: object) -> str:
        """The key from the named variable; errors never echo the name or the value."""
        key = os.environ.get(env_name, "").strip() if isinstance(env_name, str) and env_name else ""
        if not _KEY_PATTERN.fullmatch(key):
            raise BinaryUsageError(
                f"{cls.KEY_ENV} must name a set environment variable holding the 64-hex-character key"
            )
        return key
