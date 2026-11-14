"""Enterprise FeatureGroup that pseudonymizes a utf8 column with keyed HMAC-SHA256 via ``anonymizer_binary``.

The key is the ``key`` field of the credential named by ``pii_key_handle``, never an option value.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import pyarrow as pa
from mloda.provider import (
    ComputeFramework,
    CoreDeclineCode,
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    credential_handle_spec,
    escalate_match_abort,
    property_spec,
    read_credential,
    record_match_rejection,
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
    # Remove two minor releases after the first release that ships pii_key_handle.
    REMOVED_KEY_ENV = "pii_key_env"

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
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        addressed = re.match(cls.PREFIX_PATTERN, str(getattr(feature_name, "name", feature_name))) is not None
        if addressed or options.get(cls.ALGORITHM) is not None:
            if options.get(cls.REMOVED_KEY_ENV) is not None:
                # Marked: user misconfiguration; containing it would let a rival group win.
                raise escalate_match_abort(
                    ValueError(
                        f"{cls.REMOVED_KEY_ENV} was removed: register Credential(key=...) in the "
                        f"DataAccessCollection and name its handle in the group option {cls.KEY_HANDLE}."
                    )
                )
        matched = super().match_feature_group_criteria(feature_name, options, data_access_collection)
        # Recorded after super so its own rejection (e.g. an unsupported algorithm) wins the first-record slot.
        if not matched and options.get(cls.ALGORITHM) is not None and options.get(cls.KEY_HANDLE) is None:
            record_match_rejection(
                cls.__name__, f"{cls.KEY_HANDLE} is not set", code=CoreDeclineCode.REQUIRED_OPTION_ABSENT
            )
        return matched

    @classmethod
    def calculate_feature(cls, data: pa.Table, features: FeatureSet) -> pa.Table:
        for feature in features.features:
            algorithm = cls._resolve_operation(feature, cls.ALGORITHM)
            if algorithm is None:
                raise BinaryUsageError(f"{cls.ALGORITHM} is required")
            source = cls._extract_single_source_feature(feature)
            key = cls._read_key(feature.options.get(cls.KEY_HANDLE), feature.name)
            result = cls.run_binary_model(data, [source], algorithm, {"key": key}, {cls.OUTPUT_KEY: feature.name})
            data = data.append_column(feature.name, result.column(feature.name))
        return data

    @classmethod
    def _read_key(cls, handle: str, feature_name: str) -> str:
        """The key field of the named credential; errors never echo the value."""
        value = read_credential(handle)["key"]
        prefix = (
            f"The 'key' field of credential '{handle}' ({cls.KEY_HANDLE}) for feature '{feature_name}' must be "
            "64 hex characters"
        )
        if isinstance(value, str) and value != value.strip():
            raise BinaryUsageError(
                f"{prefix}, but it has leading or trailing whitespace: strip it before registering the Credential"
            )
        if not isinstance(value, str) or not _KEY_PATTERN.fullmatch(value):
            raise BinaryUsageError(prefix)
        return value
