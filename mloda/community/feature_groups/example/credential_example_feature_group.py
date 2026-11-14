"""Credential Example FeatureGroup: signs a column with a key read from a credential handle."""

import hashlib
import hmac
from typing import Any, ClassVar

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    credential_handle_spec,
    property_spec,
    read_credential,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class CredentialExampleFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Signs the input column with HMAC-SHA256 using the 'key' field of a credential."""

    KEY_HANDLE = "signing_key_handle"
    PREFIX_PATTERN = r".+__hmac_signed$"
    RECOGNITION_ONLY_PATTERN = True
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING: ClassVar = {
        KEY_HANDLE: credential_handle_spec("Credential handle whose 'key' field signs", required_fields=("key",)),
        DefaultOptionKeys.in_features: property_spec("Column to sign"),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[Any]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            key = read_credential(feature.options.get(cls.KEY_HANDLE))["key"].encode()
            source = cls._extract_single_source_feature(feature)
            data[feature.name] = [
                None if value is None else hmac.new(key, value.encode(), hashlib.sha256).hexdigest()
                for value in data[source]
            ]
        return data
