"""``BinaryExampleFeatureGroup``: the enterprise example FeatureGroup that mixes in
``BinaryModelMixin`` to run the "hash" operation via an external binary (pattern 28, Binary-Backed
Features; see ``docs/guides/feature-group-patterns/28-binary-backed-features.md``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pyarrow as pa

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet, property_spec
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.binary_model.mixin import BinaryModelMixin


def _is_column_list(value: object) -> bool:
    """A non-empty, ordered list or tuple of non-empty strings."""
    return isinstance(value, (list, tuple)) and bool(value) and all(isinstance(item, str) and item for item in value)


def _is_parameters_mapping(value: object) -> bool:
    """A dict with string keys."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


class BinaryExampleFeatureGroup(BinaryModelMixin, FeatureGroup):
    """Keyed hash of the configured columns, computed by the ``example_binary`` wheel."""

    BINARY_PLUGIN_ID = "example_binary"
    OUTPUT_KEY = "result"
    OPERATION = "binary_operation"
    INPUT_COLUMNS = "binary_input_columns"
    PARAMETERS = "binary_parameters"

    PROPERTY_MAPPING: ClassVar = {
        OPERATION: property_spec("Operation the binary runs", strict=True, allowed_values={"hash": "Keyed hash"}),
        INPUT_COLUMNS: property_spec(
            "Frame columns sent to the binary, in operation order", match_guard=_is_column_list
        ),
        PARAMETERS: property_spec(
            "Operation parameters passed through unchanged", default=None, match_guard=_is_parameters_mapping
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature.not_typed(name) for name in options.get(self.INPUT_COLUMNS)}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        if options is None:
            return False
        if options.get(cls.OPERATION) != "hash":
            return False
        input_columns = options.get(cls.INPUT_COLUMNS)
        if not _is_column_list(input_columns):
            return False
        parameters = options.get(cls.PARAMETERS)
        if parameters is not None and not _is_parameters_mapping(parameters):
            return False
        if str(feature_name) in input_columns:
            return False
        return True

    @classmethod
    def calculate_feature(cls, data: pa.Table, features: FeatureSet) -> pa.Table:
        for feature in features.features:
            columns: Sequence[str] = feature.options.get(cls.INPUT_COLUMNS)
            parameters: Mapping[str, Any] = feature.options.get(cls.PARAMETERS) or {}
            operation = feature.options.get(cls.OPERATION)
            result = cls.run_binary_model(data, columns, operation, parameters, {cls.OUTPUT_KEY: feature.name})
            data = data.append_column(feature.name, result.column(feature.name))
        return data
