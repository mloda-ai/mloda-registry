"""Base class for binning operation feature groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureSet,
    OperationFeatureGroup,
    SubtypeDeclaration,
    property_spec,
)
from mloda.user import DataType, Feature

from mloda.community.feature_groups.data_operations.base import (
    OP_TOKEN_EXPECTED,
    POSITIVE_INT_EXPECTED,
    assert_source_columns_present,
    is_op_token,
    is_positive_int,
    op_token_value,
    positive_int_value,
)

BINNING_OPS = {
    "bin": "Equal-width binning (value range divided into n equal intervals)",
    "qbin": "Quantile-based binning (rows divided into n roughly equal groups by rank)",
}


class BinningFeatureGroup(FeatureChainParserMixin, OperationFeatureGroup):
    """Base class for element-wise binning operations that preserve row count.

    Supported operations: ``bin`` (equal-width) and ``qbin`` (quantile).

    Reference Contract
    ------------------

    Null input gives null output; the PyArrow implementation is the reference. Row count and order are
    preserved. ``bin`` and ``qbin`` declare INT64. Each implementation passes ``compute_binning(table,
    feature_name, source_col, op, n_bins)`` to the group; it returns the input with the new column appended.
    """

    BINNING_OP = "binning_op"
    N_BINS = "n_bins"

    PREFIX_PATTERN = rf".*__(?P<{BINNING_OP}>bin|qbin)_(?P<{N_BINS}>[1-9]\d*)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    SUBTYPES = SubtypeDeclaration(key=BINNING_OP)

    PROPERTY_MAPPING = {
        BINNING_OP: property_spec(
            "Binning operation applied to the source column",
            strict=True,
            allowed_values=BINNING_OPS,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        N_BINS: property_spec(
            "Number of bins (positive integer)",
            match_guard=is_positive_int,
            expected=POSITIVE_INT_EXPECTED,
            name_converter=int,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Source numeric column",
        ),
    }

    @classmethod
    def _validate_n_bins(cls, n_bins: int, feature_name: str) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins} (feature: {feature_name})")

    @classmethod
    def get_binning_params(cls, feature_name: str) -> tuple[str, int]:
        bindings = cls.resolve_feature_name(feature_name).bindings
        op, n_bins = bindings.get(cls.BINNING_OP), bindings.get(cls.N_BINS)
        if op is not None and n_bins is not None:
            cls._validate_n_bins(n_bins, feature_name)
            return op, n_bins
        raise ValueError(f"Could not extract binning parameters from feature name: {feature_name}")

    @classmethod
    def _extract_binning_params(cls, feature: Feature) -> tuple[str, int]:
        feature_name = feature.name
        resolution = cls.resolve_feature_name(feature_name)
        if resolution.owned:
            name_op, name_n_bins = resolution.bindings.get(cls.BINNING_OP), resolution.bindings.get(cls.N_BINS)
            if name_op is None or name_n_bins is None:
                raise ValueError(f"Could not extract binning parameters for {feature_name}")
            cls._validate_n_bins(name_n_bins, feature_name)
            return name_op, name_n_bins
        op = feature.options.get(cls.BINNING_OP)
        n = feature.options.get(cls.N_BINS)
        if op is None or n is None:
            raise ValueError(f"Could not extract binning parameters for {feature_name}")
        # Unwrap, then coerce: the guard keeps a float out at match time, a direct call still gets an int.
        n_bins = int(positive_int_value(n))
        cls._validate_n_bins(n_bins, feature_name)
        return op_token_value(op), n_bins

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        """Declare INT64: both bin and qbin emit integer bin indices."""
        op, _ = cls._extract_binning_params(feature)
        if op in {"bin", "qbin"}:
            return DataType.INT64
        return None

    @classmethod
    def assign_bins(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_binning: Callable[[Any, str, str, str, int], Any],
    ) -> Any:
        """Shared loop: extract params from each feature, delegate to ``compute_binning``."""
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            op, n_bins = cls._extract_binning_params(feature)

            table = compute_binning(table, feature_name, source_col, op, n_bins)

        return table
