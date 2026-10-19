"""Base class for binning operation feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParser,
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
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


class BinningFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__(bin|qbin)_[1-9]\d*$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    BINNING_OP = "binning_op"
    N_BINS = "n_bins"

    PROPERTY_MAPPING = {
        BINNING_OP: property_spec(
            "Binning operation applied to the source column",
            strict=True,
            allowed_values=BINNING_OPS,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        # deferred_binding stays True: see _validate_n_bins_name_agreement for why.
        N_BINS: property_spec(
            "Number of bins (positive integer)",
            match_guard=is_positive_int,
            expected=POSITIVE_INT_EXPECTED,
            deferred_binding=True,
        ),
        DefaultOptionKeys.in_features: property_spec(
            "Source numeric column",
        ),
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        return operation_config in BINNING_OPS

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Any,
        _data_access_collection: Any = None,
    ) -> bool:
        if not super().match_feature_group_criteria(feature_name, options, _data_access_collection):
            return False
        cls._validate_n_bins_name_agreement(feature_name, options)
        return True

    @classmethod
    def _validate_n_bins_name_agreement(cls, feature_name: Any, options: Any) -> None:
        """Run core's name-agreement checks on the hand-parsed suffix, which its capture bindings never see."""
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(str(feature_name), prefix_patterns)
        if operation_config is None or source_feature is None:
            return
        bindings = {cls.N_BINS: str(cls._name_n_bins(str(feature_name)))}
        cls._validate_forwarded_name_mismatch(feature_name, bindings, options)
        cls._validate_name_agreement(feature_name, bindings, None, options)

    @staticmethod
    def _name_n_bins(feature_name: str) -> int:
        """The digit suffix PREFIX_PATTERN anchors on, e.g. 5 in ``value__bin_5``."""
        return int(feature_name.rsplit("_", 1)[-1])

    @classmethod
    def _validate_n_bins(cls, n_bins: int, feature_name: str) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins} (feature: {feature_name})")

    @classmethod
    def get_binning_params(cls, feature_name: str) -> tuple[str, int]:
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None and source_feature is not None:
            n_bins = cls._name_n_bins(feature_name)
            cls._validate_n_bins(n_bins, feature_name)
            return operation_config, n_bins
        raise ValueError(f"Could not extract binning parameters from feature name: {feature_name}")

    @classmethod
    def _extract_binning_params(cls, feature: Feature) -> tuple[str, int]:
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)
        if operation_config is not None:
            n_bins = cls._name_n_bins(feature_name)
            cls._validate_n_bins(n_bins, feature_name)
            return operation_config, n_bins
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
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            op, n_bins = cls._extract_binning_params(feature)

            table = cls._compute_binning(table, feature_name, source_col, op, n_bins)

        return table

    @classmethod
    def _compute_binning(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> Any:
        raise NotImplementedError
