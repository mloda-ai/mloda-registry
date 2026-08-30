"""Base class for binning operation feature groups."""

from __future__ import annotations

import logging
import os
from typing import Any

from mloda.core.abstract_plugins.components.utils import escalate_match_abort  # no public equivalent yet
from mloda.provider import DefaultOptionKeys, FeatureChainParser, FeatureGroup, FeatureSet, property_spec
from mloda.user import DataType, Feature, FeatureName, Options

from mloda.community.feature_groups.data_operations.base import (
    RejectionReasonMixin,
    is_op_token,
    is_positive_int,
    op_token_value,
    positive_int_value,
)

logger = logging.getLogger(__name__)

BINNING_OPS = {
    "bin": "Equal-width binning (value range divided into n equal intervals)",
    "qbin": "Quantile-based binning (rows divided into n roughly equal groups by rank)",
}


class BinningFeatureGroup(RejectionReasonMixin, FeatureGroup):
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
        ),
        # deferred_binding stays True: see _validate_forwarded_n_bins_mismatch for why.
        N_BINS: property_spec(
            "Number of bins (positive integer)",
            match_guard=is_positive_int,
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
        cls._validate_forwarded_n_bins_mismatch(feature_name, options)
        return True

    @classmethod
    def _validate_forwarded_n_bins_mismatch(cls, feature_name: Any, options: Any) -> None:
        """Reject a forwarded ``n_bins`` that contradicts the name-parsed value.

        The name's digit suffix is never captured by PREFIX_PATTERN (it is parsed by hand in
        get_binning_params), and its match_guard (is_positive_int) rejects raw capture text anyway,
        so n_bins can't bind by name like binning_op does; this compares the parsed int directly.
        """
        if cls.N_BINS not in (options.inherited_group_keys or ()):
            return
        prefix_patterns = cls._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(str(feature_name), prefix_patterns)
        if operation_config is None or source_feature is None:
            return
        name_n_bins = cls._name_n_bins(str(feature_name))
        forwarded = options.get(cls.N_BINS)
        if forwarded is None:
            return
        forwarded_n_bins = positive_int_value(forwarded)
        if forwarded_n_bins == name_n_bins:
            return
        message = (
            f"Feature '{feature_name}': option '{cls.N_BINS}' was forwarded from a consumer with value "
            f"'{forwarded}', but the feature name parses to {name_n_bins}. The name-parsed value takes "
            f"precedence, so the forwarded value would be silently ignored. Carve the key out with "
            f"forward_group_exclude={{'{cls.N_BINS}'}} on the child in the consumer's input_features, or use "
            f"an allowlist / forward_group=False. Set MLODA_ALLOW_FORWARDED_NAME_MISMATCH=1 to downgrade this "
            f"error to a warning."
        )
        if os.environ.get("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "").lower() in ("1", "true"):
            logger.warning(message)
            return
        # Marked: user misconfiguration; containing it would let a rival group win with the value ignored.
        raise escalate_match_abort(ValueError(message))

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

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        _feature_name = str(feature_name)

        prefix_patterns = self._get_prefix_patterns()
        operation_config, source_feature = FeatureChainParser.parse_feature_name(_feature_name, prefix_patterns)

        if operation_config is not None and source_feature is not None and source_feature:
            in_features = [Feature(source_feature)]
            self._validate_in_feature_count(in_features, _feature_name)
            return set(in_features)

        in_features_set = options.get_in_features()
        self._validate_in_feature_count(list(in_features_set), _feature_name)
        return set(in_features_set)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        feature_name = feature.name
        prefix_patterns = cls._get_prefix_patterns()

        operation_config, source_feature = FeatureChainParser.parse_feature_name(feature_name, prefix_patterns)

        if operation_config is not None and source_feature is not None and source_feature:
            return [source_feature]

        in_features_set = feature.options.get_in_features()
        return [str(f.name) for f in in_features_set]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        table = data

        for feature in features.features:
            feature_name = feature.name

            source_features = cls._extract_source_features(feature)
            source_col = source_features[0]
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
